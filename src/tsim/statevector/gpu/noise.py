"""The host tables of the GPU noise kernels.

A noise kernel's block owns ``S`` consecutive shots and keeps their words in
fast memory (or, rows too wide for it, in device memory). The channels with
equal fire probability and equal numbers of outcomes form a group, and within
a block the group's channels make ``S * len(group)`` Bernoulli trials, channel
major (trial ``i`` is channel ``group[i / S]`` on shot ``i % S``). The host cuts
the groups into *segments* of about ``FAULTS_PER_SEGMENT`` expected faults. In
phase 1 the threads walk the segments with geometric skips and record each
fault as an *event*: its row, the record of the alias slot it draws and the
draw's fraction; phase 2 applies the events: it keeps the slot's outcome or
takes its alias and XORs the outcome's flips into the words.

Each launch lays out its channels' outcomes in group order, so a group's
channel ``c`` has its records at ``first + c * outcomes`` and a fault reads one
aligned 32-byte record (and its alias's, in the same channel):

    alias:  bits 0-24 keep threshold (probability * 2^24), bits 25-31 alias slot
    words:  the first two flipped words (16 bits each)
    extra:  bits 0-7 number of flips, bits 8-31 start of the rest in the overflow
    masks:  the first two flip masks

The flips past an outcome's first two are 16-byte ``(word, mask)`` pairs, in
order. The samples follow ``ChannelSampler``'s distribution, not its streams:
every block draws from its own counter-based stream.

Under postselection the channels can be drawn in two passes. Those with an
outcome that flips a bit the screen, the tapes or the postselection read go
first; the others (``deferred_channels``) cannot change whether a shot is kept,
so they can be drawn after the tapes, on the kept shots only. Channels are
independent and their flips commute, so the kept shots' distribution is
unchanged.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from tsim.noise.channels import ChannelSampler, WordTables
from tsim.statevector.shared import pack_bits

if TYPE_CHECKING:
    from tsim.statevector.shots import ShotLayout, ShotPlan

FAULTS_PER_SEGMENT = 4.0
# NOTE: rows take 8 bits of an event.
MAX_BLOCK_SHOTS = 256
# NOTE: a block's event buffer holds at least this many events.
MIN_EVENTS = 256
# NOTE: a record's flip count has 8 bits.
MAX_FLIPS = 255
# NOTE: a kernel draws a skip from 24 bits of a uniform, which leave trials that no
# skip reaches once p is below about 2^-24; a segment of fewer than 2^-12 expected
# faults is drawn from 48 bits instead (the kernels' fine path).
TINY_P = 2.0**-23
FINE_FAULTS = 2.0**-13
DEFER_SHARE = 0.25
"""Share of the expected faults that postselection cannot see, from which they
are drawn on the accepted shots only, after the tapes (``split``)."""

_RECORD = np.dtype(
    [
        ("alias", "<u4"),
        ("words", "<u4"),
        ("extra", "<u4"),
        ("pad", "<u4"),
        ("m0", "<u8"),
        ("m1", "<u8"),
    ],
    align=True,
)
_FLIP = np.dtype([("word", "<u4"), ("pad", "<u4"), ("mask", "<u8")], align=True)


@dataclass(frozen=True)
class NoiseGroups:
    """A set of channels as one launch draws them: their groups, cut into segments.

    Attributes:
        max_events: Events a block records before it applies faults at once.
        segments: ``(first, outcomes, begin, end, log_keep)`` of each segment:
            its group's first record and outcomes per channel, its trial range
            and log(1 - p).
        records: The channels' outcome records, in group order.
        faults: Expected faults per shot.

    """

    max_events: int
    segments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    records: np.ndarray
    faults: float


@dataclass(frozen=True)
class NoiseTables:
    """The host tables of the noise kernels.

    Attributes:
        block_shots: Shots per block (``S``).
        groups: Every channel's groups.
        records: One 32-byte record per fault outcome, in channel order.
        overflow: The ``(word, mask)`` flips past an outcome's first two.
        p_fire: Each channel's fire probability.
        outcome_start: Each channel's first outcome (and the end).
        num_words: Words per shot.
        shared_bytes: Fast memory that a block's words and events share.
        shared_words: Whether a block's words are in that memory; if not, they
            are in device memory and the events have all of it.

    """

    block_shots: int
    groups: NoiseGroups
    records: np.ndarray
    overflow: np.ndarray
    p_fire: np.ndarray
    outcome_start: np.ndarray
    num_words: int
    shared_bytes: int
    shared_words: bool = True

    def subset(self, select: np.ndarray) -> NoiseGroups:
        """Return the groups of the channels ``select`` (a boolean mask)."""
        return _groups(self, select)


def _groups(tables: NoiseTables, select: np.ndarray) -> NoiseGroups:
    p_fire, outcome_start, S = tables.p_fire, tables.outcome_start, tables.block_shots
    outcomes = np.diff(outcome_start)
    faults = float(p_fire[select].sum())
    max_events = min(
        max(MIN_EVENTS, int(2 * faults * S + MIN_EVENTS)),
        (tables.shared_bytes - 8 * tables.num_words * S * tables.shared_words) // 8,
    )
    # NOTE: channels of equal fire probability and outcomes form a group, so a
    # group's channel c has its records at first + c * outcomes.
    chosen = np.flatnonzero(select)
    order = chosen[np.lexsort((outcomes[chosen], p_fire[chosen]))]
    p_sorted, n_sorted = p_fire[order], outcomes[order]
    change = (np.diff(p_sorted, prepend=-1.0) != 0) | (
        np.diff(n_sorted, prepend=-1) != 0
    )
    starts = np.flatnonzero(change)
    group_start = np.append(starts, len(order))
    record_start = np.concatenate([[0], np.cumsum(n_sorted)])
    first, per, begin, end, log_keep = [], [], [], [], []
    for g in range(len(starts)):
        lo, hi = int(group_start[g]), int(group_start[g + 1])
        p = float(p_sorted[lo])
        lk = -np.inf if p >= 1.0 else float(np.log1p(-p))
        trials = S * (hi - lo)
        # NOTE: float32 skips are exact below 2^24 trials only.
        pieces = max(-(-trials >> 24), round(trials * min(p, 1.0) / FAULTS_PER_SEGMENT))
        if p < TINY_P:
            pieces = max(pieces, math.ceil(trials * -lk / FINE_FAULTS))
        cuts = np.linspace(0, trials, pieces + 1).astype(np.int64)
        for a, b in itertools.pairwise(cuts):
            if b > a:
                first.append(int(record_start[lo]))
                per.append(int(n_sorted[lo]))
                begin.append(a)
                end.append(b)
                log_keep.append(lk)
    if max(end, default=0) >= 1 << 31:
        raise ValueError("too many noise channels for the GPU noise sampler")
    # NOTE: the segments with the most expected faults first, so the threads that
    # take a second segment take short ones.
    expected = (np.array(end) - np.array(begin)) * -np.expm1(np.array(log_keep))
    by_work = np.argsort(-expected, kind="stable")
    # NOTE: each chosen channel's outcomes, in group order.
    within = np.arange(int(record_start[-1])) - np.repeat(record_start[:-1], n_sorted)
    records = tables.records[np.repeat(outcome_start[order], n_sorted) + within]
    return NoiseGroups(
        max_events=max_events,
        segments=(
            np.asarray(first, np.int32)[by_work],
            np.asarray(per, np.int32)[by_work],
            np.asarray(begin, np.int32)[by_work],
            np.asarray(end, np.int32)[by_work],
            np.asarray(log_keep, np.float32)[by_work],
        ),
        records=records,
        faults=faults,
    )


def _outcome_hits(flips: WordTables, read: np.ndarray) -> np.ndarray:
    """Return which outcomes flip a bit of ``read`` (as shot words)."""
    hit = (read[flips.word] & flips.mask) != 0
    # NOTE: hits per outcome as differences of running counts.
    per_flip = np.concatenate([[0], np.cumsum(hit)])
    return per_flip[flips.start[1:]] - per_flip[flips.start[:-1]] > 0


def deferred_channels(
    channels: ChannelSampler, flips: WordTables, read: np.ndarray
) -> np.ndarray:
    """Return which channels flip none of the bits ``read`` (as shot words)."""
    hits = np.concatenate([[0], np.cumsum(_outcome_hits(flips, read))])
    start = channels._tables.outcome_start
    return hits[start[1:]] == hits[start[:-1]]


def split(
    tables: NoiseTables,
    channels: ChannelSampler,
    layout: ShotLayout,
    plan: ShotPlan,
    psm_words: np.ndarray,
    observables_from: int | None = None,
) -> tuple[NoiseGroups, NoiseGroups] | None:
    """Return a postselected batch's noise passes ``(screened, deferred)``, or None.

    The deferred channels flip none of the bits that the screen, the tapes or
    the postselection ``psm_words`` read: the postselected outputs, the keys,
    and, with ``observables_from``, the components' outputs from that index on
    (the observables, whose expectations the tapes return). They are drawn
    apart when they are at least ``DEFER_SHARE`` of the expected faults.
    """
    read = np.zeros(layout.num_words, dtype=np.uint64)
    read[: layout.output_words] = psm_words | plan.screen
    for program in plan.programs:
        start = program.key_start
        read[start : start + program.key_words] = ~np.uint64(0)
        np.bitwise_or.at(read, program.ps_word, program.ps_mask.astype(np.uint64))
    if observables_from is not None:
        bits = np.zeros(64 * layout.output_words, dtype=bool)
        for component in layout.components:
            for index in component.output_indices:
                if index >= observables_from:
                    bits[index] = True
        read[: layout.output_words] |= pack_bits(bits, layout.output_words)
    deferred = deferred_channels(channels, layout.noise, read)
    later = tables.subset(deferred)
    if later.faults > 0 and later.faults >= DEFER_SHARE * tables.groups.faults:
        return tables.subset(~deferred), later
    return None


def screen_pass_rate(
    channels: ChannelSampler, layout: ShotLayout, plan: ShotPlan
) -> float:
    """Return the estimated share of shots that pass ``plan``'s screen."""
    screen = np.zeros(layout.num_words, dtype=np.uint64)
    screen[: layout.output_words] = plan.screen
    return pass_rate(channels, layout.noise, screen) if screen.any() else 1.0


def words_fit(num_words: int, shared_bytes: int) -> bool:
    """Return whether rows of ``num_words`` words fit a block's ``shared_bytes``.

    The smallest block (32 shots) must leave room for ``MIN_EVENTS`` events.
    """
    return 8 * 32 * num_words <= shared_bytes - 8 * MIN_EVENTS


def pass_rate(channels: ChannelSampler, flips: WordTables, read: np.ndarray) -> float:
    """Return the share of shots whose faults flip no bit of ``read``, to first order.

    Each channel's chance to flip one is its fire probability times its hitting
    outcomes' probabilities (from its alias table); flips that cancel are not
    counted.
    """
    fire = channels._tables
    n = np.diff(fire.outcome_start)
    per = np.repeat(n, n).astype(np.float64)
    q = fire.alias_prob / per
    np.add.at(q, fire.alias_index, (1 - fire.alias_prob) / per)
    hits = np.concatenate([[0.0], np.cumsum(q * _outcome_hits(flips, read))])
    start = fire.outcome_start
    p = np.clip(fire.p_fire * (hits[start[1:]] - hits[start[:-1]]), 0.0, 1.0)
    return float(np.exp(np.log1p(-np.minimum(p, 1 - 1e-12)).sum()))


def noise_tables(
    channels: ChannelSampler,
    flips: WordTables,
    num_words: int,
    shared_bytes: int,
    shared_words: bool = True,
) -> NoiseTables:
    """Return the noise kernels' tables for rows of ``num_words`` shot words.

    A block's words and events share ``shared_bytes`` of fast memory, or
    without ``shared_words`` the words are in device memory and the events
    have all of it.

    Raises:
        ValueError: If the rows have no words, their words or 256 events do not
            fit, or the noise exceeds the kernels' limits.

    """
    fire = channels._tables
    if len(fire.p_fire) >= 1 << 24:
        raise ValueError("the GPU noise sampler supports fewer than 2^24 channels")
    if num_words >= 1 << 16:
        raise ValueError("the GPU noise sampler supports fewer than 2^16 shot words")
    if num_words == 0:
        raise ValueError("the GPU noise sampler needs at least one shot word")
    block_shots = max(
        32, min(MAX_BLOCK_SHOTS, (shared_bytes // 2 // (8 * num_words)) // 32 * 32)
    )
    if not shared_words:
        block_shots = MAX_BLOCK_SHOTS
        if shared_bytes < 8 * MIN_EVENTS:
            raise ValueError(
                f"the noise kernel's shared memory holds fewer than {MIN_EVENTS} events"
            )
    elif 8 * num_words * block_shots > shared_bytes - 8 * MIN_EVENTS:
        raise ValueError(
            f"{num_words} shot words do not fit the noise kernel's shared memory"
        )
    outcome_start = fire.outcome_start
    if len(flips.start) > 1 and int(np.diff(flips.start).max()) > MAX_FLIPS:
        raise ValueError(
            f"the GPU noise sampler supports outcomes of up to {MAX_FLIPS} flipped words"
        )
    if len(fire.alias_prob) >= 1 << 24:
        raise ValueError("the GPU noise sampler supports fewer than 2^24 outcomes")
    records = np.zeros(len(fire.alias_prob), dtype=_RECORD)
    overflow: list[tuple[int, int]] = []
    for c in range(len(fire.p_fire)):
        lo, hi = int(outcome_start[c]), int(outcome_start[c + 1])
        if hi - lo > 127:
            raise ValueError(
                "the GPU noise sampler supports channels of up to 127 outcomes"
            )
        for k in range(lo, hi):
            keep = (
                1 << 24
                if hi - lo == 1
                else min(round(float(fire.alias_prob[k]) * (1 << 24)), 1 << 24)
            )
            records[k]["alias"] = keep | ((int(fire.alias_index[k]) - lo) << 25)
            j0, j1 = int(flips.start[k]), int(flips.start[k + 1])
            words = flips.word[j0:j1].tolist()
            masks = flips.mask[j0:j1].tolist()
            records[k]["words"] = (words[0] if words else 0) | (
                (words[1] if len(words) > 1 else 0) << 16
            )
            records[k]["m0"] = masks[0] if masks else 0
            records[k]["m1"] = masks[1] if len(masks) > 1 else 0
            records[k]["extra"] = len(words) | (len(overflow) << 8)
            overflow.extend(zip(words[2:], masks[2:], strict=True))
    if len(overflow) >= 1 << 24:
        raise ValueError("too many noise flips for the GPU noise sampler")
    pairs = np.zeros(len(overflow) + 1, dtype=_FLIP)
    if overflow:
        pairs["word"][:-1] = [w for w, _ in overflow]
        pairs["mask"][:-1] = [m for _, m in overflow]
    empty = np.empty(0, np.int32)
    tables = NoiseTables(
        block_shots=block_shots,
        groups=NoiseGroups(0, (empty, empty, empty, empty, empty), records[:0], 0.0),
        records=records,
        overflow=pairs,
        p_fire=fire.p_fire,
        outcome_start=outcome_start,
        num_words=num_words,
        shared_bytes=shared_bytes,
        shared_words=shared_words,
    )
    every = np.ones(len(fire.p_fire), dtype=bool)
    return replace(tables, groups=_groups(tables, every))
