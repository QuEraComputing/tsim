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
from dataclasses import dataclass, replace

import numpy as np

from tsim.noise.channels import ChannelSampler, WordTables

FAULTS_PER_SEGMENT = 4.0
# NOTE: rows take 8 bits of an event.
MAX_BLOCK_SHOTS = 256

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
        max(256, int(2 * faults * S + 256)),
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
        ValueError: If the rows have no words, their words do not fit, or the
            noise exceeds the kernels' limits.

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
    elif 8 * num_words * block_shots > shared_bytes - 8 * 256:
        raise ValueError(
            f"{num_words} shot words do not fit the noise kernel's shared memory"
        )
    outcome_start = fire.outcome_start
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
