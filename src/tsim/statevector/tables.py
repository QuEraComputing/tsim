"""Outcome tables: each common key's output distribution, built once per sampler.

A component's amplitudes depend on a shot only through its key, the f-bit part
of every rotation sign (``SharingLayout.key_rows``), and the branch bits of the
measurements that split lanes. For one key, running the tape on both outcomes of
every such measurement gives every branch and its probability. The outputs
depend on a branch only through the parities of its branch bits, its pattern,
so branches with equal patterns merge into one entry. A shot draws an entry from
its key's table (Walker's alias method) and XORs the entry's pattern, its own
f-bits' pattern and its dormant coins' pattern: no state vector is simulated
per batch.

Tables cover the keys of no fault and of one fault that reaches the key, which
most shots have when noise is weak; a shot with another key is simulated on
lanes, as without tables. With postselection, a fault that a postselected direct
detector catches discards its shot before the component, so only the faults
that pass this screen count. ``table_keys`` picks the keys for a screen when, by
the noise model, building their tables costs less than the lanes they replace.

One store holds a component's tables for every screen. The first batch to meet
a picked key builds its table, and every later batch and call reuses it. Which
shots draw from tables depends only on their keys and the picked keys, and a
table only on its key, so the samples do not depend on which batch built a
table.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import numba
import numpy as np

import tsim.utils.kernels as kernels
from tsim.noise.channels import FireTables
from tsim.statevector.program import (
    MEASURE,
    MEASURE_DIAG,
    MEASURE_DORMANT,
    ROTATE,
    ROTATE_DIAG,
    StatevectorComponent,
)
from tsim.statevector.shared import (
    _TINY,
    SharingLayout,
    _chunks,
    _pack_range,
    _parity,
    _project_range,
    _rotate_range,
    _signs,
    _unpack,
    _weights_range,
    sample_shared_component,
    sharing_layout,
)
from tsim.utils.kernels import jit, kernel
from tsim.utils.rng import GOLDEN, splitmix64

# NOTE: the tables must cost less than what lanes spend on their keys in this
# many shots, in batches of _BATCH: the bookkeeping of every shot, and the tree
# of the key of no fault, as far as a batch's shots reach, in every batch.
_HORIZON = 2**20
_BATCH = 2**16
# NOTE: in amplitude updates, the per-shot grouping and output work of lanes,
# the work of finding every shot's key, which tables add, and the work per
# branch of a build beyond its amplitudes (copies, patterns, merging).
_LANE_OVERHEAD = 64
_KEY_OVERHEAD = 16
_BRANCH_OVERHEAD = 160
# NOTE: one build, and all of a component's tables, hold at most this many bytes.
_BUILD_BYTES = 2**28
# NOTE: draws resolve probabilities down to about 2^-64, so a branch whose
# outcome is less likely than this is never drawn and is dropped.
_NEGLIGIBLE = 2.0**-66
# NOTE: the largest float64 below 2^64: the threshold of a bucket kept in full,
# and the cap of the others, which would round up to 2^64 near 1.
_FULL = 2.0**64 - 2.0**11


class _Store(NamedTuple):
    """The tables of one component; a batch looks up and draws from one store.

    Entry ``e`` of a table of ``n`` entries is drawn as bucket ``e`` with
    probability 1/n, then kept with probability ``threshold[e] / 2^64``, else
    replaced by entry ``alias[e]``.

    Attributes:
        keys: Key of each table, shape ``(T,)``.
        slot_keys: Open-addressing hash of the keys, shape ``(capacity,)``.
        slot_tables: Table of each slot, -1 if empty, shape ``(capacity,)``.
        start: First entry of each table, then the entry count, shape ``(T + 1,)``.
        threshold: Each bucket's chance to keep its entry, times 2^64, shape ``(E,)``.
        alias: Each bucket's other entry, shape ``(E,)``.
        patterns: Each entry's output pattern, 64 outputs per word, shape
            ``(E, words)``.

    """

    keys: np.ndarray
    slot_keys: np.ndarray
    slot_tables: np.ndarray
    start: np.ndarray
    threshold: np.ndarray
    alias: np.ndarray
    patterns: np.ndarray


def _empty_store(num_words: int) -> _Store:
    keys = np.zeros(0, dtype=np.uint64)
    slot_keys, slot_tables = _hashed(keys)
    return _Store(
        keys=keys,
        slot_keys=slot_keys,
        slot_tables=slot_tables,
        start=np.zeros(1, dtype=np.int64),
        threshold=np.zeros(0, dtype=np.uint64),
        alias=np.zeros(0, dtype=np.int64),
        patterns=np.zeros((0, num_words), dtype=np.uint64),
    )


def _packed(bits: np.ndarray) -> np.ndarray:
    """Return the rows of a boolean matrix as patterns, bit o in word o // 64."""
    rows, num_outputs = bits.shape
    words = np.zeros((rows, (num_outputs + 63) // 64), dtype=np.uint64)
    for o in range(num_outputs):
        words[:, o >> 6] |= bits[:, o].astype(np.uint64) << np.uint64(o & 63)
    return words


class OutcomeTables:
    """The outcome tables of one component, shared by every batch and call.

    Attributes:
        component: The compiled component.
        layout: Its sharing layout without postselection: the key, the dormant
            measurements that branch, and those drawn per shot.

    """

    def __init__(self, component: StatevectorComponent) -> None:
        """Start without tables, which batches build as they meet their keys."""
        self.component = component
        self.layout = layout = sharing_layout(component)
        if len(layout.key_rows) > 64:
            raise ValueError("outcome tables need a key of at most 64 bits")
        tape, num_words = component.tape, component.num_words
        num_f = len(tape.f_selection)
        output_bits = _unpack(tape.output_row, num_words)
        self._key_columns = _key_columns(component, layout)
        self._output_columns = _packed(output_bits[:, :num_f].T)
        coins = np.flatnonzero(_unpack(layout.shot_dormant, num_words))
        self._coin_columns = _packed(output_bits[:, coins].T)
        self._store = _empty_store(self._output_columns.shape[1])
        self._lock = threading.Lock()

    @property
    def num_tables(self) -> int:
        """Return the number of tables built so far."""
        return len(self._store.keys)

    def __getstate__(self) -> dict:
        """Return the state to pickle: all but the lock."""
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore a pickled state with a new lock."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def sample(
        self,
        f_params: np.ndarray,
        seed: int,
        layout: SharingLayout,
        keys: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, dict]:
        """Sample one batch, as ``sample_shared_component`` does with ``layout``.

        Shots whose key is in ``keys`` (every key if None) draw from tables, built
        as needed; the others run on lanes. ``layout`` gives the outputs'
        reference bits and the postselected outputs: a shot that one of them
        discards reads ``False`` in the outputs set after the discarding output,
        as if its lane were pruned there.

        Returns:
            ``(samples, max_norm_deviation, stats)``, where ``stats`` counts the
            tables in total, those this batch built, and the shots on lanes,
            plus the lanes' own counts if any.

        """
        selection = self.component.tape.f_selection
        shots, num_words = f_params.shape[0], self._output_columns.shape[1]
        shot_keys, f_patterns = _shot_keys(
            shots * (len(selection) + 1),
            f_params,
            selection,
            self._key_columns,
            self._output_columns,
            kernels.PARALLEL_WORK,
            numba.get_num_threads(),
        )
        picked = None
        if keys is not None:
            picked, _ = _lookup(shot_keys, *_hashed(np.asarray(keys, np.uint64)))
        store = self._store
        tables, _ = _lookup(shot_keys, store.slot_keys, store.slot_tables)
        # NOTE: a table built for other keys is not used, so that which shots
        # take lanes doesn't depend on what earlier calls built.
        if picked is not None:
            tables[picked < 0] = -1
        deviation, built = 0.0, 0
        missing = np.flatnonzero(tables < 0)
        if picked is not None:
            missing = missing[picked[missing] >= 0]
        if len(missing):
            store, deviation, built = self._add(shot_keys, f_params, missing)
            tables, _ = _lookup(shot_keys, store.slot_keys, store.slot_tables)
            if picked is not None:
                tables[picked < 0] = -1
        postselected = layout.prune_first | layout.prune_outputs.any(axis=0)
        draws, coins, lane_seed = _streams(np.uint64(seed))
        samples = _draw(
            shots * (num_words + self.component.num_outputs + 1),
            f_patterns,
            tables,
            store.start,
            store.threshold,
            store.alias,
            store.patterns,
            self._coin_columns,
            _packed(layout.output_const[None, :])[0],
            _packed(postselected[None, :])[0],
            layout.ready,
            draws,
            coins,
            kernels.PARALLEL_WORK,
            numba.get_num_threads(),
        )
        lanes = np.flatnonzero(tables < 0)
        stats = {"tables": len(store.keys), "built": built, "lane shots": len(lanes)}
        if len(lanes):
            lane_samples, lane_deviation, lane_stats = sample_shared_component(
                self.component, layout, f_params[lanes], int(lane_seed)
            )
            samples[lanes] = lane_samples
            deviation = _worse(deviation, lane_deviation)
            stats.update(lane_stats)
        return samples, deviation, stats

    def _add(
        self, keys: np.ndarray, f_params: np.ndarray, missing: np.ndarray
    ) -> tuple[_Store, float, int]:
        """Build the tables of the keys of the ``missing`` shots; return the store."""
        with self._lock:
            store = self._store
            tables, _ = _lookup(keys[missing], store.slot_keys, store.slot_tables)
            missing = missing[tables < 0]
            if not len(missing):
                return store, 0.0, 0
            new_keys, first = np.unique(keys[missing], return_index=True)
            *entries, deviation = self._build(f_params[missing[first]])
            store = _extended(store, new_keys, *entries)
            self._store = store
            return store, deviation, len(new_keys)

    def _build(
        self, f_params: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Return the tables of the keys of ``f_params``, one shot's f-bits each.

        Returns:
            ``(start, threshold, alias, patterns, max_norm_deviation)`` as in
            ``_Store``, with aliases into these entries.

        """
        component, layout = self.component, self.layout
        tape = component.tape
        symbols = np.zeros((len(f_params), component.num_words), dtype=np.uint32)
        _pack_range(f_params, tape.f_selection, symbols, 0, len(f_params))
        splits = int(np.count_nonzero(_splitting(component, layout)))
        chunk = max(1, int(_BUILD_BYTES // (2.0**splits * _branch_bytes(component))))
        work = 2**splits * int(np.sum(1 << tape.extent.astype(np.int64)))
        starts, probabilities, patterns, deviation = [], [], [], 0.0
        for lo in range(0, len(symbols), chunk):
            lanes = symbols[lo : lo + chunk]
            leaves, probability, owner, part_deviation = _branches(
                len(lanes) * work,
                lanes,
                tape,
                layout.factor,
                layout.coupling,
                layout.i_power,
                layout.lane_dormant,
                component.width,
                _NEGLIGIBLE,
                kernels.PARALLEL_WORK,
                numba.get_num_threads(),
            )
            deviation = _worse(deviation, float(part_deviation))
            leaf_patterns = _patterns(leaves & layout.lane_branches, tape.output_row)
            start, entry_probability, entry_patterns = _entries(
                leaf_patterns, probability, owner, len(lanes)
            )
            starts.append(np.diff(start))
            probabilities.append(entry_probability)
            patterns.append(entry_patterns)
        start = np.concatenate([[0], np.cumsum(np.concatenate(starts))])
        start = start.astype(np.int64)
        threshold, alias = _alias(np.concatenate(probabilities), start)
        return start, threshold, alias, np.concatenate(patterns), deviation


def _key_columns(component: StatevectorComponent, layout: SharingLayout) -> np.ndarray:
    """Return the key bits that each of the component's f-bits flips."""
    num_f = len(component.tape.f_selection)
    key_bits = _unpack(layout.key_rows, component.num_words)[:, :num_f]
    powers = np.uint64(1) << np.arange(len(key_bits), dtype=np.uint64)
    columns = np.where(key_bits, powers[:, None], np.uint64(0))
    return np.bitwise_or.reduce(columns, axis=0).astype(np.uint64)


def _worse(deviation: float, other: float) -> float:
    """Return the larger of two norm deviations, NaN if either is NaN."""
    return other if other > deviation or other != other else deviation


def _branch_bytes(component: StatevectorComponent) -> int:
    """Return the memory of one branch of a build: amplitudes, twice, and records."""
    words = (component.num_outputs + 63) // 64
    return 32 * 2**component.width + 128 + 16 * words


def _splitting(component: StatevectorComponent, layout: SharingLayout) -> np.ndarray:
    """Return which actions branch: active measurements and keyed dormant ones."""
    measured = np.isin(component.tape.kind, (MEASURE, MEASURE_DIAG))
    return measured | layout.lane_dormant


def _entries(
    patterns: np.ndarray, probability: np.ndarray, owner: np.ndarray, num_keys: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge each key's branches by pattern.

    Returns:
        ``(start, probability, patterns)`` of the entries, grouped by key.

    """
    rows = np.ascontiguousarray(
        np.concatenate([owner.astype(np.uint64)[:, None], patterns], axis=1)
    )
    _, first, inverse = np.unique(
        rows.view(np.dtype((np.void, 8 * rows.shape[1]))).ravel(),
        return_index=True,
        return_inverse=True,
    )
    weight = np.bincount(inverse.ravel(), weights=probability, minlength=len(first))
    order = np.argsort(owner[first], kind="stable")
    counts = np.bincount(owner[first], minlength=num_keys)
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return start, weight[order], patterns[first[order]]


def _extended(
    store: _Store,
    keys: np.ndarray,
    start: np.ndarray,
    threshold: np.ndarray,
    alias: np.ndarray,
    patterns: np.ndarray,
) -> _Store:
    """Return ``store`` with the tables of ``keys`` appended."""
    all_keys = np.concatenate([store.keys, keys])
    slot_keys, slot_tables = _hashed(all_keys)
    offset = store.start[-1]
    return _Store(
        keys=all_keys,
        slot_keys=slot_keys,
        slot_tables=slot_tables,
        start=np.concatenate([store.start, offset + start[1:]]),
        threshold=np.concatenate([store.threshold, threshold]),
        alias=np.concatenate([store.alias, offset + alias]),
        patterns=np.concatenate([store.patterns, patterns]),
    )


def table_keys(
    component: StatevectorComponent,
    noise: FireTables,
    screened: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return the keys that get outcome tables, or None if lanes cost less.

    ``screened`` holds the f-bits that discard a shot before it reaches the
    component (postselected direct detectors). The tables cover the keys of no
    fault and of one fault that the screen passes (``_fault_keys``). Each costs
    its tree: every action runs on every branch alive at it, at most ``2^s``
    after ``s`` branching measurements, and every branch has its bookkeeping.
    Over ``_HORIZON`` shots, the tables also cost every shot its key, and they
    save the lanes' bookkeeping of every shot they cover and, in every batch,
    the tree of the key of no fault as far as the batch's shots reach.
    """
    layout = sharing_layout(component)
    splitting = _splitting(component, layout)
    branches = 2.0 ** np.count_nonzero(splitting)
    if len(layout.key_rows) > 64 or branches * _branch_bytes(component) > _BUILD_BYTES:
        return None
    keys, covered = _fault_keys(component, layout, noise, screened)
    words = (component.num_outputs + 63) // 64
    if len(keys) * branches * (16 + 8 * words) > _BUILD_BYTES:
        return None
    if len(keys) == 2 ** len(layout.key_rows):
        covered = 1.0
    extent = 2.0 ** component.tape.extent.astype(np.float64)
    alive = 2.0 ** (np.cumsum(splitting) - splitting)
    tree = float(np.sum(alive * extent)) + branches * _BRANCH_OVERHEAD
    batch_tree = float(np.sum(np.minimum(alive, _BATCH) * extent))
    cost = len(keys) * tree + _HORIZON * _KEY_OVERHEAD
    saved = _HORIZON * covered * _LANE_OVERHEAD + _HORIZON / _BATCH * batch_tree
    return keys if cost <= saved else None


def _fault_keys(
    component: StatevectorComponent,
    layout: SharingLayout,
    noise: FireTables,
    screened: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return the keys of no fault and of one passing fault, and roughly their chance.

    A fault passes when it flips none of the ``screened`` f-bits, which discard
    a shot before it reaches the component. The chance is that of a shot that
    passes the screen having no fault that changes the key, or one such fault
    that passes, with channels independent. It neglects shots that pass because
    two caught faults cancel on the screened bits, which can have other keys.
    """
    selection = np.asarray(component.tape.f_selection, dtype=np.int64)
    flip_cols = np.asarray(noise.flip_cols, dtype=np.int64)
    screened = np.zeros(0, np.int64) if screened is None else np.asarray(screened)
    num_columns = 1 + max(
        int(flip_cols.max(initial=-1)),
        int(selection.max(initial=-1)),
        int(screened.max(initial=-1)),
    )
    columns = np.zeros(num_columns, dtype=np.uint64)
    columns[selection] = _key_columns(component, layout)
    prefix = np.concatenate(
        [np.zeros(1, dtype=np.uint64), np.bitwise_xor.accumulate(columns[flip_cols])]
    )
    outcome_key = prefix[noise.flip_start[1:]] ^ prefix[noise.flip_start[:-1]]
    is_screened = np.zeros(num_columns, dtype=np.int64)
    is_screened[screened] = 1
    hits = np.concatenate([[0], np.cumsum(is_screened[flip_cols])])
    detected = hits[noise.flip_start[1:]] > hits[noise.flip_start[:-1]]
    starts = np.zeros(len(noise.cdf), dtype=bool)
    starts[noise.outcome_start[:-1][np.diff(noise.outcome_start) > 0]] = True
    previous = np.concatenate([[0.0], noise.cdf])[: len(noise.cdf)]
    conditional = noise.cdf - np.where(starts, 0.0, previous)
    channel = np.repeat(np.arange(len(noise.p_fire)), np.diff(noise.outcome_start))

    def chance(outcomes: np.ndarray) -> np.ndarray:
        weights = conditional * outcomes
        return noise.p_fire * np.bincount(channel, weights, len(noise.p_fire))

    caught = np.minimum(chance(detected), 1 - 1e-12)
    reaching = np.minimum(chance(~detected & (outcome_key != 0)), 1 - 1e-12 - caught)
    p_none = float(np.exp(np.sum(np.log1p(-caught - reaching) - np.log1p(-caught))))
    p_one = p_none * float(np.sum(reaching / (1 - caught - reaching)))
    passing = outcome_key[~detected]
    keys = np.unique(np.concatenate([np.zeros(1, dtype=np.uint64), passing]))
    return keys, min(1.0, p_none + p_one)


@jit
def _streams(seed):
    """Return a batch's stream starts: entry draws, dormant coins, lanes' seed.

    SplitMix64 counts up from its state, so each stream starts from its own
    mixed output of the seed, and the lanes' executor mixes its seed again. An
    array keeps them uint64: numba types a Python int below 2^63 as int64.
    """
    streams = np.empty(3, dtype=np.uint64)
    _, state = splitmix64(seed)
    for i in range(3):
        state, streams[i] = splitmix64(state)
    return streams


@jit
def _hashed(keys):
    """Return an open-addressing hash of ``keys``: slot keys and slot tables."""
    capacity = 8
    while capacity < 2 * len(keys):
        capacity *= 2
    slot_keys = np.zeros(capacity, dtype=np.uint64)
    slot_tables = np.full(capacity, -1, dtype=np.int64)
    mask = np.uint64(capacity - 1)
    for table in range(len(keys)):
        _, slot = splitmix64(keys[table])
        slot &= mask
        while slot_tables[slot] >= 0:
            slot = (slot + np.uint64(1)) & mask
        slot_keys[slot] = keys[table]
        slot_tables[slot] = table
    return slot_keys, slot_tables


@jit
def _lookup(keys, slot_keys, slot_tables):
    """Return each key's table, -1 for a key without one, and how many lack one."""
    tables = np.empty(len(keys), dtype=np.int64)
    mask = np.uint64(len(slot_keys) - 1)
    misses = 0
    for s in range(len(keys)):
        if s > 0 and keys[s] == keys[s - 1]:
            tables[s] = tables[s - 1]
        else:
            _, slot = splitmix64(keys[s])
            slot &= mask
            while slot_tables[slot] >= 0 and slot_keys[slot] != keys[s]:
                slot = (slot + np.uint64(1)) & mask
            tables[s] = slot_tables[slot]
        misses += tables[s] < 0
    return tables, misses


@jit
def _keys_range(f, selection, key_columns, output_columns, keys, f_patterns, lo, hi):
    """Write the key and the f-bits' output pattern of shots ``lo … hi-1``."""
    num_words = output_columns.shape[1]
    for s in range(lo, hi):
        key = np.uint64(0)
        for w in range(num_words):
            f_patterns[s, w] = 0
        for j in range(len(selection)):
            if f[s, selection[j]]:
                key ^= key_columns[j]
                for w in range(num_words):
                    f_patterns[s, w] ^= output_columns[j, w]
        keys[s] = key


@kernel
def _shot_keys(f, selection, key_columns, output_columns, parallel_work, threads):
    """Return each shot's key and its f-bits' output pattern."""
    shots = f.shape[0]
    keys = np.empty(shots, dtype=np.uint64)
    f_patterns = np.empty((shots, output_columns.shape[1]), dtype=np.uint64)
    chunks = _chunks(shots * (len(selection) + 1), parallel_work, threads)
    if chunks == 1:
        _keys_range(
            f, selection, key_columns, output_columns, keys, f_patterns, 0, shots
        )
    else:
        for c in numba.prange(chunks):
            lo, hi = shots * c // chunks, shots * (c + 1) // chunks
            _keys_range(
                f, selection, key_columns, output_columns, keys, f_patterns, lo, hi
            )
    return keys, f_patterns


@jit
def _patterns(branches, output_row):
    """Return the parities of each branch with the output rows, 64 per word."""
    num_outputs = output_row.shape[0]
    patterns = np.zeros((branches.shape[0], (num_outputs + 63) // 64), np.uint64)
    for b in range(branches.shape[0]):
        for o in range(num_outputs):
            acc = 0
            for w in range(branches.shape[1]):
                acc ^= branches[b, w] & output_row[o, w]
            if _parity(acc):
                patterns[b, o >> 6] |= np.uint64(1) << np.uint64(o & 63)
    return patterns


@jit
def _alias(probability, start):
    """Return Vose's alias tables of each table's entries, as in ``_Store``."""
    threshold = np.empty(len(probability), dtype=np.uint64)
    alias = np.arange(len(probability))
    small = np.empty(len(probability), dtype=np.int64)
    large = np.empty(len(probability), dtype=np.int64)
    for t in range(len(start) - 1):
        first, size = start[t], start[t + 1] - start[t]
        total = 0.0
        for e in range(first, first + size):
            total += probability[e]
        scaled = probability[first : first + size] * (size / total)
        num_small = num_large = 0
        for i in range(size):
            if scaled[i] < 1:
                small[num_small] = i
                num_small += 1
            else:
                large[num_large] = i
                num_large += 1
        while num_small and num_large:
            num_small -= 1
            num_large -= 1
            little, big = small[num_small], large[num_large]
            exact = scaled[little] * 2.0**64
            threshold[first + little] = np.uint64(min(exact + 0.5, _FULL))
            alias[first + little] = first + big
            # NOTE: in this order the rounding error stays small.
            scaled[big] = (scaled[big] + scaled[little]) - 1
            if scaled[big] < 1:
                small[num_small] = big
                num_small += 1
            else:
                large[num_large] = big
                num_large += 1
        for i in small[:num_small]:
            threshold[first + i] = np.uint64(_FULL)
        for i in large[:num_large]:
            threshold[first + i] = np.uint64(_FULL)
    return threshold, alias


@kernel
def _branches(
    symbols,
    tape,
    factor,
    coupling,
    i_power,
    lane_dormant,
    width,
    negligible,
    parallel_work,
    threads,
):
    """Run the tape on every branch of a lane per row of ``symbols``.

    Every active measurement, and every dormant one in ``lane_dormant``, splits
    each lane into its two outcomes, dropping an outcome of probability at most
    ``negligible``. Lanes hold only the amplitudes they need, as in ``_execute``.

    Returns:
        ``(symbols, probability, owner, deviation)``: each branch's symbols, its
        probability, the row it started from, and the largest deviation of a
        measured norm from 1.

    """
    # NOTE: numba's parallel loops cannot read the fields of a named tuple.
    kind, extent, branch = tape.kind, tape.extent, tape.branch
    x_mask, z_mask, cos = tape.x_mask, tape.z_mask, tape.cos
    pivot, sign_row, sign_const = tape.pivot, tape.sign_row, tape.sign_const
    size, num_words = symbols.shape
    lane_symbols = symbols.copy()
    probability = np.ones(size)
    owner = np.arange(size)
    amps = np.zeros((size, 1 << width), dtype=np.complex128)
    for lane in range(size):
        amps[lane, 0] = 1
    deviation = 0.0
    for a in range(len(kind)):
        shift = extent[a]
        span = 1 << shift
        x, z = np.int64(x_mask[a]), np.int64(z_mask[a])
        if kind[a] == ROTATE_DIAG or kind[a] == ROTATE:
            flip = _signs(lane_symbols, size, sign_row[a], np.int64(sign_const[a]))
            diagonal = kind[a] == ROTATE_DIAG
            work = size * span
            chunks = _chunks(work, parallel_work, threads)
            if chunks == 1:
                _rotate_range(
                    amps,
                    0,
                    work,
                    shift,
                    flip,
                    diagonal,
                    x,
                    z,
                    cos[a],
                    factor[a],
                    coupling[a],
                )
            else:
                for c in numba.prange(chunks):
                    lo, hi = work * c // chunks, work * (c + 1) // chunks
                    _rotate_range(
                        amps,
                        lo,
                        hi,
                        shift,
                        flip,
                        diagonal,
                        x,
                        z,
                        cos[a],
                        factor[a],
                        coupling[a],
                    )
            continue
        dormant = kind[a] == MEASURE_DORMANT
        if dormant and not lane_dormant[a]:
            continue
        half = 0.5 if dormant else 0.0
        w0 = np.full(size, half)
        w1 = np.full(size, half)
        if not dormant:
            chunks = _chunks(size * span, parallel_work, threads)
            if chunks == 1:
                _weights_range(amps, 0, size, span, x, z, i_power[a], w0, w1)
            else:
                for c in numba.prange(chunks):
                    lo, hi = size * c // chunks, size * (c + 1) // chunks
                    _weights_range(amps, lo, hi, span, x, z, i_power[a], w0, w1)
            for lane in range(size):
                d = abs(w0[lane] + w1[lane] - 1)
                # NOTE: written so that a NaN deviation sticks.
                if d > deviation or d != d:
                    deviation = d
        # NOTE: both outcomes as ratios, so that a rare one keeps its precision.
        zero = np.ones(size)
        one = np.zeros(size)
        children = 0
        for lane in range(size):
            total = w0[lane] + w1[lane]
            if total > _TINY:
                zero[lane], one[lane] = w0[lane] / total, w1[lane] / total
            children += int(zero[lane] > negligible) + int(one[lane] > negligible)
        new_amps = np.zeros((children, 1 << width), dtype=np.complex128)
        new_symbols = np.empty((children, num_words), dtype=np.uint32)
        new_probability = np.empty(children)
        new_owner = np.empty(children, dtype=np.int64)
        outcome = np.zeros(children, dtype=np.bool_)
        weight = np.zeros(children)
        child = 0
        for lane in range(size):
            for bit in range(2):
                p = one[lane] if bit else zero[lane]
                if p <= negligible:
                    continue
                for i in range(span):
                    new_amps[child, i] = amps[lane, i]
                for w in range(num_words):
                    new_symbols[child, w] = lane_symbols[lane, w]
                if bit and branch[a] >= 0:
                    word, offset = branch[a] >> 5, np.uint32(branch[a] & 31)
                    new_symbols[child, word] |= np.uint32(1) << offset
                new_probability[child] = probability[lane] * p
                new_owner[child] = owner[lane]
                outcome[child] = bit == 1
                weight[child] = w1[lane] if bit else w0[lane]
                child += 1
        if not dormant:
            chunks = _chunks(children * span, parallel_work, threads)
            if chunks == 1:
                _project_range(
                    new_amps,
                    0,
                    children,
                    span,
                    x,
                    z,
                    pivot[a],
                    i_power[a],
                    outcome,
                    weight,
                )
            else:
                for c in numba.prange(chunks):
                    lo, hi = children * c // chunks, children * (c + 1) // chunks
                    _project_range(
                        new_amps,
                        lo,
                        hi,
                        span,
                        x,
                        z,
                        pivot[a],
                        i_power[a],
                        outcome,
                        weight,
                    )
        amps, lane_symbols = new_amps, new_symbols
        probability, owner, size = new_probability, new_owner, children
    return lane_symbols, probability, owner, deviation


@jit
def _draw_range(
    f_patterns,
    tables,
    start,
    threshold,
    alias,
    patterns,
    coin_columns,
    const,
    postselected,
    ready,
    draws,
    coins,
    samples,
    lo,
    hi,
):
    """Draw an entry for shots ``lo … hi-1`` and write their outputs."""
    num_words, num_outputs = f_patterns.shape[1], samples.shape[1]
    num_coins = coin_columns.shape[0]
    coin_words = (num_coins + 63) // 64
    pattern = np.empty(num_words, dtype=np.uint64)
    for s in range(lo, hi):
        # NOTE: lanes sample the shots without a table.
        if tables[s] < 0:
            continue
        first = start[tables[s]]
        size = start[tables[s] + 1] - first
        _, z = splitmix64(draws + np.uint64(2 * s) * GOLDEN)
        _, keep = splitmix64(draws + np.uint64(2 * s + 1) * GOLDEN)
        bucket = first + np.int64((z >> np.uint64(11)) * 2.0**-53 * size)
        entry = bucket if keep < threshold[bucket] else alias[bucket]
        fired = np.uint64(0)
        for w in range(num_words):
            pattern[w] = f_patterns[s, w] ^ patterns[entry, w] ^ const[w]
        bits = np.uint64(0)
        for d in range(num_coins):
            if d % 64 == 0:
                n = np.uint64(s * coin_words + d // 64)
                _, bits = splitmix64(coins + n * GOLDEN)
            if (bits >> np.uint64(d % 64)) & np.uint64(1):
                for w in range(num_words):
                    pattern[w] ^= coin_columns[d, w]
        for w in range(num_words):
            fired |= pattern[w] & postselected[w]
        # NOTE: the earliest discarding output prunes the shot's later ones.
        pruned_at = np.iinfo(np.int64).max
        if fired:
            for o in range(num_outputs):
                word, bit = o >> 6, np.uint64(o & 63)
                discards = (pattern[word] & postselected[word]) >> bit & np.uint64(1)
                if discards and ready[o] < pruned_at:
                    pruned_at = ready[o]
        for o in range(num_outputs):
            value = (pattern[o >> 6] >> np.uint64(o & 63)) & np.uint64(1)
            samples[s, o] = value & np.uint64(ready[o] <= pruned_at)


@kernel
def _draw(
    f_patterns,
    tables,
    start,
    threshold,
    alias,
    patterns,
    coin_columns,
    const,
    postselected,
    ready,
    draws,
    coins,
    parallel_work,
    threads,
):
    """Return the outputs of shots that draw from the tables ``tables`` names.

    ``draws`` and ``coins`` start the streams of entry draws and dormant coins
    (``_streams``).
    """
    shots = f_patterns.shape[0]
    samples = np.empty((shots, len(ready)), dtype=np.bool_)
    work = shots * (f_patterns.shape[1] + len(ready) + 1)
    chunks = _chunks(work, parallel_work, threads)
    if chunks == 1:
        _draw_range(
            f_patterns,
            tables,
            start,
            threshold,
            alias,
            patterns,
            coin_columns,
            const,
            postselected,
            ready,
            draws,
            coins,
            samples,
            0,
            shots,
        )
    else:
        for c in numba.prange(chunks):
            lo, hi = shots * c // chunks, shots * (c + 1) // chunks
            _draw_range(
                f_patterns,
                tables,
                start,
                threshold,
                alias,
                patterns,
                coin_columns,
                const,
                postselected,
                ready,
                draws,
                coins,
                samples,
                lo,
                hi,
            )
    return samples
