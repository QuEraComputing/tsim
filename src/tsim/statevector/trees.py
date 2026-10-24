"""Outcome trees: each key's measurement outcomes, expanded as shots reach them.

A component's amplitudes depend on a shot only through its key and the outcomes
of the measurements that split lanes (``tables``). For one key these outcomes
form a binary tree. A node is a measurement whose outcome is random given the
outcomes before it, and it holds the chance of outcome 1. An edge holds the
output pattern of the branch bits it sets: those of the node's outcome and of
every measurement up to the next node whose outcome is certain, so a shot makes
no draw there.

A shot walks its key's tree: at each node it draws its outcome from its own
stream, XORs the edge's pattern into its outputs and tests the postselected
outputs that the edge makes ready. It stops at the end of the tape or at the
first output that discards it. That draws each outcome from its chance given the
earlier ones, as lanes do, and needs no sorting of shots.

The trees are expanded as shots reach them. A shot at a node or key without its
edge parks; after a pass over the batch the engine runs the tape on one state
per missing edge, up to the next node, and the parked shots resume. The trees
persist across batches and calls, and every postselection mask uses them, since
they key without postselection. A node's contents depend only on its key and
the outcomes before it, and a shot's draws only on its index, so the samples
don't depend on the threads, the parking rounds or what was built before.

Readers take the store's published counts and treat ids beyond them as missing;
one writer at a time appends nodes and edges and links them in, then publishes
the new counts.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import numba
import numpy as np

import tsim.utils.kernels as kernels
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
    _unpack,
    _weights_range,
    sharing_layout,
)
from tsim.statevector.tables import _packed, _streams, _worse
from tsim.utils.kernels import jit, kernel
from tsim.utils.rng import GOLDEN, splitmix64

MISSING = -1
END = -2
# NOTE: a shot's walk state: ROOT before its key's root edge, DONE at the end.
_ROOT = -3
_DONE = -2
# NOTE: what a parked shot needs: its key's root, or nothing.
_NEEDS_ROOT = -1
_NEEDS_NOTHING = -2
# NOTE: as in the tables, draws resolve 2^-64, so an outcome less likely than
# this is never drawn: its measurement counts as certain.
_NEGLIGIBLE = 2.0**-66
_FULL = 2.0**64 - 2.0**11
# NOTE: the states of nodes with a missing edge hold at most this many bytes; the
# oldest are evicted and recomputed from the root when needed.
_STATE_BYTES = 2**28
_LONGEST = np.iinfo(np.int64).max


class _Arrays(NamedTuple):
    """A tree store's arrays, with spare capacity beyond the counts.

    Attributes:
        slot_keys: Open-addressing hash of the keys, shape ``(C, key_words)``.
        slot_roots: Root edge of each slot, MISSING if empty, shape ``(C,)``.
        key_symbols: Symbols of one shot of each key, shape ``(K, num_words)``.
        node_action: Measurement of each node, shape ``(N,)``.
        node_threshold: Chance of outcome 1 times 2^64, shape ``(N,)``.
        node_child: Edge of each outcome, MISSING if not built, shape ``(N, 2)``.
        node_weight: The measured weights of both outcomes, shape ``(N, 2)``.
        node_symbols: Symbols before the measurement, shape ``(N, num_words)``.
        node_parent: Node whose outcome leads here, -1 at a root, shape ``(N,)``.
        node_outcome: That outcome, shape ``(N,)``.
        node_key: Key of each node, shape ``(N,)``.
        node_slot: State slot of each node, -1 if none, shape ``(N,)``.
        edge_pattern: Output pattern of each edge, shape ``(E, output_words)``.
        edge_next: Node at the end of each edge, END at the end of the tape,
            shape ``(E,)``.
        edge_last: The action that ends each edge, its next node's or the
            tape's length, shape ``(E,)``.
        states: Amplitudes before the measurement of slot owners, shape
            ``(S, 2^width)``.
        slot_owner: Node of each state slot, -1 if free, shape ``(S,)``.

    """

    slot_keys: np.ndarray
    slot_roots: np.ndarray
    key_symbols: np.ndarray
    node_action: np.ndarray
    node_threshold: np.ndarray
    node_child: np.ndarray
    node_weight: np.ndarray
    node_symbols: np.ndarray
    node_parent: np.ndarray
    node_outcome: np.ndarray
    node_key: np.ndarray
    node_slot: np.ndarray
    edge_pattern: np.ndarray
    edge_next: np.ndarray
    edge_last: np.ndarray
    states: np.ndarray
    slot_owner: np.ndarray


class _Fixed(NamedTuple):
    """A component's arrays for the engine and the walk, fixed per component.

    Attributes:
        branch_pattern: Output pattern of each action's branch bit, shape
            ``(A, output_words)``; zero for rotations and per-shot coins.
        lane_dormant: Dormant measurements that branch, shape ``(A,)``.
        factor: As in ``SharingLayout``.
        coupling: As in ``SharingLayout``.
        i_power: As in ``SharingLayout``.

    """

    branch_pattern: np.ndarray
    lane_dormant: np.ndarray
    factor: np.ndarray
    coupling: np.ndarray
    i_power: np.ndarray


# NOTE: counts[0] nodes, counts[1] edges, counts[2] keys, counts[3] slots in use,
# counts[4] the next slot to evict.
_NODES, _EDGES, _KEYS, _SLOTS, _EVICT = range(5)


def _grown(array: np.ndarray, rows: int, fill: int | float = 0) -> np.ndarray:
    """Return ``array`` with ``rows`` rows, the new ones set to ``fill``."""
    out = np.full((rows, *array.shape[1:]), fill, dtype=array.dtype)
    out[: len(array)] = array
    return out


class OutcomeTrees:
    """The outcome trees of one component, shared by every batch and call.

    Attributes:
        component: The compiled component.
        layout: Its sharing layout without postselection.

    """

    def __init__(self, component: StatevectorComponent) -> None:
        """Start with empty trees."""
        self.component = component
        self.layout = layout = sharing_layout(component)
        tape, num_words = component.tape, component.num_words
        num_f = len(tape.f_selection)
        output_bits = _unpack(tape.output_row, num_words)
        key_bits = _unpack(layout.key_rows, num_words)[:, :num_f]
        self.key_words = max(1, (len(key_bits) + 63) // 64)
        self._key_columns = np.zeros((num_f, self.key_words), np.uint64)
        if len(key_bits):
            self._key_columns[:] = _packed(key_bits.T)
        self._output_columns = _packed(output_bits[:, :num_f].T)
        coins = np.flatnonzero(_unpack(layout.shot_dormant, num_words))
        self._coin_columns = _packed(output_bits[:, coins].T)
        output_words = self._output_columns.shape[1]
        lane_bits = _unpack(layout.lane_branches, num_words)
        branch_pattern = np.zeros((len(tape.kind), output_words), np.uint64)
        for a in np.flatnonzero(tape.branch >= 0):
            if lane_bits[tape.branch[a]]:
                column = output_bits[:, tape.branch[a]][None, :]
                branch_pattern[a] = _packed(column)[0]
        self._fixed = _Fixed(
            branch_pattern=branch_pattern,
            lane_dormant=layout.lane_dormant,
            factor=layout.factor,
            coupling=layout.coupling,
            i_power=layout.i_power,
        )
        splitting = np.isin(tape.kind, (MEASURE, MEASURE_DIAG)) | layout.lane_dormant
        self._stride = int(np.count_nonzero(splitting)) + 1
        self._max_slots = max(16, min(2**20, _STATE_BYTES // (16 << component.width)))
        self._arrays = self._empty(num_words, output_words)
        self._counts = np.zeros(5, dtype=np.int64)
        self._lock = threading.Lock()

    def _empty(self, num_words: int, output_words: int) -> _Arrays:
        rows, width = 64, self.component.width
        return _Arrays(
            slot_keys=np.zeros((128, self.key_words), np.uint64),
            slot_roots=np.full(128, MISSING, np.int64),
            key_symbols=np.zeros((rows, num_words), np.uint32),
            node_action=np.zeros(rows, np.int64),
            node_threshold=np.zeros(rows, np.uint64),
            node_child=np.full((rows, 2), MISSING, np.int64),
            node_weight=np.zeros((rows, 2), np.float64),
            node_symbols=np.zeros((rows, num_words), np.uint32),
            node_parent=np.zeros(rows, np.int64),
            node_outcome=np.zeros(rows, np.int64),
            node_key=np.zeros(rows, np.int64),
            node_slot=np.full(rows, -1, np.int64),
            edge_pattern=np.zeros((rows, output_words), np.uint64),
            edge_next=np.zeros(rows, np.int64),
            edge_last=np.zeros(rows, np.int64),
            states=np.zeros((16, 1 << width), np.complex128),
            slot_owner=np.full(16, -1, np.int64),
        )

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes built so far."""
        return int(self._counts[_NODES])

    @property
    def num_keys(self) -> int:
        """Return the number of keys met so far."""
        return int(self._counts[_KEYS])

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
        self, f_params: np.ndarray, seed: int, layout: SharingLayout
    ) -> tuple[np.ndarray, float, dict]:
        """Sample one batch, as ``sample_shared_component`` does with ``layout``.

        ``layout`` gives the outputs' reference bits and the postselected outputs:
        a shot that one of them discards reads ``False`` in the outputs set after
        the discarding output, as if its lane were pruned there.

        Returns:
            ``(samples, max_norm_deviation, stats)``, where ``stats`` counts the
            nodes in total, those this batch built, its parking rounds and its
            parked shots.

        """
        tape = self.component.tape
        shots = f_params.shape[0]
        output_words = self._output_columns.shape[1]
        threads = numba.get_num_threads()
        shot_keys, f_patterns = _shot_keys(
            shots * (len(tape.f_selection) + 1),
            f_params,
            tape.f_selection,
            self._key_columns,
            self._output_columns,
            kernels.PARALLEL_WORK,
            threads,
        )
        postselected = layout.prune_first | layout.prune_outputs.any(axis=0)
        checks = np.flatnonzero(postselected)
        checks = checks[np.argsort(layout.ready[checks], kind="stable")]
        draws, coins, _ = _streams(np.uint64(seed))
        at, outcome, depth, pruned, patterns = _start(
            shots * (output_words + len(self._coin_columns) + 1),
            f_patterns,
            self._coin_columns,
            _packed(layout.output_const[None, :])[0],
            checks,
            layout.ready,
            coins,
            kernels.PARALLEL_WORK,
            threads,
        )
        ptr = np.searchsorted(layout.ready[checks], 0).repeat(shots).astype(np.int64)
        active = np.flatnonzero(at != _DONE)
        need = np.full(shots, _NEEDS_NOTHING, dtype=np.int64)
        deviation, built, rounds, parked = 0.0, 0, 0, 0
        while len(active):
            with self._lock:
                arrays, counts = self._arrays, self._counts.copy()
            _walk(
                len(active) * self._stride,
                active,
                shot_keys,
                at,
                outcome,
                depth,
                pruned,
                ptr,
                patterns,
                need,
                checks,
                layout.ready,
                arrays.slot_keys,
                arrays.slot_roots,
                arrays.node_threshold,
                arrays.node_child,
                arrays.edge_pattern,
                arrays.edge_next,
                arrays.edge_last,
                counts[_NODES],
                counts[_EDGES],
                draws,
                self._stride,
                kernels.PARALLEL_WORK,
                threads,
            )
            active = active[need[active] != _NEEDS_NOTHING]
            if not len(active):
                break
            rounds += 1
            parked += len(active)
            part_deviation, part_built = self._expand(active, need, shot_keys, f_params)
            deviation = _worse(deviation, part_deviation)
            built += part_built
        samples = _outputs(
            shots * self.component.num_outputs, patterns, layout.ready, pruned
        )
        stats = {
            "tree shots": shots,
            "nodes": self.num_nodes,
            "nodes built": built,
            "rounds": rounds,
            "parked": parked,
        }
        return samples, deviation, stats

    def _expand(
        self,
        active: np.ndarray,
        need: np.ndarray,
        shot_keys: np.ndarray,
        f_params: np.ndarray,
    ) -> tuple[float, int]:
        """Build what the ``active`` shots need; return the deviation and nodes built."""
        wants = need[active]
        children = np.unique(wants[wants >= 0])
        rooted = active[wants == _NEEDS_ROOT]
        _, first = np.unique(shot_keys[rooted], axis=0, return_index=True)
        rooted = rooted[np.sort(first)]
        tape = self.component.tape
        symbols = np.zeros((len(rooted), self.component.num_words), dtype=np.uint32)
        _pack_range(f_params[rooted], tape.f_selection, symbols, 0, len(rooted))
        with self._lock:
            self._reserve(len(children) + len(rooted), len(rooted))
            before = int(self._counts[_NODES])
            deviation = _build(
                children >> 1,
                children & 1,
                shot_keys[rooted],
                symbols,
                tape,
                self._fixed,
                self._arrays,
                self._counts,
                self._max_slots,
                _NEGLIGIBLE,
            )
            return float(deviation), int(self._counts[_NODES]) - before

    def _reserve(self, edges: int, keys: int) -> None:
        """Make room for ``edges`` more edges and nodes and ``keys`` more keys."""
        arrays, counts = self._arrays, self._counts
        changes: dict[str, np.ndarray] = {}
        nodes = counts[_NODES] + edges
        if nodes > len(arrays.node_action):
            rows = max(int(nodes), 2 * len(arrays.node_action))
            for name in _Arrays._fields:
                if name.startswith("node_"):
                    fill = MISSING if name in ("node_child", "node_slot") else 0
                    changes[name] = _grown(getattr(arrays, name), rows, fill)
        if counts[_EDGES] + edges > len(arrays.edge_next):
            rows = max(int(counts[_EDGES] + edges), 2 * len(arrays.edge_next))
            for name in ("edge_pattern", "edge_next", "edge_last"):
                changes[name] = _grown(getattr(arrays, name), rows)
        if counts[_KEYS] + keys > len(arrays.key_symbols):
            rows = max(int(counts[_KEYS] + keys), 2 * len(arrays.key_symbols))
            changes["key_symbols"] = _grown(arrays.key_symbols, rows)
        capacity = len(arrays.slot_roots)
        if 2 * (counts[_KEYS] + keys) > capacity:
            while 2 * (counts[_KEYS] + keys) > capacity:
                capacity *= 2
            slot_keys = np.zeros((capacity, self.key_words), np.uint64)
            slot_roots = np.full(capacity, MISSING, np.int64)
            _rehash(arrays.slot_keys, arrays.slot_roots, slot_keys, slot_roots)
            changes["slot_keys"], changes["slot_roots"] = slot_keys, slot_roots
        slots = len(arrays.slot_owner)
        if counts[_SLOTS] + edges > slots and slots < self._max_slots:
            rows = min(self._max_slots, max(int(counts[_SLOTS] + edges), 2 * slots))
            changes["states"] = _grown(arrays.states, rows)
            changes["slot_owner"] = _grown(arrays.slot_owner, rows, -1)
        if changes:
            self._arrays = arrays._replace(**changes)


@jit
def _hash(words):
    """Return the hash of a key's words."""
    h = np.uint64(0)
    for w in range(len(words)):
        _, h = splitmix64(h ^ words[w])
    return h


@jit
def _find(words, slot_keys, slot_roots, num_edges):
    """Return the root edge of a key, or MISSING if it has none below ``num_edges``."""
    mask = np.uint64(len(slot_roots) - 1)
    slot = _hash(words) & mask
    while True:
        root = slot_roots[slot]
        if root == MISSING:
            return MISSING
        if root < num_edges:
            same = True
            for w in range(len(words)):
                if slot_keys[slot, w] != words[w]:
                    same = False
                    break
            if same:
                return root
        slot = (slot + np.uint64(1)) & mask


@jit
def _rehash(old_keys, old_roots, slot_keys, slot_roots):
    """Insert every key of the old hash into the new one."""
    mask = np.uint64(len(slot_roots) - 1)
    for old in range(len(old_roots)):
        if old_roots[old] == MISSING:
            continue
        slot = _hash(old_keys[old]) & mask
        while slot_roots[slot] != MISSING:
            slot = (slot + np.uint64(1)) & mask
        slot_keys[slot] = old_keys[old]
        slot_roots[slot] = old_roots[old]


@jit
def _keys_range(f, selection, key_columns, output_columns, keys, f_patterns, lo, hi):
    """Write the key words and the f-bits' output pattern of shots ``lo … hi-1``."""
    for s in range(lo, hi):
        for w in range(keys.shape[1]):
            keys[s, w] = 0
        for w in range(f_patterns.shape[1]):
            f_patterns[s, w] = 0
        for j in range(len(selection)):
            if f[s, selection[j]]:
                for w in range(keys.shape[1]):
                    keys[s, w] ^= key_columns[j, w]
                for w in range(f_patterns.shape[1]):
                    f_patterns[s, w] ^= output_columns[j, w]


@kernel
def _shot_keys(f, selection, key_columns, output_columns, parallel_work, threads):
    """Return each shot's key words and its f-bits' output pattern."""
    shots = f.shape[0]
    keys = np.empty((shots, key_columns.shape[1]), dtype=np.uint64)
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
def _start_range(
    f_patterns, coin_columns, const, checks, ready, coins, at, patterns, pruned, lo, hi
):
    """Start shots ``lo … hi-1`` from their f-bits, coins and constants.

    Tests the postselected outputs of f-bits and coins alone.
    """
    num_words, num_coins = f_patterns.shape[1], coin_columns.shape[0]
    coin_words = (num_coins + 63) // 64
    for s in range(lo, hi):
        for w in range(num_words):
            patterns[s, w] = f_patterns[s, w] ^ const[w]
        bits = np.uint64(0)
        for d in range(num_coins):
            if d % 64 == 0:
                n = np.uint64(s * coin_words + d // 64)
                _, bits = splitmix64(coins + n * GOLDEN)
            if (bits >> np.uint64(d % 64)) & np.uint64(1):
                for w in range(num_words):
                    patterns[s, w] ^= coin_columns[d, w]
        at[s], pruned[s] = _ROOT, _LONGEST
        for c in range(len(checks)):
            o = checks[c]
            if ready[o] >= 0:
                break
            if (patterns[s, o >> 6] >> np.uint64(o & 63)) & np.uint64(1):
                at[s], pruned[s] = _DONE, ready[o]
                break


@kernel
def _start(
    f_patterns, coin_columns, const, checks, ready, coins, parallel_work, threads
):
    """Return the walk state of every shot before its key's root."""
    shots, num_words = f_patterns.shape
    at = np.empty(shots, dtype=np.int64)
    outcome = np.full(shots, -1, dtype=np.int64)
    depth = np.zeros(shots, dtype=np.int64)
    pruned = np.empty(shots, dtype=np.int64)
    patterns = np.empty((shots, num_words), dtype=np.uint64)
    chunks = _chunks(shots * (num_words + 1), parallel_work, threads)
    if chunks == 1:
        _start_range(
            f_patterns,
            coin_columns,
            const,
            checks,
            ready,
            coins,
            at,
            patterns,
            pruned,
            0,
            shots,
        )
    else:
        for c in numba.prange(chunks):
            lo, hi = shots * c // chunks, shots * (c + 1) // chunks
            _start_range(
                f_patterns,
                coin_columns,
                const,
                checks,
                ready,
                coins,
                at,
                patterns,
                pruned,
                lo,
                hi,
            )
    return at, outcome, depth, pruned, patterns


@jit
def _walk_range(
    active,
    shot_keys,
    at,
    outcome,
    depth,
    pruned,
    ptr,
    patterns,
    need,
    checks,
    ready,
    slot_keys,
    slot_roots,
    node_threshold,
    node_child,
    edge_pattern,
    edge_next,
    edge_last,
    num_nodes,
    num_edges,
    draws,
    stride,
    lo,
    hi,
):
    """Walk the shots ``active[lo … hi-1]`` until each ends or needs an edge."""
    num_words = patterns.shape[1]
    for i in range(lo, hi):
        s = active[i]
        need[s] = _NEEDS_NOTHING
        while at[s] != _DONE:
            if at[s] == _ROOT:
                edge = _find(shot_keys[s], slot_keys, slot_roots, num_edges)
                if edge == MISSING:
                    need[s] = _NEEDS_ROOT
                    break
            else:
                node = at[s]
                if outcome[s] < 0:
                    n = np.uint64(s * stride + depth[s])
                    _, z = splitmix64(draws + n * GOLDEN)
                    depth[s] += 1
                    outcome[s] = 1 if z < node_threshold[node] else 0
                edge = node_child[node, outcome[s]]
                if edge < 0 or edge >= num_edges:
                    need[s] = 2 * node + outcome[s]
                    break
                outcome[s] = -1
            for w in range(num_words):
                patterns[s, w] ^= edge_pattern[edge, w]
            last = edge_last[edge]
            # NOTE: the earliest discarding output prunes the shot's later ones.
            while ptr[s] < len(checks) and ready[checks[ptr[s]]] < last:
                o = checks[ptr[s]]
                ptr[s] += 1
                if (patterns[s, o >> 6] >> np.uint64(o & 63)) & np.uint64(1):
                    pruned[s] = ready[o]
                    at[s] = _DONE
                    break
            if at[s] == _DONE:
                break
            # NOTE: an edge within the published counts ends at a published node.
            following = edge_next[edge]
            at[s] = _DONE if following == END else following


@kernel
def _walk(
    active,
    shot_keys,
    at,
    outcome,
    depth,
    pruned,
    ptr,
    patterns,
    need,
    checks,
    ready,
    slot_keys,
    slot_roots,
    node_threshold,
    node_child,
    edge_pattern,
    edge_next,
    edge_last,
    num_nodes,
    num_edges,
    draws,
    stride,
    parallel_work,
    threads,
):
    """Walk the ``active`` shots until each ends or needs an edge not yet built."""
    chunks = _chunks(len(active) * stride, parallel_work, threads)
    if chunks == 1:
        _walk_range(
            active,
            shot_keys,
            at,
            outcome,
            depth,
            pruned,
            ptr,
            patterns,
            need,
            checks,
            ready,
            slot_keys,
            slot_roots,
            node_threshold,
            node_child,
            edge_pattern,
            edge_next,
            edge_last,
            num_nodes,
            num_edges,
            draws,
            stride,
            0,
            len(active),
        )
    else:
        for c in numba.prange(chunks):
            lo, hi = len(active) * c // chunks, len(active) * (c + 1) // chunks
            _walk_range(
                active,
                shot_keys,
                at,
                outcome,
                depth,
                pruned,
                ptr,
                patterns,
                need,
                checks,
                ready,
                slot_keys,
                slot_roots,
                node_threshold,
                node_child,
                edge_pattern,
                edge_next,
                edge_last,
                num_nodes,
                num_edges,
                draws,
                stride,
                lo,
                hi,
            )


@kernel
def _outputs(patterns, ready, pruned):
    """Return the outputs; those set after a shot's pruning action read False."""
    shots, num_outputs = patterns.shape[0], len(ready)
    samples = np.empty((shots, num_outputs), dtype=np.bool_)
    for s in numba.prange(shots):
        for o in range(num_outputs):
            value = (patterns[s, o >> 6] >> np.uint64(o & 63)) & np.uint64(1)
            samples[s, o] = value & np.uint64(ready[o] <= pruned[s])
    return samples


@jit
def _advance(amps, symbols, a, tape, fixed, pattern, negligible, measured):
    """Run the tape on one state from action ``a`` to the next random outcome.

    A measurement whose outcome is certain is projected onto it, and its branch
    bit goes into ``symbols`` and its pattern into ``pattern``. ``measured``
    receives the largest deviation of a measured norm from 1 (its entry 0) and
    the weights of the stopping measurement (entries 1 and 2).

    Returns:
        The stopping action, or the tape's length.

    """
    kind, extent, branch = tape.kind, tape.extent, tape.branch
    flip = np.zeros(1, dtype=np.int64)
    w0, w1 = np.zeros(1), np.zeros(1)
    chosen = np.zeros(1, dtype=np.bool_)
    weight = np.zeros(1)
    while a < len(kind):
        span = 1 << extent[a]
        x, z = np.int64(tape.x_mask[a]), np.int64(tape.z_mask[a])
        if kind[a] == ROTATE_DIAG or kind[a] == ROTATE:
            acc = 0
            for w in range(len(symbols)):
                acc ^= symbols[w] & tape.sign_row[a, w]
            flip[0] = _parity(acc) ^ np.int64(tape.sign_const[a])
            _rotate_range(
                amps,
                0,
                span,
                extent[a],
                flip,
                kind[a] == ROTATE_DIAG,
                x,
                z,
                tape.cos[a],
                fixed.factor[a],
                fixed.coupling[a],
            )
            a += 1
            continue
        if kind[a] == MEASURE_DORMANT:
            if fixed.lane_dormant[a]:
                measured[1], measured[2] = 0.5, 0.5
                return a
            a += 1
            continue
        w0[0], w1[0] = 0.0, 0.0
        _weights_range(amps, 0, 1, span, x, z, fixed.i_power[a], w0, w1)
        d = abs(w0[0] + w1[0] - 1)
        # NOTE: written so that a NaN deviation sticks.
        if d > measured[0] or d != d:
            measured[0] = d
        total = w0[0] + w1[0]
        zero, one = (w0[0] / total, w1[0] / total) if total > _TINY else (1.0, 0.0)
        if zero > negligible and one > negligible:
            measured[1], measured[2] = w0[0], w1[0]
            return a
        chosen[0] = one > negligible
        weight[0] = w1[0] if chosen[0] else w0[0]
        _project_range(
            amps, 0, 1, span, x, z, tape.pivot[a], fixed.i_power[a], chosen, weight
        )
        if chosen[0] and branch[a] >= 0:
            symbols[branch[a] >> 5] |= np.uint32(1) << np.uint32(branch[a] & 31)
            for w in range(len(pattern)):
                pattern[w] ^= fixed.branch_pattern[a, w]
        a += 1
    return a


@jit
def _take(amps, symbols, a, bit, weight, tape, fixed, pattern):
    """Apply outcome ``bit`` of the random measurement ``a`` to one state."""
    if tape.kind[a] != MEASURE_DORMANT:
        chosen = np.full(1, bit == 1)
        weights = np.full(1, weight)
        span = 1 << tape.extent[a]
        x, z = np.int64(tape.x_mask[a]), np.int64(tape.z_mask[a])
        _project_range(
            amps, 0, 1, span, x, z, tape.pivot[a], fixed.i_power[a], chosen, weights
        )
    if bit and tape.branch[a] >= 0:
        b = tape.branch[a]
        symbols[b >> 5] |= np.uint32(1) << np.uint32(b & 31)
        for w in range(len(pattern)):
            pattern[w] ^= fixed.branch_pattern[a, w]


@jit
def _state(node, tape, fixed, arrays, amps, negligible):
    """Write the state of ``node`` before its measurement into ``amps``.

    Copies it from the node's slot, or recomputes it from the root along the
    node's outcomes.
    """
    slot = arrays.node_slot[node]
    if slot >= 0:
        amps[0] = arrays.states[slot]
        return
    path = []
    n = node
    while n >= 0:
        path.append(n)
        n = arrays.node_parent[n]
    amps[0] = 0
    amps[0, 0] = 1
    symbols = arrays.key_symbols[arrays.node_key[node]].copy()
    pattern = np.zeros(arrays.edge_pattern.shape[1], dtype=np.uint64)
    measured = np.zeros(3)
    a = 0
    for i in range(len(path) - 1, -1, -1):
        a = _advance(amps, symbols, a, tape, fixed, pattern, negligible, measured)
        if i == 0:
            return
        on = path[i]
        bit = arrays.node_outcome[path[i - 1]]
        _take(amps, symbols, a, bit, arrays.node_weight[on, bit], tape, fixed, pattern)
        a += 1


@jit
def _keep(node, amps, arrays, counts, max_slots):
    """Keep the state of a new node in a slot, evicting the oldest if all are used."""
    if counts[_SLOTS] < len(arrays.slot_owner):
        slot = counts[_SLOTS]
        counts[_SLOTS] += 1
    else:
        slot = counts[_EVICT] % len(arrays.slot_owner)
        counts[_EVICT] += 1
        owner = arrays.slot_owner[slot]
        if owner >= 0:
            arrays.node_slot[owner] = -1
    arrays.states[slot] = amps[0]
    arrays.slot_owner[slot] = node
    arrays.node_slot[node] = slot


@jit
def _emit(
    amps,
    symbols,
    a,
    pattern,
    measured,
    key,
    parent,
    bit,
    arrays,
    counts,
    max_slots,
    num_actions,
):
    """Append the edge that ends at action ``a`` and its node; return the edge."""
    edge = counts[_EDGES]
    counts[_EDGES] += 1
    arrays.edge_pattern[edge] = pattern
    arrays.edge_last[edge] = a
    if a == num_actions:
        arrays.edge_next[edge] = END
        return edge
    node = counts[_NODES]
    arrays.node_action[node] = a
    w0, w1 = measured[1], measured[2]
    one = w1 / (w0 + w1) if w0 + w1 > _TINY else 0.0
    arrays.node_threshold[node] = np.uint64(min(one * 2.0**64, _FULL))
    arrays.node_weight[node, 0], arrays.node_weight[node, 1] = w0, w1
    arrays.node_symbols[node] = symbols
    arrays.node_parent[node] = parent
    arrays.node_outcome[node] = bit
    arrays.node_key[node] = key
    arrays.node_child[node, 0] = MISSING
    arrays.node_child[node, 1] = MISSING
    arrays.node_slot[node] = -1
    _keep(node, amps, arrays, counts, max_slots)
    arrays.edge_next[edge] = node
    counts[_NODES] += 1
    return edge


@jit
def _build(
    nodes, bits, keys, key_symbols, tape, fixed, arrays, counts, max_slots, negligible
):
    """Build the child edges ``(nodes, bits)`` and the roots of ``keys``.

    Each new edge's node is linked last, so that a reader that finds the link
    within the published counts finds the node and edge complete.

    Returns:
        The largest deviation of a measured norm from 1.

    """
    num_actions = len(tape.kind)
    amps = np.zeros((1, arrays.states.shape[1]), dtype=np.complex128)
    pattern = np.zeros(arrays.edge_pattern.shape[1], dtype=np.uint64)
    measured = np.zeros(3)
    for r in range(len(keys)):
        if _find(keys[r], arrays.slot_keys, arrays.slot_roots, counts[_EDGES]) >= 0:
            continue
        key = counts[_KEYS]
        counts[_KEYS] += 1
        arrays.key_symbols[key] = key_symbols[r]
        symbols = key_symbols[r].copy()
        amps[0] = 0
        amps[0, 0] = 1
        pattern[:] = 0
        a = _advance(amps, symbols, 0, tape, fixed, pattern, negligible, measured)
        edge = _emit(
            amps,
            symbols,
            a,
            pattern,
            measured,
            key,
            -1,
            0,
            arrays,
            counts,
            max_slots,
            num_actions,
        )
        mask = np.uint64(len(arrays.slot_roots) - 1)
        slot = _hash(keys[r]) & mask
        while arrays.slot_roots[slot] != MISSING:
            slot = (slot + np.uint64(1)) & mask
        arrays.slot_keys[slot] = keys[r]
        arrays.slot_roots[slot] = edge
    for r in range(len(nodes)):
        node, bit = nodes[r], bits[r]
        if arrays.node_child[node, bit] >= 0:
            continue
        _state(node, tape, fixed, arrays, amps, negligible)
        symbols = arrays.node_symbols[node].copy()
        pattern[:] = 0
        a = arrays.node_action[node]
        _take(
            amps, symbols, a, bit, arrays.node_weight[node, bit], tape, fixed, pattern
        )
        a = _advance(amps, symbols, a + 1, tape, fixed, pattern, negligible, measured)
        edge = _emit(
            amps,
            symbols,
            a,
            pattern,
            measured,
            arrays.node_key[node],
            node,
            bit,
            arrays,
            counts,
            max_slots,
            num_actions,
        )
        arrays.node_child[node, bit] = edge
        if arrays.node_child[node, 1 - bit] >= 0 and arrays.node_slot[node] >= 0:
            arrays.slot_owner[arrays.node_slot[node]] = -1
            arrays.node_slot[node] = -1
    return measured[0]
