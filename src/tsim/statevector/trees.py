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
the new counts. The node, edge and state arrays are allocated once, at their
capacity, as zeros, which take no memory until written, so that they never
grow by copying; links store an id plus one, so that zero means missing.
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
    sample_shared_component,
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
# NOTE: the states that one round grows hold at most about this many bytes.
_ROUND_BYTES = 2**26
# NOTE: a store holds at most this many nodes and edges.
_CAPACITY = 2**22


class _Arrays(NamedTuple):
    """A tree store's arrays, with spare capacity beyond the counts.

    Attributes:
        slot_keys: Open-addressing hash of the keys, shape ``(C, key_words)``.
        slot_roots: Root edge of each slot plus one, 0 if empty, shape ``(C,)``.
        key_symbols: Symbols of one shot of each key, shape ``(K, num_words)``.
        node_action: Measurement of each node, shape ``(N,)``.
        node_threshold: Chance of outcome 1 times 2^64, shape ``(N,)``.
        node_child: Edge of each outcome plus one, 0 if not built, shape
            ``(N, 2)``.
        node_weight: The measured weights of both outcomes, shape ``(N, 2)``.
        node_symbols: Symbols before the measurement, shape ``(N, num_words)``.
        node_parent: Node whose outcome leads here, -1 at a root, shape ``(N,)``.
        node_outcome: That outcome, shape ``(N,)``.
        node_key: Key of each node, shape ``(N,)``.
        node_slot: State slot of each node plus one, 0 if none, shape ``(N,)``.
        edge_pattern: Output pattern of each edge, shape ``(E, output_words)``.
        edge_next: Node at the end of each edge, END at the end of the tape,
            shape ``(E,)``.
        edge_last: The action that ends each edge, its next node's or the
            tape's length, shape ``(E,)``.
        states: Amplitudes before the measurement of slot owners, shape
            ``(S, 2^width)``.
        slot_owner: Node of each state slot plus one, 0 if free, shape ``(S,)``.

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
        # NOTE: readers take this pair without the lock; the writer replaces it.
        self._published = (self._arrays, self._counts.copy())
        self._lock = threading.Lock()

    def _empty(self, num_words: int, output_words: int) -> _Arrays:
        rows, width = _CAPACITY, self.component.width
        return _Arrays(
            slot_keys=np.zeros((128, self.key_words), np.uint64),
            slot_roots=np.zeros(128, np.int64),
            key_symbols=np.zeros((rows, num_words), np.uint32),
            node_action=np.zeros(rows, np.int64),
            node_threshold=np.zeros(rows, np.uint64),
            node_child=np.zeros((rows, 2), np.int64),
            node_weight=np.zeros((rows, 2), np.float64),
            node_symbols=np.zeros((rows, num_words), np.uint32),
            node_parent=np.zeros(rows, np.int64),
            node_outcome=np.zeros(rows, np.int64),
            node_key=np.zeros(rows, np.int64),
            node_slot=np.zeros(rows, np.int64),
            edge_pattern=np.zeros((rows, output_words), np.uint64),
            edge_next=np.zeros(rows, np.int64),
            edge_last=np.zeros(rows, np.int64),
            states=np.zeros((self._max_slots, 1 << width), np.complex128),
            slot_owner=np.zeros(self._max_slots, np.int64),
        )

    @property
    def full(self) -> bool:
        """Return whether the store holds half its capacity of nodes or edges."""
        return 2 * int(max(self._counts[_NODES], self._counts[_EDGES])) > _CAPACITY

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes built so far."""
        return int(self._counts[_NODES])

    @property
    def num_keys(self) -> int:
        """Return the number of keys met so far."""
        return int(self._counts[_KEYS])

    def __getstate__(self) -> dict:
        """Return the state to pickle: the used rows of the arrays, no lock."""
        state = self.__dict__.copy()
        del state["_lock"], state["_published"]
        counts = self._counts
        used = {"key_symbols": counts[_KEYS], "states": counts[_SLOTS]}
        used["slot_owner"] = counts[_SLOTS]
        for name in _Arrays._fields:
            if name.startswith("node_"):
                used[name] = counts[_NODES]
            elif name.startswith("edge_"):
                used[name] = counts[_EDGES]
        state["_arrays"] = {
            name: array[: used[name]] if name in used else array
            for name, array in self._arrays._asdict().items()
        }
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore a pickled state into arrays at full capacity, with a new lock."""
        used = state.pop("_arrays")
        self.__dict__.update(state)
        output_words = self._output_columns.shape[1]
        arrays = self._empty(self.component.num_words, output_words)
        for name, array in used.items():
            target = getattr(arrays, name)
            if target.shape == array.shape:
                arrays = arrays._replace(**{name: array.copy()})
            else:
                target[: len(array)] = array
        self._arrays = arrays
        self._published = (self._arrays, self._counts.copy())
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
        draws, coins, lanes = _streams(np.uint64(seed))
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
            arrays, counts = self._published
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
            expanded = self._expand(
                arrays,
                active,
                need,
                shot_keys,
                f_params,
                (depth, patterns, ptr, checks, layout.ready, draws),
            )
            if expanded is None:
                break
            deviation = _worse(deviation, expanded[0])
            built += expanded[1]
        samples = _outputs(
            shots * self.component.num_outputs, patterns, layout.ready, pruned
        )
        if len(active):
            # NOTE: a full store sends the shots still walking to lanes.
            lane_samples, lane_deviation, _ = sample_shared_component(
                self.component, layout, f_params[active], int(lanes)
            )
            samples[active] = lane_samples
            deviation = _worse(deviation, lane_deviation)
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
        arrays: _Arrays,
        active: np.ndarray,
        need: np.ndarray,
        shot_keys: np.ndarray,
        f_params: np.ndarray,
        walks: tuple,
    ) -> tuple[float, int] | None:
        """Build what the ``active`` shots need; return the deviation and nodes built.

        Returns None if the store has no room for them.

        Each missing edge or key grows along the path of one of the shots that
        need it, so that a batch needs few rounds. The tape runs outside the lock,
        on the published ``arrays`` that the shots walked, so that batches expand
        their trees at once; the lock covers copying cached states and linking
        in what is still missing.
        """
        depth, patterns, ptr, checks, ready, draws = walks
        wants = need[active]
        children, first = np.unique(wants[wants >= 0], return_index=True)
        child_shots = active[wants >= 0][first]
        rooted = active[wants == _NEEDS_ROOT]
        _, first = np.unique(shot_keys[rooted], axis=0, return_index=True)
        rooted = rooted[np.sort(first)]
        nodes = np.concatenate([children >> 1, np.full(len(rooted), -1)])
        bits = np.concatenate([children & 1, np.zeros(len(rooted), np.int64)])
        shots = np.concatenate([child_shots, rooted])
        tape, component = self.component.tape, self.component
        key_symbols = np.zeros((len(shots), component.num_words), np.uint32)
        _pack_range(f_params[shots], tape.f_selection, key_symbols, 0, len(shots))
        starts = np.zeros((len(shots), 1 << component.width), np.complex128)
        has_state = np.zeros(len(shots), dtype=np.bool_)
        with self._lock:
            missing = _gather(nodes, bits, self._arrays, starts, has_state)
        nodes, bits, shots = nodes[missing], bits[missing], shots[missing]
        key_symbols, starts = key_symbols[missing], starts[missing]
        has_state, keys = has_state[missing], shot_keys[shots]
        state_bytes = 16 << component.width
        capacity = min(
            len(nodes) * self._stride,
            max(len(nodes), _ROUND_BYTES // state_bytes),
        )
        grown = self._rows(capacity)
        rows, deviation = _grow(
            nodes,
            bits,
            key_symbols,
            starts,
            has_state,
            shots,
            depth,
            patterns,
            ptr,
            checks,
            ready,
            draws,
            self._stride,
            grown,
            arrays,
            tape,
            self._fixed,
            _NEGLIGIBLE,
        )
        with self._lock:
            if not self._reserve(rows, int(np.count_nonzero(nodes < 0))):
                return None
            before = int(self._counts[_NODES])
            _link(
                nodes,
                bits,
                keys,
                key_symbols,
                grown,
                rows,
                self._arrays,
                self._counts,
                len(tape.kind),
            )
            self._published = (self._arrays, self._counts.copy())
            built = int(self._counts[_NODES]) - before
        return float(deviation), built

    def _rows(self, rows: int) -> _Grown:
        """Return empty rows for edges and nodes computed outside the store."""
        component = self.component
        return _Grown(
            request=np.zeros(rows, np.int64),
            outcome=np.zeros(rows, np.int64),
            states=np.zeros((rows, 1 << component.width), np.complex128),
            patterns=np.zeros((rows, self._output_columns.shape[1]), np.uint64),
            last=np.zeros(rows, np.int64),
            symbols=np.zeros((rows, component.num_words), np.uint32),
            weights=np.zeros((rows, 2)),
        )

    def _reserve(self, edges: int, keys: int) -> bool:
        """Make room for ``keys`` more keys; return whether ``edges`` more fit."""
        arrays, counts = self._arrays, self._counts
        if counts[_NODES] + edges > _CAPACITY or counts[_EDGES] + edges > _CAPACITY:
            return False
        capacity = len(arrays.slot_roots)
        if 2 * (counts[_KEYS] + keys) > capacity:
            while 2 * (counts[_KEYS] + keys) > capacity:
                capacity *= 2
            slot_keys = np.zeros((capacity, self.key_words), np.uint64)
            slot_roots = np.zeros(capacity, np.int64)
            _rehash(arrays.slot_keys, arrays.slot_roots, slot_keys, slot_roots)
            self._arrays = arrays._replace(slot_keys=slot_keys, slot_roots=slot_roots)
        return True


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
        root = slot_roots[slot] - 1
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
        if old_roots[old] == 0:
            continue
        slot = _hash(old_keys[old]) & mask
        while slot_roots[slot] != 0:
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
                edge = node_child[node, outcome[s]] - 1
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
def _replay(node, tape, fixed, arrays, amps, negligible):
    """Write the state of ``node`` before its measurement into ``amps[0]``.

    Recomputes it from the root along the node's outcomes, reading only what a
    published node never changes.
    """
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


class _Grown(NamedTuple):
    """Edges and nodes computed outside the store, along the paths of shots.

    Row ``r`` is an edge of request ``request[r]``, and unless the edge ends
    the tape, the node it ends at. The first row of a request starts at the
    request's node or key; every other row starts at the previous row's node.

    Attributes:
        request: Request of each row, shape ``(R,)``.
        outcome: Outcome that starts each row's edge, shape ``(R,)``.
        states: Each node's state before its measurement, shape ``(R, 2^width)``.
        patterns: Each edge's output pattern, shape ``(R, output_words)``.
        last: The action that ends each edge, shape ``(R,)``.
        symbols: Each node's symbols, shape ``(R, num_words)``.
        weights: Each node's measured weights, shape ``(R, 2)``.

    """

    request: np.ndarray
    outcome: np.ndarray
    states: np.ndarray
    patterns: np.ndarray
    last: np.ndarray
    symbols: np.ndarray
    weights: np.ndarray


@jit
def _gather(nodes, bits, arrays, states, has_state):
    """Copy the cached states of the parents of the edges ``(nodes, bits)``.

    Returns which of the edges are still missing.
    """
    missing = np.zeros(len(nodes), dtype=np.bool_)
    for r in range(len(nodes)):
        if nodes[r] < 0:
            missing[r] = True
            continue
        if arrays.node_child[nodes[r], bits[r]] > 0:
            continue
        missing[r] = True
        slot = arrays.node_slot[nodes[r]] - 1
        if slot >= 0:
            states[r] = arrays.states[slot]
            has_state[r] = True
    return missing


@jit
def _threshold(w0, w1):
    """Return the chance of outcome 1 times 2^64, as a node stores it."""
    one = w1 / (w0 + w1) if w0 + w1 > _TINY else 0.0
    return np.uint64(min(one * 2.0**64, _FULL))


@jit
def _grow(
    nodes,
    bits,
    key_symbols,
    starts,
    has_state,
    shots,
    depth,
    patterns,
    ptr,
    checks,
    ready,
    draws,
    stride,
    grown,
    arrays,
    tape,
    fixed,
    negligible,
):
    """Build each request's missing edges along the path of one of its shots.

    Request ``q`` starts at outcome ``bits[q]`` of node ``nodes[q]``, or at the
    root of a key with symbols ``key_symbols[q]`` if the node is -1, from the
    state ``starts[q]`` (replayed unless ``has_state[q]``). It follows the path
    that shot ``shots[q]`` takes, drawing its outcomes from its stream as its
    walk would, until the tape ends, the shot is discarded or the rows run out.

    Returns:
        ``(rows, deviation)``: the rows written and the largest deviation of a
        measured norm from 1.

    """
    num_actions = len(tape.kind)
    measured = np.zeros(3)
    shot_pattern = np.zeros(patterns.shape[1], dtype=np.uint64)
    rows = 0
    for q in range(len(nodes)):
        # NOTE: every later request keeps at least one row.
        room = len(grown.last) - rows - (len(nodes) - 1 - q)
        if room <= 0:
            break
        amps = starts[q : q + 1]
        if nodes[q] < 0:
            amps[0] = 0
            amps[0, 0] = 1
            symbols = key_symbols[q].copy()
            a, bit = 0, 0
        else:
            node, bit = nodes[q], bits[q]
            if not has_state[q]:
                _replay(node, tape, fixed, arrays, amps, negligible)
            symbols = arrays.node_symbols[node].copy()
            a = arrays.node_action[node]
        s = shots[q]
        shot_pattern[:] = patterns[s]
        check, d = ptr[s], depth[s]
        first = True
        while room > 0:
            r = rows
            rows += 1
            room -= 1
            pattern = grown.patterns[r]
            pattern[:] = 0
            grown.request[r], grown.outcome[r] = q, bit
            if not first or nodes[q] >= 0:
                weight = (
                    arrays.node_weight[nodes[q], bit]
                    if first
                    else grown.weights[r - 1, bit]
                )
                _take(amps, symbols, a, bit, weight, tape, fixed, pattern)
                a += 1
            first = False
            a = _advance(amps, symbols, a, tape, fixed, pattern, negligible, measured)
            grown.last[r] = a
            grown.symbols[r] = symbols
            grown.weights[r, 0], grown.weights[r, 1] = measured[1], measured[2]
            grown.states[r] = amps[0]
            if a == num_actions:
                break
            discarded = False
            for w in range(len(shot_pattern)):
                shot_pattern[w] ^= pattern[w]
            while check < len(checks) and ready[checks[check]] < a:
                o = checks[check]
                check += 1
                if (shot_pattern[o >> 6] >> np.uint64(o & 63)) & np.uint64(1):
                    discarded = True
                    break
            if discarded:
                break
            _, z = splitmix64(draws + np.uint64(s * stride + d) * GOLDEN)
            d += 1
            bit = 1 if z < _threshold(measured[1], measured[2]) else 0
    return rows, measured[0]


@jit
def _keep(node, state, arrays, counts):
    """Keep the state of a new node in a slot, evicting the oldest if all are used."""
    if counts[_SLOTS] < len(arrays.slot_owner):
        slot = counts[_SLOTS]
        counts[_SLOTS] += 1
    else:
        slot = counts[_EVICT] % len(arrays.slot_owner)
        counts[_EVICT] += 1
        owner = arrays.slot_owner[slot] - 1
        if owner >= 0:
            arrays.node_slot[owner] = 0
    arrays.states[slot] = state
    arrays.slot_owner[slot] = node + 1
    arrays.node_slot[node] = slot + 1


@jit
def _append(grown, r, key, parent, arrays, counts, num_actions):
    """Append the edge of row ``r`` and its node; return the edge and the node."""
    edge = counts[_EDGES]
    counts[_EDGES] += 1
    arrays.edge_pattern[edge] = grown.patterns[r]
    a = grown.last[r]
    arrays.edge_last[edge] = a
    if a == num_actions:
        arrays.edge_next[edge] = END
        return edge, -1
    node = counts[_NODES]
    counts[_NODES] += 1
    arrays.node_action[node] = a
    w0, w1 = grown.weights[r, 0], grown.weights[r, 1]
    arrays.node_threshold[node] = _threshold(w0, w1)
    arrays.node_weight[node, 0], arrays.node_weight[node, 1] = w0, w1
    arrays.node_symbols[node] = grown.symbols[r]
    arrays.node_parent[node] = parent
    arrays.node_outcome[node] = grown.outcome[r]
    arrays.node_key[node] = key
    arrays.node_child[node, 0] = 0
    arrays.node_child[node, 1] = 0
    arrays.node_slot[node] = 0
    _keep(node, grown.states[r], arrays, counts)
    arrays.edge_next[edge] = node
    return edge, node


@jit
def _link(nodes, bits, keys, key_symbols, grown, rows, arrays, counts, num_actions):
    """Append the rows of every request whose first edge is still missing.

    A link is written after its edge and node, so that a reader that finds it
    within the published counts finds them complete.
    """
    r = 0
    while r < rows:
        q = grown.request[r]
        end = r
        while end < rows and grown.request[end] == q:
            end += 1
        node = nodes[q]
        if node < 0:
            if _find(keys[q], arrays.slot_keys, arrays.slot_roots, counts[_EDGES]) >= 0:
                r = end
                continue
            key = counts[_KEYS]
            counts[_KEYS] += 1
            arrays.key_symbols[key] = key_symbols[q]
        else:
            if arrays.node_child[node, bits[q]] > 0:
                r = end
                continue
            key = arrays.node_key[node]
        parent = node
        for row in range(r, end):
            edge, child = _append(grown, row, key, parent, arrays, counts, num_actions)
            if parent < 0:
                mask = np.uint64(len(arrays.slot_roots) - 1)
                slot = _hash(keys[q]) & mask
                while arrays.slot_roots[slot] != 0:
                    slot = (slot + np.uint64(1)) & mask
                arrays.slot_keys[slot] = keys[q]
                arrays.slot_roots[slot] = edge + 1
            else:
                bit = grown.outcome[row]
                arrays.node_child[parent, bit] = edge + 1
                if arrays.node_child[parent, 1 - bit] > 0:
                    slot = arrays.node_slot[parent] - 1
                    if slot >= 0:
                        arrays.slot_owner[slot] = 0
                        arrays.node_slot[parent] = 0
            parent = child
        r = end
