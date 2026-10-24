"""Compiled samplers for measurements and detectors."""

from __future__ import annotations

import threading
import warnings
from collections import deque
from collections.abc import Callable
from math import ceil, inf
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, overload

import jax
import jax.numpy as jnp
import numba
import numpy as np
import psutil

from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.core.types import CompiledComponent, CompiledProgram, SamplingStrategy
from tsim.noise.channels import _BLOCK_SHOTS, ChannelSampler, FireTables
from tsim.statevector.program import STABILIZER_RANK_HINT, StatevectorComponent
from tsim.statevector.shared import (
    SharingLayout,
    lane_bound,
    lane_bytes,
    sample_shared_component,
    sharing_layout,
)
from tsim.statevector.shots import (
    ShotLayout,
    gather,
    gather_symbols,
    sample_shots,
    scatter,
    shot_layout,
)
from tsim.statevector.tables import OutcomeTables, table_keys
from tsim.statevector.trees import MIN_SLOTS, OutcomeTrees
from tsim.utils.cuda_helpers import copy_d2h
from tsim.utils.kernels import kernel, run_tasks
from tsim.utils.rng import GOLDEN

if TYPE_CHECKING:
    from jax import Array as PRNGKey

    from tsim.circuit import Circuit

# NOTE: a component's trees fall back to lanes after a call, past the first
# under its mask, in which they built more nodes per shot of the component than
# this, over at least TREE_MIN_SHOTS shots. At about half a node per shot, trees cost about what lanes do on one
# thread and more on several (cultivation d=5 without postselection, M4 Pro).
TREE_MAX_NODES_PER_SHOT = 0.3
TREE_MIN_SHOTS = 2**10
# NOTE: and after a call that left its trees larger than this.
TREE_MAX_BYTES = 2**31


class ComponentPlan(NamedTuple):
    """How a shared statevector component samples a batch.

    Attributes:
        tables: Its outcome tables, if some keys draw from them; the other shots
            run on lanes.
        keys: The keys that draw from the tables, every key if None.
        trees: Its outcome trees, if it has no tables, or None for lanes.

    """

    tables: OutcomeTables | None
    keys: np.ndarray | None
    trees: OutcomeTrees | None


ComponentPlans = tuple[ComponentPlan | None, ...]

# NOTE: beyond 2^16 lanes, statevector batches cost memory without being faster.
STATEVECTOR_MAX_BATCH_SIZE = 2**16

# NOTE: without a given batch size, a call is split into a batch per thread when
# every batch gets at least MIN_THREAD_BATCH shots, whose fixed costs would
# dominate otherwise. A smaller call stays one batch, which the parallel kernels
# spread over the threads instead, at about 0.1 ms per parallel loop. That is
# slower than a batch per thread of MIN_LANE_BATCH shots or more, unless sharing
# keeps the shots on fewer lanes than such a batch has shots.
MIN_THREAD_BATCH = 2**12
MIN_LANE_BATCH = 2**8

# NOTE: postselection screens shots for direct discards in chunks of at most
# this many, several chunks per thread at a time, so that calls of a few batches
# still keep every thread busy and slow cores take fewer chunks.
SCREEN_SHOTS = 2**14
# NOTE: the fused shot kernel splits a call into about this many chunks, of at
# most FUSED_SHOTS shots and FUSED_BYTES of shot states each.
FUSED_CHUNKS = 16
FUSED_SHOTS = 2**16
FUSED_BYTES = 2**22
SCREEN_CHUNKS_PER_THREAD = 2


def _sample_component(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey, jax.Array]:
    """Sample from component using autoregressive sampling.

    Args:
        component: The compiled component to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Tuple of (samples, next_key, max_norm_deviation) where samples has
        shape (batch_size, num_outputs_for_component).

    """
    batch_size = f_params.shape[0]
    num_outputs = len(component.compiled_scalar_graphs) - 1

    f_selected = f_params[:, component.f_selection].astype(jnp.bool_)

    # Pre-allocate output array with final shape to avoid dynamic hstack
    m_accumulated = jnp.zeros((batch_size, num_outputs), dtype=jnp.bool_)

    # First circuit is normalization (only f-params)
    prev = jnp.abs(evaluate(component.compiled_scalar_graphs[0], f_selected))

    ones = jnp.ones((batch_size, 1), dtype=jnp.bool_)
    zero = jnp.zeros((1, 1), dtype=jnp.bool_)

    max_norm_deviation = jnp.array(0.0)

    # Autoregressive sampling for remaining circuits
    for i, circuit in enumerate(component.compiled_scalar_graphs[1:]):
        # Evaluate the real batch with trying_bit=1, plus one extra row for the
        # first sample's prefix with trying_bit=0 used by the norm check.
        params = jnp.hstack([f_selected, m_accumulated[:, :i], ones])
        check_row = jnp.hstack([f_selected[:1], m_accumulated[:1, :i], zero])
        probs = jnp.abs(evaluate(circuit, jnp.vstack([params, check_row])))
        p1 = probs[:batch_size]
        p0_single = probs[-1]

        norm = (p0_single + p1[0]) / prev[0]
        max_norm_deviation = jnp.maximum(max_norm_deviation, jnp.abs(norm - 1.0))

        key, subkey = jax.random.split(key)
        bits = jax.random.bernoulli(subkey, p=p1 / prev)
        m_accumulated = m_accumulated.at[:, i].set(bits)

        # Update prev using chain rule
        prev = jnp.where(bits, p1, prev - p1)

    return m_accumulated, key, max_norm_deviation


@jax.jit
def _sample_component_jit(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey, jax.Array]:
    """JIT-compiled version of _sample_component."""
    return _sample_component(component, f_params, key)


def sample_component(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey, jax.Array]:
    """Sample outputs from a single component.

    Args:
        component: The compiled component to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Tuple of (samples, next_key, max_norm_deviation) where samples has shape
        (batch_size, num_outputs_for_component).

    """
    # Skip JIT for small components (overhead not worth it)
    if len(component.output_indices) <= 1:
        return _sample_component(component, f_params, key)
    return _sample_component_jit(component, f_params, key)


def _check_norm(max_norm_deviation: float | jax.Array, threshold: float) -> None:
    """Raise on a vanishing normalization; warn when it drifts beyond ``threshold``."""
    if np.isclose(max_norm_deviation, 1):
        raise ValueError(
            "A vanishing marginal probability distribution was encountered (normalization 0). "
            "This is likely the result of an underflow error. Please report this "
            "as a bug at https://github.com/QuEraComputing/tsim/issues/new."
        )  # pragma: no cover
    # NOTE: written so that a NaN deviation also warns.
    if not max_norm_deviation <= threshold:
        warnings.warn(
            "A marginal probability was not normalized correctly "
            f"(normalization deviated from 1 by {max_norm_deviation:.1e}). "
            "This is likely a floating point precision issue.",
            stacklevel=3,
        )


def _direct_bits(program: CompiledProgram, f_params: jax.Array) -> jax.Array:
    """Evaluate direct outputs, including constants with f-index -1."""
    indices = program.direct_f_indices
    has_f = indices >= 0
    bits = jnp.broadcast_to(program.direct_flips, (f_params.shape[0], len(indices)))
    return bits.at[:, has_f].set(
        bits[:, has_f] ^ f_params[:, indices[has_f]].astype(jnp.bool_)
    )


def sample_program(
    program: CompiledProgram, f_params: jax.Array, key: PRNGKey
) -> jax.Array:
    """Sample all outputs of a program of stabilizer-rank components with JAX.

    Args:
        program: The compiled program to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Samples array of shape (batch_size, num_outputs), reordered to
        match the original output indices.

    """
    if program.num_outputs == 0:
        return jnp.zeros((f_params.shape[0], 0), dtype=jnp.bool_)
    results = []
    if len(program.direct_f_indices) > 0:
        results.append(_direct_bits(program, f_params))
    for component in program.components:
        assert isinstance(component, CompiledComponent)
        samples, key, max_norm_deviation = sample_component(component, f_params, key)
        _check_norm(max_norm_deviation, component.norm_warning_threshold)
        results.append(samples)
    combined = jnp.concatenate(results, axis=1)
    if program.output_reindex is not None:
        combined = combined[:, program.output_reindex]
    return combined


def sample_statevector_program(
    program: CompiledProgram,
    f_params: np.ndarray,
    seed: int,
    layouts: tuple[SharingLayout, ...],
    direct_flips: np.ndarray,
    *,
    share: bool,
    sharing_stats: deque[dict] | None = None,
    plans: ComponentPlans | None = None,
    record: Callable[[int, int, dict], None] | None = None,
) -> np.ndarray:
    """Sample all outputs of a program of statevector components, without JAX.

    Args:
        program: The compiled program to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        seed: Seed of the batch's random numbers, below 2^64.
        layouts: The sharing layout of each component.
        direct_flips: The flip of each direct output, with its reference bit.
        share: Whether shots with equal histories share a lane.
        sharing_stats: If given, the executor appends its lane counts, and a
            component with outcome tables or trees their counts.
        plans: How each component samples, or None for lanes.
        record: If given, called with each component's index, the batch's
            shots and the component's counts.

    Returns:
        Samples array of shape (batch_size, num_outputs) in the original order.

    """
    rows = f_params.shape[0]
    results = []
    for index, (component, layout) in enumerate(
        zip(program.components, layouts, strict=True)
    ):
        assert isinstance(component, StatevectorComponent)
        component_seed = (seed + index * int(GOLDEN)) % 2**64
        plan = None if plans is None else plans[index]
        if plan is not None and plan.tables is not None:
            samples, max_norm_deviation, stats = plan.tables.sample(
                f_params, component_seed, layout, plan.keys
            )
        elif plan is not None and plan.trees is not None:
            samples, max_norm_deviation, stats = plan.trees.sample(
                f_params, component_seed, layout
            )
        else:
            samples, max_norm_deviation, stats = sample_shared_component(
                component, layout, f_params, component_seed, share
            )
        if sharing_stats is not None:
            sharing_stats.append(stats)
        if record is not None:
            record(index, rows, stats)
        _check_norm(max_norm_deviation, component.norm_warning_threshold)
        results.append(samples)
    component_bits = (
        np.concatenate(results, axis=1) if results else np.zeros((rows, 0), np.bool_)
    )
    out = np.empty((rows, program.num_outputs), dtype=np.bool_)
    _gather_outputs(
        out.size,
        f_params,
        program.direct_f_indices,
        direct_flips,
        component_bits,
        program.output_order,
        out,
    )
    return out


@kernel
def _write_direct_detectors(f, f_indices, flips, columns, postselect, out):
    """Clear every row of ``out`` and write its direct detector ``j`` at ``columns[j]``.

    Detector ``j`` is ``f[row, f_indices[j]] ^ flips[j]``, or just ``flips[j]``
    when its index is -1. Returns which rows have a postselected detector of 1.
    """
    # NOTE: clearing a row element by element, and testing it for discards
    # without a branch per detector, is several times faster.
    discarded = np.zeros(f.shape[0], dtype=np.bool_)
    for row in numba.prange(f.shape[0]):
        for column in range(out.shape[1]):
            out[row, column] = False
        discard = False
        for j in range(len(f_indices)):
            bit = flips[j]
            if f_indices[j] >= 0:
                bit ^= f[row, f_indices[j]] != 0
            out[row, columns[j]] = bit
            discard |= postselect[j] & bit
        discarded[row] = discard
    return discarded


@kernel
def _gather_outputs(f, f_indices, flips, component_bits, order, out):
    """Write every row's outputs into ``out`` in their original order.

    Output ``k`` of the concatenation goes to column ``order[k]``: first the
    direct outputs (just ``flips[j]`` when ``f_indices[j]`` is -1), then
    ``component_bits``.
    """
    num_direct = len(f_indices)
    for row in numba.prange(f.shape[0]):
        for j in range(num_direct):
            bit = flips[j]
            if f_indices[j] >= 0:
                bit ^= f[row, f_indices[j]] != 0
            out[row, order[j]] = bit
        for j in range(component_bits.shape[1]):
            out[row, order[num_direct + j]] = component_bits[row, j]


def _nbytes(tree: object) -> int:
    if isinstance(tree, StatevectorComponent):
        return sum(array.nbytes for array in tree.tape)
    return sum(
        v.nbytes for v in jax.tree_util.tree_leaves(tree) if isinstance(v, jax.Array)
    )


class _Backend(Protocol):
    """Samples batches of a program's non-direct components.

    Attributes:
        prunes: Whether shots that a masked component detector discards stop
            being simulated.
        max_batch_size: The largest batch worth sampling at once, if any.

    """

    prunes: bool
    max_batch_size: int | None

    def max_threads(self) -> int:
        """Return how many batches may run at once."""
        ...

    def finish_call(self) -> None:
        """Note that a call's batches are done."""
        ...

    def next_seed(self) -> Any:
        """Return the seed of the next batch, in the order of the batches."""
        ...

    def memory_available(self) -> int:
        """Return the free memory of the device that holds the batches."""
        ...

    def peak_bytes_per_shot(self, num_f: int) -> int:
        """Return the peak memory of one shot of a batch."""
        ...

    def max_lanes(
        self, mask: np.ndarray | None, flip: np.ndarray | None = None
    ) -> float:
        """Return the most states a batch simulates, however many shots it has.

        ``mask`` marks the postselected detectors and ``flip`` the reference bits
        of the outputs, as in ``sample``; ``inf`` means one per shot.
        """
        ...

    def sample(
        self,
        f_params: np.ndarray,
        seed: Any,
        *,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
        pad_to: int | None = None,
    ) -> Any:
        """Return the outputs of one batch, XOR ``flip`` (one bit per output).

        ``mask`` marks the postselected detectors and ``pad_to`` the number of
        rows the batch may be padded to.
        """
        ...

    def concatenate(self, batches: list) -> np.ndarray:
        """Return the batches' outputs as one host array."""
        ...


class _StabilizerRankBackend:
    """Samples stabilizer-rank programs with JAX, one batch at a time."""

    prunes: bool = False
    max_batch_size: int | None = None

    def __init__(self, program: CompiledProgram, seed: int) -> None:
        self._program = program
        self._key = jax.random.key(seed)

    def max_threads(self) -> int:
        return 1

    def finish_call(self) -> None:
        pass

    def next_seed(self) -> PRNGKey:
        self._key, key = jax.random.split(self._key)
        return key

    def memory_available(self) -> int:
        device = jax.devices()[0]
        if device.platform != "gpu":
            return psutil.virtual_memory().available
        stats = device.memory_stats()
        return stats.get("bytes_limit", 8 * 1024**3) - stats.get("bytes_in_use", 0)

    def peak_bytes_per_shot(self, num_f: int) -> int:
        peak = 0
        for component in self._program.components:
            assert isinstance(component, CompiledComponent)
            for circuit in component.compiled_scalar_graphs:
                G = circuit.num_graphs
                max_a = circuit.node_phases.phases.shape[1]
                max_b = circuit.phase_terms.coeffs.shape[1]
                max_c = circuit.pi_products.psi_params.shape[1]
                max_d = circuit.phase_pairs.alpha.shape[1]
                max_e = circuit.float_terms.table.shape[1]
                largest = max(max_a * 16, max_b * 4, max_c * 4, max_d * 16, max_e * 16)
                peak = max(peak, G * largest * 3)
        return peak

    def max_lanes(
        self, mask: np.ndarray | None, flip: np.ndarray | None = None
    ) -> float:
        return inf

    def sample(
        self,
        f_params: np.ndarray,
        seed: PRNGKey,
        *,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
        pad_to: int | None = None,
    ) -> jax.Array:
        """Sample the batch, padded with its first row to ``pad_to`` rows.

        Padding keeps the batch shapes, and so JAX's compilations, few.
        """
        count = len(f_params)
        if pad_to is not None and count < pad_to:
            padding = np.repeat(f_params[:1], pad_to - count, axis=0)
            f_params = np.concatenate([f_params, padding])
        samples = sample_program(self._program, jnp.asarray(f_params), seed)
        if len(f_params) > count:
            samples = samples[:count]
        return samples if flip is None else samples ^ flip

    def concatenate(self, batches: list) -> np.ndarray:
        # NOTE: one device-to-host copy of the concatenation is much faster than
        # one per batch for large outputs.
        combined = batches[0] if len(batches) == 1 else jnp.concatenate(batches)
        jax.block_until_ready(combined)
        return copy_d2h(combined)


class _StatevectorBackend:
    """Samples statevector programs with numba kernels on the host.

    Up to ``numba.get_num_threads()`` batches run at once. With sharing, a
    component samples from outcome tables when the noise model says that they
    cost less than lanes (``table_keys``), and otherwise its shots walk its
    outcome trees. The choice of tables depends on the postselection mask,
    since the shots that the direct detectors discard never reach a component,
    so it is made per mask; each component's tables and trees are shared by
    every mask and persist across calls. Under a mask whose trees built more
    than ``TREE_MAX_NODES_PER_SHOT`` nodes per shot in a call, the component's
    later calls run on lanes, which share within a batch at less cost.
    """

    prunes: bool = True
    max_batch_size: int | None = STATEVECTOR_MAX_BATCH_SIZE

    def __init__(
        self,
        program: CompiledProgram,
        seed: int,
        num_detectors: int,
        share: bool,
        noise: FireTables,
    ) -> None:
        self._program = program
        self._num_detectors = num_detectors
        self._share = share
        self._seeds = np.random.default_rng([seed, 1])
        self._noise = noise
        self._layouts: dict[bytes, tuple[tuple[SharingLayout, ...], np.ndarray]] = {}
        self._stores: dict[int, OutcomeTables] = {}
        self._trees: dict[int, OutcomeTrees] = {}
        self._plans: dict[bytes, ComponentPlans] = {}
        self._on_lanes: set[tuple[bytes, int]] = set()
        self._warm: set[tuple[bytes, int]] = set()
        # NOTE: per screen and component, the shots of the batches in this call in
        # which some shots walked its trees, and the nodes they built.
        self._walked: dict[tuple[bytes, int], list[int]] = {}
        self._fused: dict[tuple[bytes], tuple] = {}
        self._lock = threading.Lock()
        self.sharing_stats: deque[dict] = deque(maxlen=10_000)

    def max_threads(self) -> int:
        return numba.get_num_threads()

    def next_seed(self) -> int:
        return int(self._seeds.integers(0, 2**64, dtype=np.uint64))

    def memory_available(self) -> int:
        return psutil.virtual_memory().available

    def peak_bytes_per_shot(self, num_f: int) -> int:
        # NOTE: plus the f-params and the output records, twice over.
        records = 2 * num_f + 2 * self._program.num_outputs
        return max(
            (
                lane_bytes(component) + records
                for component in self._program.components
                if isinstance(component, StatevectorComponent)
            ),
            default=0,
        )

    def max_lanes(
        self, mask: np.ndarray | None, flip: np.ndarray | None = None
    ) -> float:
        if not self._share:
            return inf
        layouts, _ = self.layouts(mask, flip)
        return max(map(lane_bound, layouts), default=0)

    def sample(
        self,
        f_params: np.ndarray,
        seed: int,
        *,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
        pad_to: int | None = None,
    ) -> np.ndarray:
        """Sample the batch; shots that a ``mask`` detector discards are pruned."""
        layouts, direct_flips = self.layouts(mask, flip)
        screen = self.screened(mask, flip).tobytes()

        def record(index: int, shots: int, stats: dict) -> None:
            if "tree shots" in stats:
                with self._lock:
                    counts = self._walked.setdefault((screen, index), [0, 0])
                    counts[0] += shots
                    counts[1] += stats["nodes built"]

        return sample_statevector_program(
            self._program,
            f_params,
            seed,
            layouts,
            direct_flips,
            share=self._share,
            sharing_stats=self.sharing_stats,
            plans=self.plans(mask, flip),
            record=record,
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def fused(
        self, mask: np.ndarray | None, flip: np.ndarray | None
    ) -> tuple[ShotLayout, ComponentPlans, tuple[SharingLayout, ...]] | None:
        """Return the shot layout, plans and layouts for the fused kernel, if any.

        The fused kernel needs noise to fold, and every statevector component on
        tables or trees: lanes share within a batch, and the kernel's batches
        are small.
        """
        program = self._program
        if not len(self._noise.p_fire) or not program.components:
            return None
        plans = self.plans(mask, flip)
        if any(plan is None for plan in plans):
            return None
        layouts, direct_flips = self.layouts(mask, flip)
        # NOTE: the layouts are cached per mask and flips, and the plans per
        # screen; a fused layout holds both, so it is valid while both are.
        key = (_layout_key(program.num_outputs, self._num_detectors, mask, flip),)
        cached = self._fused.get(key)
        if cached is not None and cached[1] is plans and cached[2] is layouts:
            return cached
        columns = []
        for component, plan in zip(program.components, plans, strict=True):
            assert isinstance(component, StatevectorComponent) and plan is not None
            engine = plan.tables if plan.tables is not None else plan.trees
            assert engine is not None
            key_columns = engine.key_columns
            if key_columns.ndim == 1:
                key_columns = key_columns[:, None]
            columns.append(
                (
                    np.asarray(component.tape.f_selection),
                    component.num_words,
                    key_columns,
                    engine.output_columns,
                )
            )
        num_f = 1 + max(
            int(self._noise.flip_cols.max(initial=-1)),
            int(program.direct_f_indices.max(initial=-1)),
            *(int(selection.max(initial=-1)) for selection, *_ in columns),
        )
        layout = shot_layout(
            self._noise,
            num_f,
            program.direct_f_indices,
            direct_flips,
            program.output_order[: len(program.direct_f_indices)],
            program.num_outputs,
            self._num_detectors,
            mask,
            columns,
        )
        fused = (layout, plans, layouts)
        self._fused[key] = fused
        return fused

    def sample_components(
        self,
        fused: tuple[ShotLayout, ComponentPlans, tuple[SharingLayout, ...]],
        state: np.ndarray,
        survivors: np.ndarray,
        out: np.ndarray,
        seed: int,
        screen: bytes,
    ) -> None:
        """Sample the components of the ``survivors`` from their shot states.

        Writes their outputs into the rows ``survivors`` of ``out``.
        """
        layout, plans, layouts = fused
        program = self._program
        for index, (component, plan, words, component_layout) in enumerate(
            zip(program.components, plans, layout.components, layouts, strict=True)
        ):
            assert isinstance(component, StatevectorComponent)
            assert plan is not None and words is not None
            component_seed = (seed + index * int(GOLDEN)) % 2**64
            keys = gather(state, survivors, words.key, words.key_words)
            patterns = gather(state, survivors, words.pattern, words.pattern_words)

            def symbols_of(
                chosen: np.ndarray, words=words, num_words=component.num_words
            ) -> np.ndarray:
                return gather_symbols(
                    state, survivors[chosen], words.symbols, num_words
                )

            if plan.tables is not None:
                samples, deviation, stats = plan.tables.sample_keyed(
                    keys,
                    patterns,
                    symbols_of,
                    component_seed,
                    component_layout,
                    plan.keys,
                )
            else:
                assert plan.trees is not None
                samples, deviation, stats = plan.trees.sample_keyed(
                    keys, patterns, symbols_of, component_seed, component_layout
                )
            self.sharing_stats.append(stats)
            if "tree shots" in stats:
                with self._lock:
                    counts = self._walked.setdefault((screen, index), [0, 0])
                    counts[0] += len(survivors)
                    counts[1] += stats["nodes built"]
            _check_norm(deviation, component.norm_warning_threshold)
            columns = np.asarray(component.output_indices, dtype=np.int64)
            scatter(out, survivors, columns, samples)

    def finish_call(self) -> None:
        """Move to lanes the trees that built too many nodes per shot in this call.

        Only the call's totals count, which don't depend on how its batches
        were spread over the threads.
        """
        with self._lock:
            walked, self._walked = self._walked, {}
        for (screen, index), (shots, built) in walked.items():
            many = shots >= TREE_MIN_SHOTS and built > TREE_MAX_NODES_PER_SHOT * shots
            warm = (screen, index) in self._warm
            if shots >= TREE_MIN_SHOTS:
                self._warm.add((screen, index))
            if (many and warm) or self._trees[index].nbytes > TREE_MAX_BYTES:
                self._on_lanes.add((screen, index))
                self._plans.pop(screen, None)

    def plans(
        self, mask: np.ndarray | None = None, flip: np.ndarray | None = None
    ) -> ComponentPlans:
        """Return how each component samples under ``mask``, None for lanes.

        The tables' keys depend on the mask only through the f-bits it screens.
        """
        screen = self.screened(mask, flip).tobytes()
        plans = self._plans.get(screen)
        if plans is not None:
            return plans
        picks: list[ComponentPlan | None] = []
        for index, component in enumerate(self._program.components):
            if not (self._share and isinstance(component, StatevectorComponent)):
                picks.append(None)
                continue
            keys = table_keys(component, self._noise, self.screened(mask, flip))
            tables = None
            if keys is not None:
                with self._lock:
                    if index not in self._stores:
                        self._stores[index] = OutcomeTables(component)
                    tables = self._stores[index]
            trees = None
            if tables is None and (screen, index) not in self._on_lanes:
                with self._lock:
                    if index not in self._trees:
                        self._trees[index] = OutcomeTrees(component)
                    trees = self._trees[index]
                if trees.max_slots < MIN_SLOTS:
                    trees = None
            if tables is None and trees is None:
                picks.append(None)
                continue
            picks.append(ComponentPlan(tables, keys, trees))
        # NOTE: batches run on threads; all of them use the first plans stored.
        return self._plans.setdefault(screen, tuple(picks))

    def screened(self, mask: np.ndarray | None, flip: np.ndarray | None) -> np.ndarray:
        """Return the f-bits that discard a shot before the components under ``mask``.

        A masked direct detector discards a shot when its f-bit differs from its
        flip. One with flip 0 discards the shots with a fault that flips it. One
        with flip 1 discards the shots without one, which the choice of tables
        doesn't model, so it is left out.
        """
        program = self._program
        _, direct_flips = self.layouts(mask, flip)
        columns = program.output_order[: len(program.direct_f_indices)]
        masked = np.zeros(len(columns), dtype=np.bool_)
        if mask is not None:
            is_detector = columns < self._num_detectors
            masked[is_detector] = mask[columns[is_detector]]
        screens = masked & ~direct_flips & (program.direct_f_indices >= 0)
        return program.direct_f_indices[screens]

    def layouts(
        self, mask: np.ndarray | None, flip: np.ndarray | None
    ) -> tuple[tuple[SharingLayout, ...], np.ndarray]:
        """Return the components' sharing layouts and the direct outputs' flips."""
        program = self._program
        postselected = np.zeros(program.num_outputs, dtype=np.bool_)
        if mask is not None:
            postselected[: self._num_detectors] = mask
        flip = np.zeros(program.num_outputs, dtype=np.bool_) if flip is None else flip
        flip = np.asarray(flip, dtype=np.bool_)
        key = _layout_key(program.num_outputs, self._num_detectors, mask, flip)
        if key not in self._layouts:
            layouts = []
            for component in program.components:
                assert isinstance(component, StatevectorComponent)
                outputs = list(component.output_indices)
                layouts.append(
                    sharing_layout(component, postselected[outputs], flip[outputs])
                )
            direct = program.output_order[: len(program.direct_f_indices)]
            self._layouts[key] = (tuple(layouts), program.direct_flips ^ flip[direct])
        return self._layouts[key]

    def concatenate(self, batches: list) -> np.ndarray:
        return np.concatenate(batches)


# NOTE: the noise buffers that threads keep between batches and calls hold at
# most this many bytes in total; larger requests get a new array each time.
SCRATCH_BYTES = 2**28
_SCRATCH: dict[int, np.ndarray] = {}
_SCRATCH_LOCK = threading.Lock()


def _scratch(rows: int, columns: int) -> np.ndarray:
    """Return this thread's reusable uint8 buffer of shape ``(rows, columns)``.

    The first write to each page of a new array faults, and the kernel serves
    those faults one at a time across threads, so batches reuse their thread's
    buffer instead of allocating one each, within ``SCRATCH_BYTES``.
    """
    thread = threading.get_ident()
    with _SCRATCH_LOCK:
        buffer = _SCRATCH.get(thread)
        if (
            buffer is not None
            and buffer.shape[0] >= rows
            and buffer.shape[1] == columns
        ):
            return buffer[:rows]
        kept = sum(b.nbytes for t, b in _SCRATCH.items() if t != thread)
        buffer = np.empty((rows, columns), dtype=np.uint8)
        if kept + buffer.nbytes <= SCRATCH_BYTES:
            _SCRATCH[thread] = buffer
        else:
            _SCRATCH.pop(thread, None)
        return buffer


def _layout_key(
    num_outputs: int,
    num_detectors: int,
    mask: np.ndarray | None,
    flip: np.ndarray | None,
) -> bytes:
    """Return the cache key of a postselection mask and the outputs' flips."""
    postselected = np.zeros(num_outputs, dtype=np.bool_)
    if mask is not None:
        postselected[:num_detectors] = mask
    flipped = np.zeros(num_outputs, dtype=np.bool_)
    if flip is not None:
        flipped[:] = flip
    return postselected.tobytes() + flipped.tobytes()


def _check_sizes(shots: int, batch_size: int | None) -> None:
    if shots < 0:
        raise ValueError(f"shots must be non-negative, got {shots}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")


class _CompiledSamplerBase:
    """Base class for compiled samplers with common initialization logic."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool,
        mode: Literal["sequential", "joint"],
        strategy: SamplingStrategy = "active_statevector",
        seed: int | None = None,
        trajectory_sharing: bool | None = None,
    ):
        """Initialize the sampler by compiling the circuit.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, sample detectors/observables instead of measurements.
            mode: Compilation mode - "sequential" for autoregressive, "joint" for probabilities.
            strategy: "active_statevector" (the default) to simulate each
                non-direct component on a dense state vector over its active
                qubits, or a stabilizer rank decomposition strategy ("cat5",
                "bss", "cutting"). "active_statevector" samples on the CPU, up
                to ``numba.get_num_threads()`` batches at once, each with its
                own memory.
            seed: Random seed. If None, a random seed is generated. Note that
                deterministic results are only guaranteed for a fixed batch size
                and fixed reference sample settings.
            trajectory_sharing: With ``strategy="active_statevector"``, simulate shots
                with equal histories on one lane. A component whose shots mostly
                share a few noise patterns draws them from outcome tables instead,
                which later calls reuse. Defaults to True for
                ``"active_statevector"`` and False otherwise.

        Raises:
            ValueError: If ``trajectory_sharing`` is combined with a
                stabilizer-rank strategy.

        """
        if trajectory_sharing is None:
            trajectory_sharing = strategy == "active_statevector"
        if trajectory_sharing and strategy != "active_statevector":
            raise ValueError("trajectory_sharing needs strategy='active_statevector'")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))
        self._trajectory_sharing = trajectory_sharing

        prepared = prepare_graph(circuit, sample_detectors=sample_detectors)
        self._program = compile_program(prepared, mode=mode, strategy=strategy)

        channel_seed = int(np.random.default_rng(seed).integers(0, 2**30))
        self._channel_sampler = ChannelSampler(
            channel_probs=prepared.channel_probs,
            error_transform=prepared.error_transform,
            seed=channel_seed,
        )

        self.circuit = circuit
        self._num_detectors = prepared.num_detectors
        self._backend: _Backend = (
            _StatevectorBackend(
                self._program,
                seed,
                self._num_detectors,
                trajectory_sharing,
                self._channel_sampler.fire_tables,
            )
            if strategy == "active_statevector"
            else _StabilizerRankBackend(self._program, seed)
        )

        program = self._program
        # NOTE: _gather_outputs writes column output_order[k] without bounds checks.
        assert np.array_equal(
            np.sort(program.output_order), np.arange(program.num_outputs)
        )
        columns = program.output_order[: len(program.direct_f_indices)]
        is_detector = columns < self._num_detectors
        self._direct_detectors = (
            program.direct_f_indices[is_detector],
            program.direct_flips[is_detector],
            columns[is_detector],
        )
        self._direct_detector_mask = np.zeros(self._num_detectors, dtype=np.bool_)
        self._direct_detector_mask[columns[is_detector]] = True

        statevector = [
            c for c in program.components if isinstance(c, StatevectorComponent)
        ]
        if statevector:
            peak = self._peak_bytes_per_sample()
            if peak > self._memory_budget():
                widest = max(statevector, key=lambda c: c.width)
                raise ValueError(
                    f"strategy='active_statevector' needs {peak} bytes per shot for the "
                    f"component with outputs {list(widest.output_indices)} (width "
                    f"{widest.width}), more than half of the available memory. "
                    f"{STABILIZER_RANK_HINT}"
                )

    def _compute_reference_sample(self) -> np.ndarray:
        """Return the noiseless reference sample (all f_params = 0).

        Does not advance the channel sampler RNG.
        """
        program = self._program
        if not program.components:
            reference = np.zeros(program.num_outputs, dtype=np.bool_)
            reference[program.output_order[: len(program.direct_f_indices)]] = (
                program.direct_flips
            )
            return reference
        num_f = self._channel_sampler.signature_matrix.shape[1]
        f_ref = np.zeros((1, num_f), dtype=np.uint8)
        samples = self._backend.sample(f_ref, self._backend.next_seed())
        return np.asarray(samples[0], dtype=np.bool_)

    def _resolve_batch_size(
        self,
        shots: int,
        batch_size: int | None,
        *,
        compute_reference: bool,
    ) -> int:
        """Choose a uniform batch size for ``shots`` samples.

        Without a ``batch_size``, batches are as large as memory allows, but split
        into a batch per thread when that pays (``_per_thread``), so that every
        thread has work.
        """
        if batch_size is None:
            batch_size = self._auto_batch_size(shots)
        if compute_reference and batch_size * ceil(shots / batch_size) == shots:
            batch_size += 1
        return batch_size

    def _auto_batch_size(self, shots: int) -> int:
        """Return the size of equal batches of ``shots``, one or more per thread."""
        num_batches = max(1, ceil(shots / self._estimate_batch_size()))
        return ceil(shots / max(num_batches, self._per_thread(shots)))

    def _per_thread(
        self,
        shots: int,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
    ) -> int:
        """Return the thread count if a batch per thread pays for ``shots``, else 1.

        It does when every batch gets at least ``MIN_THREAD_BATCH`` shots, or at
        least ``MIN_LANE_BATCH`` and the backend may need as many lanes as a
        batch has shots; ``mask`` and ``flip`` are as in ``max_lanes``.
        """
        threads = self._threads(1)
        if shots >= threads * MIN_THREAD_BATCH:
            return threads
        if shots < threads * MIN_LANE_BATCH:
            return 1
        lanes = self._backend.max_lanes(mask, flip)
        return threads if lanes >= ceil(shots / threads) else 1

    def _peak_bytes_per_sample(self) -> int:
        """Estimate peak device memory per sample from compiled program structure."""
        num_f = self._channel_sampler.signature_matrix.shape[1]
        return max(self._backend.peak_bytes_per_shot(num_f), 1)

    def _memory_budget(self) -> int:
        """Return the memory one batch may use: half of the free device memory."""
        return int(self._backend.memory_available() * 0.5)  # conservative

    def _threads(self, batch_size: int) -> int:
        """Return how many batches of ``batch_size`` shots run at once.

        That is the backend's limit, as long as the batches fit in the memory
        budget.
        """
        # NOTE: with sharing a batch may still hold a lane per shot.
        fits = self._memory_budget() // (self._peak_bytes_per_sample() * batch_size)
        return max(1, min(self._backend.max_threads(), fits))

    def _estimate_batch_size(self) -> int:
        """Estimate the largest batch size that fits in available device memory.

        Without sharing, every batch that runs at once holds a lane per shot. With
        sharing, lanes are allocated as trajectories split, so the estimate
        leaves room for a lane per shot in one batch only. The backend may cap
        the batch size further.
        """
        peak = self._peak_bytes_per_sample()
        if not self._trajectory_sharing:
            peak *= self._threads(1)
        batch_size = max(1, self._memory_budget() // peak)
        if self._backend.max_batch_size is not None:
            batch_size = min(batch_size, self._backend.max_batch_size)
        return batch_size

    @overload
    def _sample_batches(
        self,
        shots: int,
        batch_size: int | None = None,
        *,
        compute_reference: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def _sample_batches(
        self,
        shots: int,
        batch_size: int | None = None,
        *,
        compute_reference: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def _sample_batches(
        self,
        shots: int,
        batch_size: int | None = None,
        *,
        compute_reference: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample in batches and concatenate results.

        Args:
            shots: Number of samples to draw.
            batch_size: Samples per batch. Auto-determined if None.
            compute_reference: If True, add one extra sample to the first
                batch for a noiseless reference (f_params=0).

        Returns:
            Samples array, or (samples, reference) tuple when compute_reference=True.

        """
        _check_sizes(shots, batch_size)
        if shots == 0:
            empty = np.empty((0, self._program.num_outputs), dtype=np.bool_)
            if compute_reference:
                return empty, np.zeros(self._program.num_outputs, dtype=np.bool_)
            return empty

        if not self._program.components:
            samples = self._sample_direct(shots)
            if compute_reference:
                reference = self._compute_reference_sample()
                return samples, reference
            return samples
        if not compute_reference:
            fused = self._sample_fused(shots, batch_size, None, None)
            if fused is not None:
                return fused

        # NOTE: with a reference sample the batch size grows by one when the
        # batches would hold exactly ``shots``, so that the first batch's
        # reference sample fits without another batch shape for JIT.
        batch_size = self._resolve_batch_size(
            shots, batch_size, compute_reference=compute_reference
        )
        num_batches = ceil((shots + compute_reference) / batch_size)
        backend = self._backend
        tasks = [
            (index, self._channel_sampler.block_seeds(batch_size), backend.next_seed())
            for index in range(num_batches)
        ]

        num_f = self._channel_sampler.signature_matrix.shape[1]
        # NOTE: JAX may read a host buffer after the call returns, so only the
        # numba backend, which is done with it, reuses the buffers.
        reuse = isinstance(backend, _StatevectorBackend)

        def sample_batch(task: tuple) -> jax.Array | np.ndarray:
            index, block_seeds, seed = task
            out = _scratch(batch_size, num_f) if reuse else None
            f_params = self._channel_sampler.sample(batch_size, block_seeds, out)
            if compute_reference and index == 0:
                f_params[0] = 0
            return backend.sample(f_params, seed)

        batches = run_tasks(sample_batch, tasks, self._threads(batch_size))
        backend.finish_call()
        reference: np.ndarray | None = None
        if compute_reference:
            reference = np.asarray(batches[0][0])
            batches[0] = batches[0][1:]
        result = backend.concatenate(batches)[:shots]

        if compute_reference:
            assert reference is not None
            return result, reference
        return result

    def _sample_fused(
        self,
        shots: int,
        batch_size: int | None,
        mask: np.ndarray | None,
        flip: np.ndarray | None,
    ) -> np.ndarray | None:
        """Sample with the fused shot kernel, or return None if it doesn't apply.

        Each chunk of shots draws its noise into packed shot states, writes its
        direct outputs, and samples its survivors' components from their states,
        in one task: no f matrix is built and no survivors wait for a batch.
        """
        backend = self._backend
        if not isinstance(backend, _StatevectorBackend):
            return None
        fused = backend.fused(mask, flip)
        if fused is None:
            return None
        layout = fused[0]
        screen = backend.screened(mask, flip).tobytes()
        result = np.empty((shots, self._program.num_outputs), dtype=np.bool_)
        # NOTE: about FUSED_CHUNKS chunks of whole noise blocks: enough tasks for
        # the threads, few and large enough that their Python work, which runs
        # under the GIL, stays small. The chunks depend on the call and the
        # shot layout, never on the threads.
        blocks = max(1, -(-shots // (FUSED_CHUNKS * _BLOCK_SHOTS)))
        fit = max(1, FUSED_BYTES // (8 * layout.words * _BLOCK_SHOTS))
        chunk = min(blocks, fit, FUSED_SHOTS // _BLOCK_SHOTS) * _BLOCK_SHOTS
        if batch_size is not None:
            chunk = min(chunk, batch_size)
        noise = self._channel_sampler.fire_tables
        tasks = [
            (
                start,
                self._channel_sampler.block_seeds(min(chunk, shots - start)),
                backend.next_seed(),
            )
            for start in range(0, shots, chunk)
        ]

        def sample_chunk(task: tuple) -> None:
            start, block_seeds, seed = task
            stop = min(start + chunk, shots)
            state = _scratch(stop - start, 8 * layout.words).view(np.uint64)
            out = result[start:stop]
            discarded = sample_shots(
                len(out) * (len(noise.p_fire) // _BLOCK_SHOTS + layout.words + 1)
                + len(noise.p_fire) * len(block_seeds),
                out,
                state,
                block_seeds,
                _BLOCK_SHOTS,
                noise.p_fire,
                noise.outcome_start,
                noise.cdf,
                layout.signature_start,
                layout.signature_word,
                layout.signature_bits,
                layout.flips,
                layout.screens,
                layout.direct_bit,
                layout.kept,
            )
            survivors = np.flatnonzero(~discarded)
            if len(survivors):
                backend.sample_components(fused, state, survivors, out, seed, screen)

        run_tasks(sample_chunk, tasks, self._threads(chunk))
        backend.finish_call()
        return result

    def _sample_batches_with_postselection(
        self,
        shots: int,
        batch_size: int | None,
        *,
        postselection_mask: np.ndarray,
        referenced: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample with postselection, skipping the components for direct discards.

        A shot that a masked direct detector discards keeps its direct detector
        columns and reads ``False`` elsewhere. If the backend prunes, a shot that
        a masked component detector discards also reads ``False`` in the outputs
        set after that. The other shots are sampled in batches of ``batch_size``.
        Without a ``batch_size``, batches are as large as memory allows, and the
        last survivors are split into a batch per thread.

        Args:
            shots: Number of samples to draw.
            batch_size: Survivors per batch. Auto-determined if None.
            postselection_mask: The postselected detectors.
            referenced: The outputs that a noiseless reference sample is XOR-ed
                into, before the postselection test, if any.

        """
        _check_sizes(shots, batch_size)
        num_outputs = self._program.num_outputs
        if shots == 0:
            return np.empty((0, num_outputs), dtype=np.bool_)

        flip = None
        if referenced is not None:
            flip = self._compute_reference_sample() & referenced
        fused = self._sample_fused(shots, batch_size, postselection_mask, flip)
        if fused is not None:
            return fused
        split_last = batch_size is None
        if batch_size is None:
            batch_size = self._estimate_batch_size()
        f_indices, flips, columns = self._direct_detectors
        if flip is not None:
            flips = flips ^ flip[columns]
        postselect = postselection_mask[columns]
        result = np.empty((shots, num_outputs), dtype=np.bool_)
        backend = self._backend
        chunk = min(batch_size, SCREEN_SHOTS)

        def screen(task: tuple[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            """Write the direct detectors of the chunk at ``start``; return its survivors."""
            start, block_seeds = task
            stop = min(start + chunk, shots)
            # NOTE: the survivors are copied out before the buffer is reused.
            f_params = self._channel_sampler.sample(
                stop - start, block_seeds, _scratch(stop - start, num_f)
            )
            discarded = _write_direct_detectors(
                (stop - start) * len(columns),
                f_params,
                f_indices,
                flips,
                columns,
                postselect,
                result[start:stop],
            )
            survivors = np.flatnonzero(~discarded)
            return f_params[survivors], start + survivors

        def dispatch(task: tuple) -> None:
            f_batch, shot_batch, seed, rows = task
            samples = backend.sample(
                f_batch, seed, mask=postselection_mask, flip=flip, pad_to=rows
            )
            result[shot_batch] = np.asarray(samples)

        threads = self._threads(batch_size)
        num_f = self._channel_sampler.signature_matrix.shape[1]
        pending_f = [np.empty((0, num_f), dtype=np.uint8)]
        pending_shot = [np.empty(0, dtype=np.int64)]
        pending = 0
        starts = range(0, shots, chunk)
        screened_chunks = 0
        # NOTE: chunks are screened until every thread has a full batch of
        # survivors. A batch starts every batch_size survivors, and all seeds
        # are drawn in order, so the thread count doesn't change the samples.
        while screened_chunks < len(starts) or pending:
            while screened_chunks < len(starts) and pending < threads * batch_size:
                tasks = [
                    (
                        start,
                        self._channel_sampler.block_seeds(min(chunk, shots - start)),
                    )
                    for start in starts[
                        screened_chunks : screened_chunks
                        + SCREEN_CHUNKS_PER_THREAD * threads
                    ]
                ]
                screened_chunks += len(tasks)
                for f_survivors, shot_survivors in run_tasks(screen, tasks, threads):
                    pending_f.append(f_survivors)
                    pending_shot.append(shot_survivors)
                    pending += len(shot_survivors)
            ready = pending
            size, rows = batch_size, batch_size
            if screened_chunks < len(starts):
                ready -= ready % (threads * batch_size)
            elif split_last and ready:
                # NOTE: JAX pads these batches to a power of two, so that few
                # batch shapes are compiled.
                per_thread = self._per_thread(ready, postselection_mask, flip)
                size = ceil(ready / max(1, ceil(ready / batch_size), per_thread))
                rows = min(batch_size, 1 << (size - 1).bit_length())
            all_f, all_shot = np.concatenate(pending_f), np.concatenate(pending_shot)
            batches = [
                (
                    all_f[start : start + size],
                    all_shot[start : start + size],
                    backend.next_seed(),
                    rows,
                )
                for start in range(0, ready, size)
            ]
            # NOTE: only JAX pads batches to ``rows``, and it runs one at a time.
            run_tasks(dispatch, batches, self._threads(size))
            pending_f, pending_shot = [all_f[ready:]], [all_shot[ready:]]
            pending -= ready
        backend.finish_call()
        return result

    def _sample_direct(self, shots: int) -> np.ndarray:
        """Sample a program without components, whose outputs are all direct."""
        program = self._program
        f_params = self._channel_sampler.sample(shots)
        num_direct = len(program.direct_f_indices)
        # NOTE: typical surface-code detector circuits need no copy.
        if (
            program.output_reindex is None
            and not program.direct_flips.any()
            and np.array_equal(program.direct_f_indices, np.arange(num_direct))
        ):
            return f_params[:, :num_direct].view(np.bool_)
        out = np.empty((shots, program.num_outputs), dtype=np.bool_)
        _gather_outputs(
            out.size,
            f_params,
            program.direct_f_indices,
            program.direct_flips,
            np.zeros((shots, 0), dtype=np.bool_),
            program.output_order,
            out,
        )
        return out

    def __repr__(self) -> str:
        """Return a string representation with compilation statistics."""
        n_direct = len(self._program.direct_f_indices)

        c_graphs = []
        c_params = []
        c_a_terms = []
        c_b_terms = []
        c_c_terms = []
        c_d_terms = []
        c_e_terms = []
        total_memory_bytes = 0
        num_outputs = []
        statevector_widths: list[int] = []
        statevector_actions = 0

        for component in self._program.components:
            if isinstance(component, StatevectorComponent):
                statevector_widths.append(component.width)
                statevector_actions += component.num_actions
                num_outputs.append(len(component.output_indices))
                total_memory_bytes += _nbytes(component)
                continue
            for circuit in component.compiled_scalar_graphs:
                num_outputs.append(len(component.output_indices))
                c_graphs.append(circuit.num_graphs)
                c_params.append(circuit.n_params)
                c_a_terms.append(circuit.node_phases.phases.size)
                c_b_terms.append(circuit.phase_terms.coeffs.size)
                c_c_terms.append(
                    circuit.pi_products.psi_params.shape[0]
                    * circuit.pi_products.psi_params.shape[1]
                )
                c_d_terms.append(
                    circuit.phase_pairs.alpha.size + circuit.phase_pairs.beta.size
                )
                c_e_terms.append(
                    circuit.float_terms.table.shape[0]
                    * circuit.float_terms.table.shape[1]
                )

                total_memory_bytes += _nbytes(circuit)

        def _format_bytes(n: int) -> str:
            if n < 1024:
                return f"{n} B"
            if n < 1024**2:
                return f"{n / 1024:.1f} kB"
            return f"{n / (1024**2):.1f} MB"

        total_memory_str = _format_bytes(total_memory_bytes)
        error_channel_bits = sum(
            channel.num_bits for channel in self._channel_sampler.channels
        )

        statevector = (
            f"{len(statevector_widths)} statevector components "
            f"(width ≤ {max(statevector_widths)}, "
            f"{statevector_actions} actions), "
            if statevector_widths
            else ""
        )

        return (
            f"{type(self).__name__}({n_direct} direct, "
            f"{statevector}"
            f"{np.sum(c_graphs)} graphs, "
            f"{error_channel_bits} error channel bits, "
            f"{np.max(num_outputs) if num_outputs else 0} outputs for largest cc, "
            f"≤ {np.max(c_params) if c_params else 0} parameters, {np.sum(c_a_terms)} A terms, "
            f"{np.sum(c_b_terms)} B terms, "
            f"{np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms, "
            f"{np.sum(c_e_terms)} E terms, "
            f"{total_memory_str})"
        )


class CompiledMeasurementSampler(_CompiledSamplerBase):
    """Samples measurement outcomes from a quantum circuit.

    Uses sequential decomposition [0, 1, 2, ..., n] where:
    - compiled_scalar_graphs[0]: normalization (0 outputs plugged)
    - compiled_scalar_graphs[i]: cumulative probability up to bit i
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        strategy: SamplingStrategy = "active_statevector",
        seed: int | None = None,
        trajectory_sharing: bool | None = None,
    ):
        """Create a measurement sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: "active_statevector" (the default), or a stabilizer rank
                decomposition strategy ("cat5", "bss", "cutting").
            trajectory_sharing: With ``strategy="active_statevector"``, simulate shots
                with equal histories on one lane. A component whose shots mostly
                share a few noise patterns draws them from outcome tables instead,
                which later calls reuse. Defaults to True for
                ``"active_statevector"`` and False otherwise.
            seed: Random seed for the sampler. IMPORTANT: Currently, the sampler
                will only produce deterministic samples for fixed batch size. If
                deterministic samples are needed, the batch size should be set
                manually.

        """
        super().__init__(
            circuit,
            sample_detectors=False,
            mode="sequential",
            seed=seed,
            strategy=strategy,
            trajectory_sharing=trajectory_sharing,
        )

    def sample(self, shots: int, *, batch_size: int | None = None) -> np.ndarray:
        """Sample measurement outcomes from the circuit.

        Args:
            shots: The number of times to sample every measurement in the circuit.
            batch_size: The number of samples to process in each batch. Defaults to
                None, which automatically chooses a batch size based on available
                memory. With ``strategy="active_statevector"`` a call is split
                into a batch per numba thread when each gets at least 4096 shots,
                or at least 256 that trajectory sharing may not keep on fewer
                lanes. With a stabilizer-rank strategy on a GPU, setting this
                explicitly can help fully utilize VRAM.
                NOTE: Changing the batch size will affect reproducibility even with
                a fixed seed.

        Returns:
            A numpy array containing the measurement samples.

        """
        return self._sample_batches(shots, batch_size)


def _maybe_bit_pack(array: np.ndarray, *, bit_packed: bool) -> np.ndarray:
    """Optionally bit-pack a boolean array."""
    if not bit_packed:
        return array
    return np.packbits(array.astype(np.bool_), axis=1, bitorder="little")


class CompiledDetectorSampler(_CompiledSamplerBase):
    """Samples detector and observable outcomes from a quantum circuit."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        strategy: SamplingStrategy = "active_statevector",
        seed: int | None = None,
        trajectory_sharing: bool | None = None,
    ):
        """Create a detector sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: "active_statevector" (the default), or a stabilizer rank
                decomposition strategy ("cat5", "bss", "cutting").
            trajectory_sharing: With ``strategy="active_statevector"``, simulate shots
                with equal histories on one lane. A component whose shots mostly
                share a few noise patterns draws them from outcome tables instead,
                which later calls reuse. Defaults to True for
                ``"active_statevector"`` and False otherwise.
            seed: Random seed for the sampler. IMPORTANT: Currently, the sampler
                will only produce deterministic samples for fixed batch size and
                fixed reference sample settings. If deterministic samples are
                needed, the batch size should be set manually.

        """
        super().__init__(
            circuit,
            sample_detectors=True,
            mode="sequential",
            seed=seed,
            strategy=strategy,
            trajectory_sharing=trajectory_sharing,
        )

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[True],
        bit_packed: bool = False,
        use_detector_reference_sample: bool = False,
        use_observable_reference_sample: bool = False,
        postselection_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[False] = False,
        bit_packed: bool = False,
        use_detector_reference_sample: bool = False,
        use_observable_reference_sample: bool = False,
        postselection_mask: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def sample(
        self,
        shots: int,
        *,
        batch_size: int | None = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
        use_detector_reference_sample: bool = False,
        use_observable_reference_sample: bool = False,
        postselection_mask: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return detector samples from the circuit.

        The circuit must define the detectors using DETECTOR instructions. Observables
        defined by OBSERVABLE_INCLUDE instructions can also be included in the results
        as honorary detectors.

        Args:
            shots: The number of times to sample every detector in the circuit.
            batch_size: The number of samples to process in each batch. Defaults to
                None, which automatically chooses a batch size based on available
                memory. With ``strategy="active_statevector"`` a call is split
                into a batch per numba thread when each gets at least 4096 shots,
                or at least 256 that trajectory sharing may not keep on fewer
                lanes. With a stabilizer-rank strategy on a GPU, setting this
                explicitly can help fully utilize VRAM.
                NOTE: Changing the batch size will affect reproducibility even with
                a fixed seed.
            separate_observables: Defaults to False. When set to True, the return value
                is a (detection_events, observable_flips) tuple instead of a flat
                detection_events array.
            prepend_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the start of the results.
            append_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the end of the results.
            bit_packed: Defaults to false. When set, results are bit-packed.
            use_detector_reference_sample: Defaults to False. When True, a noiseless
                reference sample is computed and XORed with detector outcomes so that
                results represent deviations from the noiseless baseline. This should
                only be used when detectors are deterministic. Otherwise, it can
                unpredictably change the results.
            use_observable_reference_sample: Defaults to False. When True, a noiseless
                reference sample is computed and XORed with observable outcomes so that
                results represent deviations from the noiseless baseline. This should
                only be used when observables are deterministic. Otherwise, it can
                unpredictably change the results.
            postselection_mask: Optional boolean array of length ``num_detectors``.
                When set, shots where any masked direct detector (an error bit
                or a constant) fires are not sampled further. All ``shots`` rows
                are still returned: survivors
                contain the full sample, while discarded rows retain direct detector
                columns and fill component columns with ``False``; with
                ``strategy="active_statevector"``, a shot discarded by a masked
                component detector also reads ``False`` in the outputs set after
                it. Re-apply the mask to the returned detector columns to recover
                surviving shots.

        Returns:
            A numpy array or tuple of numpy arrays containing the samples.

        Raises:
            ValueError: If ``separate_observables`` is combined with
                ``prepend_observables`` or ``append_observables``.

        """
        if separate_observables and (prepend_observables or append_observables):
            raise ValueError(
                "Can't specify separate_observables=True with "
                "append_observables=True or prepend_observables=True"
            )

        compute_reference = (
            use_detector_reference_sample or use_observable_reference_sample
        )
        num_detectors = self._num_detectors

        if postselection_mask is not None:
            mask = np.asarray(postselection_mask, dtype=np.bool_)
            if mask.shape != (num_detectors,):
                raise ValueError(
                    f"postselection_mask must have shape ({num_detectors},), "
                    f"got {mask.shape}"
                )
            # NOTE: without pruning, only direct detectors can skip any work.
            if not self._backend.prunes:
                skips = mask & self._direct_detector_mask
            else:
                skips = mask.copy()
            indices, flips, columns = self._direct_detectors
            constant = indices < 0
            skips[columns[constant]] &= flips[constant] & (
                not use_detector_reference_sample
            )
            postselection_mask = (
                mask if skips.any() and self._program.components else None
            )

        if postselection_mask is not None:
            referenced = None
            if compute_reference:
                is_detector = np.arange(self._program.num_outputs) < num_detectors
                referenced = np.where(
                    is_detector,
                    use_detector_reference_sample,
                    use_observable_reference_sample,
                )
            samples = self._sample_batches_with_postselection(
                shots,
                batch_size,
                postselection_mask=postselection_mask,
                referenced=referenced,
            )
        elif compute_reference:
            samples, reference = self._sample_batches(
                shots, batch_size, compute_reference=True
            )
            if use_detector_reference_sample:
                samples[:, :num_detectors] ^= reference[:num_detectors]
            if use_observable_reference_sample:
                samples[:, num_detectors:] ^= reference[num_detectors:]
        else:
            samples = self._sample_batches(shots, batch_size)

        det_samples = samples[:, :num_detectors]
        obs_samples = samples[:, num_detectors:]

        if prepend_observables and append_observables:
            combined = np.concatenate([obs_samples, det_samples, obs_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if append_observables:
            return _maybe_bit_pack(samples, bit_packed=bit_packed)
        if prepend_observables:
            combined = np.concatenate([obs_samples, det_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if separate_observables:
            return (
                _maybe_bit_pack(det_samples, bit_packed=bit_packed),
                _maybe_bit_pack(obs_samples, bit_packed=bit_packed),
            )

        return _maybe_bit_pack(det_samples, bit_packed=bit_packed)


class CompiledStateProbs(_CompiledSamplerBase):
    """Computes measurement probabilities for a given state.

    Uses joint decomposition [0, n] where:
    - compiled_scalar_graphs[0]: normalization (0 outputs plugged)
    - compiled_scalar_graphs[1]: full joint probability (all outputs plugged)
    """

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool = False,
        strategy: SamplingStrategy = "cat5",
        seed: int | None = None,
    ):
        """Create a probability estimator.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, compute detector/observable probabilities.
            strategy: Stabilizer rank decomposition strategy ("cat5", "bss",
                "cutting"); "active_statevector" cannot compute probabilities.
            seed: Random seed. If None, a random seed is generated. Note that
                deterministic results are only guaranteed for a fixed batch size.

        """
        super().__init__(
            circuit,
            sample_detectors=sample_detectors,
            mode="joint",
            seed=seed,
            strategy=strategy,
        )

    def probability_of(self, state: np.ndarray, *, batch_size: int) -> np.ndarray:
        """Compute probabilities for a batch of error samples given a measurement state.

        Args:
            state: The measurement outcome state to compute probability for.
            batch_size: Number of error samples to use for estimation.

        Returns:
            Array of probabilities P(state | error_sample) for each error sample.

        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        expected_outputs = self._program.num_outputs
        if state.shape != (expected_outputs,):
            raise ValueError(
                f"state must have shape ({expected_outputs},), got {state.shape}"
            )
        f_samples = jnp.asarray(self._channel_sampler.sample(batch_size))
        p_norm = jnp.ones(batch_size)
        p_joint = jnp.ones(batch_size)

        if len(self._program.direct_f_indices) > 0:
            direct_bits = _direct_bits(self._program, f_samples)
            n_direct = len(self._program.direct_f_indices)
            targets = state[self._program.output_order[:n_direct]]
            p_joint = p_joint * (direct_bits == targets).all(axis=1)

        for component in self._program.components:
            assert isinstance(component, CompiledComponent)
            assert len(component.compiled_scalar_graphs) == 2

            f_selected = f_samples[:, component.f_selection]

            norm_circuit, joint_circuit = component.compiled_scalar_graphs

            # Normalization: only f-params
            p_norm = p_norm * jnp.abs(evaluate(norm_circuit, f_selected))

            # Joint probability: f-params + state
            component_state = state[list(component.output_indices)]
            tiled_state = jnp.tile(component_state, (batch_size, 1))
            joint_params = jnp.hstack([f_selected, tiled_state])
            p_joint = p_joint * jnp.abs(evaluate(joint_circuit, joint_params))

        return np.asarray(p_joint / p_norm)
