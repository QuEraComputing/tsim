"""Compiled samplers for measurements and detectors."""

from __future__ import annotations

import warnings
from collections import deque
from math import ceil
from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp
import numba
import numpy as np
import psutil

from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.core.types import CompiledComponent, CompiledProgram, SamplingStrategy
from tsim.noise.channels import ChannelSampler
from tsim.statevector.program import STABILIZER_RANK_HINT, StatevectorComponent
from tsim.statevector.shared import (
    SharingLayout,
    lane_bytes,
    sample_shared_component,
    sharing_layout,
)
from tsim.utils.cuda_helpers import copy_d2h
from tsim.utils.kernels import kernel, run_tasks
from tsim.utils.rng import GOLDEN

if TYPE_CHECKING:
    from jax import Array as PRNGKey

    from tsim.circuit import Circuit

# NOTE: beyond 2^16 lanes, statevector batches cost memory without being faster.
STATEVECTOR_MAX_BATCH_SIZE = 2**16

# NOTE: without a given batch size, a call is split into a batch per thread when
# every batch gets at least this many shots, whose fixed costs would dominate
# otherwise. A call too small for that stays one batch, which the parallel
# kernels spread over the threads instead.
MIN_THREAD_BATCH = 2**12

# NOTE: postselection screens shots for direct discards in chunks of at most
# this many, several chunks per thread at a time, so that calls of a few batches
# still keep every thread busy and slow cores take fewer chunks.
SCREEN_SHOTS = 2**14
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
        direct_bits = (
            f_params[:, program.direct_f_indices].astype(jnp.bool_)
            ^ program.direct_flips
        )
        results.append(direct_bits)
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
    *,
    share: bool,
    direct: tuple[np.ndarray, np.ndarray, np.ndarray],
    sharing_stats: deque[dict] | None = None,
) -> np.ndarray:
    """Sample all outputs of a program of statevector components, without JAX.

    Args:
        program: The compiled program to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        seed: Seed of the batch's random numbers, below 2^64.
        layouts: The sharing layout of each component.
        share: Whether shots with equal histories share a lane.
        direct: The program's direct f-indices, direct flips and output order.
        sharing_stats: If given, the executor appends its lane counts.

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
        samples, max_norm_deviation, stats = sample_shared_component(
            component, layout, f_params, component_seed, share
        )
        if sharing_stats is not None:
            sharing_stats.append(stats)
        _check_norm(max_norm_deviation, component.norm_warning_threshold)
        results.append(samples)
    component_bits = (
        np.concatenate(results, axis=1) if results else np.zeros((rows, 0), np.bool_)
    )
    out = np.empty((rows, program.num_outputs), dtype=np.bool_)
    _gather_outputs(out.size, f_params, *direct[:2], component_bits, direct[2], out)
    return out


@kernel
def _write_direct_detectors(f, f_indices, flips, columns, postselect, reference, out):
    """Clear every row of ``out`` and write its direct detector ``j`` at ``columns[j]``.

    Detector ``j`` is ``f[row, f_indices[j]] ^ flips[j]``. Returns which rows
    have a postselected detector that differs from its ``reference`` bit.
    """
    discarded = np.zeros(f.shape[0], dtype=np.bool_)
    for row in numba.prange(f.shape[0]):
        out[row] = False
        for j in range(len(f_indices)):
            bit = (f[row, f_indices[j]] != 0) ^ flips[j]
            out[row, columns[j]] = bit
            if postselect[j] and bit != reference[j]:
                discarded[row] = True
    return discarded


@kernel
def _gather_outputs(f, f_indices, flips, component_bits, order, out):
    """Write every row's outputs into ``out`` in their original order.

    Output ``k`` of the concatenation goes to column ``order[k]``: first the
    direct outputs ``f[row, f_indices[j]] ^ flips[j]``, then ``component_bits``.
    """
    num_direct = len(f_indices)
    for row in numba.prange(f.shape[0]):
        for j in range(num_direct):
            out[row, order[j]] = (f[row, f_indices[j]] != 0) ^ flips[j]
        for j in range(component_bits.shape[1]):
            out[row, order[num_direct + j]] = component_bits[row, j]


def _nbytes(tree: object) -> int:
    if isinstance(tree, StatevectorComponent):
        return sum(v.nbytes for v in vars(tree).values() if isinstance(v, np.ndarray))
    return sum(
        v.nbytes for v in jax.tree_util.tree_leaves(tree) if isinstance(v, jax.Array)
    )


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
                with equal histories on one lane. Defaults to True for
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

        self._key = jax.random.key(seed)
        # NOTE: statevector batches draw their seeds here instead of from the key.
        self._batch_seeds = np.random.default_rng([seed, 1])
        self._trajectory_sharing = trajectory_sharing
        self._layouts_by_key: dict[bytes, tuple[SharingLayout, ...]] = {}
        self.sharing_stats: deque[dict] = deque(maxlen=10_000)

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

        prog = self._program
        # NOTE: _gather_outputs writes column output_order[k] without bounds checks.
        assert np.array_equal(
            np.sort(np.asarray(prog.output_order)), np.arange(prog.num_outputs)
        )
        self._direct_f_indices = np.asarray(prog.direct_f_indices)
        self._direct_flips = np.asarray(prog.direct_flips, dtype=np.bool_)
        self._direct_reindex = (
            np.asarray(prog.output_reindex) if prog.output_reindex is not None else None
        )
        # Zero-copy fast path: f-indices are 0..n-1, no flips, no reindex.
        # Hit by typical surface-code detector circuits at low noise.
        n_direct = len(self._direct_f_indices)
        self._direct_zero_copy = (
            n_direct > 0
            and self._direct_reindex is None
            and not self._direct_flips.any()
            and np.array_equal(self._direct_f_indices, np.arange(n_direct))
        )
        self._direct_global_indices = np.asarray(
            prog.output_order[:n_direct], dtype=np.int32
        )
        self._direct = (
            self._direct_f_indices,
            self._direct_flips,
            np.asarray(prog.output_order),
        )
        self._direct_output_mask = np.zeros(prog.num_outputs, dtype=np.bool_)
        if n_direct > 0:
            self._direct_output_mask[self._direct_global_indices] = True
        self._direct_detector_mask = self._direct_output_mask[
            : self._num_detectors
        ].copy()
        is_detector = self._direct_global_indices < self._num_detectors
        self._direct_detector_f = self._direct_f_indices[is_detector]
        self._direct_detector_flips = self._direct_flips[is_detector]
        self._direct_detector_columns = self._direct_global_indices[is_detector]

        statevector = [
            c for c in self._program.components if isinstance(c, StatevectorComponent)
        ]
        self._has_statevector = bool(statevector)
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

    def _layouts(
        self,
        postselection_mask: np.ndarray | None,
        detector_reference: np.ndarray | None = None,
    ) -> tuple[SharingLayout, ...]:
        """Return the sharing layout of every statevector component.

        Args:
            postselection_mask: Postselected detectors, whose lanes are pruned.
            detector_reference: Reference bits XOR-ed into the detectors before
                the postselection test, if any.

        """
        num_detectors = self._num_detectors
        mask = np.zeros(num_detectors, dtype=np.bool_)
        if postselection_mask is not None:
            mask = np.asarray(postselection_mask, dtype=np.bool_)
        flip = np.zeros(num_detectors, dtype=np.bool_)
        if detector_reference is not None:
            flip = np.asarray(detector_reference[:num_detectors], dtype=np.bool_)
        key = mask.tobytes() + flip.tobytes()
        if key not in self._layouts_by_key:
            layouts = []
            for component in self._program.components:
                assert isinstance(component, StatevectorComponent)
                detector = np.asarray(component.output_indices) < num_detectors
                indices = np.minimum(component.output_indices, num_detectors - 1)
                layouts.append(
                    sharing_layout(
                        component,
                        detector & mask[indices] if num_detectors else detector,
                        detector & flip[indices] if num_detectors else detector,
                    )
                )
            self._layouts_by_key[key] = tuple(layouts)
        return self._layouts_by_key[key]

    def _sample_statevector(
        self, f_params: np.ndarray, seed: int, layouts: tuple[SharingLayout, ...]
    ) -> np.ndarray:
        """Sample one batch of a statevector program."""
        return sample_statevector_program(
            self._program,
            f_params,
            seed,
            layouts,
            share=self._trajectory_sharing,
            direct=self._direct,
            sharing_stats=self.sharing_stats,
        )

    def _next_seed(self) -> int:
        """Return the next statevector batch seed."""
        return int(self._batch_seeds.integers(0, 2**64, dtype=np.uint64))

    def _compute_direct_outputs(self, f_params_np: np.ndarray) -> np.ndarray:
        """Scatter direct output bits into a full (batch, num_outputs) bool array.

        Non-direct columns are zero.  The zero-copy fast path applies when
        direct indices are 0..n-1, there are no flips, and no reindex —
        i.e. the common surface-code case.
        """
        batch = f_params_np.shape[0]
        num_outputs = self._program.num_outputs
        n_direct = len(self._direct_f_indices)
        if n_direct == 0:
            return np.zeros((batch, num_outputs), dtype=np.bool_)
        if self._direct_zero_copy and n_direct == num_outputs:
            return f_params_np[:, :n_direct].view(np.bool_).copy()
        raw = (
            f_params_np[:, :n_direct].view(np.bool_)
            if self._direct_zero_copy
            else (f_params_np[:, self._direct_f_indices] ^ self._direct_flips).view(
                np.bool_
            )
        )
        out = np.zeros((batch, num_outputs), dtype=np.bool_)
        out[:, self._direct_global_indices] = raw
        return out

    def _compute_reference_sample(self) -> np.ndarray:
        """Return the noiseless reference sample (all f_params = 0).

        Does not advance the channel sampler RNG.
        """
        num_f = self._channel_sampler.signature_matrix.shape[1]
        f_ref = np.zeros((1, num_f), dtype=np.uint8)
        if not self._program.components:
            return self._compute_direct_outputs(f_ref)[0]
        if self._has_statevector:
            seed = self._next_seed()
            return self._sample_statevector(f_ref, seed, self._layouts(None))[0]
        self._key, subkey = jax.random.split(self._key)
        return np.asarray(
            sample_program(self._program, jnp.asarray(f_ref), subkey)[0],
            dtype=np.bool_,
        )

    def _resolve_batch_size(
        self,
        shots: int,
        batch_size: int | None,
        *,
        compute_reference: bool,
    ) -> int:
        """Choose a uniform batch size for ``shots`` samples.

        Without a ``batch_size``, batches are as large as memory allows, but split
        into a batch per thread (``_threads``) when each gets at least
        ``MIN_THREAD_BATCH`` shots, so that every thread has work.
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

    def _per_thread(self, shots: int) -> int:
        """Return the thread count if ``shots`` fill a batch per thread, else 1."""
        threads = self._threads(1)
        return threads if shots >= threads * MIN_THREAD_BATCH else 1

    def _peak_bytes_per_sample(self) -> int:
        """Estimate peak device memory per sample from compiled program structure."""
        num_f = self._channel_sampler.signature_matrix.shape[1]
        peak = 0
        for component in self._program.components:
            if isinstance(component, StatevectorComponent):
                # NOTE: plus the f-params and the output records, twice over.
                lane = lane_bytes(component) + 2 * num_f + 2 * self._program.num_outputs
                peak = max(peak, lane)
                continue
            for circuit in component.compiled_scalar_graphs:
                G = circuit.num_graphs
                max_a = circuit.node_phases.phases.shape[1]
                max_b = circuit.phase_terms.coeffs.shape[1]
                max_c = circuit.pi_products.psi_params.shape[1]
                max_d = circuit.phase_pairs.alpha.shape[1]
                max_e = circuit.float_terms.table.shape[1]
                largest = max(max_a * 16, max_b * 4, max_c * 4, max_d * 16, max_e * 16)
                peak = max(peak, G * largest * 3)
        return max(peak, 1)

    def _memory_budget(self) -> int:
        """Return the memory one batch may use: half of the free device memory.

        Statevector programs sample on the host, so their device is the host.
        """
        device = jax.devices()[0]
        if device.platform == "gpu" and not self._has_statevector:
            stats = device.memory_stats()
            available = stats.get("bytes_limit", 8 * 1024**3) - stats.get(
                "bytes_in_use", 0
            )
        else:
            available = psutil.virtual_memory().available
        return int(available * 0.5)  # conservative

    def _threads(self, batch_size: int) -> int:
        """Return how many batches of ``batch_size`` shots run at once.

        Statevector programs sample one batch per numba thread, as long as the
        batches fit in the memory budget; the stabilizer-rank programs sample
        one batch at a time.
        """
        if not self._has_statevector:
            return 1
        threads = numba.get_num_threads()
        # NOTE: with sharing a batch may still hold a lane per shot.
        fits = self._memory_budget() // (self._peak_bytes_per_sample() * batch_size)
        return max(1, min(threads, fits))

    def _estimate_batch_size(self) -> int:
        """Estimate the largest batch size that fits in available device memory.

        Without sharing, every batch that runs at once holds a lane per shot. With
        sharing, lanes are allocated as trajectories split, so the estimate
        leaves room for a lane per shot in one batch only. Programs with statevector
        components use at most ``STATEVECTOR_MAX_BATCH_SIZE`` lanes per batch.
        """
        peak = self._peak_bytes_per_sample()
        if not self._trajectory_sharing:
            peak *= self._threads(1)
        batch_size = max(1, self._memory_budget() // peak)
        if any(isinstance(c, StatevectorComponent) for c in self._program.components):
            batch_size = min(batch_size, STATEVECTOR_MAX_BATCH_SIZE)
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
        if shots < 0:
            raise ValueError(f"shots must be non-negative, got {shots}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

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

        # NOTE: with a reference sample the batch size grows by one when the
        # batches would hold exactly ``shots``, so that the first batch's
        # reference sample fits without another batch shape for JIT.
        batch_size = self._resolve_batch_size(
            shots, batch_size, compute_reference=compute_reference
        )
        num_batches = ceil((shots + compute_reference) / batch_size)

        statevector = self._has_statevector
        layouts = self._layouts(None) if statevector else ()
        tasks = []
        for index in range(num_batches):
            seeds = self._channel_sampler.block_seeds(batch_size)
            if statevector:
                tasks.append((index, seeds, self._next_seed()))
            else:
                self._key, subkey = jax.random.split(self._key)
                tasks.append((index, seeds, subkey))

        def sample_batch(task: tuple) -> jax.Array | np.ndarray:
            index, seeds, seed_or_key = task
            f_params_np = self._channel_sampler.sample(batch_size, seeds)
            if compute_reference and index == 0:
                f_params_np[0] = 0
            if statevector:
                return self._sample_statevector(f_params_np, seed_or_key, layouts)
            return sample_program(self._program, jnp.asarray(f_params_np), seed_or_key)

        batches = run_tasks(sample_batch, tasks, self._threads(batch_size))
        reference: np.ndarray | None = None
        if compute_reference:
            reference = np.asarray(batches[0][0])
            batches[0] = batches[0][1:]
        if statevector:
            result = np.concatenate(batches)[:shots]
        else:
            # Concatenate on device, then a single d2h. The prior
            # np.concatenate(batches) triggered per-batch __array__ (one d2h
            # each) plus a host-side memcpy into a fresh numpy buffer for the
            # concat output. For big bool tensors (e.g. 500k shots × 528
            # detector bits) the host memcpy alone was ~1 s on top of the PCIe
            # transfer.
            combined = (
                batches[0] if len(batches) == 1 else jnp.concatenate(batches, axis=0)
            )
            jax.block_until_ready(combined)
            result = copy_d2h(combined)[:shots]

        if compute_reference:
            assert reference is not None
            return result, reference
        return result

    def _sample_batches_with_postselection(
        self,
        shots: int,
        batch_size: int | None,
        *,
        postselection_mask: np.ndarray,
        compute_reference: bool = False,
        xor_detector_ref: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Sample with postselection, skipping the components for direct discards.

        Shots discarded by a direct masked detector are filled with their
        direct detector columns and ``False`` elsewhere; no component samples
        them.  Survivors are buffered until a full batch of ``batch_size`` is
        ready, then sampled in one call.  For JAX (the stabilizer-rank
        strategies) the final partial batch is padded to keep the batch size
        fixed (avoiding recompilation) and the padding rows are discarded;
        statevector batches are sampled as they are, and their shots discarded
        by a masked component detector read ``False`` from the action that
        discards them on.
        Chunks of shots, and then the batches of their survivors, run on
        ``_threads()`` threads at once. Without a ``batch_size``, batches are as
        large as memory allows, and the survivors left after the last chunk are
        split into a batch per thread (as in ``_resolve_batch_size``); JAX pads
        each to a power of two, so that few batch shapes are compiled.
        """
        if shots < 0:
            raise ValueError(f"shots must be non-negative, got {shots}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        num_outputs = self._program.num_outputs
        if shots == 0:
            empty = np.empty((0, num_outputs), dtype=np.bool_)
            empty_discarded = np.empty(0, dtype=np.bool_)
            if compute_reference:
                return empty, np.zeros(num_outputs, dtype=np.bool_), empty_discarded
            return empty, None, empty_discarded

        if not self._program.components:
            samples = self._sample_direct(shots)
            if compute_reference:
                reference = self._compute_reference_sample()
                if xor_detector_ref:
                    samples[:, : self._num_detectors] ^= reference[
                        : self._num_detectors
                    ]
                return samples, reference, np.zeros(shots, dtype=np.bool_)
            return samples, None, np.zeros(shots, dtype=np.bool_)

        split_last = batch_size is None
        if batch_size is None:
            batch_size = self._estimate_batch_size()

        reference: np.ndarray | None = None
        if compute_reference:
            reference = self._compute_reference_sample()

        result = np.empty((shots, num_outputs), dtype=np.bool_)
        was_discarded = np.empty(shots, dtype=np.bool_)

        columns = self._direct_detector_columns
        postselect = postselection_mask[columns]
        reference_bits = (
            reference[columns]
            if xor_detector_ref and reference is not None
            else np.zeros(len(columns), dtype=np.bool_)
        )

        statevector = self._has_statevector
        layouts = (
            self._layouts(postselection_mask, reference if xor_detector_ref else None)
            if statevector
            else ()
        )

        chunk = min(batch_size, SCREEN_SHOTS)

        def screen(task: tuple[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            """Sample the chunk of shots at ``start``; return its survivors.

            Writes the chunk's direct detectors and which of its shots a direct
            detector discards; returns the survivors' f-params and shot indices.
            """
            start, seeds = task
            stop = min(start + chunk, shots)
            f_params_np = self._channel_sampler.sample(stop - start, seeds)
            discarded = _write_direct_detectors(
                (stop - start) * len(columns),
                f_params_np,
                self._direct_detector_f,
                self._direct_detector_flips,
                columns,
                postselect,
                reference_bits,
                result[start:stop],
            )
            was_discarded[start:stop] = discarded
            survivors = np.flatnonzero(~discarded)
            return f_params_np[survivors], start + survivors

        def dispatch(task: tuple) -> None:
            """Sample the shots ``shot_batch`` in one batch.

            JAX batches are padded with row 0 to ``rows`` rows; statevector
            batches take any number of rows.
            """
            f_batch, shot_batch, seed_or_key, rows = task
            count = len(shot_batch)
            if statevector:
                samples = self._sample_statevector(f_batch, seed_or_key, layouts)
            else:
                if count < rows:
                    padding = np.repeat(f_batch[:1], rows - count, axis=0)
                    f_batch = np.concatenate([f_batch, padding])
                samples = sample_program(
                    self._program, jnp.asarray(f_batch), seed_or_key
                )
            result[shot_batch] = np.asarray(samples)[:count]

        threads = self._threads(batch_size)
        num_f = self._channel_sampler.signature_matrix.shape[1]
        pending_f = [np.empty((0, num_f), dtype=np.uint8)]
        pending_shot = [np.empty(0, dtype=np.int64)]
        pending = 0
        starts = range(0, shots, chunk)
        screened_chunks = 0
        # NOTE: chunks are screened until every thread has a full batch of
        # survivors. Batches still start every batch_size survivors and seeds and
        # seeds and keys are drawn in order, so the thread count doesn't change
        # the samples.
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
                per_thread = self._per_thread(ready)
                size = ceil(ready / max(1, ceil(ready / batch_size), per_thread))
                rows = min(batch_size, 1 << (size - 1).bit_length())
            all_f, all_shot = np.concatenate(pending_f), np.concatenate(pending_shot)
            batches = []
            for start in range(0, ready, size):
                if statevector:
                    seed_or_key = self._next_seed()
                else:
                    self._key, seed_or_key = jax.random.split(self._key)
                stop = start + size
                batch = (all_f[start:stop], all_shot[start:stop], seed_or_key, rows)
                batches.append(batch)
            run_tasks(dispatch, batches, threads)
            pending_f, pending_shot = [all_f[ready:]], [all_shot[ready:]]
            pending -= ready

        if xor_detector_ref and reference is not None:
            det_ref = reference[: self._num_detectors]
            survivors = ~was_discarded
            result[survivors, : self._num_detectors] ^= det_ref
            result[was_discarded, : self._num_detectors] ^= (
                det_ref & self._direct_detector_mask
            )

        if compute_reference:
            assert reference is not None
            return result, reference, was_discarded
        return result, None, was_discarded

    def _sample_direct(self, shots: int) -> np.ndarray:
        """Fast path when all components are direct (pure numpy, no JAX)."""
        f_params = self._channel_sampler.sample(shots)
        if self._direct_zero_copy:
            return f_params[:, : len(self._direct_f_indices)].view(np.bool_)
        result = f_params[:, self._direct_f_indices] ^ self._direct_flips
        if self._direct_reindex is not None:
            result = result[:, self._direct_reindex]
        return result.view(np.bool_)

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
                with equal histories on one lane. Defaults to True for
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
                into a batch per numba thread when each gets at least 4096 shots.
                With a stabilizer-rank strategy on a GPU, setting this explicitly
                can help fully utilize VRAM.
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
                with equal histories on one lane. Defaults to True for
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
                into a batch per numba thread when each gets at least 4096 shots.
                With a stabilizer-rank strategy on a GPU, setting this explicitly
                can help fully utilize VRAM.
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
                When set, shots where any masked direct detector fires are not
                sampled further. All ``shots`` rows are still returned: survivors
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

        if postselection_mask is not None:
            mask = np.asarray(postselection_mask, dtype=np.bool_)
            if mask.shape != (self._num_detectors,):
                raise ValueError(
                    f"postselection_mask must have shape ({self._num_detectors},), "
                    f"got {mask.shape}"
                )
            if postselection_mask is not mask:
                postselection_mask = mask
            if (
                not (postselection_mask & self._direct_detector_mask).any()
                or not self._program.components
            ):
                postselection_mask = None

        if postselection_mask is not None:
            if compute_reference:
                samples, reference, direct_discarded = (
                    self._sample_batches_with_postselection(
                        shots,
                        batch_size,
                        postselection_mask=postselection_mask,
                        compute_reference=True,
                        xor_detector_ref=use_detector_reference_sample,
                    )
                )
                assert reference is not None
                num_detectors = self._num_detectors
                if use_observable_reference_sample:
                    obs_ref = reference[num_detectors:]
                    samples[~direct_discarded, num_detectors:] ^= obs_ref
            else:
                samples, _, _ = self._sample_batches_with_postselection(
                    shots,
                    batch_size,
                    postselection_mask=postselection_mask,
                )
        elif compute_reference:
            samples, reference = self._sample_batches(
                shots, batch_size, compute_reference=True
            )
            num_detectors = self._num_detectors
            if use_detector_reference_sample:
                samples[:, :num_detectors] ^= reference[:num_detectors]
            if use_observable_reference_sample:
                samples[:, num_detectors:] ^= reference[num_detectors:]
        else:
            samples = self._sample_batches(shots, batch_size)

        num_detectors = self._num_detectors
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
            direct_bits = (
                f_samples[:, self._program.direct_f_indices].astype(jnp.bool_)
                ^ self._program.direct_flips
            )
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
