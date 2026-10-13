"""Compiled samplers for measurements and detectors."""

from __future__ import annotations

import warnings
from collections import deque
from math import ceil
from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from zxparam import DecompositionStrategy

from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.core.types import CompiledComponent, CompiledProgram, SamplingStrategy
from tsim.noise.channels import ChannelSampler
from tsim.statevector.execute import lane_bytes, sample_statevector_component
from tsim.statevector.program import STABILIZER_RANK_HINT, StatevectorComponent
from tsim.statevector.shared import Sharing, sample_shared_component, sharing_layout
from tsim.utils.cuda_helpers import copy_d2h

if TYPE_CHECKING:
    from jax import Array as PRNGKey

    from tsim.circuit import Circuit

# NOTE: statevector batches beyond 2^16 lanes cost memory without being faster:
# on CPU, 2^16 lanes sampled cultivation d3 1.7x faster than the memory-derived
# batch, at a fifth of its peak memory.
STATEVECTOR_MAX_BATCH_SIZE = 2**16


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
    component: CompiledComponent | StatevectorComponent,
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
    if isinstance(component, StatevectorComponent):
        return sample_statevector_component(component, f_params, key)
    # Skip JIT for small components (overhead not worth it)
    if len(component.output_indices) <= 1:
        return _sample_component(component, f_params, key)
    return _sample_component_jit(component, f_params, key)


def sample_program(
    program: CompiledProgram,
    f_params: jax.Array,
    key: PRNGKey,
    sharing: tuple[Sharing | None, ...] | None = None,
    sharing_stats: deque[dict] | None = None,
) -> jax.Array:
    """Sample all outputs from a compiled program.

    Args:
        program: The compiled program to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.
        sharing: One sharing state per component, to sample statevector
            components with shared trajectories; ``None`` samples one lane per
            shot.
        sharing_stats: If given, the shared executor appends its lane counts.

    Returns:
        Samples array of shape (batch_size, num_outputs), reordered to
        match the original output indices.

    """
    results: list[jax.Array] = []

    if program.num_outputs == 0:
        batch_size = f_params.shape[0]
        return jnp.zeros((batch_size, 0), dtype=jnp.bool_)

    if len(program.direct_f_indices) > 0:
        direct_bits = (
            f_params[:, program.direct_f_indices].astype(jnp.bool_)
            ^ program.direct_flips
        )
        results.append(direct_bits)

    for index, component in enumerate(program.components):
        shared = sharing[index] if sharing is not None else None
        if isinstance(component, StatevectorComponent) and shared is not None:
            samples, key, max_norm_deviation, stats = sample_shared_component(
                component, shared, f_params, key
            )
            if sharing_stats is not None:
                sharing_stats.append(stats)
        else:
            samples, key, max_norm_deviation = sample_component(
                component, f_params, key
            )
        if np.isclose(max_norm_deviation, 1):
            raise ValueError(
                "A vanishing marginal probability distribution was encountered (normalization 0). "
                "This is likely the result of an underflow error. Please report this "
                "as a bug at https://github.com/QuEraComputing/tsim/issues/new."
            )  # pragma: no cover
        if max_norm_deviation > component.norm_warning_threshold:
            warnings.warn(
                "A marginal probability was not normalized correctly "
                f"(normalization deviated from 1 by {max_norm_deviation:.1e}). "
                "This is likely a floating point precision issue.",
                stacklevel=2,
            )
        results.append(samples)

    combined = jnp.concatenate(results, axis=1)
    if program.output_reindex is not None:
        combined = combined[:, program.output_reindex]
    return combined


def _nbytes(tree: object) -> int:
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
        strategy: SamplingStrategy = "cat5",
        seed: int | None = None,
        trajectory_sharing: bool = False,
    ):
        """Initialize the sampler by compiling the circuit.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, sample detectors/observables instead of measurements.
            mode: Compilation mode - "sequential" for autoregressive, "joint" for probabilities.
            strategy: Stabilizer rank decomposition strategy ("cat5", "bss",
                "cutting"), or "statevector" to simulate each non-direct
                component as an extracted circuit on a dense state vector.
            seed: Random seed. If None, a random seed is generated. Note that
                deterministic results are only guaranteed for a fixed batch size
                and fixed reference sample settings.
            trajectory_sharing: With ``strategy="statevector"``, simulate shots
                with equal histories on one lane (experimental).

        Raises:
            ValueError: If ``trajectory_sharing`` is combined with a
                stabilizer-rank strategy.

        """
        if trajectory_sharing and strategy != "statevector":
            raise ValueError("trajectory_sharing needs strategy='statevector'")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))

        self._key = jax.random.key(seed)
        self._trajectory_sharing = trajectory_sharing
        self._sharing: dict[bytes, tuple[Sharing | None, ...]] = {}
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
        self._direct_output_mask = np.zeros(prog.num_outputs, dtype=np.bool_)
        if n_direct > 0:
            self._direct_output_mask[self._direct_global_indices] = True
        self._direct_detector_mask = self._direct_output_mask[
            : self._num_detectors
        ].copy()

        statevector = [
            c for c in self._program.components if isinstance(c, StatevectorComponent)
        ]
        if statevector:
            peak = self._peak_bytes_per_sample()
            if peak > self._memory_budget():
                widest = max(statevector, key=lambda c: c.width)
                raise ValueError(
                    f"strategy='statevector' needs {peak} bytes per shot for the "
                    f"component with outputs {list(widest.output_indices)} (width "
                    f"{widest.width}), more than half of the available memory. "
                    f"{STABILIZER_RANK_HINT}"
                )

    def _sharing_arguments(
        self,
        postselection_mask: np.ndarray | None,
        detector_reference: np.ndarray | None = None,
    ) -> dict:
        """Return the extra ``sample_program`` arguments: none without sharing.

        Args:
            postselection_mask: Postselected detectors, whose lanes are pruned.
            detector_reference: Reference bits XOR-ed into the detectors before
                the postselection test, if any.

        """
        if not self._trajectory_sharing:
            return {}
        num_detectors = self._num_detectors
        mask = np.zeros(num_detectors, dtype=np.bool_)
        if postselection_mask is not None:
            mask = np.asarray(postselection_mask, dtype=np.bool_)
        flip = np.zeros(num_detectors, dtype=np.bool_)
        if detector_reference is not None:
            flip = np.asarray(detector_reference[:num_detectors], dtype=np.bool_)
        key = mask.tobytes() + flip.tobytes()
        if key not in self._sharing:
            self._sharing[key] = tuple(
                (
                    Sharing(
                        sharing_layout(
                            component,
                            np.array(
                                [
                                    i < num_detectors and bool(mask[i])
                                    for i in component.output_indices
                                ]
                            ),
                            np.array(
                                [
                                    i < num_detectors and bool(flip[i])
                                    for i in component.output_indices
                                ]
                            ),
                        )
                    )
                    if isinstance(component, StatevectorComponent)
                    else None
                )
                for component in self._program.components
            )
        return {"sharing": self._sharing[key], "sharing_stats": self.sharing_stats}

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
        """Choose a uniform JAX batch size for ``shots`` samples."""
        if batch_size is None:
            max_batch_size = self._estimate_batch_size()
            num_batches = max(1, ceil(shots / max_batch_size))
            batch_size = ceil(shots / num_batches)
        if compute_reference and batch_size * ceil(shots / batch_size) == shots:
            batch_size += 1
        return batch_size

    def _peak_bytes_per_sample(self) -> int:
        """Estimate peak device memory per sample from compiled program structure."""
        num_f = self._channel_sampler.signature_matrix.shape[1]
        peak = 0
        for component in self._program.components:
            if isinstance(component, StatevectorComponent):
                # NOTE: plus the f-params and the output records, on host and device.
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
        """Return the memory one batch may use: half of the free device memory."""
        device = jax.devices()[0]
        if device.platform == "gpu":
            stats = device.memory_stats()
            available = stats.get("bytes_limit", 8 * 1024**3) - stats.get(
                "bytes_in_use", 0
            )
        else:
            available = psutil.virtual_memory().available
        return int(available * 0.5)  # conservative

    def _estimate_batch_size(self) -> int:
        """Estimate the largest batch size that fits in available device memory.

        Programs with statevector components use at most
        ``STATEVECTOR_MAX_BATCH_SIZE`` lanes per batch.
        """
        batch_size = max(1, self._memory_budget() // self._peak_bytes_per_sample())
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

        if batch_size is None:
            max_batch_size = self._estimate_batch_size()
            num_batches = max(1, ceil(shots / max_batch_size))
            batch_size = ceil(shots / num_batches)
        else:
            num_batches = ceil(shots / batch_size)

        if compute_reference and batch_size * num_batches == shots:
            # Bump batch_size so the first batch's reference sample fits
            # within existing batches (keeps shapes uniform for JIT).
            batch_size += 1

        batches: list[jax.Array] = []
        reference: np.ndarray | None = None

        for _ in range(num_batches):
            f_params_np = self._channel_sampler.sample(batch_size)

            if compute_reference and reference is None:
                f_params_np[0] = 0

            f_params = jnp.asarray(f_params_np)
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program(
                self._program, f_params, subkey, **self._sharing_arguments(None)
            )

            if compute_reference and reference is None:
                reference = np.asarray(samples[0])
                samples = samples[1:]

            batches.append(samples)

        # Concatenate on device, then a single d2h. The prior
        # np.concatenate(batches) triggered per-batch __array__ (one d2h each)
        # plus a host-side memcpy into a fresh numpy buffer for the concat
        # output. For big bool tensors (e.g. 500k shots × 528 detector bits)
        # the host memcpy alone was ~1 s on top of the PCIe transfer.
        combined = batches[0] if len(batches) == 1 else jnp.concatenate(batches, axis=0)
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
        """Sample with postselection, skipping JAX for direct discarded shots.

        Shots discarded by a direct masked detector are filled with their
        direct detector columns and ``False`` elsewhere; JAX is never called for
        those shots.  Survivors are buffered until a full batch of
        ``batch_size`` is ready, then dispatched to ``sample_program`` in one
        call.  The final partial batch is padded to keep the JAX batch size
        fixed (avoiding recompilation) and the padding rows are discarded.
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

        postselect_direct = postselection_mask & self._direct_detector_mask

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

        if batch_size is None:
            batch_size = self._resolve_batch_size(
                shots, batch_size, compute_reference=False
            )

        reference: np.ndarray | None = None
        if compute_reference:
            reference = self._compute_reference_sample()

        result = np.zeros((shots, num_outputs), dtype=np.bool_)
        was_discarded = np.zeros(shots, dtype=np.bool_)
        shot_idx = 0

        sharing = self._sharing_arguments(
            postselection_mask, reference if xor_detector_ref else None
        )

        # NOTE: a chunk adds at most batch_size survivors to fewer than batch_size
        # pending ones, so two batches always hold the pending survivors.
        num_f = self._channel_sampler.signature_matrix.shape[1]
        pending_f = np.empty((2 * batch_size, num_f), dtype=np.uint8)
        pending_shot = np.empty(2 * batch_size, dtype=np.int64)
        pending = 0

        def dispatch(count: int) -> None:
            """Sample the first ``count`` pending survivors in one full batch."""
            nonlocal pending
            pending_f[count:batch_size] = pending_f[0]
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program(
                self._program, jnp.asarray(pending_f[:batch_size]), subkey, **sharing
            )
            result[pending_shot[:count]] = np.asarray(samples)[:count]
            pending -= count
            pending_f[:pending] = pending_f[count : count + pending]
            pending_shot[:pending] = pending_shot[count : count + pending]

        while shot_idx < shots:
            chunk = min(batch_size, shots - shot_idx)
            f_params_np = self._channel_sampler.sample(chunk)
            direct_full = self._compute_direct_outputs(f_params_np)
            det_cols = direct_full[:, : self._num_detectors]
            if xor_detector_ref and reference is not None:
                det_cols = det_cols ^ reference[: self._num_detectors]

            discarded = (det_cols & postselect_direct).any(axis=1)

            result[shot_idx : shot_idx + chunk, : self._num_detectors] = direct_full[
                :, : self._num_detectors
            ]
            was_discarded[shot_idx : shot_idx + chunk] = discarded

            survivors = np.flatnonzero(~discarded)
            end = pending + len(survivors)
            pending_f[pending:end] = f_params_np[survivors]
            pending_shot[pending:end] = shot_idx + survivors
            pending = end

            shot_idx += chunk
            if pending >= batch_size:
                dispatch(batch_size)

        if pending:
            dispatch(pending)

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
        strategy: SamplingStrategy = "cat5",
        seed: int | None = None,
        trajectory_sharing: bool = False,
    ):
        """Create a measurement sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: Stabilizer rank decomposition strategy ("cat5", "bss",
                "cutting"), or "statevector".
            trajectory_sharing: With ``strategy="statevector"``, simulate shots
                with equal histories on one lane (experimental).
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
                memory. When using a GPU, setting this explicitly can help fully
                utilize VRAM for maximum performance. NOTE: Changing the batch size
                will affect reproducibility even with a fixed seed.

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
        strategy: SamplingStrategy = "cat5",
        seed: int | None = None,
        trajectory_sharing: bool = False,
    ):
        """Create a detector sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: Stabilizer rank decomposition strategy ("cat5", "bss",
                "cutting"), or "statevector".
            trajectory_sharing: With ``strategy="statevector"``, simulate shots
                with equal histories on one lane (experimental).
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
                memory. When using a GPU, setting this explicitly can help fully
                utilize VRAM for maximum performance. NOTE: Changing the batch size
                will affect reproducibility even with a fixed seed.
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
                When set, shots where any masked direct detector fires skip the JAX
                sampling loop. All ``shots`` rows are still returned: survivors contain
                the full sample, while discarded rows retain direct detector columns
                and fill component columns with ``False``. Re-apply the mask to the
                returned detector columns to recover surviving shots.

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
        strategy: DecompositionStrategy = "cat5",
        seed: int | None = None,
    ):
        """Create a probability estimator.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, compute detector/observable probabilities.
            strategy: Stabilizer rank decomposition strategy.
                Must be one of "cat5", "bss", "cutting".
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
