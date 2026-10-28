"""Compiled samplers for measurements and detectors."""

from __future__ import annotations

import warnings
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
from tsim.core.types import CompiledComponent, CompiledProgram, Device, SamplingStrategy
from tsim.noise.channels import ChannelSampler
from tsim.statevector.program import StatevectorComponent
from tsim.statevector.shots import ShotSampler
from tsim.utils.cuda_helpers import copy_d2h
from tsim.utils.kernels import kernel

if TYPE_CHECKING:
    from jax import Array as PRNGKey

    from tsim.circuit import Circuit
    from tsim.statevector.cuda.sampler import CudaShotSampler

# NOTE: postselection with a stabilizer-rank strategy screens shots for direct
# discards in rounds of SCREEN_CHUNKS chunks of at most SCREEN_SHOTS shots.
SCREEN_SHOTS = 2**14
SCREEN_CHUNKS = 2


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


def _nbytes(tree: object) -> int:
    if isinstance(tree, StatevectorComponent):
        return sum(array.nbytes for array in tree.tape)
    return sum(
        v.nbytes for v in jax.tree_util.tree_leaves(tree) if isinstance(v, jax.Array)
    )


class _StabilizerRankBackend:
    """Samples stabilizer-rank programs with JAX, one batch at a time."""

    def __init__(self, program: CompiledProgram, seed: int) -> None:
        self._program = program
        self._key = jax.random.key(seed)

    def next_seed(self) -> PRNGKey:
        self._key, key = jax.random.split(self._key)
        return key

    def memory_available(self) -> int:
        device = jax.devices()[0]
        if device.platform != "gpu":
            return psutil.virtual_memory().available
        stats = device.memory_stats()
        return stats.get("bytes_limit", 8 * 1024**3) - stats.get("bytes_in_use", 0)

    def peak_bytes_per_shot(self) -> int:
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

    def sample(
        self,
        f_params: np.ndarray,
        seed: PRNGKey,
        *,
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
        device: Device = "cpu",
    ):
        """Initialize the sampler by compiling the circuit.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, sample detectors/observables instead of measurements.
            mode: Compilation mode - "sequential" for autoregressive, "joint" for probabilities.
            strategy: "active_statevector" (the default) to simulate each
                non-direct component on a dense state vector over its active
                qubits, or a stabilizer rank decomposition strategy ("cat5",
                "bss", "cutting"). "active_statevector" samples on the CPU, on
                ``numba.get_num_threads()`` threads.
            seed: Random seed. If None, a random seed is generated. With
                "active_statevector", and for circuits without non-direct
                components, the samples of a seed depend on neither the batch
                size nor the thread count; with a stabilizer-rank strategy they
                are only deterministic for a fixed batch size.
            trajectory_sharing: With ``strategy="active_statevector"``, simulate shots
                with equal histories on one lane. Defaults to True for
                ``"active_statevector"`` and False otherwise. Only the CPU
                executor shares lanes.
            device: "cpu" (the default), or "cuda" to run the noise sampler and
                the statevector components as CUDA kernels in float32 (needs
                ``strategy="active_statevector"``, CuPy and an NVIDIA GPU).

        Raises:
            ValueError: If ``trajectory_sharing`` is combined with a
                stabilizer-rank strategy, or ``device="cuda"`` with one.

        """
        if trajectory_sharing is None:
            trajectory_sharing = strategy == "active_statevector"
        if trajectory_sharing and strategy != "active_statevector":
            raise ValueError("trajectory_sharing needs strategy='active_statevector'")
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
        if device == "cuda" and strategy != "active_statevector":
            raise ValueError("device='cuda' needs strategy='active_statevector'")
        self._device = device
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
        program = self._program
        # NOTE: programs of statevector components, and programs without components
        # under any strategy, are sampled as shot words; the others with JAX.
        self._shots: ShotSampler | CudaShotSampler | None = None
        self._backend: _StabilizerRankBackend | None = None
        if device == "cuda":
            from tsim.statevector.cuda.sampler import CudaShotSampler

            self._shots = CudaShotSampler(
                program, self._channel_sampler, self._num_detectors, seed
            )
        elif strategy == "active_statevector" or not program.components:
            self._shots = ShotSampler(
                program,
                self._channel_sampler,
                self._num_detectors,
                seed,
                share=trajectory_sharing,
            )
        else:
            self._backend = _StabilizerRankBackend(program, seed)

        # NOTE: _write_direct_detectors writes column output_order[k] without
        # bounds checks.
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

    def _sample_shot_outputs(
        self,
        shots: int,
        batch_size: int | None,
        columns: list[np.ndarray],
        *,
        bit_packed: bool,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Return the output bits ``columns`` of ``shots`` shots (``ShotSampler.sample``)."""
        assert self._shots is not None
        _check_sizes(shots, batch_size)
        arrays, deviation = self._shots.sample(
            shots,
            columns,
            bit_packed=bit_packed,
            batch_size=batch_size,
            mask=mask,
            flip=flip,
        )
        _check_norm(deviation, StatevectorComponent.norm_warning_threshold)
        return arrays

    def _reference_sample(self) -> np.ndarray:
        """Return a noiseless sample of every output (all f-params 0).

        Does not advance the channel sampler RNG.
        """
        if self._shots is not None:
            reference, deviation = self._shots.reference()
            _check_norm(deviation, StatevectorComponent.norm_warning_threshold)
            return reference
        assert self._backend is not None
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
        """Choose a uniform batch size for ``shots`` samples: as memory allows."""
        if batch_size is None:
            batch_size = self._auto_batch_size(shots)
        if compute_reference and batch_size * ceil(shots / batch_size) == shots:
            batch_size += 1
        return batch_size

    def _auto_batch_size(self, shots: int) -> int:
        """Return the size of equal batches of ``shots`` that fit in memory."""
        num_batches = max(1, ceil(shots / self._estimate_batch_size()))
        return ceil(shots / num_batches)

    def _peak_bytes_per_sample(self) -> int:
        """Estimate peak device memory per sample from compiled program structure."""
        assert self._backend is not None
        return max(self._backend.peak_bytes_per_shot(), 1)

    def _memory_budget(self) -> int:
        """Return the memory one batch may use: half of the free device memory."""
        assert self._backend is not None
        return int(self._backend.memory_available() * 0.5)  # conservative

    def _estimate_batch_size(self) -> int:
        """Estimate the largest batch size that fits in available device memory."""
        return max(1, self._memory_budget() // self._peak_bytes_per_sample())

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
        """Sample in batches with JAX and concatenate results.

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

        # NOTE: with a reference sample the batch size grows by one when the
        # batches would hold exactly ``shots``, so that the first batch's
        # reference sample fits without another batch shape for JIT.
        batch_size = self._resolve_batch_size(
            shots, batch_size, compute_reference=compute_reference
        )
        num_batches = ceil((shots + compute_reference) / batch_size)
        backend = self._backend
        assert backend is not None
        batches = []
        for index in range(num_batches):
            f_params = self._channel_sampler.sample(batch_size)
            if compute_reference and index == 0:
                f_params[0] = 0
            batches.append(backend.sample(f_params, backend.next_seed()))
        reference: np.ndarray | None = None
        if compute_reference:
            reference = np.asarray(batches[0][0])
            batches[0] = batches[0][1:]
        result = backend.concatenate(batches)[:shots]

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
        referenced: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample with postselection and JAX, skipping the components for direct discards.

        A shot that a masked direct detector discards keeps its direct detector
        columns and reads ``False`` elsewhere. The other shots are sampled in
        batches of ``batch_size``. Without a ``batch_size``, batches are as large
        as memory allows, and the last survivors are split into equal batches.

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

        split_last = batch_size is None
        if batch_size is None:
            batch_size = self._estimate_batch_size()
        flip = None
        if referenced is not None:
            flip = self._reference_sample() & referenced
        f_indices, flips, columns = self._direct_detectors
        if flip is not None:
            flips = flips ^ flip[columns]
        postselect = postselection_mask[columns]
        result = np.empty((shots, num_outputs), dtype=np.bool_)
        backend = self._backend
        assert backend is not None
        chunk = min(batch_size, SCREEN_SHOTS)
        num_f = self._channel_sampler.signature_matrix.shape[1]
        pending_f = [np.empty((0, num_f), dtype=np.uint8)]
        pending_shot = [np.empty(0, dtype=np.int64)]
        pending = 0
        starts = range(0, shots, chunk)
        screened_chunks = 0
        # NOTE: chunks are screened until a full batch of survivors is pending. A
        # batch starts every batch_size survivors, and all seeds are drawn in order.
        while screened_chunks < len(starts) or pending:
            while screened_chunks < len(starts) and pending < batch_size:
                round_starts = starts[screened_chunks : screened_chunks + SCREEN_CHUNKS]
                screened_chunks += len(round_starts)
                for start in round_starts:
                    stop = min(start + chunk, shots)
                    f_params = self._channel_sampler.sample(stop - start)
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
                    pending_f.append(f_params[survivors])
                    pending_shot.append(start + survivors)
                    pending += len(survivors)
            ready = pending
            size, rows = batch_size, batch_size
            if screened_chunks < len(starts):
                ready -= ready % batch_size
            elif split_last and ready:
                # NOTE: JAX pads these batches to a power of two, so that few
                # batch shapes are compiled.
                size = ceil(ready / ceil(ready / batch_size))
                rows = min(batch_size, 1 << (size - 1).bit_length())
            all_f, all_shot = np.concatenate(pending_f), np.concatenate(pending_shot)
            for start in range(0, ready, size):
                samples = backend.sample(
                    all_f[start : start + size],
                    backend.next_seed(),
                    flip=flip,
                    pad_to=rows,
                )
                result[all_shot[start : start + size]] = np.asarray(samples)
            pending_f, pending_shot = [all_f[ready:]], [all_shot[ready:]]
            pending -= ready
        return result

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
        device: Device = "cpu",
    ):
        """Create a measurement sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: "active_statevector" (the default), or a stabilizer rank
                decomposition strategy ("cat5", "bss", "cutting").
            trajectory_sharing: With ``strategy="active_statevector"``, simulate shots
                with equal histories on one lane. Defaults to True for
                ``"active_statevector"`` and False otherwise.
            seed: Random seed for the sampler. With "active_statevector" the samples
                of a seed do not depend on the batch size; with a stabilizer-rank
                strategy they are only deterministic for a fixed batch size.

            device: "cpu" (the default) or "cuda" (see ``_CompiledSamplerBase``).

        """
        super().__init__(
            circuit,
            sample_detectors=False,
            mode="sequential",
            seed=seed,
            strategy=strategy,
            trajectory_sharing=trajectory_sharing,
            device=device,
        )

    def sample(self, shots: int, *, batch_size: int | None = None) -> np.ndarray:
        """Sample measurement outcomes from the circuit.

        Args:
            shots: The number of times to sample every measurement in the circuit.
            batch_size: The number of samples to process at once. Defaults to None,
                which chooses it from the available memory. With
                ``strategy="active_statevector"`` it bounds the shots a thread
                holds and does not change the samples.
                With a stabilizer-rank strategy on a GPU, setting this explicitly
                can help fully utilize VRAM; it changes the samples of a seed.

        Returns:
            A numpy array containing the measurement samples.

        """
        if self._shots is not None:
            columns = [np.arange(self._program.num_outputs)]
            if not self._program.components:
                _check_sizes(shots, batch_size)
                return self._shots.direct_bools(shots)
            return self._sample_shot_outputs(
                shots, batch_size, columns, bit_packed=False
            )[0]
        return self._sample_batches(shots, batch_size)


def _maybe_bit_pack(array: np.ndarray, *, bit_packed: bool) -> np.ndarray:
    """Optionally bit-pack a boolean array."""
    if not bit_packed:
        return array
    return np.packbits(array.astype(np.bool_), axis=1, bitorder="little")


def _take_columns(samples: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Return ``samples[:, columns]``, as a view when ``columns`` is a range."""
    start = int(columns[0]) if len(columns) else 0
    if np.array_equal(columns, np.arange(start, start + len(columns))):
        return samples[:, start : start + len(columns)]
    return samples[:, columns]


def _output_columns(
    num_detectors: int,
    num_outputs: int,
    *,
    prepend: bool,
    append: bool,
    separate: bool,
) -> list[np.ndarray]:
    """Return the output columns of each array that a detector sample returns."""
    detectors = np.arange(num_detectors)
    observables = np.arange(num_detectors, num_outputs)
    if separate:
        return [detectors, observables]
    return [
        np.concatenate([observables] * prepend + [detectors] + [observables] * append)
    ]


class CompiledDetectorSampler(_CompiledSamplerBase):
    """Samples detector and observable outcomes from a quantum circuit."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        strategy: SamplingStrategy = "active_statevector",
        seed: int | None = None,
        trajectory_sharing: bool | None = None,
        device: Device = "cpu",
    ):
        """Create a detector sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: "active_statevector" (the default), or a stabilizer rank
                decomposition strategy ("cat5", "bss", "cutting").
            trajectory_sharing: With ``strategy="active_statevector"``, simulate shots
                with equal histories on one lane. Defaults to True for
                ``"active_statevector"`` and False otherwise.
            seed: Random seed for the sampler. With "active_statevector" the samples
                of a seed do not depend on the batch size; with a stabilizer-rank
                strategy they are only deterministic for a fixed batch size and
                fixed reference sample settings.

            device: "cpu" (the default) or "cuda" (see ``_CompiledSamplerBase``).

        """
        super().__init__(
            circuit,
            sample_detectors=True,
            mode="sequential",
            seed=seed,
            strategy=strategy,
            trajectory_sharing=trajectory_sharing,
            device=device,
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
            batch_size: The number of samples to process at once. Defaults to None,
                which chooses it from the available memory. With
                ``strategy="active_statevector"`` it bounds the shots a thread
                holds and does not change the samples.
                With a stabilizer-rank strategy on a GPU, setting this explicitly
                can help fully utilize VRAM; it changes the samples of a seed.
            separate_observables: Defaults to False. When set to True, the return value
                is a (detection_events, observable_flips) tuple instead of a flat
                detection_events array.
            prepend_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the start of the results.
            append_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the end of the results.
            bit_packed: Defaults to false. When set, results are bit-packed. The
                packed rows may be a strided view of a wider array.
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
            # NOTE: shot words discard and prune with any mask; with a
            # stabilizer-rank strategy only direct detectors that can fire skip work.
            if self._shots is not None:
                postselection_mask = mask if self._program.components else None
            else:
                skips = mask & self._direct_detector_mask
                indices, flips, columns = self._direct_detectors
                constant = indices < 0
                skips[columns[constant]] &= flips[constant] & (
                    not use_detector_reference_sample
                )
                postselection_mask = mask if skips.any() else None

        num_outputs = self._program.num_outputs
        is_detector = np.arange(num_outputs) < num_detectors
        referenced = np.where(
            is_detector, use_detector_reference_sample, use_observable_reference_sample
        )
        columns = _output_columns(
            num_detectors,
            num_outputs,
            prepend=prepend_observables,
            append=append_observables,
            separate=separate_observables,
        )
        if self._shots is not None:
            # NOTE: the reference bits are XOR-ed in before the postselection tests.
            flip = self._reference_sample() & referenced if compute_reference else None
            if self._program.components or bit_packed:
                arrays = self._sample_shot_outputs(
                    shots,
                    batch_size,
                    columns,
                    bit_packed=bit_packed,
                    mask=postselection_mask,
                    flip=flip,
                )
                return (arrays[0], arrays[1]) if separate_observables else arrays[0]
            _check_sizes(shots, batch_size)
            samples = self._shots.direct_bools(shots, flip)
        elif postselection_mask is not None:
            if not compute_reference:
                referenced = None
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
        arrays = [
            _maybe_bit_pack(_take_columns(samples, c), bit_packed=bit_packed)
            for c in columns
        ]
        return (arrays[0], arrays[1]) if separate_observables else arrays[0]

    def sample_postselected(
        self,
        shots: int,
        *,
        postselection_mask: np.ndarray | None = None,
        observable_expectations: bool = False,
        use_detector_reference_sample: bool = False,
        use_observable_reference_sample: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample ``shots`` shots and return only those that pass postselection.

        A shot passes when none of the detectors in ``postselection_mask`` fires
        (every shot passes without a mask). Unlike ``sample``, the discarded
        shots are not returned, so with ``device="cuda"`` they are never copied
        off the GPU.

        Args:
            shots: The number of shots to sample, discarded ones included.
            postselection_mask: Optional boolean array of length ``num_detectors``.
            observable_expectations: Also return, for each passing shot and each
                observable, the expectation of ``(-1)^observable`` given the
                shot's history before the measurement that settles it, as a
                ``(passed, num_observables)`` float32 array. This is the
                observable's expectation value on the shot's state (cf. Clifft's
                ``EXP_VAL``): it is ``±1`` for a deterministic observable and
                ``0`` for a fair coin. It does not condition on postselection
                that follows that measurement, so when a postselected detector
                is settled by a measurement of the observable's statevector
                component at or after it, the observable's value is the sampled
                ``±1`` (``1 - 2 * observable``) instead; either way it averages
                to the passing shots' expectation. Needs ``device="cuda"``.
            use_detector_reference_sample: As in ``sample``.
            use_observable_reference_sample: As in ``sample``.
            batch_size: Shots per batch.

        Returns:
            The passing shots' detectors then observables, bit-packed as
            ``np.packbits(..., bitorder="little")`` would, in shot order; with
            ``observable_expectations``, also their expectations.

        Raises:
            ValueError: If ``observable_expectations`` is set without
                ``device="cuda"``.

        """
        num_detectors = self._num_detectors
        num_outputs = self._program.num_outputs
        mask = None
        if postselection_mask is not None:
            mask = np.asarray(postselection_mask, dtype=np.bool_)
            if mask.shape != (num_detectors,):
                raise ValueError(
                    f"postselection_mask must have shape ({num_detectors},), "
                    f"got {mask.shape}"
                )
        compute_reference = (
            use_detector_reference_sample or use_observable_reference_sample
        )
        flip = None
        if compute_reference:
            is_detector = np.arange(num_outputs) < num_detectors
            referenced = np.where(
                is_detector,
                use_detector_reference_sample,
                use_observable_reference_sample,
            )
            flip = self._reference_sample() & referenced
        if self._device == "cuda":
            from tsim.statevector.cuda.sampler import CudaShotSampler

            assert isinstance(self._shots, CudaShotSampler)
            _check_sizes(shots, batch_size)
            rows, values, deviation = self._shots.sample_accepted(
                shots,
                mask=mask,
                flip=flip,
                expectations=observable_expectations,
                batch_size=batch_size,
            )
            _check_norm(deviation, StatevectorComponent.norm_warning_threshold)
            if observable_expectations:
                assert values is not None
                return rows, values
            return rows
        if observable_expectations:
            raise ValueError("observable_expectations needs device='cuda'")
        samples = self.sample(
            shots,
            batch_size=batch_size,
            append_observables=True,
            bit_packed=True,
            use_detector_reference_sample=use_detector_reference_sample,
            use_observable_reference_sample=use_observable_reference_sample,
            postselection_mask=mask,
        )
        if mask is None:
            return samples
        mask_bytes = np.packbits(
            np.concatenate([mask, np.zeros(num_outputs - num_detectors, bool)]),
            bitorder="little",
        )
        return samples[~(samples & mask_bytes).any(axis=1)]


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
