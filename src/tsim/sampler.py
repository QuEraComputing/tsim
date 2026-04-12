"""Compiled samplers for measurements and detectors."""

from __future__ import annotations

import warnings
from math import ceil
from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from pyzx_param.simulate import DecompositionStrategy

from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.core.types import CompiledComponent, CompiledProgram
from tsim.noise.channels import ChannelSampler

if TYPE_CHECKING:
    from jax import Array as PRNGKey

    from tsim.circuit import Circuit


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
    """Sample outputs from a single component using autoregressive sampling.

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


def sample_program(
    program: CompiledProgram,
    f_params: jax.Array,
    key: PRNGKey,
) -> jax.Array:
    """Sample all outputs from a compiled program.

    Args:
        program: The compiled program to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Samples array of shape (batch_size, num_outputs), reordered to
        match the original output indices.

    """
    if len(program.components) == 0:
        batch_size = f_params.shape[0]
        return jnp.empty((batch_size, 0), dtype=jnp.bool_)

    results: list[jax.Array] = []

    for component in program.components:
        samples, key, max_norm_deviation = sample_component(component, f_params, key)
        if np.isclose(max_norm_deviation, 1):
            raise ValueError(
                "A vanishing marginal probability distributionwas encountered (normalization 0). "
                "This is likely the result of an underflow error. Please report this "
                "as a bug at https://github.com/QuEraComputing/tsim/issues/new."
            )  # pragma: no cover
        if max_norm_deviation > 1e-5:
            warnings.warn(
                "A marginal probability was not normalized correctly "
                f"(normalization deviated from 1 by {max_norm_deviation:.1e}). "
                "This is likely a floating point precision issue."
            )
        results.append(samples)

    combined = jnp.concatenate(results, axis=1)
    return combined[:, jnp.argsort(program.output_order)]


class _CompiledSamplerBase:
    """Base class for compiled samplers with common initialization logic."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool,
        mode: Literal["sequential", "joint"],
        strategy: DecompositionStrategy = "cat5",
        seed: int | None = None,
    ):
        """Initialize the sampler by compiling the circuit.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, sample detectors/observables instead of measurements.
            mode: Compilation mode - "sequential" for autoregressive, "joint" for probabilities.
            strategy: Stabilizer rank decomposition strategy.
                Must be one of "cat5", "bss", "cutting".
            seed: Random seed. If None, a random seed is generated.

        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**30))

        self._key = jax.random.key(seed)

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

    def _peak_bytes_per_sample(self) -> int:
        """Estimate peak device memory per sample from compiled program structure."""
        peak = 0
        for component in self._program.components:
            for circuit in component.compiled_scalar_graphs:
                G = circuit.num_graphs
                max_a = circuit.a_const_phases.shape[1]
                max_b = circuit.b_term_types.shape[1]
                max_c = circuit.c_const_bits_a.shape[1]
                max_d = circuit.d_const_alpha.shape[1]
                largest = max(max_a * 16, max_b * 4, max_c * 4, max_d * 16)
                peak = max(peak, G * largest * 3)
        return max(peak, 1)

    def _estimate_batch_size(self) -> int:
        """Estimate the largest batch size that fits in available device memory."""
        device = jax.devices()[0]
        if device.platform == "gpu":
            stats = device.memory_stats()
            available = stats.get("bytes_limit", 8 * 1024**3) - stats.get(
                "bytes_in_use", 0
            )
        else:
            available = psutil.virtual_memory().available

        half_of_available = int(available * 0.5)  # conservative estimate
        return max(1, half_of_available // self._peak_bytes_per_sample())

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
        if batch_size is None:
            max_batch_size = self._estimate_batch_size()
            num_batches = max(1, ceil(shots / max_batch_size))
            batch_size = ceil(shots / num_batches)

        if compute_reference:
            num_batches = ceil(shots / batch_size)
            has_leeway = batch_size * num_batches > shots
            if not has_leeway:
                batch_size += 1

        batches: list[jax.Array] = []
        reference: np.ndarray | None = None

        for _ in range(ceil(shots / batch_size)):
            f_params_np = self._channel_sampler.sample(batch_size)

            if compute_reference and reference is None:
                f_params_np[0] = 0

            f_params = jnp.asarray(f_params_np)
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program(self._program, f_params, subkey)

            if compute_reference and reference is None:
                reference = np.asarray(samples[0])
                samples = samples[1:]

            batches.append(samples)

        result = np.concatenate(batches)[:shots]

        if compute_reference:
            assert reference is not None
            return result, reference
        return result

    def __repr__(self) -> str:
        """Return a string representation with compilation statistics."""
        c_graphs = []
        c_params = []
        c_a_terms = []
        c_b_terms = []
        c_c_terms = []
        c_d_terms = []
        num_circuits = 0
        total_memory_bytes = 0
        num_outputs = []

        for component in self._program.components:
            for circuit in component.compiled_scalar_graphs:
                num_outputs.append(len(component.output_indices))
                c_graphs.append(circuit.num_graphs)
                c_params.append(circuit.n_params)
                c_a_terms.append(circuit.a_const_phases.size)
                c_b_terms.append(circuit.b_term_types.size)
                c_c_terms.append(circuit.c_const_bits_a.size)
                c_d_terms.append(circuit.d_const_alpha.size + circuit.d_const_beta.size)
                num_circuits += 1

                total_memory_bytes += sum(
                    v.nbytes
                    for v in jax.tree_util.tree_leaves(circuit)
                    if isinstance(v, jax.Array)
                )

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

        return (
            f"{type(self).__name__}({np.sum(c_graphs)} graphs, "
            f"{error_channel_bits} error channel bits, "
            f"{np.max(num_outputs) if num_outputs else 0} outputs for largest cc, "
            f"≤ {np.max(c_params) if c_params else 0} parameters, {np.sum(c_a_terms)} A terms, "
            f"{np.sum(c_b_terms)} B terms, "
            f"{np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms, "
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
        strategy: DecompositionStrategy = "cat5",
        seed: int | None = None,
    ):
        """Create a measurement sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: Stabilizer rank decomposition strategy.
                Must be one of "cat5", "bss", "cutting".
            seed: Random seed for JAX. If None, a random seed is generated.

        """
        super().__init__(
            circuit,
            sample_detectors=False,
            mode="sequential",
            seed=seed,
            strategy=strategy,
        )

    def sample(self, shots: int, *, batch_size: int | None = None) -> np.ndarray:
        """Sample measurement outcomes from the circuit.

        Args:
            shots: The number of times to sample every measurement in the circuit.
            batch_size: The number of samples to process in each batch. Defaults to
                None, which automatically chooses a batch size based on available
                memory. When using a GPU, setting this explicitly can help fully
                utilize VRAM for maximum performance.

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
        strategy: DecompositionStrategy = "cat5",
        seed: int | None = None,
    ):
        """Create a detector sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: Stabilizer rank decomposition strategy.
                Must be one of "cat5", "bss", "cutting".
            seed: Random seed for JAX. If None, a random seed is generated.

        """
        super().__init__(
            circuit,
            sample_detectors=True,
            mode="sequential",
            seed=seed,
            strategy=strategy,
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
                utilize VRAM for maximum performance.
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

        Returns:
            A numpy array or tuple of numpy arrays containing the samples.

        """
        compute_reference = (
            use_detector_reference_sample or use_observable_reference_sample
        )

        if compute_reference:
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

        if append_observables:
            return _maybe_bit_pack(samples, bit_packed=bit_packed)

        num_detectors = self._num_detectors
        det_samples = samples[:, :num_detectors]
        obs_samples = samples[:, num_detectors:]

        if prepend_observables:
            combined = np.concatenate([obs_samples, det_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if separate_observables:
            return (
                _maybe_bit_pack(det_samples, bit_packed=bit_packed),
                _maybe_bit_pack(obs_samples, bit_packed=bit_packed),
            )

        return _maybe_bit_pack(det_samples, bit_packed=bit_packed)
        # TODO: don't compute observables if they are discarded here


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
            seed: Random seed for JAX. If None, a random seed is generated.

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
        f_samples = jnp.asarray(self._channel_sampler.sample(batch_size))
        p_norm = jnp.ones(batch_size)
        p_joint = jnp.ones(batch_size)

        for component in self._program.components:
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
