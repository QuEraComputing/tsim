from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp
import numpy as np

from tsim.channels import ChannelSampler
from tsim.evaluate import evaluate_batch
from tsim.graph_util import prepare_graph
from tsim.program import compile_program
from tsim.types import CompiledComponent, CompiledProgram, SamplingGraph

if TYPE_CHECKING:
    from jax import Array as PRNGKey

    from tsim.circuit import Circuit


def _sample_component_impl(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey]:
    """Implementation of component sampling using autoregressive sampling.

    Args:
        component: The compiled component to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Tuple of (samples, next_key) where samples has shape
        (batch_size, num_outputs_for_component).
    """
    batch_size = f_params.shape[0]

    # Select this component's f-parameters
    f_selected = f_params[:, component.f_selection].astype(jnp.bool_)

    # Initialize m_accumulated as empty
    m_accumulated = jnp.empty((batch_size, 0), dtype=jnp.bool_)

    # First circuit is normalization (only f-params)
    prev = jnp.abs(evaluate_batch(component.compiled_scalar_graphs[0], f_selected))

    # Autoregressive sampling for remaining circuits
    for circuit in component.compiled_scalar_graphs[1:]:
        # Build params: [f_selected, m_accumulated, trying_bit=1]
        ones = jnp.ones((batch_size, 1), dtype=jnp.bool_)
        params = jnp.hstack([f_selected, m_accumulated, ones])

        # Evaluate P(bit=1 | previous bits)
        p1 = jnp.abs(evaluate_batch(circuit, params))

        # Sample from Bernoulli
        key, subkey = jax.random.split(key)
        bits = jax.random.bernoulli(subkey, p=p1 / prev)

        # Update prev using chain rule: new_prev = p1 if bit=1, else prev - p1
        prev = jnp.where(bits, p1, prev - p1)

        # Accumulate the sampled bit
        m_accumulated = jnp.hstack([m_accumulated, bits[:, None]])

    return m_accumulated, key


@jax.jit
def _sample_component_jit(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey]:
    """JIT-compiled version of component sampling."""
    return _sample_component_impl(component, f_params, key)


def sample_component(
    component: CompiledComponent,
    f_params: jax.Array,
    key: PRNGKey,
) -> tuple[jax.Array, PRNGKey]:
    """Sample outputs from a single component using autoregressive sampling.

    Args:
        component: The compiled component to sample from.
        f_params: Error parameters, shape (batch_size, num_f_params).
        key: JAX random key.

    Returns:
        Tuple of (samples, next_key) where samples has shape
        (batch_size, num_outputs_for_component).
    """
    # Skip JIT for small components (overhead not worth it)
    if len(component.output_indices) <= 1:
        return _sample_component_impl(component, f_params, key)
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
    results: list[jax.Array] = []

    for component in program.components:
        samples, key = sample_component(component, f_params, key)
        results.append(samples)

    # Combine results from all components
    combined = jnp.concatenate(results, axis=1)

    # Reorder to original output order
    return combined[:, jnp.argsort(program.output_order)]


# =============================================================================
# Compiled Samplers
# =============================================================================


def _create_channel_sampler(
    prepared: SamplingGraph, seed: int | None
) -> ChannelSampler:
    """Create a channel sampler from a prepared graph."""
    return ChannelSampler(
        channel_probs=prepared.channel_probs,
        error_transform=prepared.error_transform,
        seed=seed,
    )


class CompiledMeasurementSampler:
    """Samples measurement outcomes from a quantum circuit.

    Uses sequential decomposition [0, 1, 2, ..., n] where:
    - compiled_scalar_graphs[0]: normalization (0 outputs plugged)
    - compiled_scalar_graphs[i]: cumulative probability up to bit i
    """

    def __init__(self, circuit: Circuit, *, seed: int | None = None):
        """Create a measurement sampler.

        Args:
            circuit: The quantum circuit to compile.
            seed: Random seed for JAX. If None, a random seed is generated.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))

        self._key = jax.random.key(seed)

        prepared = prepare_graph(circuit, sample_detectors=False)
        self._program = compile_program(prepared, mode="sequential")

        self._key, subkey = jax.random.split(self._key)
        channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
        self._channel_sampler = _create_channel_sampler(prepared, channel_seed)

        self.circuit = circuit

    def sample(self, shots: int, *, batch_size: int = 1024) -> np.ndarray:
        """Sample measurement outcomes from the circuit.

        Args:
            shots: The number of times to sample every measurement in the circuit.
            batch_size: The number of samples to process in each batch. When using a
                GPU, it is recommended to increase this value until VRAM is fully
                utilized for maximum performance.

        Returns:
            A numpy array containing the measurement samples.
        """
        if shots < batch_size:
            batch_size = shots

        batches: list[jax.Array] = []
        for _ in range(ceil(shots / batch_size)):
            f_params = self._channel_sampler.sample(batch_size)
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program(self._program, f_params, subkey)
            batches.append(samples)

        return np.concatenate(batches)[:shots]

    def __repr__(self) -> str:
        return _program_repr(
            self._program,
            len(self._channel_sampler.channels),
            "CompiledMeasurementSampler",
        )


def maybe_bit_pack(array: np.ndarray, *, do_nothing: bool = False) -> np.ndarray:
    """Bit pack an array of boolean values (or do nothing).

    Args:
        array: The array to bit pack.
        do_nothing: If True, do nothing and return the array as is.

    Returns:
        The bit packed array or the original array if do_nothing is True.
    """
    if do_nothing:
        return array
    return np.packbits(array.astype(np.bool_), axis=1, bitorder="little")


class CompiledDetectorSampler:
    """Samples detector and observable outcomes from a quantum circuit."""

    def __init__(self, circuit: Circuit, *, seed: int | None = None):
        """Create a detector sampler.

        Args:
            circuit: The quantum circuit to compile.
            seed: Random seed for JAX. If None, a random seed is generated.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))

        self._key = jax.random.key(seed)

        prepared = prepare_graph(circuit, sample_detectors=True)
        self._program = compile_program(prepared, mode="sequential")

        self._key, subkey = jax.random.split(self._key)
        channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
        self._channel_sampler = _create_channel_sampler(prepared, channel_seed)

        self.circuit = circuit
        self._num_detectors = prepared.num_detectors

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int = 1024,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[True],
        bit_packed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def sample(
        self,
        shots: int,
        *,
        batch_size: int = 1024,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: Literal[False] = False,
        bit_packed: bool = False,
    ) -> np.ndarray: ...

    def sample(
        self,
        shots: int,
        *,
        batch_size: int = 1024,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Returns detector samples from the circuit.

        The circuit must define the detectors using DETECTOR instructions. Observables
        defined by OBSERVABLE_INCLUDE instructions can also be included in the results
        as honorary detectors.

        Args:
            shots: The number of times to sample every detector in the circuit.
            batch_size: The number of samples to process in each batch. When using a
                GPU, it is recommended to increase this value until VRAM is fully
                utilized for maximum performance.
            separate_observables: Defaults to False. When set to True, the return value
                is a (detection_events, observable_flips) tuple instead of a flat
                detection_events array.
            prepend_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the start of the results.
            append_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the end of the results.
            bit_packed: Defaults to false. When set, results are bit-packed.

        Returns:
            A numpy array or tuple of numpy arrays containing the samples.
        """
        if shots < batch_size:
            batch_size = shots

        batches: list[jax.Array] = []
        for _ in range(ceil(shots / batch_size)):
            f_params = self._channel_sampler.sample(batch_size)
            self._key, subkey = jax.random.split(self._key)
            samples = sample_program(self._program, f_params, subkey)
            batches.append(samples)

        samples = np.concatenate(batches)[:shots]

        if append_observables:
            return maybe_bit_pack(samples, do_nothing=not bit_packed)

        num_detectors = self._num_detectors
        det_samples = samples[:, :num_detectors]
        obs_samples = samples[:, num_detectors:]

        if prepend_observables:
            return maybe_bit_pack(
                np.concatenate([obs_samples, det_samples], axis=1),
                do_nothing=not bit_packed,
            )
        if separate_observables:
            return (
                maybe_bit_pack(det_samples, do_nothing=not bit_packed),
                maybe_bit_pack(obs_samples, do_nothing=not bit_packed),
            )

        return maybe_bit_pack(det_samples, do_nothing=not bit_packed)
        # TODO: don't compute observables if they are discarded here

    def __repr__(self) -> str:
        return _program_repr(
            self._program,
            len(self._channel_sampler.channels),
            "CompiledDetectorSampler",
        )


class CompiledStateProbs:
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
        seed: int | None = None,
    ):
        """Create a probability estimator.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, compute detector/observable probabilities.
            seed: Random seed for JAX. If None, a random seed is generated.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))

        key = jax.random.key(seed)

        prepared = prepare_graph(circuit, sample_detectors=sample_detectors)
        self._program = compile_program(prepared, mode="joint")

        _, subkey = jax.random.split(key)
        channel_seed = int(jax.random.randint(subkey, (), 0, 2**30))
        self._channel_sampler = _create_channel_sampler(prepared, channel_seed)

        self.circuit = circuit

    def probability_of(self, state: np.ndarray, *, batch_size: int) -> np.ndarray:
        """Compute probabilities for a batch of error samples given a measurement state.

        Args:
            state: The measurement outcome state to compute probability for.
            batch_size: Number of error samples to use for estimation.

        Returns:
            Array of probabilities P(state | error_sample) for each error sample.
        """
        f_samples = self._channel_sampler.sample(batch_size)
        p_norm = jnp.ones(batch_size)
        p_joint = jnp.ones(batch_size)

        for component in self._program.components:
            assert (
                len(component.compiled_scalar_graphs) == 2
            )  # joint mode: [norm, full]

            # Select this component's f-params
            f_selected = f_samples[:, component.f_selection]

            norm_circuit, joint_circuit = component.compiled_scalar_graphs

            # Normalization: only f-params
            p_norm = p_norm * jnp.abs(evaluate_batch(norm_circuit, f_selected))

            # Joint probability: f-params + state
            component_state = state[list(component.output_indices)]
            tiled_state = jnp.tile(component_state, (batch_size, 1))
            joint_params = jnp.hstack([f_selected, tiled_state])
            p_joint = p_joint * jnp.abs(evaluate_batch(joint_circuit, joint_params))

        return np.asarray(p_joint / p_norm)

    def __repr__(self) -> str:
        return _program_repr(
            self._program,
            len(self._channel_sampler.channels),
            "CompiledStateProbs",
        )


def _program_repr(program: CompiledProgram, num_channels: int, class_name: str) -> str:
    """Generate a repr string for a compiled sampler."""
    c_graphs = []
    c_params = []
    c_a_terms = []
    c_b_terms = []
    c_c_terms = []
    c_d_terms = []
    num_circuits = 0
    total_memory_bytes = 0
    num_outputs = []

    for component in program.components:
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

    total_memory_mb = total_memory_bytes / (1024**2)

    return (
        f"{class_name}({np.sum(c_graphs)} graphs, "
        f"{num_channels} error channels, "
        f"{np.max(num_outputs)} outputs for largest cc, "
        f"{np.max(c_params) if c_params else 0} parameters, {np.sum(c_a_terms)} A terms, "
        f"{np.sum(c_b_terms)} B terms, "
        f"{np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms, "
        f"{total_memory_mb:.2f} MB)"
    )
