from math import ceil
from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np

import tsim.external.pyzx as zx
from tsim.channels import ChannelSampler
from tsim.circuit import Circuit
from tsim.decomposer import Decomposer, DecomposerArray, DecompositionMode
from tsim.evaluate import evaluate_batch
from tsim.graph_util import connected_components, squash_graph, transform_error_basis


class _CompiledSamplerBase:
    """Base class with common initialization logic for all compiled samplers."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        sample_detectors: bool,
        mode: DecompositionMode,
        seed: int | None,
    ):
        """Initialize the sampler with common graph preparation.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, sample detectors/observables; if False, sample measurements.
            mode: Decomposition mode:
                - "sequential": For sampling - [0, 1, 2, ..., n] per component
                - "joint": For probability estimation - [0, n] per component
            seed: Random seed for JAX. If None, a random seed is generated.
        """
        self.circuit = circuit
        graph = circuit.get_sampling_graph(sample_detectors=sample_detectors)

        zx.full_reduce(graph, paramSafe=True)
        squash_graph(graph)

        graph, error_transform = transform_error_basis(graph)

        # Remove the scalar. Since we have not started the stabilizer rank decomposition,
        # it is safe to remove the overall scalar.
        graph.scalar = zx.Scalar()

        self.channel_sampler = ChannelSampler(
            error_channels=circuit.error_channels, error_transform=error_transform
        )

        m_chars = [f"m{i}" for i in range(len(graph.outputs()))]

        decomposers: list[Decomposer] = []
        for component in connected_components(graph):
            error_chars = set()
            for v in component.graph.vertices():
                error_chars |= component.graph._phaseVars[v]

            decomposers.append(
                Decomposer(
                    graph=component.graph,
                    output_indices=component.output_indices,
                    f_chars=sorted(error_chars),
                    m_chars=m_chars,
                )
            )
        sorted_decomposers = sorted(decomposers, key=lambda x: len(x.output_indices))
        self.program = DecomposerArray(components=sorted_decomposers)

        self.program.decompose(mode=mode)

        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
        self._key = jax.random.key(seed)

    def _sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample a batch of measurement outcomes."""
        f_samples = self.channel_sampler.sample(batch_size)
        ones = jnp.ones((batch_size, 1), dtype=jnp.bool)

        component_samples = []
        output_order = self.program.output_order

        for component in self.program.components:
            assert component.f_selection is not None
            assert component.compiled_circuits is not None

            params = f_samples[:, component.f_selection]
            num_errors = params.shape[1]

            key, self._key = jax.random.split(self._key)

            # circuits[0] is normalization, circuits[1:] are for sequential bits
            norm_circuit = component.compiled_circuits[0]
            bit_circuits = component.compiled_circuits[1:]

            # Compute normalization once
            prev = np.abs(evaluate_batch(norm_circuit, params).to_numpy())

            for circuit in bit_circuits:
                # Only evaluate with bit=1 (chain rule: p0 = prev - p1)
                state_1 = jnp.hstack([params, ones])
                p1 = np.abs(evaluate_batch(circuit, state_1).to_numpy())

                # P(bit=1) = p1 / prev
                p1_norm = p1 / prev

                _, key = jax.random.split(key)
                measurement = jax.random.bernoulli(key, p=p1_norm)

                # Chain rule: new_prev = p1 if bit=1, else prev - p1
                prev = jnp.where(measurement, p1, prev - p1)

                params = jnp.hstack([params, measurement[:, None]])

            component_samples.append(params[:, num_errors:])

        return np.concatenate(component_samples, axis=1)[:, np.argsort(output_order)]

    def __repr__(self):
        c_graphs = []
        c_params = []
        c_a_terms = []
        c_b_terms = []
        c_c_terms = []
        c_d_terms = []
        num_circuits = 0
        for component in self.program.components:
            if component.compiled_circuits is None:
                continue
            for circuit in component.compiled_circuits:
                c_graphs.append(circuit.num_graphs)
                c_params.append(circuit.n_params)
                c_a_terms.append(len(circuit.a_graph_ids))
                c_b_terms.append(len(circuit.b_graph_ids))
                c_c_terms.append(len(circuit.c_graph_ids))
                c_d_terms.append(len(circuit.d_graph_ids))
                num_circuits += 1
        return (
            f"CompiledSampler({num_circuits} outputs, {np.sum(c_graphs)} graphs, "
            f"{np.sum(c_params)} parameters, {np.sum(c_a_terms)} A terms, "
            f"{np.sum(c_b_terms)} B terms, "
            f"{np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms)"
        )


class CompiledStateProbs(_CompiledSamplerBase):
    """Computes measurement probabilities for a given state.

    Uses joint decomposition [0, n] where:
    - circuits[0]: normalization (0 outputs plugged)
    - circuits[1]: full joint probability (all outputs plugged)
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
        super().__init__(
            circuit,
            sample_detectors=sample_detectors,
            mode="joint",
            seed=seed,
        )

    def probability_of(self, state: np.ndarray, *, batch_size: int) -> np.ndarray:
        """Compute probabilities for a batch of error samples given a measurement state.

        Args:
            state: The measurement outcome state to compute probability for.
            batch_size: Number of error samples to use for estimation.

        Returns:
            Array of probabilities P(state | error_sample) for each error sample.
        """
        f_samples = self.channel_sampler.sample(batch_size)
        p_norm = np.ones(batch_size, dtype=np.float64)
        p_joint = np.ones(batch_size, dtype=np.float64)

        for component in self.program.components:
            assert component.f_selection is not None
            assert component.compiled_circuits is not None
            assert len(component.compiled_circuits) == 2

            params = f_samples[:, component.f_selection]
            norm_circuit, joint_circuit = component.compiled_circuits

            # Normalization: circuit[0] with only error params
            p_norm *= np.abs(evaluate_batch(norm_circuit, params).to_numpy())

            # Joint probability: circuit[1] with error params + state
            component_state = state[component.output_indices]
            tiled_state = jnp.tile(component_state, (batch_size, 1))
            full_params = jnp.hstack([params, tiled_state])
            p_joint *= np.abs(evaluate_batch(joint_circuit, full_params).to_numpy())

        return p_joint / p_norm


class CompiledMeasurementSampler(_CompiledSamplerBase):
    """Samples measurement outcomes from a quantum circuit.

    Uses sequential decomposition [0, 1, 2, ..., n] where:
    - circuits[0]: normalization (0 outputs plugged)
    - circuits[i]: cumulative probability up to bit i
    """

    def __init__(self, circuit: Circuit, *, seed: int | None = None):
        """Create a measurement sampler.

        Args:
            circuit: The quantum circuit to compile.
            seed: Random seed for JAX. If None, a random seed is generated.
        """
        super().__init__(
            circuit,
            sample_detectors=False,
            mode="sequential",
            seed=seed,
        )

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
        batches = []
        for _ in range(ceil(shots / batch_size)):
            batches.append(self._sample_batch(batch_size))
        return np.concatenate(batches)[:shots]


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


class CompiledDetectorSampler(_CompiledSamplerBase):
    """Samples detector and observable outcomes from a quantum circuit."""

    def __init__(self, circuit: Circuit, *, seed: int | None = None):
        """Create a detector sampler.

        Args:
            circuit: The quantum circuit to compile.
            seed: Random seed for JAX. If None, a random seed is generated.
        """
        super().__init__(
            circuit,
            sample_detectors=True,
            mode="sequential",
            seed=seed,
        )

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
        batches = []
        for _ in range(ceil(shots / batch_size)):
            batches.append(self._sample_batch(batch_size))
        samples = np.concatenate(batches)[:shots]

        if append_observables:
            return maybe_bit_pack(samples, do_nothing=not bit_packed)

        num_detectors = len(self.circuit._detectors)
        det_samples = samples[:, :num_detectors]
        obs_samples = samples[:, num_detectors:]

        if prepend_observables:
            return np.concatenate([obs_samples, det_samples], axis=1)
        if separate_observables:
            return (
                maybe_bit_pack(det_samples, do_nothing=not bit_packed),
                maybe_bit_pack(obs_samples, do_nothing=not bit_packed),
            )

        return maybe_bit_pack(det_samples, do_nothing=not bit_packed)
        # TODO: don't compute observables if they are discarded here
