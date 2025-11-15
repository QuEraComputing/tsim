from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

import tsim.external.pyzx as zx
from tsim.channels import ChannelSampler
from tsim.circuit import Circuit
from tsim.decomposer import Decomposer, DecomposerArray
from tsim.evaluate import evaluate_batch
from tsim.graph_util import connected_components, transform_error_basis


class CompiledSampler:
    """Quantum circuit sampler using ZX-calculus based stabilizer rank decomposition."""

    def __init__(self, circuit: Circuit, sample_detectors: bool = False):
        """Create a sampler from pre-built sampler resources."""
        graph = circuit.get_sampling_graph(sample_detectors=sample_detectors)

        zx.full_reduce(graph)

        graph, error_transform = transform_error_basis(graph)

        self.channel_sampler = ChannelSampler(
            error_channels=circuit.errors, error_transform=error_transform
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
        self.program = DecomposerArray(components=decomposers)

        self.program.decompose()

        self._key = jax.random.key(0)

    def __repr__(self):
        c_graphs = []
        c_params = []
        c_ab_terms = []
        c_c_terms = []
        c_d_terms = []
        num_circuits = 0
        for component in self.program.components:
            if component.compiled_circuits is None:
                continue
            for circuit in component.compiled_circuits:
                c_graphs.append(circuit.num_graphs)
                c_params.append(circuit.n_params)
                c_ab_terms.append(len(circuit.ab_graph_ids))
                c_c_terms.append(len(circuit.c_graph_ids))
                c_d_terms.append(len(circuit.d_graph_ids))
                num_circuits += 1
        return (
            f"CompiledSampler({num_circuits} outputs, {np.sum(c_graphs)} graphs, "
            f"{np.sum(c_params)} parameters, {np.sum(c_ab_terms)} AB terms, "
            f"{np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms)"
        )

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample a batch of measurement/detector outcomes."""
        f_samples = self.channel_sampler.sample(batch_size)

        zeros = jnp.zeros((batch_size, 1), dtype=jnp.uint8)
        ones = jnp.ones((batch_size, 1), dtype=jnp.uint8)

        component_samples = []

        for component in self.program.components:
            if component.f_selection is None or component.compiled_circuits is None:
                raise RuntimeError("Sampling plan not decomposed before sampling.")

            s = f_samples[:, component.f_selection]
            num_errors = s.shape[1]

            key, self._key = jax.random.split(self._key)

            for circuit in component.compiled_circuits:
                state_0 = jnp.hstack([s, zeros])
                p_batch_0 = jnp.abs(evaluate_batch(circuit, state_0))

                state_1 = jnp.hstack([s, ones])
                p_batch_1 = jnp.abs(evaluate_batch(circuit, state_1))

                # normalize the probabilities
                p1 = p_batch_1 / (p_batch_0 + p_batch_1)

                _, key = jax.random.split(key)
                m = jax.random.bernoulli(key, p=p1).astype(jnp.uint8)
                s = jnp.hstack([s, m[:, None]])

            correlated_samples = s[:, num_errors:]
            component_samples.append(correlated_samples)

        return np.concatenate(component_samples, axis=1)[:, self.program.output_order]

    def sample(self, num_samples: int, batch_size: int = 100) -> np.ndarray:
        """Sample measurement/detector outcomes with specified batch size. On a GPU
        performance can be significantly improved by increasing the batch size.

        Args:
            num_samples: Total number of samples to generate
            batch_size: Size of each sampling batch

        Returns:
            Array of measurement outcomes, shape (num_samples, num_qubits)
        """
        if num_samples < batch_size:
            batch_size = num_samples
        batches = []
        for _ in range(ceil(num_samples / batch_size)):
            batches.append(self.sample_batch(batch_size))
        return np.concatenate(batches)[:num_samples]


class CompiledMeasurementSampler(CompiledSampler):
    """Measurement sampler"""

    def __init__(self, circuit: Circuit):
        super().__init__(circuit, sample_detectors=False)


class CompiledDetectorSampler(CompiledSampler):
    """Detector and observable sampler"""

    def __init__(self, circuit: Circuit):
        super().__init__(circuit, sample_detectors=True)
