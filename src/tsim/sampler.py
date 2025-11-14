from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

import tsim.external.pyzx as zx
from tsim.circuit import SamplingGraphs
from tsim.compile import compile_circuit
from tsim.evaluate import evaluate_batch
from tsim.stabrank import find_stab


class Sampler:
    """Efficient quantum circuit sampler using ZX-calculus based stabilizer rank decomposition."""

    def __init__(self, sampling_graphs: SamplingGraphs):
        """Compile graphs for fast sampling."""
        self.compiled_circuits = []
        for i, g in enumerate(sampling_graphs.graphs):
            zx.full_reduce(g, paramSafe=True)
            g.normalize()
            g_list = find_stab(g)
            circuit = compile_circuit(
                g_list, sampling_graphs.num_errors + i + 1, sampling_graphs.chars
            )
            self.compiled_circuits.append(circuit)

        self.error_sampler = sampling_graphs.error_sampler
        self.isolated_outputs = sampling_graphs.isolated_outputs
        self.correlated_outputs = sampling_graphs.correlated_outputs
        self._key = jax.random.key(0)

    def __repr__(self):
        c_graphs = [c.num_graphs for c in self.compiled_circuits]
        c_params = [c.n_params for c in self.compiled_circuits]
        c_ab_terms = [len(c.ab_graph_ids) for c in self.compiled_circuits]
        c_c_terms = [len(c.c_graph_ids) for c in self.compiled_circuits]
        c_d_terms = [len(c.d_graph_ids) for c in self.compiled_circuits]
        num_circuits = len(self.compiled_circuits)
        return f"CompiledSampler({num_circuits} qubits, {np.sum(c_graphs)} graphs, {np.sum(c_params)} parameters, {np.sum(c_ab_terms)} AB terms, {np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms)"

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample a batch of measurement outcomes."""
        dem_samples, cor_samples = self.error_sampler.sample(batch_size)
        num_errors = cor_samples.shape[1]
        zeros = jnp.zeros((batch_size, 1), dtype=jnp.uint8)
        ones = jnp.ones((batch_size, 1), dtype=jnp.uint8)

        s = cor_samples

        key, self._key = jax.random.split(self._key)

        for circuit in self.compiled_circuits:
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
        s = jnp.hstack([dem_samples, correlated_samples])
        order = np.concatenate([self.isolated_outputs, self.correlated_outputs])

        return np.array(s)[:, order]

    def sample(self, num_samples: int, batch_size: int = 100) -> np.ndarray:
        """Sample measurement outcomes with specified batch size.

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
