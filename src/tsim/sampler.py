from dataclasses import dataclass
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

import tsim.external.pyzx as zx
from tsim.circuit import SamplingGraphs
from tsim.compile import CompiledCircuit, compile_circuit
from tsim.evaluate import evaluate_batch
from tsim.stabrank import find_stab


@dataclass
class DecomposedComponent:
    circuits: list[CompiledCircuit]
    f_selection: jax.Array
    num_errors: int
    output_order: list[int]


class Sampler:
    """Efficient quantum circuit sampler using ZX-calculus based stabilizer rank decomposition."""

    def __init__(self, sampling_graphs: SamplingGraphs):
        """Compile graphs for fast sampling."""
        self.compiled_circuits_components = []
        self.error_sampler = sampling_graphs.error_sampler
        self._key = jax.random.key(0)

        f_char_to_idx = {char: i for i, char in enumerate(sampling_graphs.chars)}

        ord = []
        self.decomposed_components: list[DecomposedComponent] = []

        for component in sampling_graphs.sampling_components:
            decomposed_circuits = []
            num_errors = len(component.f_chars)
            chars = component.f_chars + component.m_chars

            for i, g in enumerate(component.graphs):
                zx.full_reduce(g, paramSafe=True)
                g.normalize()
                g_list = find_stab(g)
                circuit = compile_circuit(g_list, num_errors + i + 1, chars)
                decomposed_circuits.append(circuit)

            dec_comp = DecomposedComponent(
                circuits=decomposed_circuits,
                f_selection=jnp.array(
                    [f_char_to_idx[char] for char in component.f_chars], dtype=jnp.int32
                ),
                num_errors=len(component.f_chars),
                output_order=component.output_indices,
            )
            self.decomposed_components.append(dec_comp)

            ord.extend(component.output_indices)

        self.output_order = jnp.array(ord)

    def __repr__(self):
        c_graphs = []
        c_params = []
        c_ab_terms = []
        c_c_terms = []
        c_d_terms = []
        num_circuits = 0
        for component in self.decomposed_components:
            for circuit in component.circuits:
                c_graphs.append(circuit.num_graphs)
                c_params.append(circuit.n_params)
                c_ab_terms.append(len(circuit.ab_graph_ids))
                c_c_terms.append(len(circuit.c_graph_ids))
                c_d_terms.append(len(circuit.d_graph_ids))
                num_circuits += 1
        return (
            f"CompiledSampler({num_circuits} qubits, {np.sum(c_graphs)} graphs, "
            f"{np.sum(c_params)} parameters, {np.sum(c_ab_terms)} AB terms, "
            f"{np.sum(c_c_terms)} C terms, {np.sum(c_d_terms)} D terms)"
        )

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample a batch of measurement outcomes."""
        f_samples = self.error_sampler.sample(batch_size)

        zeros = jnp.zeros((batch_size, 1), dtype=jnp.uint8)
        ones = jnp.ones((batch_size, 1), dtype=jnp.uint8)

        component_samples = []

        for component in self.decomposed_components:
            s = f_samples[:, component.f_selection]
            num_errors = s.shape[1]

            key, self._key = jax.random.split(self._key)

            for circuit in component.circuits:
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

        return np.concatenate(component_samples, axis=1)[:, self.output_order]

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
