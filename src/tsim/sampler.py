from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

import tsim.external.pyzx as zx
from tsim.circuit import SamplingGraphs
from tsim.compile import compile_circuit
from tsim.evaluate import evaluate_batch
from tsim.external.pyzx.graph.base import BaseGraph
from tsim.simplify import full_red


def find_stab(gg: BaseGraph, printOut: bool = False) -> list[BaseGraph]:
    """Recursively decompose ZX-graph into stabilizer components."""
    if zx.simplify.tcount(gg) == 0:
        return [gg]
    gsum = zx.simulate.replace_magic_states(gg, False)

    full_red(gsum)
    output = []

    for hh in gsum.graphs:
        output.extend(find_stab(hh, printOut))

    if printOut:
        print(len(gsum.graphs), len(output))
    return output


class Sampler:
    """Efficient quantum circuit sampler using ZX-calculus."""

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
        self._key = jax.random.key(0)

    def __repr__(self):
        c_graphs = [c.num_graphs for c in self.compiled_circuits]
        c_params = [c.n_params for c in self.compiled_circuits]
        num_circuits = len(self.compiled_circuits)
        return (
            f"CompiledSampler({num_circuits} qubits, {np.sum(c_graphs)} graphs, "
            f"{np.sum(c_params)} scalars)"
        )

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample a batch of measurement outcomes."""
        if self.error_sampler is not None:
            s = self.error_sampler.sample(batch_size)
            num_errors = s.shape[1]
        else:
            s = jnp.zeros((batch_size, 0), dtype=jnp.uint8)
            num_errors = 0
        zeros = jnp.zeros((batch_size, 1), dtype=jnp.uint8)
        p_prev = jnp.ones((batch_size,), dtype=jnp.float32)

        # Split key at the start to ensure different randomness for each call
        key, self._key = jax.random.split(self._key)

        for circuit in self.compiled_circuits:
            state = jnp.hstack([s, zeros])

            p_batch = evaluate_batch(circuit, state)

            p0 = jnp.abs(p_batch / p_prev)
            _, key = jax.random.split(key)
            m = jax.random.bernoulli(key, p=1 - p0).astype(jnp.uint8)
            s = jnp.hstack([s, m[:, None]])
            p_prev = jnp.where(m == 0, p_prev * p0, p_prev * (1 - p0))
        return np.array(s[:, num_errors:])

    def sample(self, num_samples: int, batch_size: int = 100) -> np.ndarray:
        """Sample measurement outcomes with specified batch size.

        Args:
            num_samples: Total number of samples to generate
            batch_size: Size of each sampling batch

        Returns:
            Array of measurement outcomes, shape (num_samples, num_qubits)
        """
        batches = []
        for _ in range(ceil(num_samples / batch_size)):
            batches.append(self.sample_batch(batch_size))
        return np.concatenate(batches)[:num_samples]
