import jax.numpy as jnp
import numpy as np
from pyzx_param.graph.graph_s import GraphS

from tsim.compile.compile import compile_scalar_graphs
from tsim.compile.evaluate import evaluate


class TestCompileEmpty:
    def test_empty_g_list(self):
        """compile_scalar_graphs accepts an empty list and produces well-shaped arrays."""
        compiled = compile_scalar_graphs([], [])
        assert compiled.num_graphs == 0
        assert compiled.n_params == 0
        assert compiled.prefactor.floatfactor.shape == (0, 4)
        assert compiled.prefactor.phase_indices.shape == (0,)
        assert compiled.prefactor.power2.shape == (0,)

    def test_all_zero_scalars_filtered_to_empty(self):
        """A list of only zero-scalar graphs is filtered down to the empty case."""
        g = GraphS()
        g.scalar.is_zero = True
        compiled = compile_scalar_graphs([g, g, g], [])
        assert compiled.num_graphs == 0
        assert compiled.prefactor.floatfactor.shape == (0, 4)

    def test_evaluate_empty_returns_zero(self):
        """evaluate() on an empty compiled circuit returns the additive identity."""
        compiled = compile_scalar_graphs([], [])
        param_vals = jnp.zeros((5, 0), dtype=jnp.uint8)
        result = evaluate(compiled, param_vals)
        assert result.shape == (5,)
        np.testing.assert_array_equal(np.asarray(result), np.zeros(5, dtype=complex))

    def test_evaluate_empty_with_params(self):
        """evaluate() handles empty-circuit + non-empty param dimension."""
        compiled = compile_scalar_graphs([], ["a", "b"])
        assert compiled.n_params == 2
        param_vals = jnp.ones((3, 2), dtype=jnp.uint8)
        result = evaluate(compiled, param_vals)
        assert result.shape == (3,)
        np.testing.assert_array_equal(np.asarray(result), np.zeros(3, dtype=complex))
