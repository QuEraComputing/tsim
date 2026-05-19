from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
from pyzx_param.graph.base import BaseGraph
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


class TestCompilePhasevarsPiRejected:
    """compile_scalar_graphs must reject graphs carrying ``Scalar.phasevars_pi``.

    ``phasevars_pi`` represents a ``(-1)^XOR(vars)`` factor on the scalar that
    no compiled term family applies. Silently dropping it would produce wrong
    amplitudes, so we crash loudly. Today ``pyzx_param.full_reduce`` does not
    emit this set, but the contract must be enforced for future safety.
    """

    def test_single_phasevars_pi_raises(self):
        g = GraphS()
        g.scalar.add_phase_vars_pi({"f0"})
        with pytest.raises(NotImplementedError, match="phasevars_pi"):
            compile_scalar_graphs([g], ["f0"])

    def test_phasevars_pi_on_later_graph_raises(self):
        """Per-graph check fires even when only a non-first graph carries it."""
        g_clean = GraphS()
        g_with = GraphS()
        g_with.scalar.add_phase_vars_pi({"f0"})
        with pytest.raises(NotImplementedError, match="graph 1"):
            compile_scalar_graphs([g_clean, g_with], ["f0"])

    def test_empty_phasevars_pi_is_accepted(self):
        """The check must only fire on a non-empty set, not on a present-but-empty one."""
        g = GraphS()
        assert g.scalar.phasevars_pi == set()
        compile_scalar_graphs([g], [])

    def test_phasevars_pi_with_multiple_vars_in_message(self):
        g = GraphS()
        g.scalar.add_phase_vars_pi({"f0", "f1"})
        with pytest.raises(NotImplementedError) as exc_info:
            compile_scalar_graphs([g], ["f0", "f1"])
        msg = str(exc_info.value)
        assert "f0" in msg and "f1" in msg


class TestCompilePrefactorPhase:
    """Phase indices must be reduced modulo 8 so the uint8 storage and
    UNIT_PHASES lookup in compile/terms.py see canonical values 0..7."""

    def _scalar_graph(self, phase: Fraction) -> GraphS:
        g = GraphS()
        g.scalar.phase = phase
        return g

    def test_negative_phase_reduced_into_range(self):
        # phase = -1/4 -> int(-1.0) -> -1 -> -1 % 8 == 7
        compiled = compile_scalar_graphs([self._scalar_graph(Fraction(-1, 4))], [])
        assert int(compiled.prefactor.phase_indices[0]) == 7

    def test_phase_above_two_reduced_into_range(self):
        # phase = 9/4 -> int(9.0) -> 9 -> 9 % 8 == 1
        compiled = compile_scalar_graphs([self._scalar_graph(Fraction(9, 4))], [])
        assert int(compiled.prefactor.phase_indices[0]) == 1

    def test_in_range_phase_unchanged(self):
        # phase = 5/4 -> int(5.0) -> 5 -> 5 % 8 == 5
        compiled = compile_scalar_graphs([self._scalar_graph(Fraction(5, 4))], [])
        assert int(compiled.prefactor.phase_indices[0]) == 5

    def test_phase_indices_stay_within_unit_phase_table(self):
        graphs: list[BaseGraph] = [
            self._scalar_graph(Fraction(p, 4)) for p in (-7, -1, 0, 3, 7, 11)
        ]
        compiled = compile_scalar_graphs(graphs, [])
        indices = np.asarray(compiled.prefactor.phase_indices)
        assert np.all((indices >= 0) & (indices < 8))
