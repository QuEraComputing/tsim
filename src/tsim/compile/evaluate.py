"""Evaluation of compiled scalar graphs using exact arithmetic."""

import functools
import operator

import jax
import jax.numpy as jnp
from jax import Array

from tsim.compile.compile import CompiledScalarGraphs
from tsim.compile.terms import UNIT_PHASES
from tsim.core.exact_scalar import ExactScalarArray


@jax.jit
def evaluate(circuit: CompiledScalarGraphs, param_vals: Array) -> Array:
    """Evaluate compiled circuit with batched parameter values.

    Each term family (``NodePhases``, ``HalfPiPhases``, ``PiProducts``,
    ``PhasePairs``) computes its own contribution via ``.evaluate(param_vals)``.
    This function multiplies those together with the per-graph
    ``ScalarPrefactor`` and folds in ``power2`` / any approximate floatfactor.

    Args:
        circuit: Compiled circuit representation.
        param_vals: Binary parameter values (error bits + measurement/detector
            outcomes), shape ``(batch_size, n_params)``.

    Returns:
        Complex array of shape ``(batch_size,)`` — the per-sample amplitude.

    """
    prefactor = circuit.prefactor
    static_phases = ExactScalarArray(UNIT_PHASES[prefactor.phase_indices])
    float_factor = ExactScalarArray(prefactor.floatfactor)

    total = functools.reduce(
        operator.mul,
        [
            circuit.node_phases.evaluate(param_vals),
            circuit.halfpi_phases.evaluate(param_vals),
            circuit.pi_products.evaluate(param_vals),
            circuit.phase_pairs.evaluate(param_vals),
            static_phases,
            float_factor,
        ],
    )

    if not prefactor.has_approximate_floatfactors:
        total = ExactScalarArray(total.coeffs, total.power + prefactor.power2)
        return total.sum().to_complex()

    return jnp.sum(
        total.to_complex() * prefactor.approximate_floatfactors * 2.0**prefactor.power2,
        axis=-1,
    )
