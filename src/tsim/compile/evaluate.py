"""Evaluation of compiled scalar graphs."""

import functools
import operator

import jax
import jax.numpy as jnp
from jax import Array

from tsim.compile.compile import CompiledScalarGraphs
from tsim.compile.terms import UNIT_PHASES
from tsim.core.scalar import scalar_type


@jax.jit
def evaluate(circuit: CompiledScalarGraphs, param_vals: Array) -> Array:
    """Evaluate compiled circuit with batched parameter values.

    Each term family (``NodePhases``, ``HalfPiPhases``, ``PiProducts``,
    ``PhasePairs``) computes its own contribution via ``.evaluate(param_vals,
    scalar)``. This function multiplies those together with the per-graph
    ``ScalarPrefactor`` and folds in ``power2`` / any approximate floatfactor.
    The scalar backend is selected by ``circuit.precision``.

    Args:
        circuit: Compiled circuit representation.
        param_vals: Binary parameter values (error bits + measurement/detector
            outcomes), shape ``(batch_size, n_params)``.

    Returns:
        Complex array of shape ``(batch_size,)`` — the per-sample amplitude.

    """
    prefactor = circuit.prefactor
    scalar = scalar_type(circuit.precision)
    if prefactor.phase_indices.shape[0] == 0:
        return jnp.zeros(param_vals.shape[0], dtype=scalar.complex_dtype())

    static_phases = scalar.from_exact(UNIT_PHASES).take(prefactor.phase_indices)
    float_factor = scalar.from_exact(prefactor.floatfactor)

    total = functools.reduce(
        operator.mul,
        [
            circuit.node_phases.evaluate(param_vals, scalar),
            circuit.halfpi_phases.evaluate(param_vals, scalar),
            circuit.pi_products.evaluate(param_vals, scalar),
            circuit.phase_pairs.evaluate(param_vals, scalar),
            static_phases,
            float_factor,
        ],
    )

    if not prefactor.has_approximate_floatfactors:
        return total.scale_power2(prefactor.power2).sum(axis=-1).to_complex()

    return jnp.sum(
        total.to_complex() * prefactor.approximate_floatfactors * 2.0**prefactor.power2,
        axis=-1,
    )
