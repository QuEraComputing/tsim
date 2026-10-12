"""Evaluation of compiled scalar graphs using exact arithmetic."""

import functools
import operator

import jax
import jax.numpy as jnp
from jax import Array

from tsim.compile.compile import CompiledScalarGraphs
from tsim.core.exact_scalar import ExactScalarArray


@jax.jit
def evaluate(circuit: CompiledScalarGraphs, param_vals: Array) -> Array:
    """Evaluate compiled circuit with batched parameter values.

    Each exact term family (``NodePhases``, ``PhaseTerms``, ``PiProducts``,
    ``PhasePairs``) computes its own contribution via ``.evaluate(param_vals)``.
    This function multiplies those together with the per-graph
    ``ScalarPrefactor`` and merges ``power2`` into the exponent of the exact
    product before converting to floating point, so the powers of two cannot
    overflow or underflow separately. Any approximate floatfactor is multiplied
    in afterwards. The floating-point ``FloatTerms`` family (angles that are not
    multiples of π/4) is multiplied in only when the circuit has float terms.

    Args:
        circuit: Compiled circuit representation.
        param_vals: Bits of the compiled parameters (error-basis bits, then
            output bits), shape ``(batch_size, n_params)``.

    Returns:
        Complex array of shape ``(batch_size,)``: the summed scalars per sample.

    """
    prefactor = circuit.prefactor
    if prefactor.power2.shape[0] == 0:
        return jnp.zeros(param_vals.shape[0], dtype=jnp.complex64)

    exact_prefactor = ExactScalarArray(prefactor.floatfactor, prefactor.power2)

    total = functools.reduce(
        operator.mul,
        [
            circuit.node_phases.evaluate(param_vals),
            circuit.phase_terms.evaluate(param_vals),
            circuit.pi_products.evaluate(param_vals),
            circuit.phase_pairs.evaluate(param_vals),
            exact_prefactor,
        ],
    )
    if not prefactor.has_approximate_floatfactors and not circuit.has_float_terms:
        return total.sum().to_complex()

    return jnp.sum(
        total.to_complex()
        * circuit.float_terms.evaluate(param_vals)
        * prefactor.approximate_floatfactors,
        axis=-1,
    )
