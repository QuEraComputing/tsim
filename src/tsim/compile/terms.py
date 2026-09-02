"""Term-family modules for compiled scalar graphs.

Each compiled ZX scalar is the product of four term families plus a global
phase and a floatfactor. This module defines the four families as
``equinox.Module`` records, bundles the shared phase tables, and gives each
family an ``evaluate`` method that turns a batch of binary parameter values
into a :class:`~tsim.core.scalar.ScalarArray`.

Every family reduces to the same pattern: a GF(2) matrix product gives one
small integer index per (batch, graph, term), a lookup table maps the index
to an exact dyadic scalar, and the scalars are multiplied along the term
axis. The lookup tables are stored as exact coefficient arrays; the
:class:`~tsim.core.scalar.ScalarArray` backend passed to ``evaluate``
decides how they are represented and reduced (exact dyadic, complex64, or
complex128 arithmetic).

Downstream, ``compile.py`` builds instances of these classes from
``pyzx_param`` scalars, and ``evaluate.py`` orchestrates the products.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from tsim.core.exact_scalar import ExactScalarArray
from tsim.core.scalar import ScalarArray
from tsim.utils.linalg import matmul_gf2

# Powers of ω = e^(iπ/4). UNIT_PHASES[k] is the exact 4-coefficient
# representation of ω^k.
UNIT_PHASES = jnp.array(
    [
        [1, 0, 0, 0],  # omega^0 = 1
        [0, 1, 0, 0],  # omega^1
        [0, 0, 1, 0],  # omega^2 = i
        [0, 0, 0, -1],  # omega^3
        [-1, 0, 0, 0],  # omega^4 = -1
        [0, -1, 0, 0],  # omega^5
        [0, 0, -1, 0],  # omega^6 = -i
        [0, 0, 0, 1],  # omega^7
    ],
    dtype=jnp.int32,
)

# Lookup table for exact scalars (1 + ω^k).
_ONE_PLUS_PHASES = UNIT_PHASES.at[:, 0].add(1)

_IDENTITY = jnp.array([1, 0, 0, 0], dtype=jnp.int32)

# Lookup table for the signs (-1)^s, s ∈ {0, 1}.
_SIGNS = jnp.array([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=jnp.int32)


def _build_pair_table() -> Array:
    """Exact table of ``1 + ω^a + ω^b − ω^(a+b)`` indexed by ``8·a + b``."""
    unit = np.asarray(UNIT_PHASES)
    table = np.zeros((64, 4), dtype=np.int32)
    for a in range(8):
        for b in range(8):
            table[8 * a + b] = (
                np.asarray(_IDENTITY) + unit[a] + unit[b] - unit[(a + b) % 8]
            )
    return jnp.asarray(table)


# Lookup table for exact scalars (1 + ω^α + ω^β − ω^(α+β)), indexed by 8·α + β.
_PAIR_TABLE = _build_pair_table()


def _pad_mask(counts: Array, max_terms: int) -> Array:
    """Boolean ``(num_graphs, max_terms)`` mask of the real (non-padded) slots."""
    return jnp.arange(max_terms)[None, :] < counts[:, None]


class NodePhases(eqx.Module):
    """Product of ``1 + exp(i·(α + ⊕params)·π)`` terms, one factor per stored term.

    Padded slots use ``0`` for both ``phases`` and ``params``; the evaluator masks
    padded slots to the multiplicative identity using ``counts``.

    Shapes are ``(num_graphs, max_terms)`` except ``params`` which is
    ``(num_graphs, max_terms, n_params)``.
    """

    phases: Array  # uint8, values 0-7 (the constant offset α, as ``4·α``)
    params: Array  # uint8, parameter parity bitmasks
    counts: Array  # int32, number of real (non-padded) terms per graph

    def evaluate(
        self, param_vals: Array, scalar: type[ScalarArray] = ExactScalarArray
    ) -> ScalarArray:
        """Evaluate Π (1 + ω^(4·parity + phase)) per graph, batched over param_vals.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.
            scalar: Scalar backend used for the arithmetic.

        Returns:
            ``ScalarArray`` of shape ``(batch, num_graphs)``.

        """
        rowsum = matmul_gf2(self.params, param_vals)
        phase_idx = (4 * rowsum + self.phases) % 8

        mask = _pad_mask(self.counts, self.phases.shape[1])
        terms = scalar.from_exact(_ONE_PLUS_PHASES).take(phase_idx)
        return terms.where(mask, scalar.one()).prod(axis=-1)


class HalfPiPhases(eqx.Module):
    """Sum of ``exp(i·j·π·⊕params / 2)`` terms with ``j ∈ {1, 3}``.

    Terms sharing a parameter bitstring have been combined to a single stored
    coefficient ``j' ∈ {1, 2, 3}`` (see ``_compile_halfpi_phases``). Coefficients
    are stored in eighth-turn units — i.e. as ``2·j'`` — so the evaluator can
    reuse the ``ω = e^(iπ/4)`` phase table.

    Padded slots use ``0`` (the additive identity for phase sums), so padded
    entries contribute nothing to the summed exponent.

    Shapes are ``(num_graphs, max_terms)`` except ``params`` which is
    ``(num_graphs, max_terms, n_params)``.
    """

    coeffs: Array  # uint8, values in {0, 2, 4, 6}  (= 2·j', with 0 = padding)
    params: Array  # uint8, parameter parity bitmasks

    def evaluate(
        self, param_vals: Array, scalar: type[ScalarArray] = ExactScalarArray
    ) -> ScalarArray:
        """Evaluate ω^(Σ coeffs · parity) per graph, batched over param_vals.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.
            scalar: Scalar backend used for the arithmetic.

        Returns:
            ``ScalarArray`` of shape ``(batch, num_graphs)``.

        """
        rowsum = matmul_gf2(self.params, param_vals)
        phase_idx = (rowsum * self.coeffs) % 8
        total_phase = jnp.sum(phase_idx, axis=-1) % 8
        return scalar.from_exact(UNIT_PHASES).take(total_phase)


class PiProducts(eqx.Module):
    """Product of ``(-1)^(ψ · φ)`` terms, with ψ and φ each a parity expression.

    Each side is encoded as a constant bit plus a parameter bitmask. Padded slots
    use ``0`` everywhere; a padded term contributes ``(-1)^0 = 1`` to the product.

    Shapes are ``(num_graphs, max_terms)`` except ``*_params`` which are
    ``(num_graphs, max_terms, n_params)``.
    """

    psi_const: Array  # uint8, values {0, 1}
    psi_params: Array  # uint8, parameter parity bitmask for ψ
    phi_const: Array  # uint8, values {0, 1}
    phi_params: Array  # uint8, parameter parity bitmask for φ

    def evaluate(
        self, param_vals: Array, scalar: type[ScalarArray] = ExactScalarArray
    ) -> ScalarArray:
        """Evaluate Π (-1)^(ψ·φ) per graph as a real ±1 scalar.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.
            scalar: Scalar backend used for the arithmetic.

        Returns:
            ``ScalarArray`` of shape ``(batch, num_graphs)``, with values
            in {+1, -1}.

        """
        psi = (self.psi_const + matmul_gf2(self.psi_params, param_vals)) % 2
        phi = (self.phi_const + matmul_gf2(self.phi_params, param_vals)) % 2

        exponent = (psi * phi) % 2
        sum_exponents = jnp.sum(exponent, axis=-1) % 2
        # Table lookup rather than ``1 - 2·s``: the latter wraps in unsigned
        # arithmetic (s is uint8/uint32) and only recovers -1 by accident of
        # int32 truncation when JAX's x64 mode is off.
        return scalar.from_exact(_SIGNS).take(sum_exponents)


class PhasePairs(eqx.Module):
    """Product of ``1 + e^(iα) + e^(iβ) − e^(i(α+β))`` terms.

    Each of ``α`` and ``β`` combines a constant phase with a parameter parity.
    Padded slots use ``0`` and are masked to the multiplicative identity using
    ``counts``.

    Shapes are ``(num_graphs, max_terms)`` except ``*_params`` which are
    ``(num_graphs, max_terms, n_params)``.
    """

    alpha: Array  # uint8, values 0-7 (constant offset of α, as ``4·α``)
    alpha_params: Array  # uint8, parameter parity bitmask for α
    beta: Array  # uint8, values 0-7 (constant offset of β, as ``4·β``)
    beta_params: Array  # uint8, parameter parity bitmask for β
    counts: Array  # int32, number of real (non-padded) terms per graph

    def evaluate(
        self, param_vals: Array, scalar: type[ScalarArray] = ExactScalarArray
    ) -> ScalarArray:
        """Evaluate Π (1 + ω^α + ω^β - ω^(α+β)) per graph, batched.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.
            scalar: Scalar backend used for the arithmetic.

        Returns:
            ``ScalarArray`` of shape ``(batch, num_graphs)``.

        """
        rowsum_a = matmul_gf2(self.alpha_params, param_vals)
        rowsum_b = matmul_gf2(self.beta_params, param_vals)

        alpha = (self.alpha + rowsum_a * 4) % 8
        beta = (self.beta + rowsum_b * 4) % 8
        pair_idx = alpha * 8 + beta  # < 64, fits uint8

        mask = _pad_mask(self.counts, self.alpha.shape[1])
        terms = scalar.from_exact(_PAIR_TABLE).take(pair_idx)
        return terms.where(mask, scalar.one()).prod(axis=-1)


class ScalarPrefactor(eqx.Module):
    """Per-graph static scalar prefactor — independent of parameter values.

    Each graph's amplitude is the product of the four term families times the
    prefactor ``ω^phase_index · floatfactor · 2^power2``, with an additional
    complex ``approximate_floatfactor`` multiplied in when any graph's phase
    had a denominator outside ``{1, 2, 4}`` and was folded into float form.

    This class is pure data; the final fold-in with the term-family product
    (which requires branching on ``has_approximate_floatfactors``) lives in
    ``evaluate.py``.
    """

    phase_indices: Array  # uint8, shape (num_graphs,), values 0-7
    floatfactor: Array  # int32, shape (num_graphs, 4) — exact dyadic (a,b,c,d)
    power2: Array  # int32, shape (num_graphs,) — exponent on the 2^(·) scaling
    approximate_floatfactors: Array  # complex64, shape (num_graphs,)
    has_approximate_floatfactors: bool = eqx.field(static=True)
