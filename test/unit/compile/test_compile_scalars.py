"""compile_scalars: compiled evaluation equals zx_param's Scalar.evaluate."""

import itertools
import random
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
from zx_param import Dyadic, Scalar
from zx_param.phase import Bit
from zx_param.terms import NodeTerm, PhasePair, PhaseTerm, PiPair

from tsim.compile.compile import compile_scalars
from tsim.compile.evaluate import evaluate

BITS: tuple[Bit, Bit] = (0, 1)
PARAM_IDS = [3, 7, 11]
EXACT = [Fraction(k, 4) for k in range(-8, 8)]
OTHER = [Fraction(1, 10), Fraction(3, 7), Fraction(-5, 6)]
SETS = [frozenset(), frozenset({3}), frozenset({7, 11}), frozenset({3, 7, 11})]


def _compiled(scalars: list[Scalar], param_ids: list[int]) -> np.ndarray:
    n = len(param_ids)
    rows = np.array(list(itertools.product(BITS, repeat=n)), dtype=np.uint8)
    rows = rows.reshape(2**n, n)
    return np.asarray(evaluate(compile_scalars(scalars, param_ids), jnp.asarray(rows)))


def _expected(scalars: list[Scalar], param_ids: list[int]) -> np.ndarray:
    return np.array(
        [
            sum(s.evaluate(dict(zip(param_ids, bits, strict=True))) for s in scalars)
            for bits in itertools.product(BITS, repeat=len(param_ids))
        ]
    )


def _assert_matches(scalars: list[Scalar], param_ids: list[int] = PARAM_IDS) -> None:
    got = _compiled(scalars, param_ids)
    want = _expected(scalars, param_ids)
    scale = max(1.0, float(np.abs(want).max()))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * scale)


def _random_scalar(rng: random.Random, angles: list[Fraction]) -> Scalar:
    return Scalar(
        base=Dyadic(
            rng.randint(-3, 3) or 1,
            rng.randint(-2, 2),
            rng.randint(-2, 2),
            rng.randint(-2, 2),
            rng.randint(-3, 3),
        ),
        nodes=[
            NodeTerm(rng.choice(angles), rng.choice(SETS))
            for _ in range(rng.randint(0, 3))
        ],
        phases=[
            PhaseTerm(rng.choice(angles), rng.choice(SETS))
            for _ in range(rng.randint(0, 3))
        ],
        pi_pairs=[
            PiPair(rng.choice(SETS[1:]), rng.choice(SETS[1:]))
            for _ in range(rng.randint(0, 2))
        ],
        phase_pairs=[
            PhasePair(
                rng.choice(angles),
                rng.choice(angles),
                rng.choice(SETS),
                rng.choice(SETS),
            )
            for _ in range(rng.randint(0, 2))
        ],
        phase=rng.choice(angles),
    )


def test_empty_list_compiles_to_zero():
    compiled = compile_scalars([], [3])
    assert compiled.num_graphs == 0
    assert compiled.prefactor.floatfactor.shape == (0, 4)
    result = evaluate(compiled, jnp.ones((5, 1), dtype=jnp.uint8))
    np.testing.assert_array_equal(np.asarray(result), np.zeros(5, dtype=complex))


def test_zero_scalars_are_dropped():
    zero = Scalar(base=Dyadic.zero())
    assert compile_scalars([zero, Scalar.one()], []).num_graphs == 1


@pytest.mark.parametrize("seed", range(20))
def test_scalars_with_pi_over_four_angles_match(seed):
    rng = random.Random(seed)
    scalars = [_random_scalar(rng, EXACT) for _ in range(rng.randint(1, 4))]
    _assert_matches(scalars)
    assert not compile_scalars(scalars, PARAM_IDS).has_float_terms


@pytest.mark.parametrize("seed", range(20))
def test_scalars_with_other_angles_match(seed):
    rng = random.Random(1000 + seed)
    scalars = [_random_scalar(rng, EXACT + OTHER) for _ in range(rng.randint(1, 4))]
    _assert_matches(scalars)


def test_phase_terms_with_one_mask_are_combined():
    s = Scalar(
        base=Dyadic.one(),
        phases=[
            PhaseTerm(Fraction(1, 2), frozenset({3})),
            PhaseTerm(Fraction(1, 2), frozenset({3})),
        ],
    )
    compiled = compile_scalars([s], [3])
    assert compiled.halfpi_phases.coeffs.shape == (1, 1)
    assert int(compiled.halfpi_phases.coeffs[0, 0]) == 4
    _assert_matches([s], [3])


@pytest.mark.parametrize(
    "phase,index", [(Fraction(-1, 4), 7), (Fraction(9, 4), 1), (Fraction(5, 4), 5)]
)
def test_prefactor_phase_is_reduced_mod_8(phase, index):
    compiled = compile_scalars([Scalar(base=Dyadic.one(), phase=phase)], [])
    assert int(compiled.prefactor.phase_indices[0]) == index


def test_a_parameter_without_a_column_raises():
    s = Scalar(base=Dyadic.one(), nodes=[NodeTerm(Fraction(0), frozenset({99}))])
    with pytest.raises(KeyError):
        compile_scalars([s], [3])


def test_coefficients_beyond_int32_raise():
    with pytest.raises(OverflowError):
        compile_scalars([Scalar(base=Dyadic(2**31 + 1, 0, 0, 0, 0))], [])


def test_power_of_two_is_merged_before_float_conversion():
    p = frozenset({3})
    s = Scalar(
        base=Dyadic(1, 0, 0, 0, -140),
        nodes=[NodeTerm(Fraction(0), p) for _ in range(140)],
        phases=[PhaseTerm(Fraction(1, 10), p)],
    )
    got = _compiled([s], [3])
    assert np.isfinite(got).all()
    _assert_matches([s], [3])
