import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tsim.core.exact_scalar import ExactScalarArray
from tsim.core.float_scalar import Float32ScalarArray, Float64ScalarArray
from tsim.core.scalar import scalar_type

X64 = bool(jax.config.read("jax_enable_x64"))

BACKENDS = [Float32ScalarArray] + ([Float64ScalarArray] if X64 else [])


def _tol(cls):
    return 1e-5 if cls is Float32ScalarArray else 1e-12


@pytest.fixture
def coeffs():
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.integers(-2, 3, size=(6, 5, 4)).astype(np.int32))


@pytest.mark.parametrize("cls", BACKENDS)
def test_from_exact_matches_exact(cls, coeffs):
    expected = ExactScalarArray(coeffs).to_complex()
    got = cls.from_exact(coeffs).to_complex()
    assert got.dtype == cls.complex_dtype()
    np.testing.assert_allclose(got, expected, rtol=_tol(cls), atol=_tol(cls))


@pytest.mark.parametrize("cls", BACKENDS)
def test_mul_and_power(cls, coeffs):
    power = jnp.asarray(np.arange(30).reshape(6, 5) - 15, dtype=jnp.int32)
    a = cls.from_exact(coeffs, power)
    b = cls.from_exact(coeffs[::-1])
    expected = (
        ExactScalarArray(coeffs, power).to_complex()
        * ExactScalarArray(coeffs[::-1]).to_complex()
    )
    np.testing.assert_allclose(
        (a * b).to_complex(), expected, rtol=_tol(cls), atol=_tol(cls)
    )
    np.testing.assert_allclose(
        a.scale_power2(jnp.int32(3)).to_complex(), 8 * a.to_complex(), rtol=_tol(cls)
    )


@pytest.mark.parametrize("cls", BACKENDS)
def test_take_and_where(cls, coeffs):
    table = cls.from_exact(coeffs[0])  # (5,) entries
    idx = jnp.asarray([[0, 4, 2], [1, 1, 3]])
    mask = jnp.asarray([[True, False, True], [False, True, True]])
    got = table.take(idx).where(mask, cls.one()).to_complex()
    exact = ExactScalarArray(coeffs[0]).to_complex()
    expected = np.where(np.asarray(mask), np.asarray(exact)[np.asarray(idx)], 1.0)
    np.testing.assert_allclose(got, expected, rtol=_tol(cls), atol=_tol(cls))
    assert got.shape == (2, 3)


@pytest.mark.parametrize("cls", BACKENDS)
def test_prod_and_sum_match_exact(cls, coeffs):
    a = cls.from_exact(coeffs)
    exact = ExactScalarArray(coeffs)
    np.testing.assert_allclose(
        a.prod(axis=1).to_complex(),
        exact.prod(axis=1).to_complex(),
        rtol=_tol(cls),
        atol=_tol(cls),
    )
    np.testing.assert_allclose(
        a.sum(axis=1).to_complex(),
        exact.sum(axis=1).to_complex(),
        rtol=_tol(cls),
        atol=_tol(cls),
    )
    # Sum with non-trivial powers must align summands.
    power = jnp.asarray(np.arange(30).reshape(6, 5) % 7 - 3, dtype=jnp.int32)
    np.testing.assert_allclose(
        cls.from_exact(coeffs, power).sum(axis=0).to_complex(),
        ExactScalarArray(coeffs, power).sum(axis=0).to_complex(),
        rtol=_tol(cls),
        atol=_tol(cls),
    )


@pytest.mark.parametrize("cls", BACKENDS)
def test_prod_empty_axis_is_identity(cls):
    a = cls.from_exact(jnp.zeros((3, 0, 4), dtype=jnp.int32))
    np.testing.assert_array_equal(a.prod(axis=1).to_complex(), np.ones(3))


@pytest.mark.parametrize("cls", BACKENDS)
def test_long_product_does_not_overflow(cls):
    """Products longer than the dtype's exponent range are renormalised per chunk."""
    n = 4000  # 2^4000 overflows both float32 and float64 without renormalisation
    two = jnp.tile(jnp.array([[2, 0, 0, 0]], dtype=jnp.int32), (n, 1))
    prod = cls.from_exact(two).prod(axis=0)
    assert prod.power is not None
    mantissa, power = np.asarray(prod.mantissa), int(prod.power)
    assert np.isfinite(mantissa)
    # Value is exactly 2^n: mantissa is a power of two and the exponents add up.
    m_exp = int(np.log2(abs(mantissa)))
    assert 2.0**m_exp == abs(mantissa)
    assert m_exp + power == n
    # And a shrinking product does not underflow to zero.
    half = jnp.tile(jnp.array([[1, 0, 0, 0]], dtype=jnp.int32), (n, 1))
    small = cls.from_exact(half, jnp.full((n,), -1, dtype=jnp.int32)).prod(axis=0)
    assert float(np.abs(small.mantissa)) > 0
    assert int(small.power) + int(np.log2(abs(float(np.abs(small.mantissa))))) == -n


def test_scalar_type_lookup():
    assert scalar_type("exact") is ExactScalarArray
    assert scalar_type("float32") is Float32ScalarArray
    with pytest.raises(ValueError):
        scalar_type("bfloat16")
    if X64:
        assert scalar_type("float64") is Float64ScalarArray
    else:
        with pytest.raises(RuntimeError, match="x64"):
            scalar_type("float64")


def test_exact_backend_new_ops():
    coeffs = jnp.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=jnp.int32)
    table = ExactScalarArray.from_exact(coeffs)
    picked = table.take(jnp.asarray([[2, 1], [0, 0]]))
    assert picked.shape == (2, 2)
    masked = picked.where(
        jnp.asarray([[True, False], [False, True]]), ExactScalarArray.one()
    )
    np.testing.assert_allclose(masked.to_complex(), [[1j, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        masked.scale_power2(jnp.int32(2)).to_complex(), 4 * masked.to_complex()
    )
