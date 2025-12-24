import jax
import jax.numpy as jnp
import pytest

from tsim.exact_scalar import ExactScalarArray


@pytest.fixture
def random_scalars():
    key = jax.random.PRNGKey(0)
    return jax.random.randint(key, (100, 4), -2, 2)


def test_scalar_multiplication(random_scalars):
    s1 = random_scalars[0]
    s2 = random_scalars[1]

    d1 = ExactScalarArray(s1)
    d2 = ExactScalarArray(s2)

    prod_exact = d1 * d2
    prod_complex = d1.to_complex() * d2.to_complex()

    assert jnp.allclose(prod_exact.to_complex(), prod_complex)


def test_prod(random_scalars):
    """Test product along axis."""
    # Reshape to (10, 10, 4) to test product along axis 1
    scalars = random_scalars.reshape(10, 10, 4)

    # Exact computation
    dyadic_array = ExactScalarArray(scalars)
    prod_exact = dyadic_array.prod(axis=1)

    # Complex verification: manually compute product along axis 1
    complex_vals = dyadic_array.to_complex()  # (10, 10)

    # Compute product along axis 1 using reduce
    prod_complex_ref = jnp.prod(complex_vals, axis=1)  # (10,)

    assert jnp.allclose(prod_exact.to_complex(), prod_complex_ref, atol=1e-5)


def test_prod_single_element():
    """Test product of a single element."""
    scalars = jnp.array([[[1, 2, 0, -1]]])  # (1, 1, 4)
    dyadic_array = ExactScalarArray(scalars)
    prod_exact = dyadic_array.prod(axis=1)

    assert prod_exact.coeffs.shape == (1, 4)
    assert jnp.array_equal(prod_exact.coeffs, jnp.array([[1, 2, 0, -1]]))


def test_dyadic_reduce():
    coeffs = jnp.array([[2, 0, 0, 0]])
    power = jnp.array([0])
    dyadic = ExactScalarArray(coeffs, power)
    reduced = dyadic.reduce()

    assert jnp.array_equal(reduced.coeffs, jnp.array([[1, 0, 0, 0]]))
    assert jnp.array_equal(reduced.power, jnp.array([1]))
    assert jnp.isclose(reduced.to_complex(), dyadic.to_complex())

    coeffs = jnp.array([[2, 0, 4, 0], [4, 16, 0, 8], [1, 0, 0, 0]])
    power = jnp.array([0, 0, 0])
    dyadic = ExactScalarArray(coeffs, power)
    reduced = dyadic.reduce()

    expected_coeffs = jnp.array([[1, 0, 2, 0], [1, 4, 0, 2], [1, 0, 0, 0]])
    expected_power = jnp.array([1, 2, 0])

    assert jnp.array_equal(reduced.coeffs, expected_coeffs)
    assert jnp.array_equal(reduced.power, expected_power)
    assert jnp.allclose(reduced.to_complex(), dyadic.to_complex())

    # Zero should stop reducing (handled by condition check)
    # [0, 0, 0, 0] -> infinitely even, but we stop to avoid infinite loop
    coeffs = jnp.array([[0, 0, 0, 0]])
    power = jnp.array([0])
    dyadic = ExactScalarArray(coeffs, power)
    reduced = dyadic.reduce()

    assert jnp.array_equal(reduced.coeffs, coeffs)
    assert jnp.array_equal(reduced.power, power)
