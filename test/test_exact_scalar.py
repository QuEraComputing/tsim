import time

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


def test_segment_prod(random_scalars):

    N = len(random_scalars)
    num_segments = 10
    segment_ids = jnp.sort(
        jax.random.randint(jax.random.PRNGKey(1), (N,), 0, num_segments)
    )

    # Exact computation
    dyadic_array = ExactScalarArray(random_scalars)
    prod_exact = dyadic_array.segment_prod(
        segment_ids, num_segments=num_segments, indices_are_sorted=True
    )

    # Complex verification
    complex_vals = dyadic_array.to_complex()
    prod_complex_ref = jax.ops.segment_prod(
        complex_vals, segment_ids, num_segments=num_segments, indices_are_sorted=True
    )

    assert jnp.allclose(prod_exact.to_complex(), prod_complex_ref, atol=1e-5)


def test_segment_prod_unsorted(random_scalars):
    N = len(random_scalars)
    num_segments = 10
    segment_ids = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, num_segments)

    # Exact computation
    dyadic_array = ExactScalarArray(random_scalars)
    prod_exact = dyadic_array.segment_prod(
        segment_ids, num_segments=num_segments, indices_are_sorted=False
    )

    # Complex verification
    complex_vals = dyadic_array.to_complex()
    prod_complex_ref = jax.ops.segment_prod(
        complex_vals, segment_ids, num_segments=num_segments, indices_are_sorted=False
    )

    assert jnp.allclose(prod_exact.to_complex(), prod_complex_ref, atol=1e-5)


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


if __name__ == "__main__":
    # Benchmark
    N_vals = [1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    num_segments_ratio = 0.1

    print(f"{'N':<10} | {'Exact (ms)':<15} | {'Complex (ms)':<15} | {'Ratio':<10}")
    print("-" * 55)

    for N in N_vals:
        num_segments = int(N * num_segments_ratio)
        key = jax.random.PRNGKey(0)

        # Data generation
        scalars = jax.random.randint(key, (N, 4), -5, 5)
        segment_ids = jnp.sort(jax.random.randint(key, (N,), 0, num_segments))
        dyadic_vals = ExactScalarArray(scalars)
        complex_vals = dyadic_vals.to_complex()

        # Warmup
        _ = dyadic_vals.segment_prod(
            segment_ids, num_segments, True
        ).coeffs.block_until_ready()
        _ = jax.ops.segment_prod(
            complex_vals.real, segment_ids, num_segments, True
        ).block_until_ready()  # without .real this leads to segmentation fault on GPU

        # Time Exact
        start = time.time()
        for _ in range(10):
            _ = dyadic_vals.segment_prod(
                segment_ids, num_segments, True
            ).coeffs.block_until_ready()
        end = time.time()
        time_exact = (end - start) / 10 * 1000

        # Time Complex
        start = time.time()
        for _ in range(10):
            _ = jax.ops.segment_prod(
                complex_vals.real, segment_ids, num_segments, True
            ).block_until_ready()
        end = time.time()
        time_complex = (end - start) / 10 * 1000

        print(
            f"{N:<10} | {time_exact:<15.3f} | {time_complex:<15.3f} | {time_exact/time_complex:<10.2f}"
        )

        # CPU:
        # N          | Exact (ms)      | Complex (ms)    | Ratio
        # -------------------------------------------------------
        # 1000       | 0.165           | 0.128           | 1.29
        # 10000      | 0.213           | 0.204           | 1.05
        # 100000     | 0.917           | 0.519           | 1.77
        # 1000000    | 8.231           | 3.114           | 2.64
        # 10000000   | 118.212         | 27.932          | 4.23
        # 100000000  | 1305.727        | 275.444         | 4.74

        # TODO: improve performance. Can we match jax.ops.segment_prod?
