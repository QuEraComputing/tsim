import time

import jax
import jax.numpy as jnp
import pytest

from tsim.exact_scalar import scalar_mul, scalar_to_complex, segment_scalar_prod


@pytest.fixture
def random_scalars():
    key = jax.random.PRNGKey(0)
    return jax.random.randint(key, (100, 4), -2, 2)


def test_scalar_multiplication(random_scalars):
    s1 = random_scalars[0][None]
    s2 = random_scalars[1][None]

    prod_exact = scalar_mul(s1, s2)
    prod_complex = scalar_to_complex(s1) * scalar_to_complex(s2)

    assert jnp.allclose(scalar_to_complex(prod_exact), prod_complex)


def test_segment_prod(random_scalars):

    N = len(random_scalars)
    num_segments = 10
    segment_ids = jnp.sort(
        jax.random.randint(jax.random.PRNGKey(1), (N,), 0, num_segments)
    )

    # Exact computation
    prod_exact = segment_scalar_prod(
        random_scalars, segment_ids, num_segments=num_segments, indices_are_sorted=True
    )

    # Complex verification
    complex_vals = scalar_to_complex(random_scalars)
    prod_complex_ref = jax.ops.segment_prod(
        complex_vals, segment_ids, num_segments=num_segments, indices_are_sorted=True
    )

    assert jnp.allclose(scalar_to_complex(prod_exact), prod_complex_ref, atol=1e-5)


def test_segment_prod_unsorted(random_scalars):
    N = len(random_scalars)
    num_segments = 10
    segment_ids = jax.random.randint(jax.random.PRNGKey(1), (N,), 0, num_segments)

    # Exact computation
    prod_exact = segment_scalar_prod(
        random_scalars, segment_ids, num_segments=num_segments, indices_are_sorted=False
    )

    # Complex verification
    complex_vals = scalar_to_complex(random_scalars)
    prod_complex_ref = jax.ops.segment_prod(
        complex_vals, segment_ids, num_segments=num_segments, indices_are_sorted=False
    )

    assert jnp.allclose(scalar_to_complex(prod_exact), prod_complex_ref, atol=1e-5)


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
        complex_vals = scalar_to_complex(scalars)

        # Warmup
        _ = segment_scalar_prod(
            scalars, segment_ids, num_segments, True
        ).block_until_ready()
        _ = jax.ops.segment_prod(
            complex_vals, segment_ids, num_segments, True
        ).block_until_ready()

        # Time Exact
        start = time.time()
        for _ in range(10):
            _ = segment_scalar_prod(
                scalars, segment_ids, num_segments, True
            ).block_until_ready()
        end = time.time()
        time_exact = (end - start) / 10 * 1000

        # Time Complex
        start = time.time()
        for _ in range(10):
            _ = jax.ops.segment_prod(
                complex_vals, segment_ids, num_segments, True
            ).block_until_ready()
        end = time.time()
        time_complex = (end - start) / 10 * 1000

        print(
            f"{N:<10} | {time_exact:<15.3f} | {time_complex:<15.3f} | {time_exact/time_complex:<10.2f}"
        )

        # N          | Exact (ms)      | Complex (ms)    | Ratio
        # -------------------------------------------------------
        # 1000       | 0.041           | 0.177           | 0.23
        # 10000      | 0.164           | 0.222           | 0.74
        # 100000     | 0.773           | 0.648           | 1.19
        # 1000000    | 7.087           | 5.016           | 1.41
        # 10000000   | 111.526         | 42.580          | 2.62
        # 100000000  | 1250.639        | 466.575         | 2.68

        # TODO: can exact be improved to outperform complex?
