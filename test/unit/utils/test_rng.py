import numba
import numpy as np
import pytest

from tsim.utils.rng import log1m, next_skip, splitmix64


@numba.njit
def _worst_log1m_error(count, seed, scale):
    state = np.uint64(seed)
    worst = 0.0
    for _ in range(count):
        state, z = splitmix64(state)
        x = z * 2.0**-64 * scale
        if x >= 1.0:
            continue
        exact = np.log1p(-x)
        if exact != 0.0:
            worst = max(worst, abs(log1m(x) / exact - 1.0))
    return worst


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-12, 1e-300])
def test_log1m_is_within_a_few_ulp_of_log1p(scale: float):
    # NOTE: 2 ulp on the development machine; the bound leaves room for other libms.
    assert _worst_log1m_error(200_000, 7, scale) <= 4 * np.finfo(np.float64).eps


def test_log1m_edges():
    assert log1m(0.0) == 0.0
    assert np.isfinite(log1m(1.0)) and log1m(1.0) < -700


@numba.njit
def _skip_counts(count, seed, log_keep):
    state = np.uint64(seed)
    zeros = 0
    total = 0.0
    for _ in range(count):
        state, skip = next_skip(state, log_keep)
        zeros += skip == 0.0
        total += skip
    return zeros, total


@pytest.mark.parametrize("p", [0.5, 1e-3])
def test_next_skip_is_geometric(p: float):
    """No failure before the first success with probability p; mean (1 - p) / p."""
    count = 400_000
    zeros, total = _skip_counts(count, 11, np.log1p(-p))
    assert abs(zeros / count - p) < 5 * np.sqrt(p * (1 - p) / count)
    mean, sigma = (1 - p) / p, np.sqrt(1 - p) / p
    assert abs(total / count - mean) < 5 * sigma / np.sqrt(count)


def test_next_skip_with_certain_success_is_zero():
    assert next_skip(np.uint64(3), -np.inf)[1] == 0.0
