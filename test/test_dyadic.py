import random

import numpy as np

from tsim.dyadic import DyadicNumber


def test_mul():
    d1 = DyadicNumber(1, 1, 0, 0, 1)
    d2 = DyadicNumber(1, 0, 1, 1, 0)
    d3 = d1 * d2
    assert np.isclose(d3.to_complex(), d1.to_complex() * d2.to_complex())

    d1 = DyadicNumber(0, 0, 1, 0, 1)
    d2 = DyadicNumber(0, 0, 1, 0, 1)
    d3 = d1 * d2
    assert np.isclose(d3.to_complex(), d1.to_complex() * d2.to_complex())


def test_mul_with_random_scalars():
    for _ in range(20):
        d1 = DyadicNumber(*[random.randint(-20, 20) for _ in range(5)])
        d2 = DyadicNumber(*[random.randint(-20, 20) for _ in range(5)])
        d3 = d1 * d2
        assert np.isclose(d3.to_complex(), d1.to_complex() * d2.to_complex())


def test_sqrt2():
    sqrt2 = DyadicNumber.sqrt2()
    assert np.isclose(sqrt2.to_complex(), np.sqrt(2))
