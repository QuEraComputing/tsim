import math
import random

import numpy as np

from tsim.external.pyzx.graph.scalar import DyadicNumber


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


def test_one():
    one = DyadicNumber.one()
    assert np.isclose(one.to_complex(), 1)


def test_conjugate():
    for _ in range(20):
        d1 = DyadicNumber(*[random.randint(-20, 20) for _ in range(5)])
        d1_conj = d1.conjugate()
        assert np.isclose(d1.to_complex(), d1_conj.to_complex().conjugate())


def test_magic_constants():
    sq2 = math.sqrt(2)

    MAGIC_GLOBAL = -(7 + 5 * sq2) / (2 + 2j)
    MAGIC_B60 = -16 + 12 * sq2
    MAGIC_B66 = 96 - 68 * sq2
    MAGIC_E6 = 10 - 7 * sq2
    MAGIC_O6 = -14 + 10 * sq2
    MAGIC_K6 = 7 - 5 * sq2
    MAGIC_PHI = 10 - 7 * sq2

    MAGIC_GLOBAL2 = DyadicNumber(k=2, a=-7, b=0, c=7, d=-10)
    MAGIC_B602 = DyadicNumber(k=-2, a=-4, b=3, c=0, d=3)
    MAGIC_B662 = DyadicNumber(k=-2, a=24, b=-17, c=0, d=-17)
    MAGIC_E62 = DyadicNumber(k=0, a=10, b=-7, c=0, d=-7)
    MAGIC_O62 = DyadicNumber(k=-1, a=-7, b=5, c=0, d=5)
    MAGIC_K62 = DyadicNumber(k=0, a=7, b=-5, c=0, d=-5)
    MAGIC_PHI2 = DyadicNumber(k=0, a=10, b=-7, c=0, d=-7)

    assert MAGIC_GLOBAL == MAGIC_GLOBAL2.to_complex()
    assert MAGIC_B60 == MAGIC_B602.to_complex()
    assert MAGIC_B66 == MAGIC_B662.to_complex()
    assert MAGIC_E6 == MAGIC_E62.to_complex()
    assert MAGIC_O6 == MAGIC_O62.to_complex()
    assert MAGIC_K6 == MAGIC_K62.to_complex()
    assert MAGIC_PHI == MAGIC_PHI2.to_complex()
