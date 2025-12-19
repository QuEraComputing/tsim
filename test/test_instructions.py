from fractions import Fraction
from test.helpers.gate_matrices import (
    ROT_GATE_MATRICES,
    SINGLE_QUBIT_GATE_MATRICES,
    TWO_QUBIT_GATE_MATRICES,
)
from typing import Callable

import numpy as np
import pytest

from tsim import _instructions


def _build_and_get_matrix(gate_func: Callable | tuple[Callable, Callable], *args):
    """Helper to build a graph with a single gate and get its matrix."""
    b = _instructions.GraphRepresentation()
    if isinstance(gate_func, tuple):
        gate_func[0](b, *args)
        gate_func[1](b, *args)
    else:
        gate_func(b, *args)
    b.graph.normalize()
    return b.graph.to_matrix()


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        (_instructions.i, SINGLE_QUBIT_GATE_MATRICES["I"]),
        (_instructions.x, SINGLE_QUBIT_GATE_MATRICES["X"]),
        (_instructions.y, SINGLE_QUBIT_GATE_MATRICES["Y"]),
        (_instructions.z, SINGLE_QUBIT_GATE_MATRICES["Z"]),
        (_instructions.t, SINGLE_QUBIT_GATE_MATRICES["T"]),
        (_instructions.t_dag, SINGLE_QUBIT_GATE_MATRICES["T_DAG"]),
        (_instructions.c_xyz, SINGLE_QUBIT_GATE_MATRICES["C_XYZ"]),
        (_instructions.c_zyx, SINGLE_QUBIT_GATE_MATRICES["C_ZYX"]),
        (_instructions.h, SINGLE_QUBIT_GATE_MATRICES["H"]),
        (_instructions.h_xy, SINGLE_QUBIT_GATE_MATRICES["H_XY"]),
        (_instructions.h_yz, SINGLE_QUBIT_GATE_MATRICES["H_YZ"]),
        (_instructions.s, SINGLE_QUBIT_GATE_MATRICES["S"]),
        (_instructions.sqrt_x, SINGLE_QUBIT_GATE_MATRICES["SQRT_X"]),
        (_instructions.sqrt_x_dag, SINGLE_QUBIT_GATE_MATRICES["SQRT_X_DAG"]),
        (_instructions.sqrt_y, SINGLE_QUBIT_GATE_MATRICES["SQRT_Y"]),
        (_instructions.sqrt_y_dag, SINGLE_QUBIT_GATE_MATRICES["SQRT_Y_DAG"]),
        (_instructions.s_dag, SINGLE_QUBIT_GATE_MATRICES["S_DAG"]),
    ],
)
def test_single_qubit_instruction(gate_func, matrix: np.ndarray):
    result = _build_and_get_matrix(gate_func, 0)
    assert np.allclose(result, matrix)


@pytest.mark.parametrize("frac", [Fraction(1, 5), Fraction(-1, 3), Fraction(1, 7)])
def test_r_z(frac: Fraction):
    frac = Fraction(1, 5)
    result = _build_and_get_matrix(_instructions.r_z, 0, frac)
    expected = ROT_GATE_MATRICES["R_Z"](frac)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("frac", [Fraction(1, 5), Fraction(-1, 3), Fraction(1, 7)])
def test_r_x(frac: Fraction):
    frac = Fraction(1, 5)
    result = _build_and_get_matrix(_instructions.r_x, 0, frac)
    expected = ROT_GATE_MATRICES["R_X"](frac)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("frac_theta", [Fraction(1, 5), Fraction(-1, 3)])
@pytest.mark.parametrize("frac_phi", [Fraction(1, 5), Fraction(1, 7)])
@pytest.mark.parametrize("frac_lambda", [Fraction(-1, 13), Fraction(1, 7)])
def test_u3(frac_theta: Fraction, frac_phi: Fraction, frac_lambda: Fraction):
    result = _build_and_get_matrix(
        _instructions.u3, 0, frac_theta, frac_phi, frac_lambda
    )
    expected = ROT_GATE_MATRICES["U3"](frac_theta, frac_phi, frac_lambda)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        (_instructions.cnot, TWO_QUBIT_GATE_MATRICES["CNOT"]),
        (_instructions.cy, TWO_QUBIT_GATE_MATRICES["CY"]),
        (_instructions.cz, TWO_QUBIT_GATE_MATRICES["CZ"]),
        (_instructions.iswap, TWO_QUBIT_GATE_MATRICES["ISWAP"]),
        (_instructions.iswap_dag, TWO_QUBIT_GATE_MATRICES["ISWAP_DAG"]),
        (_instructions.sqrt_xx, TWO_QUBIT_GATE_MATRICES["SQRT_XX"]),
        (_instructions.sqrt_xx_dag, TWO_QUBIT_GATE_MATRICES["SQRT_XX_DAG"]),
        (_instructions.sqrt_yy, TWO_QUBIT_GATE_MATRICES["SQRT_YY"]),
        (_instructions.sqrt_yy_dag, TWO_QUBIT_GATE_MATRICES["SQRT_YY_DAG"]),
        (_instructions.sqrt_zz, TWO_QUBIT_GATE_MATRICES["SQRT_ZZ"]),
        (_instructions.sqrt_zz_dag, TWO_QUBIT_GATE_MATRICES["SQRT_ZZ_DAG"]),
        (_instructions.swap, TWO_QUBIT_GATE_MATRICES["SWAP"]),
        (_instructions.xcx, TWO_QUBIT_GATE_MATRICES["XCX"]),
        (_instructions.xcy, TWO_QUBIT_GATE_MATRICES["XCY"]),
        (_instructions.xcz, TWO_QUBIT_GATE_MATRICES["XCZ"]),
        (_instructions.ycx, TWO_QUBIT_GATE_MATRICES["YCX"]),
        (_instructions.ycy, TWO_QUBIT_GATE_MATRICES["YCY"]),
        (_instructions.ycz, TWO_QUBIT_GATE_MATRICES["YCZ"]),
    ],
)
def test_two_qubit_instruction(gate_func, matrix: np.ndarray):
    result = _build_and_get_matrix(gate_func, 0, 1)
    assert np.allclose(result, matrix)


def test_rz_mrz():
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)

    plus = (zero + one) / np.sqrt(2)

    result = _build_and_get_matrix((_instructions.i, _instructions.r), 0)
    assert np.allclose(result, np.outer(zero, plus.conj()))

    result = _build_and_get_matrix(_instructions.mr, 0)
    assert np.allclose(result, np.outer(zero, plus.conj()))

    result = _build_and_get_matrix(_instructions.r, 0)
    assert np.allclose(result, np.outer(zero, 1))


def test_rx_mrx():
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)

    plus = (zero + one) / np.sqrt(2)

    result = _build_and_get_matrix((_instructions.i, _instructions.rx), 0)
    assert np.allclose(result, np.outer(plus, zero.conj()))

    result = _build_and_get_matrix(_instructions.mrx, 0)
    assert np.allclose(result, np.outer(plus, zero.conj()))

    result = _build_and_get_matrix(_instructions.rx, 0)
    assert np.allclose(result, np.outer(plus, 1))


def test_ry_mry():
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)

    plus = (zero + one) / np.sqrt(2)

    plus_i = (zero + 1j * one) / np.sqrt(2)

    h_yz = SINGLE_QUBIT_GATE_MATRICES["H_YZ"]

    result = _build_and_get_matrix((_instructions.i, _instructions.ry), 0)
    assert np.allclose(result, np.outer(plus_i, (h_yz @ plus).conj()))

    result = _build_and_get_matrix(_instructions.mry, 0)

    assert np.allclose(result, np.outer(plus_i, (h_yz @ plus).conj()))

    result = _build_and_get_matrix(_instructions.ry, 0)
    assert np.allclose(result, np.outer(plus_i, 1))


def test_m():
    id = np.eye(2)
    result = _build_and_get_matrix(_instructions.m, 0)
    assert np.allclose(result, id / np.sqrt(2))


def tes_mx():
    id = np.eye(2)
    result = _build_and_get_matrix(_instructions.mx, 0)
    assert np.allclose(result, id / np.sqrt(2))
    print(result)


def test_my():
    id = np.eye(2)
    result = _build_and_get_matrix(_instructions.my, 0)
    assert np.allclose(result, id / np.sqrt(2))
