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
        (_instructions.i, np.array([[1, 0], [0, 1]])),
        (_instructions.x, np.array([[0, 1], [1, 0]])),
        (_instructions.y, np.array([[0, -1j], [1j, 0]])),
        (_instructions.z, np.array([[1, 0], [0, -1]])),
        (_instructions.t, np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])),
        (_instructions.t_dag, np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])),
        (_instructions.c_xyz, np.array([[1 - 1j, -1 - 1j], [1 - 1j, 1 + 1j]]) / 2),
        (_instructions.c_zyx, np.array([[1 + 1j, 1 + 1j], [-1 + 1j, 1 - 1j]]) / 2),
        (_instructions.h, np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
        (_instructions.h_xy, np.array([[0, 1 - 1j], [1 + 1j, 0]]) / np.sqrt(2)),
        (_instructions.h_yz, np.array([[1, -1j], [1j, -1]]) / np.sqrt(2)),
        (_instructions.s, np.array([[1, 0], [0, 1j]])),
        (_instructions.sqrt_x, np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2),
        (_instructions.sqrt_x_dag, np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2),
        (_instructions.sqrt_y, np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2),
        (_instructions.sqrt_y_dag, np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2),
        (_instructions.s_dag, np.array([[1, 0], [0, -1j]])),
    ],
)
def test_single_qubit_instruction(gate_func, matrix: np.ndarray):
    result = _build_and_get_matrix(gate_func, 0)
    assert np.allclose(result, matrix)


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        (
            _instructions.cnot,
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        ),
        (
            _instructions.cy,
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        ),
        (
            _instructions.cz,
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        ),
        (
            _instructions.iswap,
            np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
        ),
        (
            _instructions.iswap_dag,
            np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]),
        ),
        (
            _instructions.sqrt_xx,
            np.array(
                [
                    [1 + 1j, 0, 0, 1 - 1j],
                    [0, 1 + 1j, 1 - 1j, 0],
                    [0, 1 - 1j, 1 + 1j, 0],
                    [1 - 1j, 0, 0, 1 + 1j],
                ]
            )
            / 2,
        ),
        (
            _instructions.sqrt_xx_dag,
            np.array(
                [
                    [1 - 1j, 0, 0, 1 + 1j],
                    [0, 1 - 1j, 1 + 1j, 0],
                    [0, 1 + 1j, 1 - 1j, 0],
                    [1 + 1j, 0, 0, 1 - 1j],
                ]
            )
            / 2,
        ),
        (
            _instructions.sqrt_yy,
            np.array(
                [
                    [1 + 1j, 0, 0, -1 + 1j],
                    [0, 1 + 1j, 1 - 1j, 0],
                    [0, 1 - 1j, 1 + 1j, 0],
                    [-1 + 1j, 0, 0, 1 + 1j],
                ]
            )
            / 2,
        ),
        (
            _instructions.sqrt_yy_dag,
            np.array(
                [
                    [1 - 1j, 0, 0, -1 - 1j],
                    [0, 1 - 1j, 1 + 1j, 0],
                    [0, 1 + 1j, 1 - 1j, 0],
                    [-1 - 1j, 0, 0, 1 - 1j],
                ]
            )
            / 2,
        ),
        (
            _instructions.sqrt_zz,
            np.array([[1, 0, 0, 0], [0, 1j, 0, 0], [0, 0, 1j, 0], [0, 0, 0, 1]]),
        ),
        (
            _instructions.sqrt_zz_dag,
            np.array([[1, 0, 0, 0], [0, -1j, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1]]),
        ),
        (
            _instructions.swap,
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        ),
        (
            _instructions.xcx,
            np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]) / 2,
        ),
        (
            _instructions.xcy,
            np.array(
                [[1, -1j, 1, 1j], [1j, 1, -1j, 1], [1, 1j, 1, -1j], [-1j, 1, 1j, 1]]
            )
            / 2,
        ),
        (
            _instructions.xcz,
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
        ),
        (
            _instructions.ycx,
            np.array(
                [[1, 1, -1j, 1j], [1, 1, 1j, -1j], [1j, -1j, 1, 1], [-1j, 1j, 1, 1]]
            )
            / 2,
        ),
        (
            _instructions.ycy,
            np.array(
                [[1, -1j, -1j, 1], [1j, 1, -1, -1j], [1j, -1, 1, -1j], [1, 1j, 1j, 1]]
            )
            / 2,
        ),
        (
            _instructions.ycz,
            np.array([[1, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1, 0], [0, 1j, 0, 0]]),
        ),
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

    h_yz = np.array([[1, -1j], [1j, -1]]) / np.sqrt(2)

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
