import numpy as np
import pytest

from tsim.circuit import Circuit


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        ("x", np.array([[0, 1], [1, 0]])),
        ("y", np.array([[0, -1j], [1j, 0]])),
        ("i", np.array([[1, 0], [0, 1]])),
        ("z", np.array([[1, 0], [0, -1]])),
        ("h", np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
        ("s", np.array([[1, 0], [0, 1j]])),
        ("s_dag", np.array([[1, 0], [0, -1j]])),
        ("t", np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])),
        ("t_dag", np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])),
        ("sqrt_x", np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2),
        ("sqrt_x_dag", np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2),
        ("sqrt_y", np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2),
        ("sqrt_y_dag", np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2),
    ],
)
def test_single_qubit_gate(gate_func: str, matrix: np.ndarray):
    c = Circuit()
    c.__getattribute__(gate_func)(0)
    assert np.allclose(c.to_matrix(), matrix)


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        ("cnot", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
        ("cz", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])),
    ],
)
def test_two_qubit_gate(gate_func: str, matrix: np.ndarray):
    c = Circuit()
    c.__getattribute__(gate_func)(0, 1)
    assert np.allclose(c.to_matrix(), matrix)
