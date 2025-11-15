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
        ("cy", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])),
    ],
)
def test_two_qubit_gate(gate_func: str, matrix: np.ndarray):
    c = Circuit()
    c.__getattribute__(gate_func)(0, 1)
    assert np.allclose(c.to_matrix(), matrix)


def test_num_measurements():
    c = Circuit()
    assert c.num_measurements == 0
    c.m(0)
    assert c.num_measurements == 1
    c.m([1, 2])
    assert c.num_measurements == 3


def test_num_detectors():
    c = Circuit()
    assert c.num_detectors == 0
    c.m(0)
    c.detector([0])
    assert c.num_detectors == 1
    c.m(1)
    c.detector([1])
    assert c.num_detectors == 2


def test_num_observables():
    c = Circuit()
    c.m(0)
    assert c.num_observables == 0
    c.m(1)
    c.observable_include([0], 0)
    assert c.num_observables == 1
    c.observable_include([1], 2)
    assert c.num_observables == 3
    c.observable_include([0, 1], 5)
    assert c.num_observables == 6


def test_num_qubits():
    c = Circuit()
    assert c.num_qubits == 0
    c.h(0)
    assert c.num_qubits == 1
    c.x(5)
    assert c.num_qubits == 6
    c.cnot(2, 3)
    assert c.num_qubits == 6
