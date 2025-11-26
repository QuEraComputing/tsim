import numpy as np
import pytest
import stim

from tsim import _instructions
from tsim.circuit import Circuit


@pytest.mark.parametrize(
    "stim_gate, matrix",
    [
        ("X", np.array([[0, 1], [1, 0]])),
        ("Y", np.array([[0, -1j], [1j, 0]])),
        ("I", np.array([[1, 0], [0, 1]])),
        ("Z", np.array([[1, 0], [0, -1]])),
        ("H", np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
        ("S", np.array([[1, 0], [0, 1j]])),
        ("S_DAG", np.array([[1, 0], [0, -1j]])),
        ("S[T]", np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])),
        ("S_DAG[T]", np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])),
        ("SQRT_X", np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2),
        ("SQRT_X_DAG", np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2),
        ("SQRT_Y", np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2),
        ("SQRT_Y_DAG", np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2),
    ],
)
def test_single_qubit_gate(stim_gate: str, matrix: np.ndarray):
    c = Circuit(f"{stim_gate} 0")
    assert np.allclose(c.to_matrix(), matrix)


@pytest.mark.parametrize(
    "stim_gate, matrix",
    [
        (
            "CNOT",
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        ),
        (
            "CZ",
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        ),
        (
            "CY",
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        ),
    ],
)
def test_two_qubit_gate(stim_gate: str, matrix: np.ndarray):
    c = Circuit(f"{stim_gate} 0 1")
    assert np.allclose(c.to_matrix(), matrix)


def _build_and_get_matrix(gate_func, *args):
    """Helper to build a graph with a single gate and get its matrix."""
    b = _instructions.GraphRepresentation()
    gate_func(b, *args)
    b.graph.normalize()
    return b.graph.to_matrix()


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        (_instructions.x, np.array([[0, 1], [1, 0]])),
        (_instructions.y, np.array([[0, -1j], [1j, 0]])),
        (_instructions.i, np.array([[1, 0], [0, 1]])),
        (_instructions.z, np.array([[1, 0], [0, -1]])),
        (_instructions.h, np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
        (_instructions.s, np.array([[1, 0], [0, 1j]])),
        (_instructions.s_dag, np.array([[1, 0], [0, -1j]])),
        (_instructions.t, np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])),
        (_instructions.t_dag, np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])),
        (_instructions.sqrt_x, np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2),
        (_instructions.sqrt_x_dag, np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2),
        (_instructions.sqrt_y, np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2),
        (_instructions.sqrt_y_dag, np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2),
    ],
)
def test_internal_single_qubit_gate(gate_func, matrix: np.ndarray):
    result = _build_and_get_matrix(gate_func, 0)
    assert np.allclose(result, matrix)


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        (
            _instructions.cx,
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        ),
        (
            _instructions.cz,
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        ),
        (
            _instructions.cy,
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        ),
    ],
)
def test_internal_two_qubit_gate(gate_func, matrix: np.ndarray):
    result = _build_and_get_matrix(gate_func, 0, 1)
    assert np.allclose(result, matrix)


def test_num_measurements():
    c = Circuit()
    assert c.num_measurements == 0

    c = Circuit("M 0")
    assert c.num_measurements == 1

    c = Circuit("M 0 1 2")
    assert c.num_measurements == 3


def test_num_detectors():
    c = Circuit()
    assert c.num_detectors == 0

    c = Circuit("M 0\nDETECTOR rec[-1]")
    assert c.num_detectors == 1

    c = Circuit("M 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]")
    assert c.num_detectors == 2


def test_num_observables():
    c = Circuit("M 0")
    assert c.num_observables == 0

    c = Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]")
    assert c.num_observables == 1

    c = Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(2) rec[-2]")
    assert c.num_observables == 3

    c = Circuit(
        "M 0 1 2\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
        "OBSERVABLE_INCLUDE(2) rec[-2]\n"
        "OBSERVABLE_INCLUDE(5) rec[-1] rec[-2]"
    )
    assert c.num_observables == 6


def test_num_qubits():
    c = Circuit()
    assert c.num_qubits == 0

    c = Circuit("H 0")
    assert c.num_qubits == 1

    c = Circuit("H 0\nX 5")
    assert c.num_qubits == 6

    c = Circuit("H 0\nX 5\nCNOT 2 3")
    assert c.num_qubits == 6


def test_from_stim_program():
    stim_circ = stim.Circuit("H 0\nCNOT 0 1\nM 0 1")
    c = Circuit.from_stim_program(stim_circ)
    assert c._stim_circ == stim_circ


def test_from_stim_program_text():
    c = Circuit("H 0\nCNOT 0 1\nM 0 1")
    assert c._stim_circ == stim.Circuit("H 0\nCNOT 0 1\nM 0 1")


def test_circuit_copy():
    c1 = Circuit("H 0\nCNOT 0 1")
    c2 = c1.copy()
    assert c1 == c2
    assert c1 is not c2


def test_circuit_add():
    c1 = Circuit("H 0")
    c2 = Circuit("CNOT 0 1")
    c3 = c1 + c2
    assert c3._stim_circ == c1._stim_circ + c2._stim_circ


def test_circuit_iadd():
    c1 = Circuit("H 0")
    c2 = Circuit("CNOT 0 1")

    c1_stim = c1._stim_circ.copy()
    c2_stim = c2._stim_circ.copy()

    c1 += c2
    assert c1._stim_circ == c1_stim + c2_stim


def test_circuit_mul():
    c1 = Circuit("H 0")
    c1_stim = c1._stim_circ.copy()
    c2 = c1 * 3
    assert c2._stim_circ == (c1_stim * 3).flattened()


def test_circuit_without_noise():
    c = Circuit("H 0\nDEPOLARIZE1(0.01) 0\nM 0")
    c_clean = c.without_noise()
    assert c_clean._stim_circ == c._stim_circ.without_noise()


def test_circuit_without_annotations():
    c = Circuit("H 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nDETECTOR rec[-1]\nM 0")
    c_clean = c.without_annotations()
    assert c_clean._stim_circ == stim.Circuit("H 0\nM 0")


def test_circuit_eq():
    c1 = Circuit("H 0")
    c2 = Circuit("H 0")
    c3 = Circuit("X 0")
    assert c1 == c2
    assert c1 != c3
