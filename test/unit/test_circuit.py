from typing import Any

import numpy as np
import pytest
import stim

from tsim.circuit import Circuit


def unitaries_equal_up_to_global_phase(
    u1: np.ndarray, u2: np.ndarray[Any, Any]
) -> bool:
    product = u1 @ u2.conj().T
    # If u1 = e^(i*phi) * u2, then product = e^(i*phi) * I
    phase = product[0, 0]
    expected = phase * np.eye(u1.shape[0])
    return np.allclose(product, expected)


@pytest.mark.parametrize(
    "stim_gate",
    [
        # Pauli gates
        "I",
        "X",
        "Y",
        "Z",
        # Single-qubit Clifford gates
        "C_XYZ",
        "C_ZYX",
        "H",
        "H_XY",
        "H_XZ",
        "H_YZ",
        "S",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
        "SQRT_Z",
        "SQRT_Z_DAG",
        "S_DAG",
    ],
)
def test_single_qubit_gate(stim_gate: str):
    c = Circuit(f"{stim_gate} 0")
    stim_c = stim.Circuit(f"{stim_gate} 0")
    stim_c_matrix = stim_c.to_tableau().to_unitary_matrix(endian="big")
    assert unitaries_equal_up_to_global_phase(c.to_matrix(), stim_c_matrix)


def test_t_gate():
    c = Circuit("S[T] 0")
    t_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    assert unitaries_equal_up_to_global_phase(c.to_matrix(), t_matrix)


def test_t_gate_shorthand():
    """Test that T shorthand is equivalent to S[T]."""
    c1 = Circuit("T 0")
    c2 = Circuit("S[T] 0")
    assert c1._stim_circ == c2._stim_circ


def test_t_dag_gate():
    c = Circuit("S_DAG[T] 0")
    t_dag_matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
    assert unitaries_equal_up_to_global_phase(c.to_matrix(), t_dag_matrix)


def test_t_dag_gate_shorthand():
    """Test that T_DAG shorthand is equivalent to S_DAG[T]."""
    c1 = Circuit("T_DAG 0")
    c2 = Circuit("S_DAG[T] 0")
    assert c1._stim_circ == c2._stim_circ


def test_rotation_gate_shorthand():
    """Test that R_Z(angle) shorthand is converted correctly."""
    c1 = Circuit("R_Z(0.25) 0")
    c2 = Circuit("I[R_Z(theta=0.25*pi)] 0")
    assert c1._stim_circ == c2._stim_circ

    c1 = Circuit("R_X(-0.5) 0")
    c2 = Circuit("I[R_X(theta=-0.5*pi)] 0")
    assert c1._stim_circ == c2._stim_circ

    c1 = Circuit("R_Y(0.333) 0")
    c2 = Circuit("I[R_Y(theta=0.333*pi)] 0")
    assert c1._stim_circ == c2._stim_circ


def test_u3_gate_shorthand():
    """Test that U3(theta, phi, lambda) shorthand is converted correctly."""
    c1 = Circuit("U3(0.3, 0.24, 0.49) 0")
    c2 = Circuit("I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0")
    assert c1._stim_circ == c2._stim_circ


@pytest.mark.parametrize(
    "stim_gate",
    [
        "CNOT",
        "CX",
        "CY",
        "CZ",
        "ISWAP",
        "ISWAP_DAG",
        "SQRT_XX",
        "SQRT_XX_DAG",
        "SQRT_YY",
        "SQRT_YY_DAG",
        "SQRT_ZZ",
        "SQRT_ZZ_DAG",
        "SWAP",
        "XCX",
        "XCY",
        "XCZ",
        "YCX",
        "YCY",
        "YCZ",
        "ZCX",
        "ZCY",
        "ZCZ",
    ],
)
def test_two_qubit_gate(stim_gate: str):
    c = Circuit(f"{stim_gate} 0 1")
    stim_c = stim.Circuit(f"{stim_gate} 0 1")
    stim_c_matrix = stim_c.to_tableau().to_unitary_matrix(endian="big")
    assert unitaries_equal_up_to_global_phase(c.to_matrix(), stim_c_matrix)


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


def test_from_file_preprocesses_shorthand(tmp_path):
    path = tmp_path / "prog.stim"
    path.write_text("T 0\nT_DAG 1\nR_Z(0.25) 0\n", encoding="utf-8")

    loaded = Circuit.from_file(path)
    expected = Circuit("T 0\nT_DAG 1\nR_Z(0.25) 0\n")

    assert loaded._stim_circ == expected._stim_circ
