"""Test gate correctness using the Bell state trick.

This module tests gate implementations by exploiting channel-state duality.
By preparing a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 and applying a gate U to one
half, the resulting state encodes the gate matrix in its amplitudes:

    (U ⊗ I)|Φ⁺⟩ = Σᵢⱼ Uᵢⱼ |i⟩|j⟩ / √d

Measuring both qubits then yields probabilities P(i,j) = |Uᵢⱼ|²/d, allowing us
to extract |U|² element-wise from a single state vector measurement.

For two-qubit gates, we use two Bell pairs (4 qubits total) and apply the gate
to one qubit from each pair.
"""

from test.helpers import get_matrix
from test.helpers.gate_matrices import (
    ROT_GATE_MATRICES,
    SINGLE_QUBIT_GATE_MATRICES,
    TWO_QUBIT_GATE_MATRICES,
)

import numpy as np
import pytest

from tsim.circuit import Circuit
from tsim.sampler import CompiledStateProbs


def get_heralded_matrix(
    sampler: CompiledStateProbs, herald_value: int, batch_size: int = 1
) -> np.ndarray:
    """Extract the effective gate matrix from a heralded Bell state circuit.

    The herald measurement is the first bit in the measurement record. This
    function fixes the herald to ``herald_value`` and iterates over the
    remaining qubit measurement outcomes to reconstruct |U|².
    """
    num_qubits = sampler.circuit.num_qubits
    probs = []
    for i in range(2**num_qubits):
        state = np.zeros(num_qubits, dtype=np.bool_)
        for j in range(num_qubits):
            state[j] = (i >> j) & 1
        state = state[::-1]
        full_state = np.concatenate([[herald_value], state])
        probs.append(np.mean(sampler.probability_of(full_state, batch_size=batch_size)))
    probs = np.array(probs)
    dim = 2 ** (num_qubits // 2)
    return (probs * dim).reshape((dim, dim))


@pytest.mark.parametrize("instruction", SINGLE_QUBIT_GATE_MATRICES.keys())
def test_single_qubit_instructions(instruction: str):
    unitary = SINGLE_QUBIT_GATE_MATRICES[instruction]
    c = Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        {instruction} 0
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.abs(unitary) ** 2)


@pytest.mark.parametrize("instruction", TWO_QUBIT_GATE_MATRICES.keys())
def test_two_qubit_instructions(instruction: str):
    unitary = TWO_QUBIT_GATE_MATRICES[instruction]
    c = Circuit(f"""
        R 0 1 2 3
        H 0 1
        CNOT 0 2 1 3
        {instruction} 0 1
        M 0 1 2 3
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.abs(unitary) ** 2)


@pytest.mark.parametrize("instruction", ["R_X", "R_Y", "R_Z"])
def test_rot_instructions(instruction: str):
    c = Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        {instruction}(0.345) 0
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    expected = np.abs(ROT_GATE_MATRICES[instruction](0.345)) ** 2
    assert np.allclose(mat, expected)


SPP_SINGLE_QUBIT_EQUIVALENCES = {
    "SPP Z0": SINGLE_QUBIT_GATE_MATRICES["S"],
    "SPP_DAG Z0": SINGLE_QUBIT_GATE_MATRICES["S_DAG"],
    "SPP !Z0": SINGLE_QUBIT_GATE_MATRICES["S_DAG"],
    "SPP_DAG !Z0": SINGLE_QUBIT_GATE_MATRICES["S"],
    "SPP X0": SINGLE_QUBIT_GATE_MATRICES["SQRT_X"],
    "SPP_DAG X0": SINGLE_QUBIT_GATE_MATRICES["SQRT_X_DAG"],
    "SPP !X0": SINGLE_QUBIT_GATE_MATRICES["SQRT_X_DAG"],
    "SPP_DAG !X0": SINGLE_QUBIT_GATE_MATRICES["SQRT_X"],
    "SPP Y0": SINGLE_QUBIT_GATE_MATRICES["SQRT_Y"],
    "SPP_DAG Y0": SINGLE_QUBIT_GATE_MATRICES["SQRT_Y_DAG"],
    "SPP !Y0": SINGLE_QUBIT_GATE_MATRICES["SQRT_Y_DAG"],
    "SPP_DAG !Y0": SINGLE_QUBIT_GATE_MATRICES["SQRT_Y"],
    # Pair cancellation: P·P = I → no-op
    "SPP X0*X0": SINGLE_QUBIT_GATE_MATRICES["I"],
    "SPP_DAG Y0*Y0": SINGLE_QUBIT_GATE_MATRICES["I"],
    # Anticommutation: Z·X·Z = -X → sign flips dagger
    "SPP Z0*X0*Z0": SINGLE_QUBIT_GATE_MATRICES["SQRT_X_DAG"],
    "SPP_DAG Z0*X0*Z0": SINGLE_QUBIT_GATE_MATRICES["SQRT_X"],
}

SPP_TWO_QUBIT_EQUIVALENCES = {
    "SPP X0*X1": TWO_QUBIT_GATE_MATRICES["SQRT_XX"],
    "SPP_DAG X0*X1": TWO_QUBIT_GATE_MATRICES["SQRT_XX_DAG"],
    "SPP !X0*X1": TWO_QUBIT_GATE_MATRICES["SQRT_XX_DAG"],
    "SPP Y0*Y1": TWO_QUBIT_GATE_MATRICES["SQRT_YY"],
    "SPP_DAG Y0*Y1": TWO_QUBIT_GATE_MATRICES["SQRT_YY_DAG"],
    "SPP !Y0*Y1": TWO_QUBIT_GATE_MATRICES["SQRT_YY_DAG"],
    "SPP Z0*Z1": TWO_QUBIT_GATE_MATRICES["SQRT_ZZ"],
    "SPP_DAG Z0*Z1": TWO_QUBIT_GATE_MATRICES["SQRT_ZZ_DAG"],
    "SPP !Z0*Z1": TWO_QUBIT_GATE_MATRICES["SQRT_ZZ_DAG"],
    # Both qubits' Paulis cancel → identity on two qubits
    "SPP X0*X1*X0*X1": np.eye(4),
    "SPP_DAG Z0*Z1*Z0*Z1": np.eye(4),
    # Anticommutation on qubit 0 flips dagger: -X0·X1 → SPP_DAG X0*X1
    "SPP Z0*X0*Z0*X1": TWO_QUBIT_GATE_MATRICES["SQRT_XX_DAG"],
    "SPP_DAG Z0*X0*Z0*X1": TWO_QUBIT_GATE_MATRICES["SQRT_XX"],
}


@pytest.mark.parametrize("instruction", SPP_SINGLE_QUBIT_EQUIVALENCES.keys())
def test_spp_single_qubit(instruction: str):
    unitary = SPP_SINGLE_QUBIT_EQUIVALENCES[instruction]
    c = Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        {instruction}
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.abs(unitary) ** 2)


@pytest.mark.parametrize("instruction", SPP_TWO_QUBIT_EQUIVALENCES.keys())
def test_spp_two_qubit(instruction: str):
    unitary = SPP_TWO_QUBIT_EQUIVALENCES[instruction]
    c = Circuit(f"""
        R 0 1 2 3
        H 0 1
        CNOT 0 2 1 3
        {instruction}
        M 0 1 2 3
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.abs(unitary) ** 2)


def _tpp_unitary(pauli_matrix: np.ndarray, dagger: bool = False) -> np.ndarray:
    """Compute the TPP unitary from first principles.

    TPP(P) phases the -1 eigenspace of P by exp(iπ/4):
        TPP(P) = (1 + e^{iπ/4})/2 · I + (1 - e^{iπ/4})/2 · P

    Up to phase, this is equivalent to:
        TPP(P) = exp(iπ/8 · P) = cos(π/8) · I + sin(π/8) · P
    """
    phase = np.exp(-1j * np.pi / 4) if dagger else np.exp(1j * np.pi / 4)
    identity = np.eye(pauli_matrix.shape[0])
    return ((1 + phase) * identity + (1 - phase) * pauli_matrix) / 2


_X = SINGLE_QUBIT_GATE_MATRICES["X"]
_Y = SINGLE_QUBIT_GATE_MATRICES["Y"]
_Z = SINGLE_QUBIT_GATE_MATRICES["Z"]

TPP_SINGLE_QUBIT_EQUIVALENCES = {
    # TPP Z equals the named T gate (and TPP_DAG Z equals T_DAG)
    "TPP Z0": SINGLE_QUBIT_GATE_MATRICES["T"],
    "TPP_DAG Z0": SINGLE_QUBIT_GATE_MATRICES["T_DAG"],
    "TPP !Z0": SINGLE_QUBIT_GATE_MATRICES["T_DAG"],
    "TPP_DAG !Z0": SINGLE_QUBIT_GATE_MATRICES["T"],
    # TPP X and TPP Y have no named equivalents — use the first-principles formula
    "TPP X0": _tpp_unitary(_X),
    "TPP_DAG X0": _tpp_unitary(_X, dagger=True),
    "TPP !X0": _tpp_unitary(_X, dagger=True),
    "TPP_DAG !X0": _tpp_unitary(_X),
    "TPP Y0": _tpp_unitary(_Y),
    "TPP_DAG Y0": _tpp_unitary(_Y, dagger=True),
    "TPP !Y0": _tpp_unitary(_Y, dagger=True),
    "TPP_DAG !Y0": _tpp_unitary(_Y),
    # Pair cancellation: P·P = I → no-op
    "TPP X0*X0": SINGLE_QUBIT_GATE_MATRICES["I"],
    "TPP_DAG Y0*Y0": SINGLE_QUBIT_GATE_MATRICES["I"],
    # Anticommutation: Z·X·Z = -X → sign flips dagger
    "TPP Z0*X0*Z0": _tpp_unitary(_X, dagger=True),
    "TPP_DAG Z0*X0*Z0": _tpp_unitary(_X),
}

TPP_TWO_QUBIT_EQUIVALENCES = {
    "TPP X0*X1": _tpp_unitary(np.kron(_X, _X)),
    "TPP_DAG X0*X1": _tpp_unitary(np.kron(_X, _X), dagger=True),
    "TPP !X0*X1": _tpp_unitary(np.kron(_X, _X), dagger=True),
    "TPP Y0*Y1": _tpp_unitary(np.kron(_Y, _Y)),
    "TPP_DAG Y0*Y1": _tpp_unitary(np.kron(_Y, _Y), dagger=True),
    "TPP !Y0*Y1": _tpp_unitary(np.kron(_Y, _Y), dagger=True),
    "TPP Z0*Z1": _tpp_unitary(np.kron(_Z, _Z)),
    "TPP_DAG Z0*Z1": _tpp_unitary(np.kron(_Z, _Z), dagger=True),
    "TPP !Z0*Z1": _tpp_unitary(np.kron(_Z, _Z), dagger=True),
    # Both qubits' Paulis cancel → identity on two qubits
    "TPP X0*X1*X0*X1": np.eye(4),
    "TPP_DAG Z0*Z1*Z0*Z1": np.eye(4),
    # Anticommutation on qubit 0 flips dagger: -X0·X1 → TPP_DAG X0*X1
    "TPP Z0*X0*Z0*X1": _tpp_unitary(np.kron(_X, _X), dagger=True),
    "TPP_DAG Z0*X0*Z0*X1": _tpp_unitary(np.kron(_X, _X)),
}


@pytest.mark.parametrize("instruction", TPP_SINGLE_QUBIT_EQUIVALENCES.keys())
def test_tpp_single_qubit(instruction: str):
    unitary = TPP_SINGLE_QUBIT_EQUIVALENCES[instruction]
    c = Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        {instruction}
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.abs(unitary) ** 2)


@pytest.mark.parametrize("instruction", TPP_TWO_QUBIT_EQUIVALENCES.keys())
def test_tpp_two_qubit(instruction: str):
    unitary = TPP_TWO_QUBIT_EQUIVALENCES[instruction]
    c = Circuit(f"""
        R 0 1 2 3
        H 0 1
        CNOT 0 2 1 3
        {instruction}
        M 0 1 2 3
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.abs(unitary) ** 2)


def test_tpp_z_equals_t():
    """TPP Z0 should be equivalent to a T gate."""
    unitary = TPP_SINGLE_QUBIT_EQUIVALENCES["TPP Z0"]
    t_matrix = SINGLE_QUBIT_GATE_MATRICES["T"]
    assert np.allclose(unitary, t_matrix)


def test_u3_instruction():
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        U3(0.345, 0.245, 0.495) 0
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    expected = np.abs(ROT_GATE_MATRICES["U3"](0.345, 0.245, 0.495)) ** 2
    assert np.allclose(mat, expected)


HERALDED_PAULI_CASES = {
    "I": ("HERALDED_PAULI_CHANNEL_1(1, 0, 0, 0)", SINGLE_QUBIT_GATE_MATRICES["I"]),
    "X": ("HERALDED_PAULI_CHANNEL_1(0, 1, 0, 0)", SINGLE_QUBIT_GATE_MATRICES["X"]),
    "Y": ("HERALDED_PAULI_CHANNEL_1(0, 0, 1, 0)", SINGLE_QUBIT_GATE_MATRICES["Y"]),
    "Z": ("HERALDED_PAULI_CHANNEL_1(0, 0, 0, 1)", SINGLE_QUBIT_GATE_MATRICES["Z"]),
}


@pytest.mark.parametrize("pauli", HERALDED_PAULI_CASES.keys())
def test_heralded_pauli_channel_1_bell_state(pauli: str):
    """Verify heralded channel applies the correct Pauli using the Bell state trick.

    With one probability set to 1, the channel is deterministic: herald always
    fires and the corresponding Pauli is applied. We reconstruct |U|² from the
    qubit measurement probabilities conditioned on herald=1.
    """
    instruction, unitary = HERALDED_PAULI_CASES[pauli]
    c = Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        {instruction} 0
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_heralded_matrix(sampler, herald_value=1)
    assert np.allclose(mat, np.abs(unitary) ** 2)


def test_heralded_pauli_channel_1_no_fire():
    """With all probabilities zero the channel never fires (identity, herald=0)."""
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        HERALDED_PAULI_CHANNEL_1(0, 0, 0, 0) 0
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_heralded_matrix(sampler, herald_value=0)
    expected = np.abs(SINGLE_QUBIT_GATE_MATRICES["I"]) ** 2
    assert np.allclose(mat, expected)


def test_heralded_erase_bell_state():
    """HERALDED_ERASE(1) applies each Pauli with probability 1/4.

    The effective matrix is (|I|² + |X|² + |Y|² + |Z|²) / 4 = [[0.5, 0.5], [0.5, 0.5]].
    """
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        HERALDED_ERASE(1) 0
        M 0 1
        """)
    sampler = CompiledStateProbs(c, seed=42)
    mat = get_heralded_matrix(sampler, herald_value=1, batch_size=10000)
    expected = np.full((2, 2), 0.5)
    assert np.allclose(mat, expected, atol=0.05)
