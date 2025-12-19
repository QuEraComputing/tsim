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


@pytest.mark.parametrize("instruction", SINGLE_QUBIT_GATE_MATRICES.keys())
def test_single_qubit_instructions(instruction: str):
    unitary = SINGLE_QUBIT_GATE_MATRICES[instruction]
    c = Circuit(
        f"""
        R 0 1
        H 0
        CNOT 0 1
        {instruction} 0
        M 0 1
        """
    )
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.abs(unitary) ** 2)


@pytest.mark.parametrize("instruction", TWO_QUBIT_GATE_MATRICES.keys())
def test_two_qubit_instructions(instruction: str):
    unitary = TWO_QUBIT_GATE_MATRICES[instruction]
    c = Circuit(
        f"""
        R 0 1 2 3
        H 0 1
        CNOT 0 2 1 3
        {instruction} 0 1
        M 0 1 2 3
        """
    )
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.abs(unitary) ** 2)


@pytest.mark.parametrize("instruction", ["R_X", "R_Y", "R_Z"])
def test_rot_instructions(instruction: str):
    c = Circuit(
        f"""
        R 0 1
        H 0
        CNOT 0 1
        {instruction}(0.345) 0
        M 0 1
        """
    )
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    expected = np.abs(ROT_GATE_MATRICES[instruction](0.345)) ** 2
    assert np.allclose(mat, expected)


def test_u3_instruction():
    c = Circuit(
        """
        R 0 1
        H 0
        CNOT 0 1
        U3(0.345, 0.245, 0.495) 0
        M 0 1
        """
    )
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    expected = np.abs(ROT_GATE_MATRICES["U3"](0.345, 0.245, 0.495)) ** 2
    assert np.allclose(mat, expected)
