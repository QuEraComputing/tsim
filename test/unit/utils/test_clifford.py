"""Comprehensive tests for parametric-to-Clifford gate conversion."""

from fractions import Fraction

import numpy as np
import pytest
import stim

from tsim.circuit import Circuit
from tsim.utils.clifford import (
    _to_half_pi_index,
    expand_clifford_rotations,
    parametric_to_clifford_gates,
)


def _unitaries_equal_up_to_global_phase(u1: np.ndarray, u2: np.ndarray) -> bool:
    product = u1 @ u2.conj().T
    phase = product[0, 0]
    if abs(phase) < 1e-10:
        return False
    return np.allclose(product, phase * np.eye(u1.shape[0]), atol=1e-10)


def _rotation_matrix(axis: str, theta_pi: float) -> np.ndarray:
    """R_axis(θ) unitary with θ given in units of π."""
    t = theta_pi * np.pi
    if axis == "Z":
        return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])
    if axis == "X":
        return np.array(
            [
                [np.cos(t / 2), -1j * np.sin(t / 2)],
                [-1j * np.sin(t / 2), np.cos(t / 2)],
            ]
        )
    if axis == "Y":
        return np.array(
            [
                [np.cos(t / 2), -np.sin(t / 2)],
                [np.sin(t / 2), np.cos(t / 2)],
            ]
        )
    raise ValueError(axis)


def _u3_matrix(theta_pi: float, phi_pi: float, lam_pi: float) -> np.ndarray:
    """U3(θ, φ, λ) = R_Z(φ) · R_Y(θ) · R_Z(λ), angles in units of π."""
    return (
        _rotation_matrix("Z", phi_pi)
        @ _rotation_matrix("Y", theta_pi)
        @ _rotation_matrix("Z", lam_pi)
    )


def _clifford_matrix(gate_names: list[str]) -> np.ndarray:
    """Build unitary from stim Clifford gate names."""
    circ = stim.Circuit()
    circ.append("I", [0])  # type: ignore
    for g in gate_names:
        circ.append(g, [0])  # type: ignore
    return circ.to_tableau().to_unitary_matrix(endian="big")  # type: ignore


class TestToHalfPiIndex:
    @pytest.mark.parametrize(
        "phase,expected",
        [
            (Fraction(0), 0),
            (Fraction(1, 2), 1),
            (Fraction(1), 2),
            (Fraction(3, 2), 3),
            (Fraction(-1, 2), 3),
            (Fraction(-1), 2),
            (Fraction(-3, 2), 1),
            (Fraction(2), 0),
            (Fraction(5, 2), 1),
            (Fraction(-2), 0),
        ],
    )
    def test_valid(self, phase, expected):
        assert _to_half_pi_index(phase) == expected

    @pytest.mark.parametrize(
        "phase",
        [Fraction(1, 3), Fraction(1, 4), Fraction(3, 4), Fraction(1, 7)],
    )
    def test_non_half_pi_returns_none(self, phase):
        assert _to_half_pi_index(phase) is None


class TestSingleAxisConversions:
    def test_non_clifford_returns_none(self):
        assert parametric_to_clifford_gates("R_Z", {"theta": Fraction(1, 4)}) is None

    def test_unknown_gate_returns_none(self):
        assert parametric_to_clifford_gates("UNKNOWN", {"theta": Fraction(0)}) is None


class TestSingleAxisUnitaries:
    @pytest.mark.parametrize(
        "axis,half_pi_idx",
        [(a, i) for a in ("X", "Y", "Z") for i in range(4)],
    )
    def test_unitary_matches(self, axis, half_pi_idx):
        phase = Fraction(half_pi_idx, 2)
        original = _rotation_matrix(axis, float(phase))
        clifford_gates = parametric_to_clifford_gates(f"R_{axis}", {"theta": phase})
        assert clifford_gates is not None
        assert _unitaries_equal_up_to_global_phase(
            original, _clifford_matrix(clifford_gates)
        )


class TestU3Conversions:
    @pytest.mark.parametrize(
        "theta_idx,phi_idx,lam_idx",
        [(t, p, lam) for t in range(4) for p in range(4) for lam in range(4)],
    )
    def test_unitary_matches_all_half_pi(self, theta_idx, phi_idx, lam_idx):
        theta = Fraction(theta_idx, 2)
        phi = Fraction(phi_idx, 2)
        lam = Fraction(lam_idx, 2)

        params = {"theta": theta, "phi": phi, "lambda": lam}
        clifford_gates = parametric_to_clifford_gates("U3", params)
        assert (
            clifford_gates is not None
        ), f"U3({theta}, {phi}, {lam}) should be convertible"

        original = _u3_matrix(float(theta), float(phi), float(lam))
        assert _unitaries_equal_up_to_global_phase(
            original, _clifford_matrix(clifford_gates)
        ), f"U3({theta_idx},{phi_idx},{lam_idx}) → {clifford_gates} mismatch"

    def test_non_clifford_returns_none(self):
        params = {"theta": Fraction(1, 4), "phi": Fraction(0), "lambda": Fraction(0)}
        assert parametric_to_clifford_gates("U3", params) is None

    def test_partially_non_clifford_returns_none(self):
        params = {"theta": Fraction(1, 2), "phi": Fraction(1, 3), "lambda": Fraction(0)}
        assert parametric_to_clifford_gates("U3", params) is None


class TestStimCircuitProperty:
    def test_clifford_rotation_expanded(self):
        c = Circuit("R_Z(0.5) 0")
        stim_str = str(c.stim_circuit)
        assert "I[" not in stim_str
        assert "S 0" in stim_str

    def test_non_clifford_rotation_preserved(self):
        c = Circuit("R_Z(0.25) 0")
        stim_str = str(c.stim_circuit)
        assert "I[R_Z" in stim_str

    def test_t_gate_preserved(self):
        c = Circuit("T 0")
        stim_str = str(c.stim_circuit)
        assert "[T]" in stim_str

    def test_u3_clifford_expanded(self):
        c = Circuit("U3(0.5, 0.0, 1.0) 0")
        stim_str = str(c.stim_circuit)
        assert "I[" not in stim_str
        assert "H 0" in stim_str

    def test_identity_rotation_becomes_i(self):
        c = Circuit("R_Z(0.0) 0\nH 0")
        stim_str = str(c.stim_circuit)
        assert "I[" not in stim_str
        assert "I 0" in stim_str

    def test_pure_clifford_circuit_unchanged(self):
        c = Circuit("H 0\nCNOT 0 1\nS 0")
        assert c.stim_circuit == c._stim_circ

    def test_mixed_circuit_unitary(self):
        c = Circuit("H 0\nR_Z(0.5) 0\nCNOT 0 1")
        expected = Circuit("H 0\nS 0\nCNOT 0 1")
        np.testing.assert_allclose(c.to_tensor(), expected.to_tensor(), atol=1e-9)

    def test_multi_target_rotation(self):
        c = Circuit("R_Z(1.0) 0 1 2")
        stim_str = str(c.stim_circuit)
        assert "I[" not in stim_str
        assert "Z 0 1 2" in stim_str

    def test_multi_target_u3_stays_compact(self):
        c = Circuit("U3(0.5, 0.5, 1.0) 0 1")
        assert str(c.stim_circuit) == "S_DAG 0 1\nSQRT_X_DAG 0 1"

    @pytest.mark.parametrize(
        "program",
        [
            # stim merges these into one instruction with repeated targets.
            "U3(-1.5, -1.5, -1.0) 0\nU3(-1.5, -1.5, -1.0) 0",
            "U3(0.5, 0.5, 1.0) 0 1\nU3(0.5, 0.5, 1.0) 1 0",
            "R_PAULI(1.0) X0\nR_PAULI(1.0) Z0",
            "R_PAULI(0.5) X0*Z1\nR_PAULI(0.5) Z0*X1\nR_PAULI(0.5) Y2",
            "R_PAULI(-0.5) X0*Y1 Z2\nR_PAULI(-0.5) Z0 X1*X2",
            "R_PAULI(0.5) X0*!Z1\nR_PAULI(0.5) Z0",
        ],
    )
    def test_merged_rotations_preserve_unitary(self, program):
        c = Circuit(program)
        assert c.is_clifford
        ref = stim.Circuit()
        for line in program.splitlines():
            ref += Circuit(line).stim_circuit
        assert c.stim_circuit.to_tableau() == ref.to_tableau()

    def test_merged_spp_dag(self):
        merged = stim.Circuit("SPP_DAG[R_PAULI(theta=1.0*pi)] X0 Z0*Y1")
        expanded = expand_clifford_rotations(merged)
        assert expanded.to_tableau() == stim.Circuit("X 0\nZ 0\nY 1").to_tableau()

    def test_merged_rotation_in_repeat_block(self):
        line = "U3(0.5, 0.5, 1.0) 0"
        c = Circuit(f"REPEAT 2 {{\n{line}\n{line}\n}}")
        ref = Circuit(line).stim_circuit * 4
        assert c.stim_circuit.flattened().to_tableau() == ref.to_tableau()

    @pytest.mark.parametrize("theta_idx", range(4))
    @pytest.mark.parametrize("phi_idx", range(4))
    @pytest.mark.parametrize("lam_idx", range(4))
    def test_merged_u3_all_half_pi(self, theta_idx, phi_idx, lam_idx):
        line = f"U3({theta_idx / 2}, {phi_idx / 2}, {lam_idx / 2}) 0"
        c = Circuit(f"{line}\n{line}")
        gates = parametric_to_clifford_gates(
            "U3",
            {
                "theta": Fraction(theta_idx, 2),
                "phi": Fraction(phi_idx, 2),
                "lambda": Fraction(lam_idx, 2),
            },
        )
        assert gates is not None
        ref = stim.Circuit()
        for g in gates + gates:
            ref.append(g, [0])  # type: ignore
        assert c.stim_circuit.to_tableau() == ref.to_tableau()


class TestIsCliffordUserTags:
    """Identity instructions with non-parametric user tags (e.g. ``I[mytag]``)
    are still Clifford — only parametric rotation tags can flip the result.
    """

    @pytest.mark.parametrize(
        "program",
        [
            "I[mytag] 0",
            "I[abc123] 0",
            "H 0\nI[mytag] 0\nCNOT 0 1\nM 0 1",
        ],
    )
    def test_user_tag_is_clifford(self, program):
        assert Circuit(program).is_clifford
