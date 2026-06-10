"""Tests for parametric two-qubit Pauli rotation gates: R_XX, R_YY, R_ZZ, R_PAULI.

Acceptance criteria from the issue:
  - Clifford-angle cases match stim's reference behaviour.
  - Arbitrary-angle sampling validated against analytic probabilities.
  - Inverse + dagger round-trip correctly.
  - Shorthand text → stim → shorthand round-trips.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import stim

from tsim import Circuit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unitaries_equal_up_to_global_phase(
    a: np.ndarray, b: np.ndarray, atol: float = 1e-8
) -> bool:
    """Return True iff a and b are equal up to a global phase factor."""
    if a.shape != b.shape:
        return False
    # Find first non-trivially-small entry to get the phase
    flat_a = a.flatten()
    flat_b = b.flatten()
    idx = np.argmax(np.abs(flat_b))
    if abs(flat_b[idx]) < 1e-12:
        return bool(np.allclose(a, 0, atol=atol))
    phase = flat_a[idx] / flat_b[idx]
    return bool(np.allclose(a, phase * b, atol=atol))


def _analytic_r_xx_matrix(alpha: float) -> np.ndarray:
    """Return the 4×4 unitary of R_XX(alpha) = exp(-i alpha pi/2 XX).

    R_XX(α) = cos(α π/2) I⊗I − i sin(α π/2) X⊗X
    in the computational basis {|00⟩, |01⟩, |10⟩, |11⟩}.
    """
    c = math.cos(alpha * math.pi / 2)
    s = math.sin(alpha * math.pi / 2)
    return np.array(
        [
            [c, 0, 0, -1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [-1j * s, 0, 0, c],
        ],
        dtype=complex,
    )


def _analytic_r_zz_matrix(alpha: float) -> np.ndarray:
    """Return the 4×4 unitary of R_ZZ(alpha) = exp(-i alpha pi/2 ZZ)."""
    c = math.cos(alpha * math.pi / 2)
    s = math.sin(alpha * math.pi / 2)
    return np.diag(
        [
            c - 1j * s,
            c + 1j * s,
            c + 1j * s,
            c - 1j * s,
        ]
    )


# ---------------------------------------------------------------------------
# 1. Shorthand parsing and round-trips
# ---------------------------------------------------------------------------


class TestShorthandRoundTrip:
    def test_r_xx_shorthand_parses(self):
        c = Circuit("R_XX(0.25) 0 1")
        assert str(c) == "R_XX(0.25) 0 1"

    def test_r_yy_shorthand_parses(self):
        c = Circuit("R_YY(-0.5) 2 3")
        assert str(c) == "R_YY(-0.5) 2 3"

    def test_r_zz_shorthand_parses(self):
        c = Circuit("R_ZZ(0.3) 1 2")
        assert str(c) == "R_ZZ(0.3) 1 2"

    def test_r_pauli_shorthand_parses(self):
        c = Circuit("R_PAULI(0.25) X0*Y1*Z2")
        assert str(c) == "R_PAULI(0.25) X0*Y1*Z2"

    def test_r_xx_round_trips_through_stim(self):
        c1 = Circuit("R_XX(0.25) 0 1")
        c2 = Circuit("SPP[R_XX(theta=0.25*pi)] X0*X1")
        assert c1._stim_circ == c2._stim_circ

    def test_r_yy_round_trips_through_stim(self):
        c1 = Circuit("R_YY(0.3) 0 1")
        c2 = Circuit("SPP[R_YY(theta=0.3*pi)] Y0*Y1")
        assert c1._stim_circ == c2._stim_circ

    def test_r_zz_round_trips_through_stim(self):
        c1 = Circuit("R_ZZ(-0.1) 0 1")
        c2 = Circuit("SPP[R_ZZ(theta=-0.1*pi)] Z0*Z1")
        assert c1._stim_circ == c2._stim_circ

    def test_r_pauli_round_trips_through_stim(self):
        c1 = Circuit("R_PAULI(0.25) X0*Y1")
        c2 = Circuit("SPP[R_PAULI(theta=0.25*pi)] X0*Y1")
        assert c1._stim_circ == c2._stim_circ

    def test_str_round_trip_r_xx(self):
        c = Circuit("R_XX(0.25) 0 1")
        assert Circuit(str(c)) == c

    def test_str_round_trip_r_pauli_multi(self):
        c = Circuit("R_PAULI(0.3) X0*X1*X2")
        assert Circuit(str(c)) == c

    def test_r_pauli_three_qubits_str(self):
        c = Circuit("R_PAULI(0.3) X0*X1*X2")
        # Must NOT collapse to R_XX shorthand for >2 qubits
        assert str(c) == "R_PAULI(0.3) X0*X1*X2"

    def test_negative_angle_round_trip(self):
        c = Circuit("R_ZZ(-0.3) 0 1")
        assert Circuit(str(c)) == c

    def test_scientific_notation_angle(self):
        c = Circuit("R_XX(0.25) 0 1")
        c2 = Circuit("SPP[R_XX(theta=2.5e-1*pi)] X0*X1")
        assert c == c2


# ---------------------------------------------------------------------------
# 2. Circuit.append API
# ---------------------------------------------------------------------------


class TestAppendAPI:
    def test_append_r_xx(self):
        c = Circuit()
        c.append("R_XX", [0, 1], arg=0.25)
        assert str(c) == "R_XX(0.25) 0 1"

    def test_append_r_yy(self):
        c = Circuit()
        c.append("R_YY", [2, 3], arg=-0.5)
        assert str(c) == "R_YY(-0.5) 2 3"

    def test_append_r_zz(self):
        c = Circuit()
        c.append("R_ZZ", [0, 1], arg=[0.3])
        assert str(c) == "R_ZZ(0.3) 0 1"

    def test_append_r_pauli(self):
        c = Circuit()
        c.append(
            "R_PAULI",
            [stim.target_x(0), stim.target_combiner(), stim.target_y(1)],
            arg=0.25,
        )
        assert str(c) == "R_PAULI(0.25) X0*Y1"

    def test_append_r_xx_no_angle_raises(self):
        c = Circuit()
        with pytest.raises(ValueError, match="angle"):
            c.append("R_XX", [0, 1])

    def test_append_r_xx_duplicate_qubits_raises(self):
        c = Circuit()
        with pytest.raises(ValueError, match="Duplicate target qubits"):
            c.append("R_XX", [3, 3], arg=0.5)

    def test_append_r_xx_wrong_target_count_raises(self):
        c = Circuit()
        with pytest.raises(ValueError, match="exactly two"):
            c.append("R_XX", [0], arg=0.5)

    def test_append_r_pauli_no_angle_raises(self):
        c = Circuit()
        with pytest.raises(ValueError, match="angle"):
            c.append(
                "R_PAULI",
                [stim.target_x(0), stim.target_combiner(), stim.target_z(1)],
            )


# ---------------------------------------------------------------------------
# 3. Clifford-angle parity with stim
# ---------------------------------------------------------------------------


class TestCliffordAngleParity:
    """At integer alpha, R_PP(alpha) = exp(-i alpha pi/2 PP).

    alpha=0 (mod 4): identity (up to global phase)
    alpha=1 (mod 4): PP rotation by π/2 — equivalent to SPP
    alpha=2 (mod 4): PP^2 = I applied with a phase, effectively II (up to gphase)
    alpha=3 (mod 4): inverse of SPP
    """

    @pytest.mark.parametrize("gate", ["R_XX", "R_YY", "R_ZZ"])
    @pytest.mark.parametrize("alpha", [0, 1, 2, 3])
    def test_clifford_angle_matches_stim(self, gate: str, alpha: int):
        pauli = gate[2]  # 'X', 'Y', or 'Z'
        # For even alpha, result is identity (mod global phase); for odd, it's PP.
        if alpha % 2 == 0:
            stim_prog = "I 0\nI 1"
        else:
            stim_prog = f"{pauli} 0\n{pauli} 1"
        c = Circuit(f"{gate}({alpha}) 0 1")
        ref = np.array(
            stim.Circuit(stim_prog).to_tableau().to_unitary_matrix(endian="big")
        )
        assert unitaries_equal_up_to_global_phase(c.to_matrix(), ref)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_r_xx_is_clifford_for_half_pi_multiples(self, alpha: float):
        c = Circuit(f"R_XX({alpha}) 0 1")
        assert c.is_clifford

    @pytest.mark.parametrize("alpha", [0.3, 0.1, 0.7, 1.3])
    def test_r_xx_not_clifford_for_arbitrary_angles(self, alpha: float):
        c = Circuit(f"R_XX({alpha}) 0 1")
        assert not c.is_clifford

    def test_r_pauli_clifford_half_pi(self):
        c = Circuit("R_PAULI(0.5) X0*Z1")
        assert c.is_clifford

    def test_r_pauli_not_clifford_arbitrary(self):
        c = Circuit("R_PAULI(0.3) X0*Z1")
        assert not c.is_clifford


# ---------------------------------------------------------------------------
# 4. Correctness against analytic matrices
# ---------------------------------------------------------------------------


class TestAnalyticCorrectness:
    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.5, 0.7, 1.0, 1.3, 1.5])
    def test_r_xx_matches_analytic_formula(self, alpha: float):
        c = Circuit(f"R_XX({alpha}) 0 1")
        expected = _analytic_r_xx_matrix(alpha)
        assert unitaries_equal_up_to_global_phase(c.to_matrix(), expected), (
            f"R_XX({alpha}) unitary mismatch"
        )

    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.5, 0.7, 1.0, 1.3, 1.5])
    def test_r_zz_matches_analytic_formula(self, alpha: float):
        c = Circuit(f"R_ZZ({alpha}) 0 1")
        expected = _analytic_r_zz_matrix(alpha)
        assert unitaries_equal_up_to_global_phase(c.to_matrix(), expected), (
            f"R_ZZ({alpha}) unitary mismatch"
        )

    def test_r_yy_matches_r_xx_conjugated_by_s(self):
        """R_YY(α) = (S⊗S) R_XX(α) (S†⊗S†), so unitaries differ by S-conjugation."""
        alpha = 0.3
        c_yy = Circuit(f"R_YY({alpha}) 0 1")
        # Validate via: exp(-i α π/2 YY) where Y = S X S†
        # So R_YY(α) conjugated by S†⊗S† on both sides should give R_XX(α).
        u_yy = c_yy.to_matrix()
        # Build S and S† matrices
        s_mat = np.array([[1, 0], [0, 1j]], dtype=complex)
        s_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
        s2 = np.kron(s_mat, s_mat)
        s2d = np.kron(s_dag, s_dag)
        u_mapped = s2d @ u_yy @ s2
        expected_xx = _analytic_r_xx_matrix(alpha)
        assert unitaries_equal_up_to_global_phase(u_mapped, expected_xx)

    def test_r_pauli_single_qubit_reduces_to_r_z(self):
        """R_PAULI with a single Z0 should equal R_Z(alpha)."""
        alpha = 0.37
        c_rpauli = Circuit(f"R_PAULI({alpha}) Z0")
        c_rz = Circuit(f"R_Z({alpha}) 0")
        assert unitaries_equal_up_to_global_phase(c_rpauli.to_matrix(), c_rz.to_matrix())

    def test_r_pauli_single_x_reduces_to_r_x(self):
        alpha = 0.42
        c_rpauli = Circuit(f"R_PAULI({alpha}) X0")
        c_rx = Circuit(f"R_X({alpha}) 0")
        assert unitaries_equal_up_to_global_phase(c_rpauli.to_matrix(), c_rx.to_matrix())


# ---------------------------------------------------------------------------
# 5. Inverse and dagger
# ---------------------------------------------------------------------------


class TestInverse:
    @pytest.mark.parametrize("alpha", [0.25, -0.3, 0.5, 1.0, 1.37])
    def test_r_xx_inverse_is_identity(self, alpha: float):
        c = Circuit(f"R_XX({alpha}) 0 1")
        c_inv = c.inverse()
        combined = (c + c_inv).to_matrix()
        n = combined.shape[0]
        phase = combined[0, 0]
        assert np.allclose(combined, phase * np.eye(n), atol=1e-8)

    @pytest.mark.parametrize("alpha", [0.25, -0.3, 0.5, 1.37])
    def test_r_yy_inverse_is_identity(self, alpha: float):
        c = Circuit(f"R_YY({alpha}) 0 1")
        c_inv = c.inverse()
        combined = (c + c_inv).to_matrix()
        n = combined.shape[0]
        phase = combined[0, 0]
        assert np.allclose(combined, phase * np.eye(n), atol=1e-8)

    @pytest.mark.parametrize("alpha", [0.25, -0.3, 0.5, 1.37])
    def test_r_zz_inverse_is_identity(self, alpha: float):
        c = Circuit(f"R_ZZ({alpha}) 0 1")
        c_inv = c.inverse()
        combined = (c + c_inv).to_matrix()
        n = combined.shape[0]
        phase = combined[0, 0]
        assert np.allclose(combined, phase * np.eye(n), atol=1e-8)

    @pytest.mark.parametrize("alpha", [0.25, -0.3, 0.5, 0.123])
    def test_r_pauli_inverse_is_identity(self, alpha: float):
        c = Circuit(f"R_PAULI({alpha}) X0*Y1*Z2")
        c_inv = c.inverse()
        combined = (c + c_inv).to_matrix()
        n = combined.shape[0]
        phase = combined[0, 0]
        assert np.allclose(combined, phase * np.eye(n), atol=1e-8)

    def test_inverse_r_xx_negates_rotation(self):
        """Inverse of R_XX(alpha) should apply the opposite rotation.

        Stim represents the inverse as SPP_DAG with the same theta tag.
        The key thing is that the combined circuit C + C^-1 is identity.
        """
        alpha = 0.3
        c = Circuit(f"R_XX({alpha}) 0 1")
        c_inv = c.inverse()
        # The inverse is stored as SPP_DAG (stim's convention), same theta
        assert "SPP_DAG" in str(c_inv._stim_circ)
        # And the combination is identity
        combined = (c + c_inv).to_matrix()
        n = combined.shape[0]
        phase = combined[0, 0]
        assert np.allclose(combined, phase * np.eye(n), atol=1e-8)

    def test_r_pauli_spp_dag_in_stim_round_trips(self):
        """SPP_DAG with R_PAULI tag from stim must also invert correctly."""
        inner = stim.Circuit("SPP_DAG[R_PAULI(theta=0.25*pi)] X0")
        c = Circuit.from_stim_program(inner)
        c_inv = c.inverse()
        combined = (c + c_inv).to_matrix()
        n = combined.shape[0]
        phase = combined[0, 0]
        assert np.allclose(combined, phase * np.eye(n), atol=1e-8)

    def test_r_xx_inverse_str(self):
        """str(R_XX(α).inverse()) should be parseable and produce the right gate."""
        c = Circuit("R_XX(0.25) 0 1")
        c_inv = c.inverse()
        # The inverse must be parseable
        c_inv2 = Circuit(str(c_inv))
        assert unitaries_equal_up_to_global_phase(c_inv.to_matrix(), c_inv2.to_matrix())


# ---------------------------------------------------------------------------
# 6. Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_duplicate_qubits_in_shorthand_r_xx_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Circuit("R_XX(0.5) 3 3")

    def test_duplicate_qubits_in_shorthand_r_yy_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Circuit("R_YY(0.5) 0 0")

    def test_duplicate_qubits_in_shorthand_r_zz_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Circuit("R_ZZ(0.5) 5 5")


# ---------------------------------------------------------------------------
# 7. Mixed circuits and composition
# ---------------------------------------------------------------------------


class TestMixedCircuits:
    def test_r_xx_composed_with_cnot(self):
        """Composition with a CNOT should produce a valid matrix."""
        c = Circuit("R_XX(0.25) 0 1\nCNOT 0 1")
        mat = c.to_matrix()
        assert mat.shape == (4, 4)
        assert np.allclose(mat @ mat.conj().T, np.eye(4), atol=1e-8)

    def test_r_pauli_three_qubit_circuit(self):
        """R_PAULI on three qubits should produce an 8×8 unitary."""
        c = Circuit("R_PAULI(0.3) X0*Y1*Z2")
        mat = c.to_matrix()
        assert mat.shape == (8, 8)
        assert np.allclose(mat @ mat.conj().T, np.eye(8), atol=1e-8)

    def test_r_xx_twice_same_angle_opposite_sign(self):
        """R_XX(α) followed by R_XX(-α) should be identity."""
        alpha = 0.37
        c = Circuit(f"R_XX({alpha}) 0 1\nR_XX(-{alpha}) 0 1")
        mat = c.to_matrix()
        n = mat.shape[0]
        phase = mat[0, 0]
        assert np.allclose(mat, phase * np.eye(n), atol=1e-8)

    def test_r_pauli_in_repeat_block(self):
        """R_PAULI inside a REPEAT block must survive the full pipeline."""
        c = Circuit("REPEAT 3 {\n  R_ZZ(0.1) 0 1\n}")
        mat = c.to_matrix()
        # Three repetitions of R_ZZ(0.1) should equal R_ZZ(0.3)
        expected = Circuit("R_ZZ(0.3) 0 1").to_matrix()
        assert unitaries_equal_up_to_global_phase(mat, expected)

    def test_r_xx_r_zz_mixed(self):
        """R_XX followed by R_ZZ: result is unitary."""
        c = Circuit("R_XX(0.25) 0 1\nR_ZZ(0.13) 0 1")
        mat = c.to_matrix()
        assert np.allclose(mat @ mat.conj().T, np.eye(4), atol=1e-8)
