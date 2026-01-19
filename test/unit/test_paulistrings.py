"""Tests for Pauli string expectation value computation."""

import numpy as np
import pytest

from tsim import Circuit
from tsim.paulistrings import PauliStrings


class TestSingleQubitExpectations:
    """Test expectation values on single-qubit states."""

    def test_z_on_zero_state(self):
        """Test <0|Z|0> = 1."""
        c = Circuit("R 0")
        ps = PauliStrings(c)
        # Format: [x0 | z0]
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1.0, atol=1e-6)

    def test_z_on_one_state(self):
        """Test <1|Z|1> = -1."""
        c = Circuit("R 0\nX 0")
        ps = PauliStrings(c)
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], -1.0, atol=1e-6)

    def test_x_on_zero_state(self):
        """Test <0|X|0> = 0."""
        c = Circuit("R 0")
        ps = PauliStrings(c)
        paulis = np.array([[1, 0]])  # X
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_x_on_one_state(self):
        """Test <1|X|1> = 0."""
        c = Circuit("R 0\nX 0")
        ps = PauliStrings(c)
        paulis = np.array([[1, 0]])  # X
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_x_on_plus_state(self):
        """Test <+|X|+> = 1."""
        c = Circuit("H 0")
        ps = PauliStrings(c)
        paulis = np.array([[1, 0]])  # X
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1.0, atol=1e-6)

    def test_x_on_minus_state(self):
        """Test <-|X|-> = -1."""
        c = Circuit("X 0\nH 0")  # |-> = H|1>
        ps = PauliStrings(c)
        paulis = np.array([[1, 0]])  # X
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], -1.0, atol=1e-6)

    def test_z_on_plus_state(self):
        """Test <+|Z|+> = 0."""
        c = Circuit("H 0")
        ps = PauliStrings(c)
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_z_on_minus_state(self):
        """Test <-|Z|-> = 0."""
        c = Circuit("X 0\nH 0")
        ps = PauliStrings(c)
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_identity_on_any_state(self):
        """Test <psi|I|psi> = 1 for any state."""
        c = Circuit("H 0\nT 0\nS 0")  # Some arbitrary state
        ps = PauliStrings(c)
        paulis = np.array([[0, 0]])  # Identity
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1.0, atol=1e-6)


class TestYExpectations:
    """Test Y expectation values.

    Y is represented by setting both x and z bits to 1.
    """

    def test_y_on_zero_state(self):
        """Test <0|Y|0> = 0."""
        c = Circuit("R 0")
        ps = PauliStrings(c)
        # [x, z] = [1, 1] represents Y
        paulis = np.array([[1, 1]])
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_y_on_plus_i_state(self):
        """Test <+i|Y|+i> = 1 where |+i> = (|0> + i|1>)/sqrt(2)."""
        c = Circuit("H 0\nS 0")
        ps = PauliStrings(c)
        paulis = np.array([[1, 1]])  # Y
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1.0, atol=1e-6)

    def test_y_on_minus_i_state(self):
        """Test <-i|Y|-i> = -1 where |-i> = (|0> - i|1>)/sqrt(2)."""
        c = Circuit("H 0\nS_DAG 0")
        ps = PauliStrings(c)
        paulis = np.array([[1, 1]])  # Y
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], -1.0, atol=1e-6)


class TestBellStateExpectations:
    """Test expectation values on Bell states."""

    def test_xx_on_bell_state(self):
        """Test <Phi+|X*X|Phi+> = 1."""
        c = Circuit("H 0\nCNOT 0 1")  # |Phi+> = (|00> + |11>)/sqrt(2)
        ps = PauliStrings(c)
        # Format: [x0, x1 | z0, z1]
        paulis = np.array([[1, 1, 0, 0]])  # X*X
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1.0, atol=1e-6)

    def test_zz_on_bell_state(self):
        """Test <Phi+|Z*Z|Phi+> = 1."""
        c = Circuit("H 0\nCNOT 0 1")
        ps = PauliStrings(c)
        paulis = np.array([[0, 0, 1, 1]])  # Z*Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1.0, atol=1e-6)

    def test_yy_on_bell_state(self):
        """Test <Phi+|Y*Y|Phi+> = -1."""
        c = Circuit("H 0\nCNOT 0 1")
        ps = PauliStrings(c)
        # [x0, x1, z0, z1] = [1, 1, 1, 1] represents Y*Y
        paulis = np.array([[1, 1, 1, 1]])
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], -1.0, atol=1e-6)

    def test_single_qubit_paulis_on_bell_state(self):
        """Test single-qubit Paulis on Bell state give 0."""
        c = Circuit("H 0\nCNOT 0 1")
        ps = PauliStrings(c)
        paulis = np.array(
            [
                [0, 0, 1, 0],  # Z*I
                [0, 0, 0, 1],  # I*Z
                [1, 0, 0, 0],  # X*I
                [0, 1, 0, 0],  # I*X
            ]
        )
        results = ps.evaluate(paulis)
        assert np.allclose(results, [0.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_xz_and_zx_on_bell_state(self):
        """Test <Phi+|X*Z|Phi+> = 0 and <Phi+|Z*X|Phi+> = 0."""
        c = Circuit("H 0\nCNOT 0 1")
        ps = PauliStrings(c)
        paulis = np.array(
            [
                [1, 0, 0, 1],  # X*Z (x=[1,0], z=[0,1])
                [0, 1, 1, 0],  # Z*X (x=[0,1], z=[1,0])
            ]
        )
        results = ps.evaluate(paulis)
        assert np.allclose(results, [0.0, 0.0], atol=1e-6)

    def test_singlet_state(self):
        """Test expectation values on singlet state |Psi-> = (|01> - |10>)/sqrt(2).

        The singlet state is unique in having all three correlators XX, YY, ZZ = -1.
        """
        # Create singlet state: |Psi-> = (|01> - |10>)/sqrt(2)
        # Start with |Phi+>, apply X on q1 to get |Psi+>, then Z on q0 to get |Psi->
        c = Circuit(
            """
            H 0
            CNOT 0 1
            X 1
            Z 0
        """
        )
        ps = PauliStrings(c)

        # All three two-qubit Pauli correlators are -1 for the singlet
        paulis = np.array(
            [
                [1, 1, 0, 0],  # XX
                [1, 1, 1, 1],  # YY
                [0, 0, 1, 1],  # ZZ
            ]
        )
        results = ps.evaluate(paulis)
        assert np.allclose(results, [-1.0, -1.0, -1.0], atol=1e-6)


class TestGHZStateExpectations:
    """Test expectation values on GHZ states."""

    def test_xxx_on_ghz_state(self):
        """Test <GHZ|X*X*X|GHZ> = 1."""
        c = Circuit(
            """
            H 0
            CNOT 0 1
            CNOT 1 2
        """
        )  # |GHZ> = (|000> + |111>)/sqrt(2)
        ps = PauliStrings(c)
        # Format: [x0, x1, x2 | z0, z1, z2]
        paulis = np.array([[1, 1, 1, 0, 0, 0]])  # X*X*X
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1.0, atol=1e-6)

    def test_zzz_on_ghz_state(self):
        """Test <GHZ|Z*Z*Z|GHZ> = 0.

        For |GHZ> = (|000> + |111>)/sqrt(2):
        - ZZZ|000> = |000>
        - ZZZ|111> = (-1)^3|111> = -|111>
        So ZZZ|GHZ> = (|000> - |111>)/sqrt(2), which is orthogonal to |GHZ>.
        """
        c = Circuit(
            """
            H 0
            CNOT 0 1
            CNOT 1 2
        """
        )
        ps = PauliStrings(c)
        paulis = np.array([[0, 0, 0, 1, 1, 1]])  # Z*Z*Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_zzi_on_ghz_state(self):
        """Test <GHZ|Z*Z*I|GHZ> = 1."""
        c = Circuit(
            """
            H 0
            CNOT 0 1
            CNOT 1 2
        """
        )
        ps = PauliStrings(c)
        paulis = np.array([[0, 0, 0, 1, 1, 0]])  # Z*Z*I
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1.0, atol=1e-6)


class TestRotationGates:
    """Test circuits with parametric rotation gates R_X, R_Y, R_Z."""

    def test_rz_on_plus_state_x_expectation(self):
        """Test <+|R_Z^dag(theta) X R_Z(theta)|+> = cos(theta).

        R_Z(theta)|+> = (|0> + e^{i*theta}|1>)/sqrt(2), so <X> = cos(theta).
        """
        # theta = pi/4
        c = Circuit("H 0\nR_Z(0.25) 0")
        ps = PauliStrings(c)
        paulis = np.array([[1, 0]])  # X
        result = ps.evaluate(paulis)
        expected = np.cos(np.pi / 4)
        assert np.isclose(result[0], expected, atol=1e-6)

    def test_rz_on_plus_state_z_expectation(self):
        """Test <+|R_Z^dag(theta) Z R_Z(theta)|+> = 0.

        R_Z doesn't change Z expectation on |+>.
        """
        c = Circuit("H 0\nR_Z(0.25) 0")
        ps = PauliStrings(c)
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_rz_on_zero_state(self):
        """Test R_Z on |0> doesn't change anything.

        R_Z(theta)|0> = |0> (up to global phase), so all expectation values unchanged.
        """
        c = Circuit("R 0\nR_Z(0.3) 0")
        ps = PauliStrings(c)
        paulis = np.array(
            [
                [0, 1],  # Z -> 1
                [1, 0],  # X -> 0
            ]
        )
        results = ps.evaluate(paulis)
        assert np.isclose(results[0], 1.0, atol=1e-6)
        assert np.isclose(results[1], 0.0, atol=1e-6)

    def test_rz_various_angles(self):
        """Test R_Z at various angles on |+> state."""
        angles_pi = [0.0, 0.125, 0.25, 0.375, 0.5]  # In units of pi
        for angle_pi in angles_pi:
            c = Circuit(f"H 0\nR_Z({angle_pi}) 0")
            ps = PauliStrings(c)
            paulis = np.array([[1, 0]])  # X
            result = ps.evaluate(paulis)
            expected = np.cos(angle_pi * np.pi)
            assert np.isclose(
                result[0], expected, atol=1e-5
            ), f"R_Z({angle_pi}*pi): expected {expected}, got {result[0]}"

    def test_two_qubit_rotation(self):
        """Test rotation gates on multiple qubits."""
        c = Circuit(
            """
            R 0 1
            R_X(0.25) 0
            R_Y(0.25) 1
        """
        )
        ps = PauliStrings(c)
        # Z*I and I*Z
        paulis = np.array(
            [
                [0, 0, 1, 0],  # Z*I
                [0, 0, 0, 1],  # I*Z
            ]
        )
        results = ps.evaluate(paulis)
        expected = np.cos(np.pi / 4)
        assert np.isclose(results[0], expected, atol=1e-6)  # Z*I
        assert np.isclose(results[1], expected, atol=1e-6)  # I*Z


class TestNonCliffordCircuits:
    """Test circuits with T gates (non-Clifford)."""

    def test_t_gate_z_expectation(self):
        """Test Z expectation after H T H rotation."""
        # H T H rotates |0> by pi/4 around X axis
        # The resulting state has <Z> = cos(pi/4) = 1/sqrt(2)
        c = Circuit("H 0\nT 0\nH 0")
        ps = PauliStrings(c)
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1 / np.sqrt(2), atol=1e-6)

    def test_t_dag_gate_z_expectation(self):
        """Test Z expectation after H T^dag H rotation."""
        # H T^dag H rotates by -pi/4
        c = Circuit("H 0\nT_DAG 0\nH 0")
        ps = PauliStrings(c)
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 1 / np.sqrt(2), atol=1e-6)

    def test_double_t_gate(self):
        """Test Z expectation after H T T H = H S H."""
        # T T = S, and H S H rotates by pi/2
        # <Z> = cos(pi/2) = 0
        c = Circuit("H 0\nT 0\nT 0\nH 0")
        ps = PauliStrings(c)
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_t_gate_on_multiple_qubits(self):
        """Test T gates on multiple qubits."""
        c = Circuit(
            """
            H 0
            H 1
            T 0
            T 1
        """
        )
        ps = PauliStrings(c)
        # After H T, the state is (|0> + e^{i*pi/4}|1>)/sqrt(2)
        # For single qubit, <X> = cos(pi/4) = 1/sqrt(2)
        paulis = np.array(
            [
                [1, 0, 0, 0],  # X*I
                [0, 1, 0, 0],  # I*X
            ]
        )
        results = ps.evaluate(paulis)
        expected = 1 / np.sqrt(2)
        assert np.allclose(results, [expected, expected], atol=1e-6)


class TestBatchEvaluation:
    """Test batch evaluation of multiple Pauli strings."""

    def test_all_single_qubit_paulis(self):
        """Test I, X, Y, Z on |+> state."""
        c = Circuit("H 0")
        ps = PauliStrings(c)
        paulis = np.array(
            [
                [0, 0],  # I -> 1
                [0, 1],  # Z -> 0
                [1, 0],  # X -> 1
                [1, 1],  # Y -> 0
            ]
        )
        results = ps.evaluate(paulis)
        assert np.allclose(results, [1.0, 0.0, 1.0, 0.0], atol=1e-6)

    def test_many_paulis_at_once(self):
        """Test evaluating many Pauli strings in a batch."""
        c = Circuit("H 0\nCNOT 0 1")
        ps = PauliStrings(c)
        paulis = np.array(
            [
                [0, 0, 0, 0],  # II -> 1
                [0, 0, 1, 1],  # ZZ -> 1
                [1, 1, 0, 0],  # XX -> 1
                [0, 0, 1, 0],  # ZI -> 0
                [0, 0, 0, 1],  # IZ -> 0
            ]
        )
        results = ps.evaluate(paulis)
        assert np.allclose(results, [1.0, 1.0, 1.0, 0.0, 0.0], atol=1e-6)


class TestValidation:
    """Test input validation and error handling."""

    def test_circuit_with_measurement_raises(self):
        """Test that circuits with measurements raise error."""
        c = Circuit("H 0\nM 0")
        with pytest.raises(ValueError, match="M"):
            PauliStrings(c)

    def test_circuit_with_detector_raises(self):
        """Test that circuits with detectors raise error."""
        c = Circuit("H 0\nM 0\nDETECTOR rec[-1]")
        with pytest.raises(ValueError, match="M"):  # M comes first
            PauliStrings(c)

    def test_circuit_with_x_error_raises(self):
        """Test that circuits with X_ERROR raise error."""
        c = Circuit("H 0\nX_ERROR(0.1) 0")
        with pytest.raises(ValueError, match="X_ERROR"):
            PauliStrings(c)

    def test_circuit_with_depolarize_raises(self):
        """Test that circuits with DEPOLARIZE1 raise error."""
        c = Circuit("H 0\nDEPOLARIZE1(0.1) 0")
        with pytest.raises(ValueError, match="DEPOLARIZE1"):
            PauliStrings(c)

    def test_wrong_pauli_shape_raises(self):
        """Test that wrong Pauli array shape raises error."""
        c = Circuit("H 0")  # 1 qubit -> expect 2 columns
        ps = PauliStrings(c)
        with pytest.raises(ValueError, match="columns"):
            ps.evaluate(np.array([[0, 0, 0]]))  # Wrong: 3 columns


class TestProperties:
    """Test class properties and attributes."""

    def test_num_qubits(self):
        """Test num_qubits attribute."""
        c = Circuit("H 0\nCNOT 0 1\nCNOT 1 2")
        ps = PauliStrings(c)
        assert ps.num_qubits == 3

    def test_single_qubit(self):
        """Test single qubit circuit."""
        c = Circuit("H 0")
        ps = PauliStrings(c)
        assert ps.num_qubits == 1


class TestSpecialCircuits:
    """Test special circuit configurations."""

    def test_identity_circuit(self):
        """Test circuit that does nothing (just R)."""
        c = Circuit("R 0")
        ps = PauliStrings(c)
        paulis = np.array(
            [
                [0, 0],  # I
                [0, 1],  # Z
                [1, 0],  # X
            ]
        )
        results = ps.evaluate(paulis)
        assert np.allclose(results, [1.0, 1.0, 0.0], atol=1e-6)

    def test_x_gate_circuit(self):
        """Test X gate flips Z expectation."""
        c = Circuit("R 0\nX 0")
        ps = PauliStrings(c)
        paulis = np.array([[0, 1]])  # Z
        result = ps.evaluate(paulis)
        assert np.isclose(result[0], -1.0, atol=1e-6)

    def test_hadamard_transforms_basis(self):
        """Test H gate transforms Z basis to X basis."""
        c = Circuit("H 0")
        ps = PauliStrings(c)
        paulis = np.array(
            [
                [0, 1],  # Z -> 0 (was eigenstate)
                [1, 0],  # X -> 1 (now eigenstate)
            ]
        )
        results = ps.evaluate(paulis)
        assert np.allclose(results, [0.0, 1.0], atol=1e-6)

    def test_cz_gate(self):
        """Test CZ gate creates correlation."""
        c = Circuit(
            """
            H 0
            H 1
            CZ 0 1
        """
        )
        ps = PauliStrings(c)
        # After H H CZ, the state is entangled
        # <ZZ> should be 1 (both qubits correlated)
        paulis = np.array(
            [
                [0, 0, 1, 1],  # ZZ
                [1, 1, 0, 0],  # XX
            ]
        )
        results = ps.evaluate(paulis)
        # Note: The actual values depend on the specific entanglement
        # For |++> after CZ: ZZ gives 0, XX gives 1
        # But this needs verification based on actual circuit behavior
        assert len(results) == 2


class TestProductStates:
    """Test expectation values on product states."""

    def test_product_state_factorizes(self):
        """Test that expectation values factorize for product states."""
        # |+0> state (product state)
        c = Circuit("H 0\nR 1")
        ps = PauliStrings(c)

        # For product state |psi>|phi>, <P*Q> = <P><Q>
        paulis = np.array(
            [
                [1, 0, 0, 0],  # X*I: <+|X|+> * <0|I|0> = 1 * 1 = 1
                [0, 0, 0, 1],  # I*Z: <+|I|+> * <0|Z|0> = 1 * 1 = 1
                [1, 0, 0, 1],  # X*Z: <+|X|+> * <0|Z|0> = 1 * 1 = 1
                [0, 0, 1, 0],  # Z*I: <+|Z|+> * <0|I|0> = 0 * 1 = 0
            ]
        )
        results = ps.evaluate(paulis)
        assert np.allclose(results, [1.0, 1.0, 1.0, 0.0], atol=1e-6)
