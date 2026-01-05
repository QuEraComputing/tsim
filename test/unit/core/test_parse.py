import stim
from numpy.testing import assert_allclose

from tsim.core.parse import parse_stim_circuit


class TestParseCorrelatedError:
    """Tests for parsing correlated error circuits."""

    def test_parse_single_correlated_error(self):
        """Parse a single CORRELATED_ERROR instruction."""
        circuit = stim.Circuit("CORRELATED_ERROR(0.1) X0 Z1")
        b = parse_stim_circuit(circuit)

        assert b.num_error_bits == 1
        assert len(b.channel_probs) == 1
        assert b.channel_probs[0].shape == (2,)
        assert_allclose(b.channel_probs[0], [0.9, 0.1])

    def test_parse_correlated_error_chain(self):
        """Parse a chain of CORRELATED_ERROR + ELSE_CORRELATED_ERROR."""
        circuit = stim.Circuit(
            """
            CORRELATED_ERROR(0.1) X0 Z1
            ELSE_CORRELATED_ERROR(0.2) X0 Z2
            """
        )
        b = parse_stim_circuit(circuit)

        assert b.num_error_bits == 2
        assert len(b.channel_probs) == 1
        assert b.channel_probs[0].shape == (4,)
        # P(00) = 0.9 * 0.8 = 0.72
        # P(01) = 0.1
        # P(10) = 0.9 * 0.2 = 0.18
        assert_allclose(b.channel_probs[0], [0.72, 0.1, 0.18, 0.0])

    def test_parse_two_separate_chains(self):
        """Parse two separate correlated error chains."""
        circuit = stim.Circuit(
            """
            CORRELATED_ERROR(0.1) X0
            ELSE_CORRELATED_ERROR(0.2) Z0
            CORRELATED_ERROR(0.3) X1
            """
        )
        b = parse_stim_circuit(circuit)

        # First chain: 2 bits, second chain: 1 bit
        assert b.num_error_bits == 3
        assert len(b.channel_probs) == 2
        assert b.channel_probs[0].shape == (4,)
        assert b.channel_probs[1].shape == (2,)

    def test_parse_y_error(self):
        """Parse a Y error (should create both X and Z vertices)."""
        circuit = stim.Circuit("CORRELATED_ERROR(0.1) Y0")
        b = parse_stim_circuit(circuit)

        assert b.num_error_bits == 1
        # Count vertices with error phases (stored in _phaseVars)
        e0_count = 0
        for v in b.graph.vertices():
            phase_vars = b.graph._phaseVars.get(v, set())
            if "e0" in phase_vars:
                e0_count += 1

        # Y error should create 2 vertices (X and Z), both with same phase
        assert e0_count == 2


class TestCorrelatedErrorGraph:
    """Tests for graph structure with correlated errors."""

    def test_error_vertices_have_e_phase(self):
        """Verify error vertices use 'e' prefix after finalization."""
        circuit = stim.Circuit("CORRELATED_ERROR(0.1) X0")
        b = parse_stim_circuit(circuit)

        # Find vertices with error phases (stored in _phaseVars)
        found_e0 = False
        for v in b.graph.vertices():
            phase_vars = b.graph._phaseVars.get(v, set())
            if "e0" in phase_vars:
                found_e0 = True
                break

        assert found_e0

    def test_no_c_phases_after_finalization(self):
        """Verify no 'c' prefixed phases remain after finalization."""
        circuit = stim.Circuit(
            """
            CORRELATED_ERROR(0.1) X0 Z1
            ELSE_CORRELATED_ERROR(0.2) Y0
        """
        )
        b = parse_stim_circuit(circuit)

        # Check that no vertices have "c" phases (stored in _phaseVars)
        for v in b.graph.vertices():
            phase_vars = b.graph._phaseVars.get(v, set())
            for var in phase_vars:
                if isinstance(var, str):
                    assert not var.startswith("c"), f"Found unfinalized phase: {var}"

    def test_chain_multiple_qubits(self):
        """Test a chain affecting multiple qubits."""
        circuit = stim.Circuit(
            """
            CORRELATED_ERROR(0.2) X1 Y2
            ELSE_CORRELATED_ERROR(0.25) Z2 Z3
            ELSE_CORRELATED_ERROR(0.33333333333) X1 Y2 Z3
            """
        )
        b = parse_stim_circuit(circuit)

        assert b.num_error_bits == 3
        assert len(b.channel_probs) == 1
        assert b.channel_probs[0].shape == (8,)

        # Verify the probability distribution
        probs = b.channel_probs[0]
        assert_allclose(probs[0], 0.4, rtol=1e-5)  # No error
        assert_allclose(probs[1], 0.2, rtol=1e-5)  # First error
        assert_allclose(probs[2], 0.2, rtol=1e-5)  # Second error
        assert_allclose(probs[4], 0.2, rtol=1e-5)  # Third error


class TestCorrelatedErrorState:
    """Tests for correlated error state management."""

    def test_state_reset_after_finalization(self):
        """Verify state is reset after finalization."""
        circuit = stim.Circuit("CORRELATED_ERROR(0.1) X0")
        b = parse_stim_circuit(circuit)

        # After parsing, state should be reset
        assert b.num_correlated_error_bits == 0
        assert b.correlated_error_probs == []

    def test_empty_circuit(self):
        """Test parsing an empty circuit."""
        circuit = stim.Circuit("")
        b = parse_stim_circuit(circuit)

        assert b.num_error_bits == 0
        assert len(b.channel_probs) == 0

    def test_mixed_errors(self):
        """Test correlated errors mixed with regular errors."""
        circuit = stim.Circuit(
            """
            X_ERROR(0.05) 0
            CORRELATED_ERROR(0.1) X1 Z2
            Z_ERROR(0.03) 1
            """
        )
        b = parse_stim_circuit(circuit)

        # X_ERROR: 1 bit, CORRELATED_ERROR: 1 bit, Z_ERROR: 1 bit
        assert b.num_error_bits == 3
        assert len(b.channel_probs) == 3
