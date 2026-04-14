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
        circuit = stim.Circuit("""
            CORRELATED_ERROR(0.1) X0 Z1
            ELSE_CORRELATED_ERROR(0.2) X0 Z2
            """)
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
        circuit = stim.Circuit("""
            CORRELATED_ERROR(0.1) X0
            ELSE_CORRELATED_ERROR(0.2) Z0
            CORRELATED_ERROR(0.3) X1
            """)
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
        circuit = stim.Circuit("""
            CORRELATED_ERROR(0.1) X0 Z1
            ELSE_CORRELATED_ERROR(0.2) Y0
        """)
        b = parse_stim_circuit(circuit)

        # Check that no vertices have "c" phases (stored in _phaseVars)
        for v in b.graph.vertices():
            phase_vars = b.graph._phaseVars.get(v, set())
            for var in phase_vars:
                if isinstance(var, str):
                    assert not var.startswith("c"), f"Found unfinalized phase: {var}"

    def test_chain_multiple_qubits(self):
        """Test a chain affecting multiple qubits."""
        circuit = stim.Circuit("""
            CORRELATED_ERROR(0.2) X1 Y2
            ELSE_CORRELATED_ERROR(0.25) Z2 Z3
            ELSE_CORRELATED_ERROR(0.33333333333) X1 Y2 Z3
            """)
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


class TestParseHeraldedChannels:
    """Tests for parsing HERALDED_PAULI_CHANNEL_1 and HERALDED_ERASE."""

    def test_heralded_pauli_channel_1_structure(self):
        """HERALDED_PAULI_CHANNEL_1 should produce 1 rec entry and 3 error bits."""
        circuit = stim.Circuit("HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 0")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1
        assert b.num_error_bits == 3
        assert len(b.channel_probs) == 1
        assert b.channel_probs[0].shape == (8,)
        assert_allclose(b.channel_probs[0].sum(), 1.0)

    def test_heralded_erase_structure(self):
        """HERALDED_ERASE should produce 1 rec entry and 3 error bits."""
        circuit = stim.Circuit("HERALDED_ERASE(0.04) 0")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1
        assert b.num_error_bits == 3
        assert len(b.channel_probs) == 1
        assert_allclose(b.channel_probs[0][[1, 3, 5, 7]], 0.01)

    def test_heralded_erase_multiple_targets(self):
        """HERALDED_ERASE on multiple targets should produce independent channels."""
        circuit = stim.Circuit("HERALDED_ERASE(0.01) 0 1 2")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 3
        assert b.num_error_bits == 9
        assert len(b.channel_probs) == 3


class TestParseIIError:
    """Tests for parsing II_ERROR instructions with parenthesized arguments."""

    def test_ii_error_with_probability(self):
        """II_ERROR(p) should parse without raising a TypeError."""
        circuit = stim.Circuit("II_ERROR(0.1) 0 1")
        b = parse_stim_circuit(circuit)
        assert set(b.last_vertex) == {0, 1}

    def test_ii_error_without_probability(self):
        """II_ERROR without parens should also parse correctly."""
        circuit = stim.Circuit("II_ERROR 0 1")
        b = parse_stim_circuit(circuit)
        assert set(b.last_vertex) == {0, 1}

    def test_ii_error_multiple_pairs(self):
        """II_ERROR(p) applied to multiple qubit pairs."""
        circuit = stim.Circuit("II_ERROR(0.05) 0 1 2 3")
        b = parse_stim_circuit(circuit)
        assert set(b.last_vertex) == {0, 1, 2, 3}


class TestParseWithRepeatBlocks:
    """Tests for parsing circuits that contain REPEAT blocks."""

    def test_parse_circuit_with_repeat_block(self):
        """parse_stim_circuit should flatten repeat blocks transparently."""
        flat_circuit = stim.Circuit("H 0\nCNOT 0 1\nH 0\nCNOT 0 1\nH 0\nCNOT 0 1")
        repeat_circuit = stim.Circuit("REPEAT 3 {\n    H 0\n    CNOT 0 1\n}")

        b_flat = parse_stim_circuit(flat_circuit)
        b_repeat = parse_stim_circuit(repeat_circuit)

        assert len(b_flat.graph.vertices()) == len(b_repeat.graph.vertices())
        assert list(b_flat.graph.edges()) == list(b_repeat.graph.edges())

    def test_parse_repeat_block_with_measurements(self):
        """Repeat blocks containing measurements should parse correctly."""
        circuit = stim.Circuit("REPEAT 3 {\n    H 0\n    M 0\n}")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 3

    def test_parse_nested_repeat_blocks(self):
        """Nested repeat blocks should be fully flattened by the parser."""
        circuit = stim.Circuit("REPEAT 2 {\n    REPEAT 3 {\n        H 0\n    }\n}")
        flat = stim.Circuit("H 0\nH 0\nH 0\nH 0\nH 0\nH 0")

        b_nested = parse_stim_circuit(circuit)
        b_flat = parse_stim_circuit(flat)

        assert len(b_nested.graph.vertices()) == len(b_flat.graph.vertices())


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
        circuit = stim.Circuit("""
            X_ERROR(0.05) 0
            CORRELATED_ERROR(0.1) X1 Z2
            Z_ERROR(0.03) 1
            """)
        b = parse_stim_circuit(circuit)

        # X_ERROR: 1 bit, CORRELATED_ERROR: 1 bit, Z_ERROR: 1 bit
        assert b.num_error_bits == 3
        assert len(b.channel_probs) == 3


class TestParseMPAD:
    """Tests for parsing MPAD (measurement record padding) instructions."""

    def test_mpad_single_zero(self):
        """MPAD 0 should add one measurement record entry."""
        circuit = stim.Circuit("MPAD 0")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1

    def test_mpad_single_one(self):
        """MPAD 1 should add one measurement record entry."""
        circuit = stim.Circuit("MPAD 1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1

    def test_mpad_multiple_targets(self):
        """MPAD with multiple targets should add one record per target."""
        circuit = stim.Circuit("MPAD 0 1 0 1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 4

    def test_mpad_mixed_with_measurements(self):
        """MPAD records should interleave correctly with regular measurements."""
        circuit = stim.Circuit("""
            M 0
            MPAD 1
            M 1
        """)
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 3

    def test_mpad_in_repeat_block(self):
        """MPAD inside a repeat block should be expanded correctly."""
        circuit = stim.Circuit("REPEAT 3 {\n    MPAD 0 1\n}")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 6


class TestParseMXXMYYMZZ:
    """Tests for parsing MXX, MYY, MZZ instructions."""

    def test_mxx_single_pair(self):
        """MXX on one pair should add one measurement record entry."""
        circuit = stim.Circuit("MXX 0 1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1

    def test_myy_single_pair(self):
        """MYY on one pair should add one measurement record entry."""
        circuit = stim.Circuit("MYY 0 1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1

    def test_mzz_single_pair(self):
        """MZZ on one pair should add one measurement record entry."""
        circuit = stim.Circuit("MZZ 0 1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1

    def test_mxx_multiple_pairs(self):
        """MXX with multiple pairs should add one record per pair."""
        circuit = stim.Circuit("MXX 0 1 2 3 4 5")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 3

    def test_mzz_mixed_with_measurements(self):
        """MZZ records should interleave correctly with regular measurements."""
        circuit = stim.Circuit("""
            M 0
            MZZ 1 2
            M 1
        """)
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 3


class TestParseMXXMYYMZZWithArgs:
    """Tests for MXX/MYY/MZZ with parenthesized flip probability."""

    def test_mxx_with_flip_probability(self):
        """MXX(p) should parse without misinterpreting p as invert."""
        circuit = stim.Circuit("MXX(0.01) 0 1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1
        assert_allclose(b.channel_probs[0], [0.99, 0.01])

    def test_myy_with_flip_probability(self):
        """MYY(p) should parse without misinterpreting p as invert."""
        circuit = stim.Circuit("MYY(0.01) 0 1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1
        assert_allclose(b.channel_probs[0], [0.99, 0.01])

    def test_mzz_with_flip_probability(self):
        """MZZ(p) should parse without misinterpreting p as invert."""
        circuit = stim.Circuit("MZZ(0.01) 0 1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1
        assert_allclose(b.channel_probs[0], [0.99, 0.01])

    def test_mxx_flip_prob_multiple_pairs(self):
        """MXX(p) with multiple pairs should work correctly."""
        circuit = stim.Circuit("MXX(0.05) 0 1 2 3")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 2


class TestParseMPPWithArgs:
    """Tests for MPP with parenthesized flip probability."""

    def test_mpp_with_flip_probability(self):
        """MPP(p) should parse and forward flip probability."""
        circuit = stim.Circuit("MPP(0.01) X0*Z1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 1
        assert_allclose(b.channel_probs[0], [0.99, 0.01])


class TestParseSPP:
    """Tests for parsing SPP and SPP_DAG instructions."""

    def test_spp_single_pauli(self):
        """SPP Z0 should parse without adding measurement records."""
        circuit = stim.Circuit("SPP Z0")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 0

    def test_spp_product(self):
        """SPP X0*Z1 should parse without adding measurement records."""
        circuit = stim.Circuit("SPP X0*Z1")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 0

    def test_spp_dag_single_pauli(self):
        """SPP_DAG Z0 should parse without adding measurement records."""
        circuit = stim.Circuit("SPP_DAG Z0")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 0

    def test_spp_multiple_products(self):
        """SPP with multiple products should parse correctly."""
        circuit = stim.Circuit("SPP X0 Y1*Z2")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 0
