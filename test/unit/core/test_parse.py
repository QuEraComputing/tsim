from collections import Counter
from fractions import Fraction
from unittest.mock import ANY, patch

import pytest
import stim
from numpy.testing import assert_allclose
from pyzx_param.utils import VertexType

from tsim.core.parse import parse_parametric_tag, parse_stim_circuit


def _assert_error_vertex_layout(b, expected):
    """Assert which ZX error vertices are controlled by each error bit."""
    actual = {}
    for v in b.graph.vertices():
        for phase in b.graph._phaseVars.get(v, set()):
            if isinstance(phase, str) and phase.startswith("e"):
                actual.setdefault(phase, Counter()).update(
                    [(b.graph.type(v), b.graph.qubit(v))]
                )

    expected_counters = {
        phase: Counter(vertices) for phase, vertices in expected.items()
    }
    assert actual == expected_counters


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


class TestProbabilityBearingInstructions:
    """Tests for instructions that create probabilistic error channels."""

    @pytest.mark.parametrize(
        "program",
        [
            "X_ERROR(0.07) 0",
            "Y_ERROR(0.07) 0",
            "Z_ERROR(0.07) 0",
            "M(0.07) 0",
            "MX(0.07) 0",
            "MY(0.07) 0",
            "MZ(0.07) 0",
            "MR(0.07) 0",
            "MRX(0.07) 0",
            "MRY(0.07) 0",
            "MRZ(0.07) 0",
            "MXX(0.07) 0 1",
            "MYY(0.07) 0 1",
            "MZZ(0.07) 0 1",
            "MPP(0.07) X0*Z1",
            "MPAD(0.07) 0",
            "CORRELATED_ERROR(0.07) X0",
        ],
    )
    def test_single_bit_probability_channels(self, program):
        """Single-bit probability-bearing instructions should create one channel."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert b.num_error_bits == 1
        assert len(b.channel_probs) == 1
        assert_allclose(b.channel_probs[0], [0.93, 0.07])

    @pytest.mark.parametrize("program", ["MR(0.07) 0", "MRX(0.07) 0", "MRY(0.07) 0"])
    def test_mr_family_does_not_double_count_measurement_noise(self, program):
        """MR-family measurement noise is a result flip, not an extra Pauli error."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert len(b.rec) == 1
        assert b.num_error_bits == 1
        assert len(b.channel_probs) == 1
        assert_allclose(b.channel_probs[0], [0.93, 0.07])

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("DEPOLARIZE1(0.12) 0", [0.88, 0.04, 0.04, 0.04]),
            ("PAULI_CHANNEL_1(0.01, 0.02, 0.03) 0", [0.94, 0.03, 0.01, 0.02]),
            (
                "HERALDED_ERASE(0.2) 0",
                [0.8, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05],
            ),
            (
                "HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 0",
                [0.9, 0.01, 0.0, 0.04, 0.0, 0.02, 0.0, 0.03],
            ),
        ],
    )
    def test_multi_bit_probability_channels(self, program, expected):
        """Multi-bit probability channels should use the documented outcome ordering."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert len(b.channel_probs) == 1
        assert_allclose(b.channel_probs[0], expected)

    def test_depolarize2_probability_channel(self):
        """DEPOLARIZE2(p) should distribute p uniformly over non-identity outcomes."""
        b = parse_stim_circuit(stim.Circuit("DEPOLARIZE2(0.15) 0 1"))

        assert b.num_error_bits == 4
        assert len(b.channel_probs) == 1
        assert_allclose(b.channel_probs[0], [0.85] + [0.01] * 15)

    def test_pauli_channel_2_probability_channel(self):
        """PAULI_CHANNEL_2 should preserve the expected packed Pauli outcome order."""
        args = [
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.010,
            0.011,
            0.012,
            0.013,
            0.014,
            0.015,
        ]
        program = f"PAULI_CHANNEL_2({', '.join(map(str, args))}) 0 1"
        b = parse_stim_circuit(stim.Circuit(program))

        assert b.num_error_bits == 4
        assert len(b.channel_probs) == 1
        assert_allclose(
            b.channel_probs[0],
            [
                0.88,
                0.012,
                0.004,
                0.008,
                0.003,
                0.015,
                0.007,
                0.011,
                0.001,
                0.013,
                0.005,
                0.009,
                0.002,
                0.014,
                0.006,
                0.010,
            ],
        )

    @pytest.mark.parametrize(
        "program",
        [
            "X_ERROR(0.07) 0 1",
            "M(0.07) 0 1",
            "MR(0.07) 0 1",
            "MRX(0.07) 0 1",
            "MRY(0.07) 0 1",
            "MXX(0.07) 0 1 2 3",
            "MPP(0.07) X0 X1",
            "MPAD(0.07) 0 1",
        ],
    )
    def test_repeated_probability_instructions_create_independent_channels(
        self, program
    ):
        """Repeated targets/products should each get their own probability channel."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert b.num_error_bits == 2
        assert len(b.channel_probs) == 2
        for probs in b.channel_probs:
            assert_allclose(probs, [0.93, 0.07])

    @pytest.mark.parametrize("program", ["I_ERROR(0.07) 0", "II_ERROR(0.07) 0 1"])
    def test_identity_error_instructions_do_not_create_channels(self, program):
        """Identity error instructions allocate lanes but do not affect noise state."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert b.num_error_bits == 0
        assert len(b.channel_probs) == 0


class TestChannelBitLayoutMatchesGraph:
    """Tests tying channel probability bits to ZX error vertices."""

    @pytest.mark.parametrize(
        ("program", "expected_layout"),
        [
            ("X_ERROR(0.07) 5", {"e0": [(VertexType.X, 5)]}),
            ("Y_ERROR(0.07) 5", {"e0": [(VertexType.Z, 5), (VertexType.X, 5)]}),
            ("Z_ERROR(0.07) 5", {"e0": [(VertexType.Z, 5)]}),
            ("M(0.07) 5", {"e0": [(VertexType.X, 5), (VertexType.X, 5)]}),
            ("MX(0.07) 5", {"e0": [(VertexType.X, 5), (VertexType.X, 5)]}),
            ("MY(0.07) 5", {"e0": [(VertexType.X, 5), (VertexType.X, 5)]}),
            ("MR(0.07) 5", {"e0": [(VertexType.X, 5), (VertexType.X, 5)]}),
            ("MRX(0.07) 5", {"e0": [(VertexType.X, 5), (VertexType.X, 5)]}),
            ("MRY(0.07) 5", {"e0": [(VertexType.X, 5), (VertexType.X, 5)]}),
            ("MXX(0.07) 5 7", {"e0": [(VertexType.X, -2), (VertexType.X, -2)]}),
            ("MPP(0.07) X5*Z7", {"e0": [(VertexType.X, -2), (VertexType.X, -2)]}),
            ("MPAD(0.07) 1", {"e0": [(VertexType.X, -2), (VertexType.X, -2)]}),
        ],
    )
    def test_single_bit_channel_layout_matches_graph(self, program, expected_layout):
        """Single-bit channel index 1 (0b1) should control the expected vertex."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert b.num_error_bits == 1
        assert_allclose(b.channel_probs[0], [0.93, 0.07])
        _assert_error_vertex_layout(b, expected_layout)

    @pytest.mark.parametrize(
        "program",
        ["PAULI_CHANNEL_1(0.01, 0.02, 0.03) 5", "DEPOLARIZE1(0.12) 5"],
    )
    def test_pauli_channel_1_bit_layout_matches_graph(self, program):
        """Index bit 0 is Z and index bit 1 is X for one-qubit Pauli channels."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert b.num_error_bits == 2
        _assert_error_vertex_layout(
            b,
            {
                "e0": [(VertexType.Z, 5)],
                "e1": [(VertexType.X, 5)],
            },
        )

    @pytest.mark.parametrize(
        "program",
        [
            "PAULI_CHANNEL_2("
            "0.001, 0.002, 0.003, 0.004, 0.005, "
            "0.006, 0.007, 0.008, 0.009, 0.010, "
            "0.011, 0.012, 0.013, 0.014, 0.015"
            ") 5 7",
            "DEPOLARIZE2(0.15) 5 7",
        ],
    )
    def test_pauli_channel_2_bit_layout_matches_graph(self, program):
        """Bits are Z_i, X_i, Z_j, X_j for two-qubit Pauli channels."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert b.num_error_bits == 4
        _assert_error_vertex_layout(
            b,
            {
                "e0": [(VertexType.Z, 5)],
                "e1": [(VertexType.X, 5)],
                "e2": [(VertexType.Z, 7)],
                "e3": [(VertexType.X, 7)],
            },
        )

    @pytest.mark.parametrize(
        "program",
        [
            "HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 5",
            "HERALDED_ERASE(0.2) 5",
        ],
    )
    def test_heralded_channel_bit_layout_matches_graph(self, program):
        """Bits are herald, Z, X for heralded one-qubit Pauli channels."""
        b = parse_stim_circuit(stim.Circuit(program))

        assert len(b.rec) == 1
        assert b.num_error_bits == 3
        _assert_error_vertex_layout(
            b,
            {
                "e0": [(VertexType.X, -2)],
                "e1": [(VertexType.Z, 5)],
                "e2": [(VertexType.X, 5)],
            },
        )

    def test_correlated_error_bit_layout_matches_graph(self):
        """Each correlated-error branch controls its matching Pauli vertices."""
        circuit = stim.Circuit("""
            CORRELATED_ERROR(0.1) X5 Y6 Z7
            ELSE_CORRELATED_ERROR(0.2) Z5
        """)
        b = parse_stim_circuit(circuit)

        assert b.num_error_bits == 2
        assert_allclose(b.channel_probs[0], [0.72, 0.1, 0.18, 0.0])
        _assert_error_vertex_layout(
            b,
            {
                "e0": [
                    (VertexType.X, 5),
                    (VertexType.Z, 6),
                    (VertexType.X, 6),
                    (VertexType.Z, 7),
                ],
                "e1": [(VertexType.Z, 5)],
            },
        )


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

    def test_spp_repeated_pair_cancels(self):
        """SPP X0*X0 should forward empty Pauli list (no-op)."""
        with patch("tsim.core.parse.spp") as mock_spp:
            parse_stim_circuit(stim.Circuit("SPP X0*X0"))
        mock_spp.assert_called_once_with(ANY, [], dagger=False)

    def test_spp_partial_cancel(self):
        """SPP X0*Y1*X0 should cancel X0 and forward Y1."""
        with patch("tsim.core.parse.spp") as mock_spp:
            parse_stim_circuit(stim.Circuit("SPP X0*Y1*X0"))
        mock_spp.assert_called_once_with(ANY, [("Y", 1)], dagger=False)

    def test_spp_anticommuting_sign_flips_dagger(self):
        """SPP Z0*X0*Z0 reduces to -X; sign=-1 should flip the dagger flag."""
        with patch("tsim.core.parse.spp") as mock_spp:
            parse_stim_circuit(stim.Circuit("SPP Z0*X0*Z0"))
        mock_spp.assert_called_once_with(ANY, [("X", 0)], dagger=True)

    def test_spp_dag_anticommuting_sign_flips_dagger(self):
        """SPP_DAG with sign=-1 toggles dagger back: SPP_DAG·(-1) → SPP."""
        with patch("tsim.core.parse.spp") as mock_spp:
            parse_stim_circuit(stim.Circuit("SPP_DAG Z0*X0*Z0"))
        mock_spp.assert_called_once_with(ANY, [("X", 0)], dagger=False)

    def test_spp_multiple_products(self):
        """SPP with multiple products should parse correctly."""
        circuit = stim.Circuit("SPP X0 Y1*Z2")
        b = parse_stim_circuit(circuit)
        assert len(b.rec) == 0


class TestParseEmptyAnnotations:
    """Empty DETECTOR / OBSERVABLE_INCLUDE annotations are valid Stim and represent
    deterministic-zero detector/observable bits. They must parse without crashing
    and produce a single annotation vertex with no record edges."""

    def test_empty_detector_alone(self):
        b = parse_stim_circuit(stim.Circuit("DETECTOR"))
        assert len(b.detectors) == 1
        v = b.detectors[0]
        assert b.graph.type(v) == VertexType.X
        assert list(b.graph.neighbors(v)) == []

    def test_empty_observable_alone(self):
        b = parse_stim_circuit(stim.Circuit("OBSERVABLE_INCLUDE(0)"))
        assert set(b.observables_dict) == {0}
        v = b.observables_dict[0]
        assert b.graph.type(v) == VertexType.X
        assert list(b.graph.neighbors(v)) == []

    def test_empty_detector_after_measurement(self):
        b = parse_stim_circuit(stim.Circuit("M 0\nDETECTOR rec[-1]\nDETECTOR"))
        assert len(b.detectors) == 2
        # First detector has the measurement edge, second has no edges.
        assert len(list(b.graph.neighbors(b.detectors[0]))) == 1
        assert list(b.graph.neighbors(b.detectors[1])) == []

    def test_empty_detector_with_args(self):
        """DETECTOR(coords...) with empty rec must also parse."""
        b = parse_stim_circuit(stim.Circuit("DETECTOR(1, 2)"))
        assert len(b.detectors) == 1


class TestParseObservableIncludePauliTargets:
    """OBSERVABLE_INCLUDE accepts Pauli targets in stim but Tsim only supports
    measurement-record targets; reject Pauli targets explicitly rather than
    silently mis-indexing into the measurement record."""

    @pytest.mark.parametrize(
        "program",
        [
            "H 1\nOBSERVABLE_INCLUDE(0) X1\nM 0",
            "M 0 1\nH 0\nOBSERVABLE_INCLUDE(0) X1",
            "M 0\nOBSERVABLE_INCLUDE(0) Z0",
            "M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1] Y1",
        ],
    )
    def test_pauli_target_rejected(self, program):
        with pytest.raises(ValueError, match="OBSERVABLE_INCLUDE with Pauli targets"):
            parse_stim_circuit(stim.Circuit(program))

    def test_record_targets_still_accepted(self):
        parse_stim_circuit(stim.Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]"))


class TestParseMPPCancellation:
    """Tests for MPP with duplicate/anticommuting Pauli targets.

    Mocks ``mpp`` so we can assert the exact Pauli list and invert flag that
    the parser forwards — this directly exercises the reduction and sign
    tracking done by ``_iter_pauli_products``.
    """

    def test_mpp_full_cancel_reduces_to_identity(self):
        """MPP X0*X0 should reduce to empty Pauli list (measures identity)."""
        with patch("tsim.core.parse.mpp") as mock_mpp:
            parse_stim_circuit(stim.Circuit("MPP X0*X0"))
        mock_mpp.assert_called_once_with(ANY, [], False, p=0)

    def test_mpp_full_cancel_inverted(self):
        """MPP !X0*X0 should measure identity with invert=True."""
        with patch("tsim.core.parse.mpp") as mock_mpp:
            parse_stim_circuit(stim.Circuit("MPP !X0*X0"))
        mock_mpp.assert_called_once_with(ANY, [], True, p=0)

    def test_mpp_partial_cancel_measures_y_basis(self):
        """MPP X0*Y1*X0 should cancel X0 pair and measure Y on qubit 1."""
        with patch("tsim.core.parse.mpp") as mock_mpp:
            parse_stim_circuit(stim.Circuit("MPP X0*Y1*X0"))
        mock_mpp.assert_called_once_with(ANY, [("Y", 1)], False, p=0)

    def test_mpp_reorders_to_z_basis(self):
        """MPP Y0*Y0*Z1 should reduce to just Z on qubit 1."""
        with patch("tsim.core.parse.mpp") as mock_mpp:
            parse_stim_circuit(stim.Circuit("MPP Y0*Y0*Z1"))
        mock_mpp.assert_called_once_with(ANY, [("Z", 1)], False, p=0)

    def test_mpp_anticommuting_sign_flips_invert(self):
        """MPP Z0*X0*Z0*X0 = -I: sign=-1 flips invert from False to True."""
        with patch("tsim.core.parse.mpp") as mock_mpp:
            parse_stim_circuit(stim.Circuit("MPP Z0*X0*Z0*X0"))
        mock_mpp.assert_called_once_with(ANY, [], True, p=0)

    def test_mpp_anticommuting_sign_with_explicit_invert(self):
        """MPP !Z0*X0*Z0*X0 = -I with !: two flips cancel → invert=False."""
        with patch("tsim.core.parse.mpp") as mock_mpp:
            parse_stim_circuit(stim.Circuit("MPP !Z0*X0*Z0*X0"))
        mock_mpp.assert_called_once_with(ANY, [], False, p=0)

    def test_mpp_combines_to_single_pauli_with_sign(self):
        """MPP Z0*X0*Z0 should reduce to -X on qubit 0."""
        with patch("tsim.core.parse.mpp") as mock_mpp:
            parse_stim_circuit(stim.Circuit("MPP Z0*X0*Z0"))
        mock_mpp.assert_called_once_with(ANY, [("X", 0)], True, p=0)

    def test_mpp_multiple_products_independent_state(self):
        """Each product's accumulated state should reset between products."""
        with patch("tsim.core.parse.mpp") as mock_mpp:
            parse_stim_circuit(stim.Circuit("MPP X0*X0 !Z0"))
        assert mock_mpp.call_count == 2
        # First product: X0*X0 → identity, invert=False
        assert mock_mpp.call_args_list[0] == (
            (ANY, [], False),
            {"p": 0},
        )
        # Second product: !Z0 → Z with invert=True
        assert mock_mpp.call_args_list[1] == (
            (ANY, [("Z", 0)], True),
            {"p": 0},
        )

    def test_mpp_anti_hermitian_raises(self):
        """MPP Z0*X0 = iY is anti-Hermitian and should raise."""
        with pytest.raises(ValueError, match="anti-Hermitian"):
            parse_stim_circuit(stim.Circuit("MPP Z0*X0"))

    def test_mpp_anti_hermitian_multi_qubit_raises(self):
        """MPP Z0*X0*X1 is anti-Hermitian and should raise."""
        with pytest.raises(ValueError, match="anti-Hermitian"):
            parse_stim_circuit(stim.Circuit("MPP Z0*X0*X1"))


class TestParseTPP:
    """Tests for parsing TPP and TPP_DAG instructions.

    Mocks ``tpp`` to assert the exact Pauli list and dagger flag that the
    parser forwards — directly exercising the Pauli reduction and sign
    tracking done by ``_iter_pauli_products``.
    """

    def test_tpp_single_pauli(self):
        """TPP Z0 should forward [(Z, 0)] with dagger=False."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP[T] Z0"))
        mock_tpp.assert_called_once_with(ANY, [("Z", 0)], dagger=False)

    def test_tpp_product(self):
        """TPP X0*Z1 should forward both Paulis with dagger=False."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP[T] X0*Z1"))
        mock_tpp.assert_called_once_with(ANY, [("X", 0), ("Z", 1)], dagger=False)

    def test_tpp_dag_single_pauli(self):
        """TPP_DAG Z0 should forward [(Z, 0)] with dagger=True."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP_DAG[T] Z0"))
        mock_tpp.assert_called_once_with(ANY, [("Z", 0)], dagger=True)

    def test_tpp_invert_flag_flips_dagger(self):
        """TPP !Z0 has the ! flag, which XORs into the dagger flag."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP[T] !Z0"))
        mock_tpp.assert_called_once_with(ANY, [("Z", 0)], dagger=True)

    def test_tpp_repeated_pair_cancels(self):
        """TPP Z0*Z0 should forward empty Pauli list (no-op)."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP[T] Z0*Z0"))
        mock_tpp.assert_called_once_with(ANY, [], dagger=False)

    def test_tpp_partial_cancel(self):
        """TPP X0*Y1*X0 should cancel X0 pair and forward Y1."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP[T] X0*Y1*X0"))
        mock_tpp.assert_called_once_with(ANY, [("Y", 1)], dagger=False)

    def test_tpp_anticommuting_sign_flips_dagger(self):
        """TPP Z0*X0*Z0 reduces to -X; sign=-1 should flip dagger to True."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP[T] Z0*X0*Z0"))
        mock_tpp.assert_called_once_with(ANY, [("X", 0)], dagger=True)

    def test_tpp_dag_anticommuting_sign_flips_dagger(self):
        """TPP_DAG with sign=-1 toggles dagger back: TPP_DAG·(-1) → TPP."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP_DAG[T] Z0*X0*Z0"))
        mock_tpp.assert_called_once_with(ANY, [("X", 0)], dagger=False)

    def test_tpp_anticommuting_with_invert(self):
        """TPP !Z0*X0*Z0: invert=True XOR sign=-1 → dagger flips twice → False."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP[T] !Z0*X0*Z0"))
        mock_tpp.assert_called_once_with(ANY, [("X", 0)], dagger=False)

    def test_tpp_multiple_products_independent_state(self):
        """Each product's accumulated state should reset between products."""
        with patch("tsim.core.parse.tpp") as mock_tpp:
            parse_stim_circuit(stim.Circuit("SPP[T] Z0*Z0 !X1"))
        assert mock_tpp.call_count == 2
        # First product: Z0*Z0 → identity, dagger=False
        assert mock_tpp.call_args_list[0] == ((ANY, []), {"dagger": False})
        # Second product: !X1 → X with dagger=True
        assert mock_tpp.call_args_list[1] == (
            (ANY, [("X", 1)]),
            {"dagger": True},
        )

    def test_tpp_anti_hermitian_raises(self):
        """TPP Z0*X0 = iY is anti-Hermitian and should raise."""
        with pytest.raises(ValueError, match="anti-Hermitian"):
            parse_stim_circuit(stim.Circuit("SPP[T] Z0*X0"))


class TestParseSparseObservables:
    """OBSERVABLE_INCLUDE indices are sparse and out-of-order in real circuits.
    Stim defines num_observables = max_index + 1 and emits one column per id
    in sorted order, with missing ids as deterministic-zero columns."""

    def test_sparse_observable_index_pads_missing(self):
        circuit = stim.Circuit("M 0\nOBSERVABLE_INCLUDE(2) rec[-1]")
        b = parse_stim_circuit(circuit)
        assert set(b.observables_dict) == {0, 1, 2}
        # Missing ids 0 and 1 have no record edges (deterministic zero).
        assert list(b.graph.neighbors(b.observables_dict[0])) == []
        assert list(b.graph.neighbors(b.observables_dict[1])) == []
        # Index 2 has the measurement edge.
        assert len(list(b.graph.neighbors(b.observables_dict[2]))) == 1

    def test_observables_dict_is_sorted_after_out_of_order(self):
        circuit = stim.Circuit(
            "M 0\nM 1\nOBSERVABLE_INCLUDE(2) rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        )
        b = parse_stim_circuit(circuit)
        assert list(b.observables_dict) == [0, 1, 2]

    def test_no_observables_remains_empty(self):
        b = parse_stim_circuit(stim.Circuit("M 0"))
        assert b.observables_dict == {}


def _instr(
    tag: str, name: str = "I", targets: tuple[int, ...] = (0,)
) -> stim.CircuitInstruction:
    """Build a single stim instruction with the given tag for testing."""
    return stim.CircuitInstruction(name=name, targets=list(targets), tag=tag)


class TestParseParametricTag:
    """Tag parsing must accept well-formed tags, return ``None`` for non-parametric
    tags, and raise on tags that look parametric but are malformed. Errors include
    the full source instruction for context."""

    def test_r_axis_tag_returns_theta(self):
        for axis in ("R_X", "R_Y", "R_Z"):
            assert parse_parametric_tag(_instr(f"{axis}(theta=0.5*pi)")) == (
                axis,
                {"theta": Fraction(1, 2)},
            )

    def test_u3_tag_returns_all_three_angles(self):
        result = parse_parametric_tag(
            _instr("U3(theta=0.25*pi, phi=0.5*pi, lambda=0.75*pi)")
        )
        assert result == (
            "U3",
            {
                "theta": Fraction(1, 4),
                "phi": Fraction(1, 2),
                "lambda": Fraction(3, 4),
            },
        )

    def test_negative_angle_parsed(self):
        assert parse_parametric_tag(_instr("R_Z(theta=-0.5*pi)")) == (
            "R_Z",
            {"theta": Fraction(-1, 2)},
        )

    def test_non_parametric_tag_returns_none(self):
        # Tags without the name(...) shape are not parametric-looking; these are
        # used for non-parametric annotations like S[T] / SPP[T].
        assert parse_parametric_tag(_instr("T")) is None
        assert parse_parametric_tag(_instr("")) is None
        assert parse_parametric_tag(_instr("R_Z")) is None

    def test_malformed_value_raises(self):
        with pytest.raises(ValueError, match="Malformed parametric tag"):
            parse_parametric_tag(_instr("R_Z(theta=abc)"))
        with pytest.raises(ValueError, match="Malformed parametric tag"):
            parse_parametric_tag(_instr("R_Z(theta=0.5)"))  # missing *pi

    def test_unknown_gate_name_raises(self):
        with pytest.raises(ValueError, match="Unknown parametric gate 'FOO'"):
            parse_parametric_tag(_instr("FOO(theta=0.5*pi)"))

    def test_r_axis_missing_theta_raises(self):
        with pytest.raises(ValueError, match=r"expected \['theta'\]"):
            parse_parametric_tag(_instr("R_X()"))
        with pytest.raises(ValueError, match=r"expected \['theta'\]"):
            parse_parametric_tag(_instr("R_X(phi=0.5*pi)"))

    def test_r_axis_extra_param_raises(self):
        with pytest.raises(ValueError, match=r"expected \['theta'\]"):
            parse_parametric_tag(_instr("R_Z(theta=0.5*pi, phi=0.25*pi)"))

    def test_u3_missing_required_param_raises(self):
        expected_msg = r"expected \['lambda', 'phi', 'theta'\]"
        with pytest.raises(ValueError, match=expected_msg):
            parse_parametric_tag(_instr("U3(theta=0.5*pi)"))
        with pytest.raises(ValueError, match=expected_msg):
            parse_parametric_tag(_instr("U3(theta=0.5*pi, phi=0.25*pi)"))
        with pytest.raises(ValueError, match=expected_msg):
            parse_parametric_tag(_instr("U3(phi=0.5*pi, lambda=0.25*pi)"))

    def test_u3_extra_param_raises(self):
        with pytest.raises(ValueError, match=r"expected \['lambda', 'phi', 'theta'\]"):
            parse_parametric_tag(
                _instr("U3(theta=0.5*pi, phi=0.25*pi, lambda=0.5*pi, extra=0.1*pi)")
            )

    def test_error_includes_full_instruction(self):
        instr = _instr("U3(theta=0.5*pi)", targets=(0, 1))
        with pytest.raises(
            ValueError,
            match=r"Could not parse instruction 'I\[U3\(theta=0\.5\*pi\)\] 0 1'",
        ):
            parse_parametric_tag(instr)


class TestSweepTargets:
    """Sweep targets are not supported and must raise rather than be silently lowered as qubit targets."""

    @pytest.mark.parametrize(
        "program",
        [
            "CX sweep[0] 1",
            "CZ 0 sweep[1]",
            "CY sweep[2] 0",
        ],
    )
    def test_sweep_target_raises(self, program):
        circuit = stim.Circuit(program)
        with pytest.raises(NotImplementedError, match="Sweep bit targets"):
            parse_stim_circuit(circuit)
