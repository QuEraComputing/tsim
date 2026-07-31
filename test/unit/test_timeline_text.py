"""Tests for the timeline-text diagram type."""

import pytest

import tsim


def render(circuit: tsim.Circuit) -> str:
    return str(circuit.diagram("timeline-text"))


@pytest.mark.parametrize(
    "source,expected",
    [
        ("T 0", "T"),
        ("T_DAG 0", "T_DAG"),
        ("R_X(0.125) 0", "R_X(0.125)"),
        ("R_Y(-0.25) 0", "R_Y(-0.25)"),
        ("R_Z(0.5) 0", "R_Z(0.5)"),
        ("U3(0.5,0.25,-0.125) 0", "U3(0.5,0.25,-0.125)"),
        ("TPP X0*Y1", "TPP[X]"),
        ("TPP_DAG Z0", "TPP_DAG[Z]"),
    ],
)
def test_custom_gates_render_logical_names(source, expected):
    assert expected in render(tsim.Circuit(source))


def test_t_gate_is_not_rendered_as_placeholder():
    """Regression test for #171."""
    out = render(tsim.Circuit("T 0"))
    assert "T" in out
    assert "-S-" not in out


@pytest.mark.parametrize(
    "name,arg,expected",
    [
        ("R_XX", 0.5, "R_PAULI(0.5)[X]"),
        ("R_YY", -0.25, "R_PAULI(-0.25)[Y]"),
        ("R_ZZ", 0.125, "R_PAULI(0.125)[Z]"),
    ],
)
def test_two_qubit_rotations_render_as_r_pauli(name, arg, expected):
    """R_XX/R_YY/R_ZZ share R_PAULI's metadata tag, so they render as R_PAULI."""
    circuit = tsim.Circuit()
    circuit.append(name, [0, 1], arg)
    assert expected in render(circuit)


def test_ccz_shows_its_clifford_t_decomposition():
    circuit = tsim.Circuit()
    circuit.append("CCZ", [0, 1, 2])
    out = render(circuit)
    assert "T" in out and "T_DAG" in out


def test_user_tag_is_stripped_from_label():
    circuit = tsim.Circuit()
    circuit.append("T", [0], tag="mynote")
    out = render(circuit)
    assert "T" in out
    assert "mynote" not in out


def test_standard_gates_are_unaffected():
    out = render(tsim.Circuit("H 0\nCX 0 1\nM 0 1"))
    assert "H" in out and "@" in out
