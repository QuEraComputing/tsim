import re

import pytest

from tsim import Circuit
from tsim.utils.program_text import (
    shorthand_to_stim,
    stim_to_shorthand,
)


def test_shorthand_to_stim_t_and_t_dag():
    text = "T 0 1\nT_DAG 2"
    expected = "S[T] 0 1\nS_DAG[T] 2"
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_tagged_t_and_t_dag():
    text = "T[phase] 0\nT_DAG[phase] 1"
    expected = "S[T:phase] 0\nS_DAG[T:phase] 1"
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_tpp_and_tpp_dag():
    text = "TPP X0*Y1\nTPP_DAG Z0"
    expected = "SPP[T] X0*Y1\nSPP_DAG[T] Z0"
    assert shorthand_to_stim(text) == expected


def test_stim_to_shorthand_tpp_and_tpp_dag():
    text = "SPP[T] X0*Y1\nSPP_DAG[T] Z0"
    expected = "TPP X0*Y1\nTPP_DAG Z0"
    assert stim_to_shorthand(text) == expected


def test_shorthand_tpp_roundtrip():
    text = "TPP X0*Z1\nTPP_DAG !Y0*Y1"
    assert stim_to_shorthand(shorthand_to_stim(text)) == text


def test_shorthand_to_stim_rotations():
    text = "R_X(0.25) 0\nR_Y(-0.5) 1\nR_Z(0.3) 2"
    expected = (
        "I[R_X(theta=0.25*pi)] 0\nI[R_Y(theta=-0.5*pi)] 1\nI[R_Z(theta=0.3*pi)] 2"
    )
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_u3():
    text = "U3(0.3, 0.24, 0.49) 0"
    expected = "I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0"
    assert shorthand_to_stim(text) == expected


def test_stim_to_shorthand_t_and_t_dag():
    text = "S[T] 0 1\nS_DAG[T] 2"
    expected = "T 0 1\nT_DAG 2"
    assert stim_to_shorthand(text) == expected


def test_stim_to_shorthand_tagged_t_and_t_dag():
    text = "S[T:phase] 0\nS_DAG[T:phase] 1"
    expected = "T[phase] 0\nT_DAG[phase] 1"
    assert stim_to_shorthand(text) == expected


def test_stim_to_shorthand_rotations_and_u3():
    text = (
        "I[R_X(theta=0.25*pi)] 0\n"
        "I[R_Y(theta=-0.5*pi)] 1\n"
        "I[R_Z(theta=0.3*pi)] 2\n"
        "I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 3"
    )
    expected = "R_X(0.25) 0\nR_Y(-0.5) 1\nR_Z(0.3) 2\nU3(0.3, 0.24, 0.49) 3"
    assert stim_to_shorthand(text) == expected


def test_shorthand_roundtrip():
    text = "T 0\nR_X(0.5) 1\nU3(0.1, 0.2, 0.3) 2"
    assert stim_to_shorthand(shorthand_to_stim(text)) == text


def test_shorthand_scientific_notation():
    result = shorthand_to_stim("R_Z(4e-4) 0")
    assert "I[R_Z(theta=0.0004*pi)]" in result


def test_shorthand_scientific_notation_u3():
    result = shorthand_to_stim("U3(1e-2, 2.5e1, 3e-3) 0")
    assert "I[U3(" in result


def test_circuit_scientific_notation():
    c = Circuit("R_Z(4e-4) 0")
    assert len(c) == 1


def test_stim_to_shorthand_rotation_scientific_notation():
    text = "I[R_Z(theta=1e-07*pi)] 0\nI[R_X(theta=-2.5e+02*pi)] 1"
    expected = "R_Z(1e-07) 0\nR_X(-2.5e+02) 1"
    assert stim_to_shorthand(text) == expected


def test_stim_to_shorthand_u3_scientific_notation():
    text = "I[U3(theta=1e-07*pi, phi=2e-07*pi, lambda=3e-07*pi)] 0"
    expected = "U3(1e-07, 2e-07, 3e-07) 0"
    assert stim_to_shorthand(text) == expected


def test_circuit_str_scientific_notation_roundtrip():
    for program in ["R_Z(1e-7) 0", "U3(1e-7, 2e-7, 3e-7) 0"]:
        rendered = str(Circuit(program))
        assert "I[" not in rendered, rendered


@pytest.mark.parametrize(
    "program",
    [
        "I[R_X(theta=0.5e-2*pi)] 0",
        "I[R_Z(theta=4e-4*pi)] 1",
        "I[U3(theta=0.1*pi, phi=2e-1*pi, lambda=0.3*pi)] 0",
    ],
)
def test_circuit_eq_roundtrip_scientific_notation(program):
    c1 = Circuit(program)
    assert Circuit(str(c1)) == c1


@pytest.mark.parametrize(
    "text, snippet",
    [
        ("R_Z(a) 0", "R_Z(a)"),
        ("R_Z(pi) 0", "R_Z(pi)"),
        ("R_Z(1/3) 0", "R_Z(1/3)"),
        ("R_Z() 0", "R_Z()"),
        ("R_Z 0", "R_Z"),
        ("R_Z(0.5, 0.3) 0", "R_Z(0.5, 0.3)"),
        ("R_X(abc) 0", "R_X(abc)"),
        ("U3(0.1, 0.2) 0", "U3(0.1, 0.2)"),
        ("U3(0.1, 0.2, 0.3, 0.4) 0", "U3(0.1, 0.2, 0.3, 0.4)"),
    ],
)
def test_circuit_parse_error_shows_snippet(text, snippet):
    with pytest.raises(ValueError, match=re.escape(snippet)):
        Circuit(text)


@pytest.mark.parametrize(
    "text",
    [
        "X[R_X(0.1)] 0",
        "X[U3(0.1, 0.2, 0.3)] 0",
        "H[R_XX(0.5)] 0",
        "H[my T tag] 0",
        "CX[T_DAG and TPP] 0 1",
        "H 0  # R_X(0.1) T U3(0.1, 0.2, 0.3)",
        "# T 0",
    ],
)
def test_shorthand_to_stim_ignores_tags_and_comments(text):
    assert shorthand_to_stim(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I[R_X(0.1)] 0", "I[R_X(theta=0.1*pi)] 0"),
        ("I[R_Z(-0.5e-2)] 0", "I[R_Z(theta=-0.005*pi)] 0"),
        (
            "I[U3(0.1,0.2, 0.3)] 0",
            "I[U3(theta=0.1*pi, phi=0.2*pi, lambda=0.3*pi)] 0",
        ),
        ("SPP[R_PAULI(0.25)] X0*Y1", "SPP[R_PAULI(theta=0.25*pi)] X0*Y1"),
        ("SPP_DAG[R_PAULI(0.25)] Z0", "SPP_DAG[R_PAULI(theta=0.25*pi)] Z0"),
        ("  I[R_Y(0.3)] 0  # c", "  I[R_Y(theta=0.3*pi)] 0  # c"),
        ("I[R_X(.)] 0", "I[R_X(.)] 0"),
        ("I[label] 0", "I[label] 0"),
    ],
)
def test_shorthand_to_stim_canonicalizes_parametric_tags(text, expected):
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_keeps_hash_inside_tag():
    assert shorthand_to_stim("T[a#b] 0  # T") == "S[T:a#b] 0  # T"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("REPEAT 2 { T 0\n}", "REPEAT 2 { S[T] 0\n}"),
        ("REPEAT[r] 2 {R_X(0.1) 0\n}", "REPEAT[r] 2 {I[R_X(theta=0.1*pi)] 0\n}"),
        ("REPEAT 2 {\nH 0\n} T 0", "REPEAT 2 {\nH 0\n} S[T] 0"),
        ("} REPEAT 3 { i[R_Z(0.5)] 0", "} REPEAT 3 { i[R_Z(theta=0.5*pi)] 0"),
        ("repeat 2 { T 0\n}", "repeat 2 { S[T] 0\n}"),
        ("Repeat 2 {R_Z(0.5) 0\n}", "Repeat 2 {I[R_Z(theta=0.5*pi)] 0\n}"),
    ],
)
def test_shorthand_to_stim_converts_instructions_next_to_braces(text, expected):
    assert shorthand_to_stim(text) == expected


def test_parse_error_ignores_shorthand_in_comments():
    with pytest.raises(ValueError, match=r"Could not parse 'R_X' in"):
        Circuit("H 0 # apply R_X(0.1) later\nR_X 0")


@pytest.mark.parametrize(
    "tagged, shorthand",
    [
        ("I[R_X(0.1)] 0", "R_X(0.1) 0"),
        ("I[R_Y(-0.3)] 1", "R_Y(-0.3) 1"),
        ("I[R_Z(1e-3)] 0", "R_Z(1e-3) 0"),
        ("I[U3(0.3, 0.24, 0.49)] 0", "U3(0.3, 0.24, 0.49) 0"),
        ("SPP[R_PAULI(0.1)] X0*X1", "R_XX(0.1) 0 1"),
        ("SPP[R_PAULI(0.1)] X0*Y1", "R_PAULI(0.1) X0*Y1"),
    ],
)
def test_positional_tag_matches_shorthand(tagged, shorthand):
    circuit = Circuit(tagged)
    assert circuit == Circuit(shorthand)
    assert str(circuit) == str(Circuit(shorthand))
    assert Circuit(str(circuit)) == circuit
