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


def test_shorthand_to_stim_rotations():
    text = "R_X(0.25) 0\nR_Y(-0.5) 1\nR_Z(0.3) 2"
    expected = (
        "I[R_X(theta=0.25*pi)] 0\n" "I[R_Y(theta=-0.5*pi)] 1\n" "I[R_Z(theta=0.3*pi)] 2"
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
