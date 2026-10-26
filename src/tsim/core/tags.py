"""Helpers for tags that encode tsim-specific gate metadata."""

import re
from fractions import Fraction

T_TAG = "T"
_T_USER_TAG_PREFIX = f"{T_TAG}:"

# Matches valid numeric literals including scientific notation (e.g. 0.5, 4e-4, 1.2e3)
FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

_PARAMETRIC_GATE_PARAMS: dict[str, tuple[str, ...]] = {
    "R_X": ("theta",),
    "R_Y": ("theta",),
    "R_Z": ("theta",),
    "R_PAULI": ("theta",),
    "U3": ("theta", "phi", "lambda"),
}

_PARAMETRIC_TAG_RE = re.compile(r"^(\w+)\((.*)\)$")
_KEYWORD_PARAM_RE = re.compile(rf"^(\w+)=({FLOAT_RE})\*pi$")
_POSITIONAL_PARAM_RE = re.compile(rf"^{FLOAT_RE}$")


def encode_t_tag(user_tag: str = "") -> str:
    """Encode a T-family gate tag while preserving an optional user tag."""
    if not user_tag:
        return T_TAG
    return f"{_T_USER_TAG_PREFIX}{user_tag}"


def is_t_tag(tag: str) -> bool:
    """Return whether a Stim tag encodes a tsim T-family gate."""
    return tag == T_TAG or tag.startswith(_T_USER_TAG_PREFIX)


def decode_t_user_tag(tag: str) -> str:
    """Return the user tag attached to an encoded T-family gate tag."""
    if tag == T_TAG:
        return ""
    if tag.startswith(_T_USER_TAG_PREFIX):
        return tag[len(_T_USER_TAG_PREFIX) :]
    raise ValueError(f"Tag does not encode a T-family gate: {tag!r}")


def parse_parametric_tag_text(tag: str) -> tuple[str, dict[str, Fraction]] | None:
    """Parse a parametric gate tag such as ``R_Z(theta=0.3*pi)`` or ``R_Z(0.3)``.

    Angles are in units of pi. They are given either all by keyword with an
    explicit ``*pi`` suffix, or all by position (``U3`` takes theta, phi, lambda
    in that order, as in the shorthand).

    Args:
        tag: The stim instruction tag.

    Returns:
        Tuple of (gate_name, params_dict), or ``None`` when the tag does not have
        the ``name(...)`` shape.

    Raises:
        ValueError: When the tag has the ``name(...)`` shape but is malformed: an
            unknown gate name, an unparsable angle, or the wrong parameters.

    """
    match = _PARAMETRIC_TAG_RE.match(tag)
    if not match:
        return None

    gate_name, args_text = match.groups()
    args = [arg.strip() for arg in args_text.split(",")] if args_text.strip() else []

    expected = _PARAMETRIC_GATE_PARAMS.get(gate_name)
    if expected is None:
        raise ValueError(f"Unknown parametric gate {gate_name!r}")

    keyword_matches = [_KEYWORD_PARAM_RE.match(arg) for arg in args]
    if all(keyword_matches):
        params = {
            m.group(1): Fraction(m.group(2)) for m in keyword_matches if m is not None
        }
        if params.keys() != set(expected):
            raise ValueError(
                f"Parametric tag {tag!r} has parameters "
                f"{sorted(params)}, expected {sorted(expected)}"
            )
        return gate_name, params

    if all(_POSITIONAL_PARAM_RE.match(arg) for arg in args):
        if len(args) != len(expected):
            raise ValueError(
                f"Parametric tag {tag!r} has {len(args)} angles, "
                f"expected {len(expected)} ({', '.join(expected)})"
            )
        return gate_name, dict(zip(expected, map(Fraction, args), strict=True))

    positional_example = ", ".join("0.25" for _ in expected)
    keyword_example = ", ".join(f"{name}=0.25*pi" for name in expected)
    raise ValueError(
        f"Malformed parametric tag {tag!r}. Angles are in units of pi, e.g. "
        f"'{gate_name}({positional_example})' or '{gate_name}({keyword_example})'"
    )


def format_parametric_tag(gate_name: str, params: dict[str, Fraction]) -> str:
    """Format a parametric gate tag in its canonical keyword form."""
    args = ", ".join(
        f"{name}={float(params[name])}*pi"
        for name in _PARAMETRIC_GATE_PARAMS[gate_name]
    )
    return f"{gate_name}({args})"
