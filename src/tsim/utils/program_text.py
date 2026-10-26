"""Conversion utilities between tsim shorthand and stim program text."""

import re
from collections.abc import Callable

from tsim.core.tags import (
    FLOAT_RE,
    decode_t_user_tag,
    encode_t_tag,
    format_parametric_tag,
    parse_parametric_tag_text,
)

_TSIM_GATES = {
    "CCZ",
    "CCX",
    "R_X",
    "R_Y",
    "R_Z",
    "R_XX",
    "R_YY",
    "R_ZZ",
    "R_PAULI",
    "U3",
}
_ROTATION_SHORTHAND_GATES = _TSIM_GATES - {"CCZ", "CCX"}
_T_FAMILY = {"T": "S", "T_DAG": "S_DAG", "TPP": "SPP", "TPP_DAG": "SPP_DAG"}
_PARAMETRIC_TAG_GATES = {"I", "SPP", "SPP_DAG"}
# The prefix group also takes block delimiters that share the instruction's line,
# as in `REPEAT 2 { H 0` or `} H 0`.
_INSTRUCTION_RE = re.compile(
    r"^((?:\s*(?:\}|REPEAT(?:\[[^\]\n]*\])?\s+\d+\s*\{))*\s*)"
    r"(\w+)(?:\[([^\]\n]*)\])?(.*)$"
)
_TAG_RE = re.compile(r"\[[^\]\n]*\]")
_COMMENT_RE = re.compile(r"#.*")
_GATE_NOT_FOUND_RE = re.compile(r"Gate not found: '(\w+)'")
_GATE_USAGE_RE = re.compile(
    r"(?<!\[)\b(CCZ\b|CCX\b|R_PAULI\([^)]*\)|R_[XYZ]{1,2}\([^)]*\)|R_[XYZ]\b|U3\([^)]*\)|U3\b)"
)


def controlled_gate_decomposition_lines(
    gate: str,
    control1: int | str,
    control2: int | str,
    target: int | str,
    *,
    tag: str = "",
) -> list[str]:
    """Return a Clifford+T decomposition for a controlled-controlled gate."""
    if gate not in ("CCZ", "CCX"):
        raise ValueError(f"Unsupported controlled-controlled gate: {gate!r}")

    def tagged(name: str) -> str:
        return f"{name}[{tag}]" if tag else name

    a, b, c = str(control1), str(control2), str(target)
    ccz_lines = [
        f"{tagged('CNOT')} {b} {c}",
        f"{tagged('T_DAG')} {c}",
        f"{tagged('CNOT')} {a} {c}",
        f"{tagged('T')} {c}",
        f"{tagged('CNOT')} {b} {c}",
        f"{tagged('T_DAG')} {c}",
        f"{tagged('CNOT')} {a} {c}",
        f"{tagged('T')} {b}",
        f"{tagged('T')} {c}",
        f"{tagged('CNOT')} {a} {b}",
        f"{tagged('T')} {a}",
        f"{tagged('T_DAG')} {b}",
        f"{tagged('CNOT')} {a} {b}",
    ]
    if gate == "CCZ":
        return ccz_lines
    return [f"{tagged('H')} {c}", *ccz_lines, f"{tagged('H')} {c}"]


def _expand_controlled_gates(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        body, sep, comment = line.partition("#")
        match = re.match(r"^(\s*)(CCZ|CCX)(?:\[([^\]\n]*)\])?\s+(.+?)\s*$", body)
        if not match:
            lines.append(line)
            continue

        indent, gate, tag, targets_text = match.groups()
        targets = targets_text.split()
        if len(targets) % 3 != 0 or not all(target.isdecimal() for target in targets):
            raise ValueError(
                f"{gate} expects bare qubit integer targets in groups of three."
            )

        if sep:
            lines.append(f"{indent}{sep}{comment}")
        for i in range(0, len(targets), 3):
            lines.extend(
                f"{indent}{decomp_line}"
                for decomp_line in controlled_gate_decomposition_lines(
                    gate, targets[i], targets[i + 1], targets[i + 2], tag=tag or ""
                )
            )
    return "\n".join(lines)


def _replace_t_family_shorthand(tsim_gate: str) -> Callable[[re.Match[str]], str]:
    def replace(m: re.Match[str]) -> str:
        user_tag = decode_t_user_tag(m.group(1))
        tag_suffix = f"[{user_tag}]" if user_tag else ""
        return f"{tsim_gate}{tag_suffix}"

    return replace


def enriched_stim_error(exc: ValueError, converted_text: str) -> ValueError:
    """Improve stim parse errors for tsim-specific gates.

    When stim raises a 'Gate not found' error for a gate that should have been
    converted by shorthand_to_stim, this searches the converted text for the
    unconverted usage and returns a more helpful error message.
    """
    m = _GATE_NOT_FOUND_RE.search(str(exc))
    if not m or m.group(1) not in _TSIM_GATES:
        return exc
    # Successfully converted gates live inside tags (e.g. I[R_Z(...)]), which are
    # dropped so that they can't match. Tags go first since they may contain '#'.
    code = _COMMENT_RE.sub("", _TAG_RE.sub("", converted_text))
    usage = _GATE_USAGE_RE.search(code)
    if not usage:
        return exc
    return ValueError(f"Could not parse '{usage.group()}' in program text.")


def shorthand_to_stim(text: str) -> str:
    """Convert tsim shorthand syntax to valid stim instructions.

    Converts:
        CCZ 0 1 2           → Clifford+T decomposition of CCZ
        CCX 0 1 2           → Clifford+T decomposition of CCX/Toffoli
        T 0 1               → S[T] 0 1
        T_DAG 0 1           → S_DAG[T] 0 1
        TPP X0*Y1           → SPP[T] X0*Y1
        TPP_DAG X0*Y1       → SPP_DAG[T] X0*Y1
        R_Z(0.3) 0          → I[R_Z(theta=0.3*pi)] 0
        R_X(0.25) 0         → I[R_X(theta=0.25*pi)] 0
        R_Y(-0.5) 0         → I[R_Y(theta=-0.5*pi)] 0
        R_XX(0.5) 0 1       → SPP[R_PAULI(theta=0.5*pi)] X0*X1
        R_YY(0.5) 0 1       → SPP[R_PAULI(theta=0.5*pi)] Y0*Y1
        R_ZZ(0.5) 0 1       → SPP[R_PAULI(theta=0.5*pi)] Z0*Z1
        R_PAULI(0.3) X0*Y1  → SPP[R_PAULI(theta=0.3*pi)] X0*Y1
        U3(0.3, 0.24, 0.49) 0 → I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0

    Tags and comments are left untouched, except that parametric tags on ``I``,
    ``SPP`` and ``SPP_DAG`` are canonicalized to the keyword form above, so that
    e.g. ``I[R_X(0.25)] 0`` and ``R_X(0.25) 0`` produce the same instruction.

    """
    text = _expand_controlled_gates(text)
    return "\n".join(_convert_line(line) for line in text.split("\n"))


def _convert_line(line: str) -> str:
    match = _INSTRUCTION_RE.match(line)
    if not match:
        return line
    indent, name, tag, rest = match.groups()
    rest, sep, comment = rest.partition("#")

    if name in _T_FAMILY:
        instruction = f"{_T_FAMILY[name]}[{encode_t_tag(tag or '')}]{rest}"
    elif name in _ROTATION_SHORTHAND_GATES and tag is None:
        instruction = _expand_rotation_shorthand(name + rest)
    elif name.upper() in _PARAMETRIC_TAG_GATES and tag:
        instruction = f"{name}[{_canonical_parametric_tag(tag)}]{rest}"
    else:
        return line
    return f"{indent}{instruction}{sep}{comment}"


def _canonical_parametric_tag(tag: str) -> str:
    # Canonicalize so that equivalent inputs like `I[R_X(0.005)]`,
    # `I[R_X(theta=0.5e-2*pi)]` and `I[R_X(theta=0.005*pi)]` produce the same stim
    # tag string. This keeps `tsim.Circuit(str(c)) == c` round-trip stable across
    # notations. Malformed tags are kept as-is and rejected when parsed.
    try:
        parsed = parse_parametric_tag_text(tag)
        return tag if parsed is None else format_parametric_tag(*parsed)
    except (ValueError, OverflowError):
        return tag


def _expand_rotation_shorthand(instruction: str) -> str:
    def replace_pauli_pair(m: re.Match) -> str:
        pauli = m.group(1)
        alpha = float(m.group(2))
        q0, q1 = m.group(3), m.group(4)
        if q0 == q1:
            raise ValueError(
                f"R_{pauli}{pauli} target qubits must be distinct, got {q0} {q1}."
            )
        return f"SPP[R_PAULI(theta={alpha}*pi)] {pauli}{q0}*{pauli}{q1}"

    instruction = re.sub(
        rf"^R_([XYZ])\1\(({FLOAT_RE})\)\s+(\d+)\s+(\d+)",
        replace_pauli_pair,
        instruction,
    )

    def replace_pauli(m: re.Match) -> str:
        alpha = float(m.group(1))
        return f"SPP[R_PAULI(theta={alpha}*pi)] {m.group(2)}"

    instruction = re.sub(
        rf"^R_PAULI\(({FLOAT_RE})\)\s+((?:[XYZ]\d+)(?:\*[XYZ]\d+)*)",
        replace_pauli,
        instruction,
    )

    def replace_rotation(m: re.Match) -> str:
        axis = m.group(1)
        return f"I[R_{axis}(theta={float(m.group(2))}*pi)]"

    instruction = re.sub(rf"^R_([XYZ])\(({FLOAT_RE})\)", replace_rotation, instruction)

    def replace_u3(m: re.Match) -> str:
        theta, phi, lam = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return f"I[U3(theta={theta}*pi, phi={phi}*pi, lambda={lam}*pi)]"

    return re.sub(
        rf"^U3\(({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\)",
        replace_u3,
        instruction,
    )


def stim_to_shorthand(text: str) -> str:
    """Convert expanded stim annotations back to tsim shorthand.

    Rewrites:
    - I[U3(theta=θ*pi, phi=φ*pi, lambda=λ*pi)] → U3(θ, φ, λ)
    - I[R_X(theta=α*pi)] / I[R_Y(...)] / I[R_Z(...)] → R_X(α) / R_Y(α) / R_Z(α)
    - SPP[R_PAULI(theta=α*pi)] P0*P1 → R_PP(α) 0 1 (P ∈ {X, Y, Z})
    - SPP[R_PAULI(theta=α*pi)] X0*Y1 → R_PAULI(α) X0*Y1
    - SPP[T] → TPP
    - SPP_DAG[T] → TPP_DAG
    - S[T] → T
    - S_DAG[T] → T_DAG
    """

    # Replace I[U3(theta=θ*pi, phi=φ*pi, lambda=λ*pi)] with U3(θ, φ, λ)
    def replace_u3(m: re.Match) -> str:
        theta, phi, lam = m.group(1), m.group(2), m.group(3)
        return f"U3({theta}, {phi}, {lam})"

    text = re.sub(
        rf"\bI\[U3\(theta=({FLOAT_RE})\*pi, phi=({FLOAT_RE})\*pi, lambda=({FLOAT_RE})\*pi\)\]",
        replace_u3,
        text,
    )

    # SPP[R_PAULI(...)] with a same-axis two-qubit product → R_XX/R_YY/R_ZZ shorthand.
    # Must precede the general R_PAULI rewrite below.
    def replace_pauli_pair(m: re.Match) -> str:
        alpha, pauli, q0, q1 = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"R_{pauli}{pauli}({alpha}) {q0} {q1}"

    text = re.sub(
        rf"\bSPP\[R_PAULI\(theta=({FLOAT_RE})\*pi\)\] ([XYZ])(\d+)\*\2(\d+)(?!\*)\b",
        replace_pauli_pair,
        text,
    )

    def replace_pauli(m: re.Match) -> str:
        alpha, product = m.group(1), m.group(2)
        return f"R_PAULI({alpha}) {product}"

    text = re.sub(
        rf"\bSPP\[R_PAULI\(theta=({FLOAT_RE})\*pi\)\] ((?:[XYZ]\d+)(?:\*[XYZ]\d+)*)",
        replace_pauli,
        text,
    )

    # Replace I[R_X(...)] / I[R_Y(...)] / I[R_Z(...)] with R_X(α) / R_Y(α) / R_Z(α)
    def replace_rotation(m: re.Match) -> str:
        axis = m.group(1)
        angle = m.group(2)
        return f"R_{axis}({angle})"

    text = re.sub(
        rf"\bI\[R_([XYZ])\(theta=({FLOAT_RE})\*pi\)\]",
        replace_rotation,
        text,
    )

    # Replace SPP[T] and SPP_DAG[T] with TPP and TPP_DAG
    # Must come before S[T]/S_DAG[T] to avoid partial matches
    text = re.sub(
        r"(?<!\w)SPP_DAG\[(T(?::[^\]\n]*)?)\](?!\w)",
        _replace_t_family_shorthand("TPP_DAG"),
        text,
    )
    text = re.sub(
        r"(?<!\w)SPP\[(T(?::[^\]\n]*)?)\](?!\w)",
        _replace_t_family_shorthand("TPP"),
        text,
    )

    # Replace S[T] and S_DAG[T] with T and T_DAG
    # Use non-word lookarounds because trailing ] is not a word character.
    text = re.sub(
        r"(?<!\w)S_DAG\[(T(?::[^\]\n]*)?)\](?!\w)",
        _replace_t_family_shorthand("T_DAG"),
        text,
    )
    text = re.sub(
        r"(?<!\w)S\[(T(?::[^\]\n]*)?)\](?!\w)",
        _replace_t_family_shorthand("T"),
        text,
    )

    return text
