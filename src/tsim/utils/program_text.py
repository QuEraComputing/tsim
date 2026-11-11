"""Conversion utilities between Tsim shorthand and Stim program text."""

import re
from fractions import Fraction

import stim

from tsim.core.tags import (
    FLOAT_RE,
    decode_t_user_tag,
    encode_t_tag,
    format_parametric_tag,
    format_parsed_parametric_tag,
    is_t_tag,
    parse_parametric_tag_text,
)

# Controlled gates expanded into Clifford+T, with their number of qubit targets.
CONTROLLED_GATE_ARITY = {"CS": 2, "CS_DAG": 2, "CCZ": 3, "CCX": 3}
_TSIM_GATES = {
    *CONTROLLED_GATE_ARITY,
    "R_X",
    "R_Y",
    "R_Z",
    "R_XX",
    "R_YY",
    "R_ZZ",
    "R_PAULI",
    "U3",
}
_ROTATION_SHORTHAND_GATES = _TSIM_GATES - CONTROLLED_GATE_ARITY.keys()
_T_FAMILY = {"T": "S", "T_DAG": "S_DAG", "TPP": "SPP", "TPP_DAG": "SPP_DAG"}
_PARAMETRIC_TAG_GATES = {"I", "SPP", "SPP_DAG"}
_SINGLE_QUBIT_PARAMETRIC_GATES = {"R_X", "R_Y", "R_Z", "U3"}
_SAME_AXIS_PAIR_RE = re.compile(r"([XYZ])(\d+)\*\1(\d+)")
_T_FAMILY_SHORTHAND = {stim_name: name for name, stim_name in _T_FAMILY.items()}
# A tagged instruction on one line of Stim output, split into indentation, gate
# name, tag and the rest. Stim escapes `]` in tags, so a tag never contains one.
_STIM_LINE_RE = re.compile(r"^([ \t]*)(\w+)\[([^\]\n]*)\](.*)$")
# A tagged I/SPP/SPP_DAG instruction at the start of a line of Stim output. Stim
# escapes `]` in tags, so a tag never contains one.
_STIM_PARAMETRIC_INSTRUCTION_RE = re.compile(
    r"^[ \t]*(I|SPP|SPP_DAG)\[([^\]\n]*)\]", re.MULTILINE
)
# The prefix group also takes block delimiters that share the instruction's line,
# as in `REPEAT 2 { H 0` or `} H 0`.
_INSTRUCTION_RE = re.compile(
    r"^((?:\s*(?:\}|(?i:REPEAT)(?:\[[^\]\n]*\])?\s+\d+\s*\{))*\s*)"
    r"(\w+)(?:\[([^\]\n]*)\])?(.*)$"
)
_TAG_RE = re.compile(r"\[[^\]\n]*\]")
_COMMENT_RE = re.compile(r"#.*")
_GATE_NOT_FOUND_RE = re.compile(r"Gate not found: '(\w+)'")
_GATE_USAGE_RE = re.compile(
    r"(?<!\[)\b(CS_DAG\b|CS\b|CCZ\b|CCX\b|R_PAULI\([^)]*\)|R_[XYZ]{1,2}\([^)]*\)|R_[XYZ]\b|U3\([^)]*\)|U3\b)",
    re.IGNORECASE,
)


def controlled_gate_decomposition_lines(
    gate: str, *qubits: int | str, tag: str = ""
) -> list[str]:
    """Return a Clifford+T decomposition of a controlled gate on ``qubits``.

    ``CS`` and ``CS_DAG`` take two qubits; ``CCZ`` and ``CCX`` take two controls
    followed by the target.
    """
    if gate not in CONTROLLED_GATE_ARITY:
        raise ValueError(f"Unsupported controlled gate: {gate!r}")
    if len({int(q) for q in qubits}) != len(qubits):
        targets = " ".join(str(q) for q in qubits)
        raise ValueError(f"{gate} target qubits must be distinct, got {targets}.")

    def tagged(name: str) -> str:
        return f"{name}[{tag}]" if tag else name

    if gate in ("CS", "CS_DAG"):
        # CS = diag(1, 1, 1, i): the phase i^(ab) = ω^(a + b - a⊕b) with ω = e^(iπ/4).
        a, b = str(qubits[0]), str(qubits[1])
        t, t_dag = ("T", "T_DAG") if gate == "CS" else ("T_DAG", "T")
        return [
            f"{tagged(t)} {a} {b}",
            f"{tagged('CNOT')} {a} {b}",
            f"{tagged(t_dag)} {b}",
            f"{tagged('CNOT')} {a} {b}",
        ]

    a, b, c = (str(q) for q in qubits)
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
        match = _INSTRUCTION_RE.match(line)
        gate = match.group(2).upper() if match else ""
        if not match or gate not in CONTROLLED_GATE_ARITY:
            lines.append(line)
            continue

        # NOTE: the tag is matched before splitting off the comment, since tags
        # may contain '#'.
        prefix, _, tag, rest = match.groups()
        targets_text, sep, comment = rest.partition("#")
        arity = CONTROLLED_GATE_ARITY[gate]
        targets = targets_text.split()
        if (
            not targets
            or len(targets) % arity != 0
            or not all(target.isascii() and target.isdigit() for target in targets)
        ):
            raise ValueError(
                f"{gate} expects bare qubit integer targets in groups of {arity}."
            )

        # A block delimiter sharing the line, as in `REPEAT 2 { CS 0 1`, goes on
        # its own line before the decomposition.
        indent = prefix[: len(prefix) - len(prefix.lstrip())]
        if prefix.strip():
            lines.append(prefix.rstrip())
        if sep:
            lines.append(f"{indent}{sep}{comment}")
        for i in range(0, len(targets), arity):
            lines.extend(
                f"{indent}{decomp_line}"
                for decomp_line in controlled_gate_decomposition_lines(
                    gate, *targets[i : i + arity], tag=tag or ""
                )
            )
    return "\n".join(lines)


def enriched_stim_error(exc: ValueError, converted_text: str) -> ValueError:
    """Improve Stim parse errors for Tsim-specific gates.

    When Stim raises a 'Gate not found' error for a gate that should have been
    converted by shorthand_to_stim, this searches the converted text for the
    unconverted usage and returns a more helpful error message.
    """
    m = _GATE_NOT_FOUND_RE.search(str(exc))
    if not m or m.group(1).upper() not in _TSIM_GATES:
        return exc
    # Successfully converted gates live inside tags (e.g. I[R_Z(...)]), which are
    # dropped so that they can't match. Tags go first since they may contain '#'.
    code = _COMMENT_RE.sub("", _TAG_RE.sub("", converted_text))
    usage = _GATE_USAGE_RE.search(code)
    if not usage:
        return exc
    return ValueError(f"Could not parse '{usage.group()}' in program text.")


def shorthand_to_stim(text: str) -> str:
    """Convert Tsim shorthand syntax to valid Stim instructions.

    Converts:
        CS 0 1              → Clifford+T decomposition of CS
        CS_DAG 0 1          → Clifford+T decomposition of CS_DAG
        CCZ 0 1 2           → Clifford+T decomposition of CCZ
        CCX 0 1 2           → Clifford+T decomposition of CCX/Toffoli
        T 0 1               → S[T] 0 1
        T_DAG 0 1           → S_DAG[T] 0 1
        TPP X0*Y1           → SPP[T] X0*Y1
        TPP_DAG X0*Y1       → SPP_DAG[T] X0*Y1
        R_Z(0.3) 0          → I[R_Z(0.3)] 0
        R_X(0.25) 0         → I[R_X(0.25)] 0
        R_Y(-0.5) 0         → I[R_Y(-0.5)] 0
        R_XX(0.5) 0 1       → SPP[R_PAULI(0.5)] X0*X1
        R_YY(0.5) 0 1       → SPP[R_PAULI(0.5)] Y0*Y1
        R_ZZ(0.5) 0 1       → SPP[R_PAULI(0.5)] Z0*Z1
        R_PAULI(0.3) X0*Y1  → SPP[R_PAULI(0.3)] X0*Y1
        R_PAULI(0.3) !X0*Y1 → SPP[R_PAULI(0.3)] !X0*Y1
        U3(0.3, 0.24, 0.49) 0 → I[U3(0.3, 0.24, 0.49)] 0

    The tag of a parametric gate is the Tsim instruction itself, with angles in
    units of pi.

    Tags and comments are left untouched, except that parametric tags on ``I``,
    ``SPP`` and ``SPP_DAG`` are canonicalized to the form above, so that e.g.
    ``I[R_X(0.25e0)] 0``, ``I[R_X(theta=0.25*pi)] 0`` (the legacy keyword form) and
    ``R_X(0.25) 0`` produce the same instruction.

    """
    text = _expand_controlled_gates(text)
    return "\n".join(_convert_line(line) for line in text.split("\n"))


def _convert_line(line: str) -> str:
    match = _INSTRUCTION_RE.match(line)
    if not match:
        return line
    indent, name, tag, rest = match.groups()
    rest, sep, comment = rest.partition("#")
    # NOTE: Stim gate names are case-insensitive, and so are Tsim's.
    upper = name.upper()

    if upper in _T_FAMILY:
        instruction = f"{_T_FAMILY[upper]}[{encode_t_tag(tag or '')}]{rest}"
    elif upper in _ROTATION_SHORTHAND_GATES and tag is None:
        instruction = _expand_rotation_shorthand(upper + rest)
    elif upper in _PARAMETRIC_TAG_GATES and tag:
        instruction = f"{name}[{canonical_parametric_tag(tag)}]{rest}"
    else:
        return line
    return f"{indent}{instruction}{sep}{comment}"


def canonical_parametric_tag(tag: str) -> str:
    """Return ``tag`` in canonical positional form if it is a parametric tag."""
    # Canonicalize so that equivalent inputs like `I[R_X(0.005)]`,
    # `I[R_X(0.5e-2)]` and the legacy `I[R_X(theta=0.005*pi)]` produce the same
    # Stim tag string. This keeps `tsim.Circuit(str(c)) == c` round-trip stable
    # across notations. Malformed tags are kept as-is and rejected when parsed.
    parsed = _try_parse_parametric_tag(tag)
    return tag if parsed is None else format_parsed_parametric_tag(*parsed)


def canonicalize_parametric_tags(circuit: stim.Circuit) -> stim.Circuit:
    """Return a copy of ``circuit`` with parametric tags in canonical form.

    Rewrites tags on ``I``, ``SPP`` and ``SPP_DAG`` (including inside ``REPEAT``
    blocks), e.g. the legacy ``I[R_X(theta=0.1*pi)]`` to ``I[R_X(0.1)]``, so that
    circuits built from a ``stim.Circuit`` compare equal to the same circuit
    written as text. Circuits without parametric tags are only copied.
    """
    if not any(
        m.group(2) != canonical_parametric_tag(m.group(2))
        for m in _STIM_PARAMETRIC_INSTRUCTION_RE.finditer(str(circuit))
    ):
        return circuit.copy()
    return _rebuild_with_canonical_tags(circuit)


def _rebuild_with_canonical_tags(circuit: stim.Circuit) -> stim.Circuit:
    result = stim.Circuit()
    for instr in circuit:
        if isinstance(instr, stim.CircuitRepeatBlock):
            body = _rebuild_with_canonical_tags(instr.body_copy())
            result.append(
                stim.CircuitRepeatBlock(instr.repeat_count, body, tag=instr.tag)
            )
        elif instr.name in _PARAMETRIC_TAG_GATES and instr.tag:
            result.append(
                stim.CircuitInstruction(
                    instr.name,
                    instr.targets_copy(),
                    instr.gate_args_copy(),
                    tag=canonical_parametric_tag(instr.tag),
                )
            )
        else:
            result.append(instr)
    return result


def _try_parse_parametric_tag(tag: str) -> tuple[str, dict[str, Fraction]] | None:
    try:
        return parse_parametric_tag_text(tag)
    except (ValueError, OverflowError):
        return None


def _expand_rotation_shorthand(instruction: str) -> str:
    def replace_pauli_pair(m: re.Match) -> str:
        pauli = m.group(1)
        alpha = float(m.group(2))
        q0, q1 = m.group(3), m.group(4)
        if q0 == q1:
            raise ValueError(
                f"R_{pauli}{pauli} target qubits must be distinct, got {q0} {q1}."
            )
        return f"SPP[{format_parametric_tag('R_PAULI', alpha)}] {pauli}{q0}*{pauli}{q1}"

    instruction = re.sub(
        rf"^R_([XYZ])\1\(({FLOAT_RE})\)\s+(\d+)\s+(\d+)",
        replace_pauli_pair,
        instruction,
    )

    def replace_pauli(m: re.Match) -> str:
        alpha = float(m.group(1))
        return f"SPP[{format_parametric_tag('R_PAULI', alpha)}] {m.group(2)}"

    instruction = re.sub(
        rf"^R_PAULI\(({FLOAT_RE})\)\s+(!?[XYZxyz]\d+(?:\*!?[XYZxyz]\d+)*)",
        replace_pauli,
        instruction,
    )

    def replace_rotation(m: re.Match) -> str:
        axis = m.group(1)
        return f"I[{format_parametric_tag(f'R_{axis}', float(m.group(2)))}]"

    instruction = re.sub(rf"^R_([XYZ])\(({FLOAT_RE})\)", replace_rotation, instruction)

    def replace_u3(m: re.Match) -> str:
        theta, phi, lam = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return f"I[{format_parametric_tag('U3', theta, phi, lam)}]"

    return re.sub(
        rf"^U3\(({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\)",
        replace_u3,
        instruction,
    )


def stim_to_shorthand(text: str) -> str:
    """Convert expanded Stim annotations back to Tsim shorthand.

    Rewrites (the legacy keyword tag form, e.g. ``R_X(theta=α*pi)``, is also read):
    - I[U3(θ, φ, λ)] → U3(θ, φ, λ)
    - I[R_X(α)] / I[R_Y(α)] / I[R_Z(α)] → R_X(α) / R_Y(α) / R_Z(α)
    - SPP[R_PAULI(α)] P0*P1 → R_PP(α) 0 1 (P ∈ {X, Y, Z})
    - SPP[R_PAULI(α)] X0*Y1 → R_PAULI(α) X0*Y1
    - SPP[R_PAULI(α)] !X0*Y1 → R_PAULI(α) !X0*Y1
    - SPP[T] → TPP
    - SPP_DAG[T] → TPP_DAG
    - S[T] → T
    - S_DAG[T] → T_DAG
    """
    return "\n".join(_shorthand_line(line) for line in text.split("\n"))


def _shorthand_line(line: str) -> str:
    # Stim prints one instruction per line, so only the gate at the start of a line
    # is rewritten, never text inside another instruction's tag.
    match = _STIM_LINE_RE.match(line)
    if not match:
        return line
    indent, name, tag, rest = match.groups()

    if name in _T_FAMILY_SHORTHAND and is_t_tag(tag):
        user_tag = decode_t_user_tag(tag)
        tag_suffix = f"[{user_tag}]" if user_tag else ""
        return f"{indent}{_T_FAMILY_SHORTHAND[name]}{tag_suffix}{rest}"

    parsed = _try_parse_parametric_tag(tag)
    if parsed is None:
        return line
    shorthand = format_parsed_parametric_tag(*parsed)
    if name == "I" and parsed[0] in _SINGLE_QUBIT_PARAMETRIC_GATES:
        return f"{indent}{shorthand}{rest}"
    if name == "SPP" and parsed[0] == "R_PAULI":
        # Only a lone same-axis pair is R_XX/R_YY/R_ZZ; further products would be
        # misread as qubit targets of the two-qubit shorthand.
        pair = _SAME_AXIS_PAIR_RE.fullmatch(rest.strip())
        if pair:
            pauli, q0, q1 = pair.groups()
            angle = shorthand.removeprefix("R_PAULI")
            return f"{indent}R_{pauli}{pauli}{angle} {q0} {q1}"
        return f"{indent}{shorthand}{rest}"
    return line
