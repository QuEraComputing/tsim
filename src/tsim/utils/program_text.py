"""Conversion utilities between tsim shorthand and stim program text."""

import re

# Matches valid numeric literals including scientific notation (e.g. 0.5, 4e-4, 1.2e3)
FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

_TSIM_GATES = {"R_X", "R_Y", "R_Z", "R_XX", "R_YY", "R_ZZ", "R_PAULI", "U3"}
_GATE_NOT_FOUND_RE = re.compile(r"Gate not found: '(\w+)'")
_GATE_USAGE_RE = re.compile(
    r"(?<!\[)\b(R_(?:XX|YY|ZZ|PAULI|[XYZ])\([^)]*\)|R_(?:XX|YY|ZZ|PAULI|[XYZ])\b|U3\([^)]*\)|U3\b)"
)


def enriched_stim_error(exc: ValueError, converted_text: str) -> ValueError:
    """Improve stim parse errors for tsim-specific gates.

    When stim raises a 'Gate not found' error for a gate that should have been
    converted by shorthand_to_stim, this searches the converted text for the
    unconverted usage and returns a more helpful error message.
    """
    m = _GATE_NOT_FOUND_RE.search(str(exc))
    if not m or m.group(1) not in _TSIM_GATES:
        return exc
    # Successfully converted gates live inside brackets (e.g. I[R_Z(...)]) and won't match.
    usage = _GATE_USAGE_RE.search(converted_text)
    if not usage:
        return exc
    return ValueError(f"Could not parse '{usage.group()}' in program text.")


def shorthand_to_stim(text: str) -> str:
    """Convert tsim shorthand syntax to valid stim instructions.

    Converts::

        T 0 1               → S[T] 0 1
        T_DAG 0 1           → S_DAG[T] 0 1
        TPP X0*Y1           → SPP[T] X0*Y1
        TPP_DAG X0*Y1       → SPP_DAG[T] X0*Y1
        R_Z(0.3) 0          → I[R_Z(theta=0.3*pi)] 0
        R_X(0.25) 0         → I[R_X(theta=0.25*pi)] 0
        R_Y(-0.5) 0         → I[R_Y(theta=-0.5*pi)] 0
        U3(0.3, 0.24, 0.49) 0 → I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0
        R_XX(0.25) 0 1      → SPP[R_XX(theta=0.25*pi)] X0*X1
        R_YY(0.25) 0 1      → SPP[R_YY(theta=0.25*pi)] Y0*Y1
        R_ZZ(0.25) 0 1      → SPP[R_ZZ(theta=0.25*pi)] Z0*Z1
        R_PAULI(0.25) X0*Y1 → SPP[R_PAULI(theta=0.25*pi)] X0*Y1

    """
    # TPP_DAG/TPP must come before T_DAG/T to avoid partial matches
    # (?<!\[) ensures we don't match T inside [T]
    text = re.sub(r"(?<!\[)\bTPP_DAG\b(?!\[)", "SPP_DAG[T]", text)
    text = re.sub(r"(?<!\[)\bTPP\b(?!\[)", "SPP[T]", text)
    text = re.sub(r"(?<!\[)\bT_DAG\b(?!\[)", "S_DAG[T]", text)
    text = re.sub(r"(?<!\[)\bT\b(?!\[)", "S[T]", text)

    # R_XX/R_YY/R_ZZ must come before R_X/R_Y/R_Z to avoid partial matches
    def replace_r_pp(m: re.Match) -> str:
        pauli = m.group(1)  # "XX", "YY", or "ZZ"
        angle = float(m.group(2))
        q0, q1 = m.group(3), m.group(4)
        if q0 == q1:
            raise ValueError(
                f"Duplicate target qubits in R_{pauli}: both targets are qubit {q0}."
            )
        p = pauli[0]
        return f"SPP[R_{pauli}(theta={angle}*pi)] {p}{q0}*{p}{q1}"

    text = re.sub(
        rf"\bR_(XX|YY|ZZ)\(({FLOAT_RE})\)\s+(\d+)\s+(\d+)",
        replace_r_pp,
        text,
    )

    def replace_r_pauli(m: re.Match) -> str:
        angle = float(m.group(1))
        targets = m.group(2)
        return f"SPP[R_PAULI(theta={angle}*pi)] {targets}"

    text = re.sub(
        rf"\bR_PAULI\(({FLOAT_RE})\)\s+((?:[XYZ]\d+)(?:\*[XYZ]\d+)*)",
        replace_r_pauli,
        text,
    )

    def replace_rotation(m: re.Match) -> str:
        axis = m.group(1)
        return f"I[R_{axis}(theta={float(m.group(2))}*pi)]"

    text = re.sub(rf"\bR_([XYZ])\(({FLOAT_RE})\)", replace_rotation, text)

    def replace_u3(m: re.Match) -> str:
        theta, phi, lam = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return f"I[U3(theta={theta}*pi, phi={phi}*pi, lambda={lam}*pi)]"

    text = re.sub(
        rf"\bU3\(({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\)",
        replace_u3,
        text,
    )

    # Canonicalize literals inside already-expanded parametric tags so that
    # equivalent inputs like `I[R_X(theta=0.5e-2*pi)]` and
    # `I[R_X(theta=0.005*pi)]` produce the same stim tag string. This keeps
    # `tsim.Circuit(str(c)) == c` round-trip stable across notations.
    def _canonicalize_param(m: re.Match) -> str:
        return f"{m.group(1)}={float(m.group(2))}*pi"

    text = re.sub(
        rf"\b(theta|phi|lambda)=({FLOAT_RE})\*pi",
        _canonicalize_param,
        text,
    )

    return text


def stim_to_shorthand(text: str) -> str:
    """Convert expanded stim annotations back to tsim shorthand.

    Rewrites:
    - I[U3(theta=θ*pi, phi=φ*pi, lambda=λ*pi)] → U3(θ, φ, λ)
    - I[R_X(theta=α*pi)] / I[R_Y(...)] / I[R_Z(...)] → R_X(α) / R_Y(α) / R_Z(α)
    - SPP[R_XX(theta=α*pi)] X0*X1 → R_XX(α) 0 1
    - SPP[R_YY(theta=α*pi)] Y0*Y1 → R_YY(α) 0 1
    - SPP[R_ZZ(theta=α*pi)] Z0*Z1 → R_ZZ(α) 0 1
    - SPP[R_PAULI(theta=α*pi)] X0*Y1*Z2 → R_PAULI(α) X0*Y1*Z2
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

    # Replace SPP[R_XX/R_YY/R_ZZ(...)] P0*P1 with R_XX/R_YY/R_ZZ(α) q0 q1
    # Must come before the general R_PAULI rewrite below.
    def replace_spp_r_pp(m: re.Match) -> str:
        pauli = m.group(1)  # "XX", "YY", or "ZZ"
        angle = m.group(2)
        q0 = m.group(3)
        q1 = m.group(4)
        return f"R_{pauli}({angle}) {q0} {q1}"

    text = re.sub(
        rf"\bSPP\[R_(XX|YY|ZZ)\(theta=({FLOAT_RE})\*pi\)\]\s+[XYZ](\d+)\*[XYZ](\d+)",
        replace_spp_r_pp,
        text,
    )

    # Replace SPP[R_PAULI(...)] with R_PAULI(α) <pauli-product>
    def replace_spp_r_pauli(m: re.Match) -> str:
        angle = m.group(1)
        targets = m.group(2)
        return f"R_PAULI({angle}) {targets}"

    text = re.sub(
        rf"\bSPP\[R_PAULI\(theta=({FLOAT_RE})\*pi\)\]\s+((?:[XYZ]\d+)(?:\*[XYZ]\d+)*)",
        replace_spp_r_pauli,
        text,
    )

    # Replace SPP[T] and SPP_DAG[T] with TPP and TPP_DAG
    # Must come before S[T]/S_DAG[T] to avoid partial matches
    text = re.sub(r"(?<!\w)SPP_DAG\[T\](?!\w)", "TPP_DAG", text)
    text = re.sub(r"(?<!\w)SPP\[T\](?!\w)", "TPP", text)

    # Replace S[T] and S_DAG[T] with T and T_DAG
    # Use non-word lookarounds because trailing ] is not a word character.
    text = re.sub(r"(?<!\w)S_DAG\[T\](?!\w)", "T_DAG", text)
    text = re.sub(r"(?<!\w)S\[T\](?!\w)", "T", text)

    return text
