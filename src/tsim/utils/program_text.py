import re


def shorthand_to_stim(text: str) -> str:
    """Convert tsim shorthand syntax to valid stim instructions.

    Converts:
        T 0 1           → S[T] 0 1
        T_DAG 0 1       → S_DAG[T] 0 1
        R_Z(0.3) 0      → I[R_Z(theta=0.3*pi)] 0
        R_X(0.25) 0     → I[R_X(theta=0.25*pi)] 0
        R_Y(-0.5) 0     → I[R_Y(theta=-0.5*pi)] 0
        U3(0.3, 0.24, 0.49) 0 → I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0
    """
    # T_DAG must come before T to avoid partial matches
    # (?<!\[) ensures we don't match T inside [T]
    text = re.sub(r"(?<!\[)\bT_DAG\b(?!\[)", "S_DAG[T]", text)
    text = re.sub(r"(?<!\[)\bT\b(?!\[)", "S[T]", text)

    # R_Z(angle), R_X(angle), R_Y(angle)
    def replace_rotation(m: re.Match) -> str:
        axis = m.group(1)
        angle = m.group(2)
        return f"I[R_{axis}(theta={angle}*pi)]"

    text = re.sub(r"\bR_([XYZ])\(([-+]?[\d.]+)\)", replace_rotation, text)

    # U3(theta, phi, lambda)
    def replace_u3(m: re.Match) -> str:
        theta, phi, lam = m.group(1), m.group(2), m.group(3)
        return f"I[U3(theta={theta}*pi, phi={phi}*pi, lambda={lam}*pi)]"

    text = re.sub(
        r"\bU3\(([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\)",
        replace_u3,
        text,
    )

    return text


def stim_to_shorthand(text: str) -> str:
    """Convert expanded stim annotations back to tsim shorthand.

    Rewrites:
    - I[U3(theta=θ*pi, phi=φ*pi, lambda=λ*pi)] → U3(θ, φ, λ)
    - I[R_X(theta=α*pi)] / I[R_Y(...)] / I[R_Z(...)] → R_X(α) / R_Y(α) / R_Z(α)
    - S[T] → T
    - S_DAG[T] → T_DAG
    """

    # Replace I[U3(theta=θ*pi, phi=φ*pi, lambda=λ*pi)] with U3(θ, φ, λ)
    def replace_u3(m: re.Match) -> str:
        theta, phi, lam = m.group(1), m.group(2), m.group(3)
        return f"U3({theta}, {phi}, {lam})"

    text = re.sub(
        r"\bI\[U3\(theta=([-+]?[\d.]+)\*pi, phi=([-+]?[\d.]+)\*pi, lambda=([-+]?[\d.]+)\*pi\)\]",
        replace_u3,
        text,
    )

    # Replace I[R_X(...)] / I[R_Y(...)] / I[R_Z(...)] with R_X(α) / R_Y(α) / R_Z(α)
    def replace_rotation(m: re.Match) -> str:
        axis = m.group(1)
        angle = m.group(2)
        return f"R_{axis}({angle})"

    text = re.sub(
        r"\bI\[R_([XYZ])\(theta=([-+]?[\d.]+)\*pi\)\]",
        replace_rotation,
        text,
    )

    # Replace S[T] and S_DAG[T] with T and T_DAG
    # Use non-word lookarounds because trailing ] is not a word character.
    text = re.sub(r"(?<!\w)S_DAG\[T\](?!\w)", "T_DAG", text)
    text = re.sub(r"(?<!\w)S\[T\](?!\w)", "T", text)

    return text
