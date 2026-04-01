"""Mapping tables for converting parametric rotations with half-pi angles to Clifford gates."""

from __future__ import annotations

from fractions import Fraction

# Clifford decompositions for U3(θ, φ, λ) = R_Z(φ) · R_Y(θ) · R_Z(λ).
# Keys: (θ_idx, φ_idx, λ_idx) where each index ∈ {0,1,2,3} is the angle in half-pi units.
# Values: stim gate names in circuit (time) order.
U3_CLIFFORD: dict[tuple[int, int, int], list[str]] = {
    (0, 0, 0): ["I"],
    (0, 0, 1): ["S"],
    (0, 0, 2): ["Z"],
    (0, 0, 3): ["S_DAG"],
    (0, 1, 0): ["S"],
    (0, 1, 1): ["Z"],
    (0, 1, 2): ["S_DAG"],
    (0, 1, 3): ["I"],
    (1, 0, 0): ["SQRT_Y"],
    (1, 0, 1): ["S", "SQRT_Y"],
    (1, 0, 2): ["H"],
    (1, 0, 3): ["S_DAG", "SQRT_Y"],
    (1, 1, 0): ["S", "SQRT_X_DAG"],
    (1, 1, 1): ["Z", "SQRT_X_DAG"],
    (1, 1, 2): ["S_DAG", "SQRT_X_DAG"],
    (1, 1, 3): ["SQRT_X_DAG"],
    (1, 2, 0): ["Z", "SQRT_Y_DAG"],
    (1, 2, 1): ["S_DAG", "SQRT_Y_DAG"],
    (1, 2, 2): ["SQRT_Y_DAG"],
    (1, 2, 3): ["S", "SQRT_Y_DAG"],
    (1, 3, 0): ["S_DAG", "SQRT_X"],
    (1, 3, 1): ["SQRT_X"],
    (1, 3, 2): ["S", "SQRT_X"],
    (1, 3, 3): ["Z", "SQRT_X"],
    (2, 0, 0): ["Y"],
    (2, 0, 1): ["S", "Y"],
    (2, 0, 2): ["X"],
    (2, 0, 3): ["S_DAG", "Y"],
    (2, 1, 0): ["Y", "S"],
    (2, 1, 1): ["Y"],
    (2, 1, 2): ["S", "Y"],
    (2, 1, 3): ["X"],
}

RZ_CLIFFORD: dict[int, str] = {0: "I", 1: "S", 2: "Z", 3: "S_DAG"}
RX_CLIFFORD: dict[int, str] = {0: "I", 1: "SQRT_X", 2: "X", 3: "SQRT_X_DAG"}
RY_CLIFFORD: dict[int, str] = {0: "I", 1: "SQRT_Y", 2: "Y", 3: "SQRT_Y_DAG"}


def _to_half_pi_index(phase: Fraction) -> int | None:
    """Convert a phase (in units of π) to a half-π index 0–3, or *None*."""
    if phase.denominator > 2:
        return None
    return int(phase * 2) % 4


def _equivalent_u3_key(t: int, p: int, l: int) -> tuple[int, int, int]:
    """U3(θ, φ, λ) ≡ U3(2π-θ, φ+π, λ+π) up to global phase."""
    return ((4 - t) % 4, (p + 2) % 4, (l + 2) % 4)


def parametric_to_clifford_gates(
    gate_name: str, params: dict[str, Fraction]
) -> list[str] | None:
    """Convert a parametric gate with half-π angles to stim Clifford gate names.

    Args:
        gate_name: One of ``"R_X"``, ``"R_Y"``, ``"R_Z"``, ``"U3"``.
        params: Dict as returned by :func:`~tsim.core.parse.parse_parametric_tag`.

    Returns:
        Stim gate names in circuit order,
        or ``None`` when the angles are not half-π multiples.

    """
    if gate_name in ("R_X", "R_Y", "R_Z"):
        idx = _to_half_pi_index(params["theta"])
        if idx is None:
            return None
        table = {"R_Z": RZ_CLIFFORD, "R_X": RX_CLIFFORD, "R_Y": RY_CLIFFORD}[gate_name]
        return [table[idx]]

    if gate_name == "U3":
        theta_idx = _to_half_pi_index(params["theta"])
        phi_idx = _to_half_pi_index(params["phi"])
        lam_idx = _to_half_pi_index(params["lambda"])
        if theta_idx is None or phi_idx is None or lam_idx is None:
            return None

        key = (theta_idx, phi_idx, lam_idx)
        gates = U3_CLIFFORD.get(key)
        if gates is None:
            gates = U3_CLIFFORD.get(_equivalent_u3_key(*key))
        assert gates is not None
        return list(gates)

    return None
