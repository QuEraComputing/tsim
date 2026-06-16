"""Mapping tables for converting parametric rotations with half-pi angles to Clifford gates."""

from __future__ import annotations

from fractions import Fraction

import stim

from tsim.core.parse import parse_parametric_tag
from tsim.core.tags import is_t_tag

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
RXX_CLIFFORD: dict[int, str] = {0: "I", 1: "SQRT_XX", 3: "SQRT_XX_DAG"}
RYY_CLIFFORD: dict[int, str] = {0: "I", 1: "SQRT_YY", 3: "SQRT_YY_DAG"}
RZZ_CLIFFORD: dict[int, str] = {0: "I", 1: "SQRT_ZZ", 3: "SQRT_ZZ_DAG"}


def _to_half_pi_index(phase: Fraction) -> int | None:
    """Convert a phase (in units of π) to a half-π index 0–3, or *None*."""
    if phase.denominator > 2:
        return None
    return int(phase * 2) % 4


def _equivalent_u3_key(t: int, p: int, lam: int) -> tuple[int, int, int]:
    """U3(θ, φ, λ) ≡ U3(2π-θ, φ+π, λ+π) up to global phase."""
    return ((4 - t) % 4, (p + 2) % 4, (lam + 2) % 4)


def parametric_to_clifford_gates(
    gate_name: str, params: dict[str, Fraction]
) -> list[str] | None:
    """Convert a parametric gate with half-π angles to stim Clifford gate names.

    For single-qubit gates (``R_X``, ``R_Y``, ``R_Z``) and ``U3``, returns
    stim gate names in circuit order.  For two-qubit gates (``R_XX``, ``R_YY``,
    ``R_ZZ``) use :func:`_expand_clifford_gates` which handles per-gate targets.

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

    if gate_name in ("R_XX", "R_YY", "R_ZZ"):
        idx = _to_half_pi_index(params["theta"])
        if idx is None:
            return None
        if idx == 2:
            return None  # decomposed via _expand_clifford_gates
        table = {
            "R_XX": RXX_CLIFFORD,
            "R_YY": RYY_CLIFFORD,
            "R_ZZ": RZZ_CLIFFORD,
        }[gate_name]
        return [table[idx]]

    if gate_name == "R_PAULI":
        return None

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


def _expand_clifford_gates(
    gate_name: str,
    params: dict[str, Fraction],
    targets: list[int],
) -> list[tuple[str, list[int]]] | None:
    """Convert a parametric gate with half-π angles to a sequence of stim
    Clifford gate operations.

    Each returned tuple is ``(stim_gate_name, qubit_targets)``.

    Args:
        gate_name: One of ``"R_X"``, ``"R_Y"``, ``"R_Z"``, ``"R_XX"``,
            ``"R_YY"``, ``"R_ZZ"``, ``"R_PAULI"``, ``"U3"``.
        params: Dict as returned by :func:`~tsim.core.parse.parse_parametric_tag`.
        targets: Qubit target indices from the original instruction.

    Returns:
        List of ``(gate_name, targets)`` tuples in circuit order,
        or ``None`` when the angles are not half-π multiples.

    """
    if gate_name in ("R_X", "R_Y", "R_Z"):
        idx = _to_half_pi_index(params["theta"])
        if idx is None:
            return None
        table = {"R_Z": RZ_CLIFFORD, "R_X": RX_CLIFFORD, "R_Y": RY_CLIFFORD}[gate_name]
        return [(table[idx], targets)]

    if gate_name in ("R_XX", "R_YY", "R_ZZ"):
        idx = _to_half_pi_index(params["theta"])
        if idx is None:
            return None
        if idx == 2:
            # Full Pauli product: decompose into stim-native gate sequences
            # since single-gate names like "XX" are not stim gates.
            # Handle multi-pair by iterating over pairs.
            result: list[tuple[str, list[int]]] = []
            if gate_name == "R_ZZ":
                # ZZ = CNOT·(I⊗Z)·CNOT
                for i in range(0, len(targets), 2):
                    q0, q1 = targets[i], targets[i + 1]
                    result += [("CNOT", [q0, q1]), ("Z", [q1]), ("CNOT", [q0, q1])]
            elif gate_name == "R_XX":
                # XX = CNOT·(X⊗I)·CNOT
                for i in range(0, len(targets), 2):
                    q0, q1 = targets[i], targets[i + 1]
                    result += [("CNOT", [q0, q1]), ("X", [q0]), ("CNOT", [q0, q1])]
            elif gate_name == "R_YY":
                # YY = S·S · XX · S_DAG·S_DAG
                for i in range(0, len(targets), 2):
                    q0, q1 = targets[i], targets[i + 1]
                    result += [
                        ("S", [q0]), ("S", [q1]),
                        ("CNOT", [q0, q1]), ("X", [q0]), ("CNOT", [q0, q1]),
                        ("S_DAG", [q0]), ("S_DAG", [q1]),
                    ]
            return result
        table = {
            "R_XX": RXX_CLIFFORD,
            "R_YY": RYY_CLIFFORD,
            "R_ZZ": RZZ_CLIFFORD,
        }[gate_name]
        return [(table[idx], targets)]

    if gate_name == "R_PAULI":
        return None

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
        return [(g, targets) for g in gates]

    return None


def is_clifford(source: stim.Circuit) -> bool:
    """Return True iff every instruction in ``source`` is Clifford.

    Recurses into ``REPEAT`` block bodies.
    """

    def is_half_pi_multiple(phase: Fraction) -> bool:
        return phase.denominator <= 2

    for instr in source:
        if isinstance(instr, stim.CircuitRepeatBlock):
            if not is_clifford(instr.body_copy()):
                return False
            continue

        if instr.name in ["SPP", "SPP_DAG"] and instr.tag:
            result = parse_parametric_tag(instr)
            if result is not None:
                gate_name, params = result
                if gate_name == "R_PAULI":
                    if not is_half_pi_multiple(params["theta"]):
                        return False
            elif is_t_tag(instr.tag):
                return False

        if instr.name in ["S", "S_DAG"] and is_t_tag(instr.tag):
            return False

        if instr.name == "I" and instr.tag:
            result = parse_parametric_tag(instr)
            if result is None:
                continue

            gate_name, params = result
            if gate_name in ["R_X", "R_Y", "R_Z", "R_XX", "R_YY", "R_ZZ"]:
                if not is_half_pi_multiple(params["theta"]):
                    return False
            elif gate_name == "R_PAULI":
                if not is_half_pi_multiple(params["theta"]):
                    return False
            elif gate_name == "U3":
                if not all(
                    is_half_pi_multiple(params[name])
                    for name in ("theta", "phi", "lambda")
                ):
                    return False
            else:
                return False

    return True


def expand_clifford_rotations(source: stim.Circuit) -> stim.Circuit:
    """Return ``source`` with half-π parametric rotations expanded to Clifford gates.

    ``REPEAT`` blocks are preserved structurally and expanded recursively.
    """
    out = stim.Circuit()
    for instr in source:
        if isinstance(instr, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(
                    instr.repeat_count, expand_clifford_rotations(instr.body_copy())
                )
            )
            continue
        expansion = _try_clifford_expansion(instr)
        if expansion is not None:
            for gate_name, gate_targets in expansion:
                out.append(gate_name, gate_targets, [])
        else:
            out.append(instr)
    return out


def _try_clifford_expansion(
    instr: stim.CircuitInstruction,
) -> list[tuple[str, list[int]]] | None:
    """Try to expand a tagged ``I`` instruction into equivalent Clifford gates.

    Returns:
        List of ``(gate_name, targets)`` tuples in circuit order,
        or ``None`` if the instruction is not an expandable parametric rotation.

    """
    if instr.name != "I" or not instr.tag:
        return None

    parsed = parse_parametric_tag(instr)
    if parsed is None:
        return None

    gate_name, params = parsed
    targets = [t.value for t in instr.targets_copy()]
    return _expand_clifford_gates(gate_name, params, targets)
