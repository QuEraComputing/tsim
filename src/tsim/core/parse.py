"""Parser for converting stim circuits to ZX graph representations."""

import re
from collections.abc import Iterator
from fractions import Fraction
from typing import Literal

import stim

from tsim.core.instructions import (
    GATE_TABLE,
    GraphRepresentation,
    correlated_error,
    detector,
    finalize_correlated_error,
    mpad,
    mpp,
    observable_include,
    r_x,
    r_y,
    r_z,
    spp,
    tick,
    tpp,
    u3,
)


def parse_parametric_tag(tag: str) -> tuple[str, dict[str, Fraction]] | None:
    """Parse a parametric gate tag like R_Z(theta=0.3*pi).

    Supports gates: R_Z, R_X, R_Y, U3.

    Args:
        tag: The instruction tag to parse, e.g. "R_Z(theta=0.3*pi)" or
             "U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)".

    Returns:
        Tuple of (gate_name, params_dict) or None if not a valid parametric tag.

    """
    match = re.match(r"^(\w+)\((.*)\)$", tag)
    if not match:
        return None

    gate_name = match.group(1)
    params_str = match.group(2)

    params = {}
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        # Match param=value*pi (value can be negative/decimal/scientific)
        param_match = re.match(
            r"^(\w+)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\*pi$", param
        )
        if not param_match:
            return None
        param_name = param_match.group(1)
        value = Fraction(param_match.group(2))
        params[param_name] = value

    return gate_name, params


_PAULI_PRODUCT: dict[
    tuple[Literal["X", "Y", "Z"], Literal["X", "Y", "Z"]],
    tuple[Literal["X", "Y", "Z"], int],
] = {
    ("X", "Y"): ("Z", 1),  # XY = iZ
    ("X", "Z"): ("Y", 3),  # XZ = -iY
    ("Y", "X"): ("Z", 3),  # YX = -iZ
    ("Y", "Z"): ("X", 1),  # YZ = iX
    ("Z", "X"): ("Y", 1),  # ZX = iY
    ("Z", "Y"): ("X", 3),  # ZY = -iX
}


def _iter_pauli_products(
    instruction: stim.CircuitInstruction,
) -> Iterator[tuple[list[tuple[Literal["X", "Y", "Z"], int]], bool]]:
    """Yield (paulis, invert) for each Pauli product in an instruction.

    Tracks Pauli algebra when the same qubit appears multiple times:
    same Pauli cancels (P·P = I), different Paulis combine with a sign
    (e.g. Z·X = iY).  An accumulated sign of -1 flips the invert flag;
    ±i raises ValueError (anti-Hermitian product), matching stim.
    """
    qubit_pauli: dict[int, Literal["X", "Y", "Z"]] = {}
    sign = 0  # accumulated power of i (mod 4)
    invert = False
    targets = instruction.targets_copy()

    for i, target in enumerate(targets):
        if target.is_combiner:
            continue

        if target.is_x_target:
            pauli_type: Literal["X", "Y", "Z"] = "X"
        elif target.is_y_target:
            pauli_type = "Y"
        elif target.is_z_target:
            pauli_type = "Z"
        else:
            raise ValueError(
                f"Invalid Pauli target in instruction {instruction.name}: {target}"
            )

        invert ^= target.is_inverted_result_target
        qubit = target.value

        if qubit not in qubit_pauli:
            qubit_pauli[qubit] = pauli_type
        elif qubit_pauli[qubit] == pauli_type:
            # P·P = I — cancel with no sign change
            del qubit_pauli[qubit]
        else:
            # Different Paulis on same qubit — combine with sign
            result, delta = _PAULI_PRODUCT[qubit_pauli[qubit], pauli_type]
            qubit_pauli[qubit] = result
            sign = (sign + delta) % 4

        next_idx = i + 1
        if next_idx >= len(targets) or not targets[next_idx].is_combiner:
            if sign % 2 == 1:
                raise ValueError(f"{instruction} acted on an anti-Hermitian operator")
            paulis: list[tuple[Literal["X", "Y", "Z"], int]] = [
                (p, q) for q, p in sorted(qubit_pauli.items())
            ]
            yield paulis, invert ^ (sign == 2)
            qubit_pauli = {}
            sign = 0
            invert = False


def parse_stim_circuit(
    stim_circuit: stim.Circuit,
    track_classical_wires: bool = False,
) -> GraphRepresentation:
    """Parse a stim circuit into a GraphRepresentation.

    Args:
        stim_circuit: The stim circuit to convert.
        track_classical_wires: Whether to track classical wires.

    Returns:
        A GraphRepresentation containing the ZX graph and all auxiliary data.

    """
    b = GraphRepresentation(track_classical_wires=track_classical_wires)

    for instruction in stim_circuit.flattened():
        assert not isinstance(instruction, stim.CircuitRepeatBlock)

        name = instruction.name
        if name == "SHIFT_COORDS":

            # TODO: handle visualization annotations in ZX diagrams
            continue

        if name == "S" and instruction.tag == "T":
            name = "T"
        elif name == "S_DAG" and instruction.tag == "T":
            name = "T_DAG"

        # Handle parametric gates via tags (e.g., I with tag "R_Z(theta=0.3*pi)")
        if name == "I" and instruction.tag:
            result = parse_parametric_tag(instruction.tag)
            if result is not None:
                gate_name, params = result
                targets = [t.value for t in instruction.targets_copy()]
                for qubit in targets:
                    if gate_name == "R_Z":
                        r_z(b, qubit, params["theta"])
                    elif gate_name == "R_X":
                        r_x(b, qubit, params["theta"])
                    elif gate_name == "R_Y":
                        r_y(b, qubit, params["theta"])
                    elif gate_name == "U3":
                        u3(b, qubit, params["theta"], params["phi"], params["lambda"])
                    else:
                        raise ValueError(f"Unknown parametric gate: {gate_name}")
                continue

        if name == "TICK":
            tick(b)
            continue
        if name == "MPP":
            args = instruction.gate_args_copy()
            p = args[0] if args else 0
            for paulis, invert in _iter_pauli_products(instruction):
                mpp(b, paulis, invert, p=p)
            continue
        if name in ("SPP", "SPP_DAG") and instruction.tag == "T":
            is_dag = name == "SPP_DAG"
            for paulis, invert in _iter_pauli_products(instruction):
                tpp(b, paulis, dagger=is_dag ^ invert)
            continue
        if name in ("SPP", "SPP_DAG"):
            is_dag = name == "SPP_DAG"
            for paulis, invert in _iter_pauli_products(instruction):
                spp(b, paulis, dagger=is_dag ^ invert)
            continue
        if name == "MPAD":
            args = instruction.gate_args_copy()
            p = args[0] if args else 0
            for target in instruction.targets_copy():
                mpad(b, target.value, p=p)
            continue
        if name == "E" or name == "ELSE_CORRELATED_ERROR":
            if name == "E":
                finalize_correlated_error(b)
            targets = [t.value for t in instruction.targets_copy()]
            types: list[Literal["X", "Y", "Z"]] = []
            for t in instruction.targets_copy():
                if t.is_x_target:
                    types.append("X")
                elif t.is_y_target:
                    types.append("Y")
                elif t.is_z_target:
                    types.append("Z")
                else:
                    raise ValueError(f"Invalid target: {t}")
            correlated_error(b, targets, types, instruction.gate_args_copy()[0])
            continue
        if name == "DETECTOR":
            targets = [t.value for t in instruction.targets_copy()]
            detector(b, targets)
            continue
        if name == "OBSERVABLE_INCLUDE":
            targets = [t.value for t in instruction.targets_copy()]
            args = instruction.gate_args_copy()
            observable_include(b, targets, int(args[0]))
            continue

        # instruction dispatch
        if name not in GATE_TABLE:
            raise ValueError(f"Unknown gate: {name}")

        gate_func, num_qubits = GATE_TABLE[name]
        targets = [t.value for t in instruction.targets_copy()]
        invert = [t.is_inverted_result_target for t in instruction.targets_copy()]
        is_classically_controlled = [
            t.is_measurement_record_target for t in instruction.targets_copy()
        ]
        args = instruction.gate_args_copy()

        for i_target in range(0, len(targets), num_qubits):
            chunk = targets[i_target : i_target + num_qubits]
            cc_chunk = is_classically_controlled[i_target : i_target + num_qubits]
            chunk_inverted = False
            for j in range(num_qubits):
                chunk_inverted ^= invert[i_target + j]
            assert not (invert[i_target] and is_classically_controlled[i_target])
            if chunk_inverted:
                gate_func(b, *chunk, *args, invert=True)
            elif any(cc_chunk):
                gate_func(b, *chunk, *args, classically_controlled=cc_chunk)
            else:
                gate_func(b, *chunk, *args)

    finalize_correlated_error(b)
    return b
