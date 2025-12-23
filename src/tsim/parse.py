import re
from fractions import Fraction

import stim

from tsim._instructions import (
    GATE_TABLE,
    GraphRepresentation,
    detector,
    mpp,
    observable_include,
    r_x,
    r_y,
    r_z,
    tick,
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
        # Match param=value*pi (value can be negative/decimal)
        param_match = re.match(r"^(\w+)=([-+]?[\d.]+)\*pi$", param)
        if not param_match:
            return None
        param_name = param_match.group(1)
        value = Fraction(param_match.group(2))
        params[param_name] = value

    return gate_name, params


def parse_stim_circuit(
    stim_circuit: stim.Circuit,
) -> GraphRepresentation:
    """Parse a stim circuit into a GraphRepresentation.

    Args:
        stim_circuit: The stim circuit to convert.

    Returns:
        A GraphRepresentation containing the ZX graph and all auxiliary data.
    """
    b = GraphRepresentation()

    for instruction in stim_circuit.flattened():
        assert not isinstance(instruction, stim.CircuitRepeatBlock)

        name = instruction.name
        if name in ["QUBIT_COORDS", "SHIFT_COORDS"]:
            # TODO: handle these visualization annotations
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
            args = str(instruction).split(" ")[1:]
            mpp(b, args)
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
            assert not (invert[i_target] and is_classically_controlled[i_target])
            if invert[i_target]:
                gate_func(b, *chunk, *args, invert=True)
            elif any(cc_chunk):
                gate_func(b, *chunk, *args, classically_controlled=cc_chunk)
            else:
                gate_func(b, *chunk, *args)

    return b
