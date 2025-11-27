import stim

from tsim._instructions import (
    GATE_TABLE,
    GraphRepresentation,
    detector,
    mpp,
    observable_include,
    tick,
)


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
        args = instruction.gate_args_copy()

        for i_target in range(0, len(targets), num_qubits):
            chunk = targets[i_target : i_target + num_qubits]
            if invert[i_target]:
                gate_func(b, *chunk, *args, invert=True)
            else:
                gate_func(b, *chunk, *args)

    return b
