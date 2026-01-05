import stim

import tsim


def broadcast_targets(
    groups: list[list[stim.GateTarget]], *, stride: int, offsets: list[int]
) -> list[int]:
    """Broadcast gate target groups with a stride and set of offsets."""
    out: list[int] = []
    for g in groups:
        for off in offsets:
            out.extend([t.value * stride + off for t in g])
    return out


def _transform_circuit(
    program_text: str,
    *,
    stride: int,
    offsets: list[int],
    gate_expansions: dict[str, list[str]] | None = None,
    used_qubits: set[int] | None = None,
    stabilizer_generators: list[list[int]] | None = None,
    observables: list[list[int]] | None = None,
) -> stim.Circuit:
    """Generic helper to expand/duplicate instructions with broadcast_targets."""
    stim_circ = tsim.Circuit(program_text)._stim_circ
    stim_circ = tsim.Circuit(program_text)._stim_circ
    mod_circ = stim.Circuit()

    for instr in stim_circ:
        assert not isinstance(instr, stim.CircuitRepeatBlock)

        if len(instr.targets_copy()) == 0:
            mod_circ.append_operation(instr)
            continue

        if used_qubits is not None:
            used_qubits |= {t.value for g in instr.target_groups() for t in g}

        # Special handling for detectors/observables using stabilizer structure.
        if instr.name == "DETECTOR" and stabilizer_generators:
            for gen in stabilizer_generators:
                targets = []
                for g in instr.target_groups():
                    for t in g:
                        targets.extend(
                            [stim.target_rec(t.value * stride + off) for off in gen]
                        )
                mod_circ.append(
                    instr.name, targets, instr.gate_args_copy(), tag=instr.tag
                )
            continue

        if instr.name == "OBSERVABLE_INCLUDE" and observables:
            for obs in observables:
                targets = []
                for g in instr.target_groups():
                    for t in g:
                        targets.extend(
                            [stim.target_rec(t.value * stride + off) for off in obs]
                        )
                mod_circ.append(
                    instr.name, targets, instr.gate_args_copy(), tag=instr.tag
                )
            continue

        new_ts = broadcast_targets(
            instr.target_groups(), stride=stride, offsets=offsets
        )

        gate_seq = (
            gate_expansions.get(instr.name, [instr.name])
            if gate_expansions
            else [instr.name]
        )

        for g in gate_seq:
            mod_circ.append(
                g,
                new_ts,
                instr.gate_args_copy(),
                tag=instr.tag,
            )
    return mod_circ


class TransversalEncoder:
    n: int
    encoding_qubit: int

    def __init__(
        self,
        n: int,
        encoding_qubit: int,
        encoding_program_text: str | None,
        stabilizer_generators: list[list[int]],
        observables: list[list[int]],
        logical_gate_expansions: dict[str, list[str]] | None = None,
    ):
        self.n = n
        self.encoding_qubit = encoding_qubit
        self.circuit = tsim.Circuit()
        self.used_qubits: set[int] = set()
        self.encoding_program_text = encoding_program_text
        self.logical_gate_expansions = logical_gate_expansions or {}
        self.stabilizer_generators = stabilizer_generators
        self.observables = observables

    def initialize(
        self, program_text: str, encoding_program_text: str | None = None
    ) -> None:
        """
        Provide a state preparation program for k qubits. The encoder will apply
        this program and then apply an encoding circuit to encode the state into n qubits.
        Optionally, the encoding program can be provided separately.

        Args:
            program_text: The state preparation program for k qubits. Generally, this
                should be a simple program that prepares each of the k qubits in a
                single-qubit state.
            encoding_program_text (optional): An encoding circuit for a single logical
                qubit. This should encode a single logical qubit at input
                `self.encoding_qubit` into a state of n qubits.
                If not provided, the encoder will use a noiseless default encoding.
        """

        encoding = encoding_program_text or self.encoding_program_text
        if not encoding:
            raise ValueError("Encoding program text is required")

        mod_circ = _transform_circuit(
            program_text,
            stride=self.n,
            offsets=[self.encoding_qubit],
            used_qubits=self.used_qubits,
            stabilizer_generators=self.stabilizer_generators,
            observables=self.observables,
        )

        self.circuit.append_from_stim_program_text(str(mod_circ))
        self.circuit.append_from_stim_program_text(
            str(
                _transform_circuit(
                    encoding,
                    stride=1,
                    offsets=[self.n * off for off in sorted(self.used_qubits)],
                    stabilizer_generators=self.stabilizer_generators,
                    observables=self.observables,
                )
            )
        )

    def encode_transversally(self, program_text: str) -> None:
        """
        Encode a program on m qubits transversally into a program on n * m qubits
        by replacing each gate with a transversal gate.

        Args:
            program_text: The program to encode transversally.
        """
        mod_circ = _transform_circuit(
            program_text,
            stride=self.n,
            offsets=list(range(self.n)),
            gate_expansions=self.logical_gate_expansions,
            stabilizer_generators=self.stabilizer_generators,
            observables=self.observables,
        )
        self.circuit.append_from_stim_program_text(str(mod_circ))

    def diagram(self, **kwargs):
        return self.circuit.diagram("timeline-svg", **kwargs)

    def encoding_flow_generators(self):
        assert self.encoding_program_text is not None
        return stim.Circuit(self.encoding_program_text).flow_generators()


class SteaneEncoder(TransversalEncoder):

    def __init__(self):
        encoding_program = """
        R 0 1 2 3 4 5
        TICK
        SQRT_Y_DAG 0 1 2 3 4 5
        TICK
        CZ 1 2 3 4 5 6
        TICK
        SQRT_Y 6
        TICK
        CZ 0 3 2 5 4 6
        TICK
        SQRT_Y 2 3 4 5 6
        TICK
        CZ 0 1 2 3 4 5
        TICK
        SQRT_Y 1 2 4
        TICK
        X 3
        Z 5 1
        TICK
        """
        super().__init__(
            n=7,
            encoding_qubit=6,
            encoding_program_text=encoding_program,
            logical_gate_expansions={
                "SQRT_X": ["SQRT_X", "X"],
                "SQRT_X_DAG": ["SQRT_X_DAG", "X"],
                "S": ["S", "Z"],
                "S_DAG": ["S_DAG", "Z"],
            },
            stabilizer_generators=[[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 4, 6]],
            observables=[[0, 1, 5]],
        )


class ColorEncoder5(TransversalEncoder):
    def __init__(self):
        encoding_program = """
        R 0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16
        SQRT_Y 0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16
        TICK
        CZ 1 3 7 10 12 14 13 16
        TICK
        SQRT_Y_DAG 7 16
        TICK
        CZ 4 7 8 10 11 14 15 16
        TICK
        SQRT_Y_DAG 4 10 14 16
        TICK
        CZ 2 4 6 8 7 9 10 13
        CZ 14 16
        TICK
        SQRT_Y 3 6 9 10 12 13
        TICK
        CZ 0 2 3 6 5 8 10 12 11 13
        TICK
        SQRT_Y 1 2 3 4 6 7 8 9 11 12 14
        TICK
        CZ 0 1 2 3 4 5 6 7 8 9 12 15
        TICK
        SQRT_Y_DAG 0 2 5 6 8 10 12
        X 14 7 5 2 1 4
        Z 11 6 4 2
        """
        stabs = [
            [0, 1, 2, 3],
            [0, 2, 4, 5],
            [4, 5, 6, 7],
            [6, 7, 8, 9],
            [11, 13, 14, 16],
            [10, 11, 12, 14],
            [12, 14, 15, 16],
            [2, 3, 5, 6, 8, 10, 11, 13],
        ]
        obs = [[1, 3, 10, 12, 15]]
        super().__init__(
            n=17,
            encoding_qubit=7,
            encoding_program_text=encoding_program,
            stabilizer_generators=stabs,
            observables=obs,
        )


ColorEncoder3 = SteaneEncoder
