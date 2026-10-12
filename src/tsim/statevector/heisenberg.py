"""Heisenberg-frame operations of an extracted circuit.

``heisenberg_ops`` rewrites an ``ExtractedCircuit`` as rotations and
measurements whose Paulis are written in the frame of the initial ``|0…0⟩``
state. The parameter part of a Z phase is noise: a Z applied when a parity of
f-bits is 1. Moving it past every later operation to the end of the circuit,
where it changes no outcome, flips the sign of each operation it anticommutes
with. So every operation carries a fixed GF(2) sign row, and operations that
commute can be reordered without touching their signs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction
from math import pi

import numpy as np
import stim
from zxparam import Graph
from zxparam.extract import ExtractedCircuit, Gate

from tsim.core.params import ParamLayout

ROTATION = 0
MEASUREMENT = 1

_CLIFFORD_GATES = {"H": "h", "CNOT": "cx", "CZ": "cz"}
_CLIFFORD_QUARTERS = {1: "s", 2: "z", 3: "s_dag"}


@dataclass(frozen=True)
class ComponentLayout:
    """Where a component's parameters and outputs go.

    Attributes:
        f_selection: Global f-bit index of each component f-bit, ascending.
        f_position: Position in ``f_selection`` of each phase parameter ID.
        output_indices: Global output index of each output slot.

    """

    f_selection: tuple[int, ...]
    f_position: Mapping[int, int]
    output_indices: tuple[int, ...]

    @classmethod
    def of(
        cls, component: Graph, output_indices: Sequence[int], layout: ParamLayout
    ) -> ComponentLayout:
        """Return the layout of ``component``, whose outputs are ``output_indices``."""
        f_selection = tuple(
            sorted(layout.basis_index(p) for p in component.parameters())
        )
        f_position = {layout.basis(i): pos for pos, i in enumerate(f_selection)}
        return cls(f_selection, f_position, tuple(output_indices))


def anticommute(
    x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray
) -> np.ndarray:
    """Return which Paulis ``(x1, z1)`` anticommute with which Paulis ``(x2, z2)``.

    Each row of an argument is a Pauli; a 1-D argument is one Pauli, whose axis
    the result drops.
    """
    f = np.float32
    return (x1.astype(f) @ z2.T.astype(f) + z1.astype(f) @ x2.T.astype(f)) % 2 == 1


@dataclass(frozen=True)
class HeisenbergOps:
    """Rotations and measurements in the frame of the initial ``|0…0⟩`` state.

    Op ``j`` is the rotation ``exp(-iθ (-1)^s P)`` or the measurement of
    ``(-1)^s P``, where ``P`` has the bit rows ``x[j]`` and ``z[j]`` (``Y``
    where both are set) and ``s`` is the parity of ``sign[j]`` against
    ``[1, f_0, f_1, …]``. A measurement records its eigenvalue bit.

    Attributes:
        kind: ``ROTATION`` or ``MEASUREMENT``, shape ``(N,)``.
        x: X part of each Pauli, shape ``(N, num_lines)``.
        z: Z part of each Pauli, shape ``(N, num_lines)``.
        angle: θ of each rotation, 0 for measurements, shape ``(N,)``.
        sign: Constant, then one column per f-bit, shape ``(N, 1 + num_f)``.
        slot: Output slot of a recorded measurement, -1 for a forgotten
            measurement or a rotation, shape ``(N,)``.

    """

    kind: np.ndarray
    x: np.ndarray
    z: np.ndarray
    angle: np.ndarray
    sign: np.ndarray
    slot: np.ndarray

    @property
    def num_ops(self) -> int:
        """Return the number of operations."""
        return len(self.kind)

    @property
    def num_lines(self) -> int:
        """Return the number of qubit lines."""
        return self.x.shape[1]

    @property
    def num_f(self) -> int:
        """Return the number of component f-bits the signs depend on."""
        return self.sign.shape[1] - 1

    @property
    def num_outputs(self) -> int:
        """Return the number of recorded measurements."""
        return int(np.count_nonzero(self.slot >= 0))

    def take(self, order: np.ndarray) -> HeisenbergOps:
        """Return the operations in the given order.

        The result describes the same distribution when ``order`` keeps every
        pair of anticommuting operations in their original order.
        """
        return HeisenbergOps(*(getattr(self, f.name)[order] for f in fields(self)))


def heisenberg_ops(circuit: ExtractedCircuit, layout: ComponentLayout) -> HeisenbergOps:
    """Return the operations of ``circuit`` in the ``|0…0⟩`` frame, in time order.

    Args:
        circuit: A circuit extracted from a closed sampling graph: every line
            starts with a ``|+⟩`` preparation and every output is classical.
        layout: The component's f-bits and output slots.

    Raises:
        ValueError: If a line starts at an input port, ends at a quantum
            output, or a gate kind is unknown.

    """
    _check_lines(circuit)
    builder = _Builder(circuit.n_lines, layout)
    for gate in circuit.gates:
        builder.gate(gate)
    slot_of_port = {port: slot for slot, port in enumerate(layout.output_indices)}
    for line in range(circuit.n_lines):
        port = circuit.line_end[line]
        if port is not None:
            builder.measure(line, slot_of_port[port])
    return builder.finish()


def _check_lines(circuit: ExtractedCircuit) -> None:
    for line in range(circuit.n_lines):
        start = circuit.line_start[line]
        if start is not None:
            raise ValueError(
                f"line {line} starts at input port {start}; "
                "sampling graphs have no inputs"
            )
        port = circuit.line_end[line]
        if port is not None and port not in circuit.classical_ports:
            raise ValueError(f"line {line} ends at the quantum output port {port}")


class _Builder:
    """Walks the gates with a stim simulator that holds the Clifford prefix U."""

    def __init__(self, num_lines: int, layout: ComponentLayout) -> None:
        """Start every line in ``|+⟩`` (``H`` on ``|0⟩``)."""
        self.num_lines = num_lines
        self.num_f = len(layout.f_selection)
        self.f_position = layout.f_position
        self.sim = stim.TableauSimulator()
        self.sim.set_num_qubits(num_lines)
        for line in range(num_lines):
            self.sim.h(line)
        self.inverse: stim.Tableau | None = None
        self.ops: list[tuple[int, np.ndarray, np.ndarray, bool, float, int]] = []
        self.noise: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

    def _mapped(self, line: int) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return U†·Z_line·U as bit rows and a sign flag."""
        if self.inverse is None:
            self.inverse = self.sim.current_inverse_tableau()
        image = self.inverse.z_output(line)
        x, z = image.to_numpy()
        return x, z, bool(image.sign == -1)

    def _clifford(self, name: str, *lines: int) -> None:
        getattr(self.sim, name)(*lines)
        self.inverse = None

    def gate(self, gate: Gate) -> None:
        """Absorb a Clifford gate, or record a rotation, noise or measurement."""
        if gate.kind in _CLIFFORD_GATES:
            self._clifford(_CLIFFORD_GATES[gate.kind], *gate.lines)
        elif gate.kind == "Z":
            self._phase(gate)
        elif gate.kind in ("MEASURE", "DISCARD"):
            self.measure(gate.lines[0], -1)
        else:
            raise ValueError(f"unknown gate kind {gate.kind!r}")

    def _phase(self, gate: Gate) -> None:
        (line,) = gate.lines
        phase = gate.phase
        assert phase is not None
        quarters = round(2 * phase.value)
        remainder = phase.value - Fraction(quarters, 2)
        clifford = _CLIFFORD_QUARTERS.get(quarters % 4)
        if clifford is not None:
            self._clifford(clifford, line)
        if remainder:
            x, z, negative = self._mapped(line)
            self.ops.append((ROTATION, x, z, negative, pi * float(remainder) / 2, -1))
        if phase.parameters:
            x, z, _ = self._mapped(line)
            condition = np.zeros(self.num_f, dtype=bool)
            condition[[self.f_position[p] for p in phase.parameters]] = True
            self.noise.append((x, z, condition, len(self.ops)))

    def measure(self, line: int, slot: int) -> None:
        """Record a Z measurement of ``line`` (``slot`` -1 if forgotten)."""
        x, z, negative = self._mapped(line)
        self.ops.append((MEASUREMENT, x, z, negative, 0.0, slot))

    def finish(self) -> HeisenbergOps:
        """Return the ops, with every noise event folded into their sign rows.

        A noise event flips the sign of every later op it anticommutes with.
        """
        num_ops, num_lines = len(self.ops), self.num_lines
        x = np.array([op[1] for op in self.ops], dtype=bool).reshape(num_ops, num_lines)
        z = np.array([op[2] for op in self.ops], dtype=bool).reshape(num_ops, num_lines)
        sign = np.zeros((num_ops, 1 + self.num_f), dtype=bool)
        sign[:, 0] = [op[3] for op in self.ops]
        if self.noise:
            noise_x, noise_z, condition, start = map(
                np.array, zip(*self.noise, strict=True)
            )
            later = start[:, None] <= np.arange(num_ops)[None, :]
            flips = (anticommute(noise_x, noise_z, x, z) & later).astype(np.float32)
            sign[:, 1:] = (flips.T @ condition.astype(np.float32)) % 2 == 1
        return HeisenbergOps(
            kind=np.array([op[0] for op in self.ops], dtype=np.uint8),
            x=x,
            z=z,
            angle=np.array([op[4] for op in self.ops], dtype=np.float64),
            sign=sign,
            slot=np.array([op[5] for op in self.ops], dtype=np.int32),
        )
