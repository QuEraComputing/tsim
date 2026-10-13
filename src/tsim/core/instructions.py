"""ZX graph representations of quantum gates and instructions.

Qubit wires are quantum. Every wire vertex starts as a quantum boundary from
``add_lane`` or ``add_quantum_boundary``, and gates turn a wire's trailing
boundary into a spider in place (see ``_wire_spider``), so the spider stays
quantum. Only measurements (``merge_mirror_copies`` in ``_m``) and detector and
observable annotations create classical vertices. NOTE: zxparam's
``Graph.add_vertex`` defaults to the classical domain, so do not call it directly
for wire vertices.

Gates are built up to a global phase: every graph is a doubled diagram (a
channel, ``U ⊗ Ū`` for a unitary), where global phases cancel, so none are
tracked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Literal

import numpy as np
from zxparam import Domain, EdgeType, Graph, Phase, Vertex, VertexType

from tsim.noise.channels import (
    correlated_error_probs,
    error_probs,
    heralded_pauli_channel_1_probs,
    pauli_channel_1_probs,
    pauli_channel_2_probs,
)

HELPER_LANE = -1
"""Qubit index, and row, of the helper wire used by MPP, MPAD and heralded channels."""

ANNOTATION_ROW = -2.0
"""Row of the detector and observable spiders, above the helper wire."""


@dataclass
class GraphBuilder:
    """ZX graph built from a stim circuit, with quantum wires and classical records.

    Qubit wires are quantum; measurement records, detectors and observables are
    classical spiders, found through the lists below (their labels are only for
    display).
    """

    graph: Graph = field(default_factory=Graph)
    rec: list[Vertex] = field(default_factory=list)
    silent_rec: list[Vertex] = field(default_factory=list)
    detectors: list[Vertex] = field(default_factory=list)
    observables_dict: dict[int, Vertex] = field(default_factory=dict)
    first_vertex: dict[int, Vertex] = field(default_factory=dict)
    last_vertex: dict[int, Vertex] = field(default_factory=dict)
    channel_probs: list[np.ndarray] = field(default_factory=list)
    correlated_error_probs: list[float] = field(default_factory=list)
    pending_correlated: list[tuple[Vertex, int]] = field(default_factory=list)
    num_error_bits: int = 0
    num_correlated_error_bits: int = 0

    @property
    def observables(self) -> list[Vertex]:
        """Get list of observable vertices sorted by index."""
        return [self.observables_dict[i] for i in sorted(self.observables_dict)]


def _error_phase(error_bit: int) -> Phase:
    """Return the phase of a noise spider for error bit ``error_bit`` (ID = bit)."""
    return Phase(Fraction(0), frozenset({error_bit}))


def last_col(builder: GraphBuilder, qubit: int) -> float:
    """Get the column of the last vertex for a qubit."""
    return builder.graph.col(builder.last_vertex[qubit])


def last_edge(builder: GraphBuilder, qubit: int) -> tuple[Vertex, Vertex]:
    """Get the last edge for a qubit."""
    edges = builder.graph.incident_edges(builder.last_vertex[qubit])
    assert len(edges) == 1
    return edges[0]


def add_quantum_boundary(
    builder: GraphBuilder, qubit: int, col: float | None = None
) -> Vertex:
    """Add a dummy quantum boundary vertex at the end of a qubit's wire."""
    if col is None:
        col = last_col(builder, qubit) + 1
    v = builder.graph.add_vertex(
        VertexType.BOUNDARY, col=col, row=qubit, domain=Domain.QUANTUM
    )
    builder.last_vertex[qubit] = v
    return v


def add_lane(builder: GraphBuilder, qubit: int) -> Vertex:
    """Initialize a qubit lane: two quantum boundaries joined by a wire."""
    v1 = builder.graph.add_vertex(
        VertexType.BOUNDARY, col=0, row=qubit, domain=Domain.QUANTUM
    )
    builder.first_vertex[qubit] = v1
    builder.graph.add_edge(v1, add_quantum_boundary(builder, qubit, col=1))
    return v1


def ensure_lane(builder: GraphBuilder, qubit: int) -> None:
    """Ensure qubit lane exists."""
    if qubit not in builder.last_vertex:
        add_lane(builder, qubit)


def _wire_spider(
    builder: GraphBuilder,
    qubit: int,
    vertex_type: VertexType,
    phase: Phase | Fraction | None = None,
    col: float | None = None,
) -> Vertex:
    """Turn the trailing boundary of a qubit's wire into a quantum spider.

    The spider keeps the boundary's quantum domain, moves to column ``col`` if
    given, and a new quantum boundary is appended after it. Returns the spider.
    """
    ensure_lane(builder, qubit)
    v1 = builder.last_vertex[qubit]
    builder.graph.set_vertex_type(v1, vertex_type)
    if col is not None:
        builder.graph.set_col(v1, col)
    if phase is not None:
        builder.graph.set_phase(v1, phase)
    v2 = add_quantum_boundary(builder, qubit)
    builder.graph.add_edge(v1, v2)
    return v1


# =============================================================================
# Non-Clifford Gates
# =============================================================================


def t(builder: GraphBuilder, qubit: int) -> None:
    """Apply T gate (π/4 Z rotation)."""
    r_z(builder, qubit, Fraction(1, 4))


def t_dag(builder: GraphBuilder, qubit: int) -> None:
    """Apply T† gate (-π/4 Z rotation)."""
    r_z(builder, qubit, Fraction(-1, 4))


def r_z(builder: GraphBuilder, qubit: int, phase: Fraction) -> None:
    """Apply R_Z rotation gate with given phase (in units of π)."""
    _wire_spider(builder, qubit, VertexType.Z, phase)


def r_x(builder: GraphBuilder, qubit: int, phase: Fraction) -> None:
    """Apply R_X rotation gate with given phase (in units of π)."""
    _wire_spider(builder, qubit, VertexType.X, phase)


def r_y(builder: GraphBuilder, qubit: int, phase: Fraction) -> None:
    """Apply R_Y rotation gate with given phase (in units of π)."""
    h_yz(builder, qubit)
    r_z(builder, qubit, phase)
    h_yz(builder, qubit)


def u3(
    builder: GraphBuilder,
    qubit: int,
    theta: Fraction,
    phi: Fraction,
    lambda_: Fraction,
) -> None:
    """Apply U3 gate: U3(θ,φ,λ) = R_Z(φ)·R_Y(θ)·R_Z(λ)."""
    r_z(builder, qubit, lambda_)
    r_y(builder, qubit, theta)
    r_z(builder, qubit, phi)


# =============================================================================
# Pauli Gates
# =============================================================================


def i(builder: GraphBuilder, qubit: int, *_args: float) -> None:
    """Apply identity (advances the column)."""
    ensure_lane(builder, qubit)
    v = builder.last_vertex[qubit]
    builder.graph.set_col(v, last_col(builder, qubit) + 1)


def ii(builder: GraphBuilder, qubit1: int, qubit2: int, *_args: float) -> None:
    """Apply two-qubit identity (advances the column on both qubits)."""
    i(builder, qubit1)
    i(builder, qubit2)


def x(builder: GraphBuilder, qubit: int) -> None:
    """Apply Pauli X gate."""
    r_x(builder, qubit, Fraction(1, 1))


def y(builder: GraphBuilder, qubit: int) -> None:
    """Apply Pauli Y gate."""
    z(builder, qubit)
    x(builder, qubit)


def z(builder: GraphBuilder, qubit: int) -> None:
    """Apply Pauli Z gate."""
    r_z(builder, qubit, Fraction(1, 1))


# =============================================================================
# Single-Qubit Clifford gates
# =============================================================================


def c_xyz(builder: GraphBuilder, qubit: int) -> None:
    """Right handed period 3 axis cycling gate, sending X -> Y -> Z -> X."""
    s_dag(builder, qubit)
    h(builder, qubit)


def c_nxyz(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending -X -> Y -> Z -> -X."""
    s_dag(builder, qubit)
    sqrt_y_dag(builder, qubit)


def c_xnyz(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending X -> -Y -> Z -> X."""
    s(builder, qubit)
    h(builder, qubit)


def c_xynz(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending X -> Y -> -Z -> X."""
    s(builder, qubit)
    sqrt_y_dag(builder, qubit)


def c_zyx(builder: GraphBuilder, qubit: int) -> None:
    """Left handed period 3 axis cycling gate, sending Z -> Y -> X -> Z."""
    h(builder, qubit)
    s(builder, qubit)


def c_nzyx(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending -Z -> Y -> X -> -Z."""
    s_dag(builder, qubit)
    sqrt_x(builder, qubit)


def c_znyx(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending Z -> -Y -> X -> Z."""
    s(builder, qubit)
    sqrt_x(builder, qubit)


def c_zynx(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending Z -> Y -> -X -> Z."""
    s(builder, qubit)
    sqrt_x_dag(builder, qubit)


def h(builder: GraphBuilder, qubit: int) -> None:
    """Apply Hadamard gate."""
    ensure_lane(builder, qubit)
    builder.graph.toggle_edge(*last_edge(builder, qubit))


def h_xy(builder: GraphBuilder, qubit: int) -> None:
    """Apply variant of Hadamard gate that swaps the X and Y axes (instead of X and Z)."""
    x(builder, qubit)
    s(builder, qubit)


def h_nxy(builder: GraphBuilder, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -X <-> Y, Z -> -Z."""
    x(builder, qubit)
    s_dag(builder, qubit)


def h_nxz(builder: GraphBuilder, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -X <-> Z."""
    z(builder, qubit)
    sqrt_y_dag(builder, qubit)


def h_yz(builder: GraphBuilder, qubit: int) -> None:
    """Apply variant of Hadamard gate that swaps the Y and Z axes (instead of X and Z)."""
    sqrt_x(builder, qubit)
    z(builder, qubit)


def h_nyz(builder: GraphBuilder, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -Y <-> Z, X -> -X."""
    z(builder, qubit)
    sqrt_x(builder, qubit)


def s(builder: GraphBuilder, qubit: int) -> None:
    """Apply S gate (π/2 Z rotation)."""
    r_z(builder, qubit, Fraction(1, 2))


def sqrt_x(builder: GraphBuilder, qubit: int) -> None:
    """Apply √X gate (π/2 X rotation)."""
    r_x(builder, qubit, Fraction(1, 2))


def sqrt_x_dag(builder: GraphBuilder, qubit: int) -> None:
    """Apply √X† gate (-π/2 X rotation)."""
    r_x(builder, qubit, Fraction(-1, 2))


def sqrt_y(builder: GraphBuilder, qubit: int) -> None:
    """Apply √Y gate (π/2 Y rotation)."""
    z(builder, qubit)
    h(builder, qubit)


def sqrt_y_dag(builder: GraphBuilder, qubit: int) -> None:
    """Apply √Y† gate (-π/2 Y rotation)."""
    h(builder, qubit)
    z(builder, qubit)


def s_dag(builder: GraphBuilder, qubit: int) -> None:
    """Apply S† gate (-π/2 Z rotation)."""
    r_z(builder, qubit, Fraction(-1, 2))


# =============================================================================
# Two-Qubit Gates
# =============================================================================


def _cx_cz(
    builder: GraphBuilder,
    is_cx: bool,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Implement CNOT or CZ gate depending on is_cx flag."""
    edge_type = EdgeType.SIMPLE if is_cx else EdgeType.HADAMARD
    vertex_type = VertexType.X if is_cx else VertexType.Z

    m_vertex: Vertex | None = None
    if classically_controlled and any(classically_controlled):
        assert len(classically_controlled) == 2
        if all(classically_controlled) and not is_cx:
            # NOTE: a CZ between two measurement records only phases classical
            # bits, so it does nothing (as in Stim).
            return
        if classically_controlled[1] and not is_cx:
            # Only control is allowed to be classically controlled, swap control and target for symmetric CZ gate
            classically_controlled = classically_controlled[::-1]
            control, target = target, control
        if classically_controlled[1]:
            raise ValueError("Measurement record editing is not supported.")
        m_vertex = builder.rec[control]
        control = int(builder.graph.row(m_vertex))
    ensure_lane(builder, control)
    ensure_lane(builder, target)

    col = max(last_col(builder, control), last_col(builder, target))
    v1 = _wire_spider(builder, control, VertexType.Z, col=col)
    if control == target:
        col += 1
    v2 = _wire_spider(builder, target, vertex_type, col=col)
    builder.graph.add_edge(v1 if m_vertex is None else m_vertex, v2, edge_type)
    builder.graph.mul_sqrt2(1, Domain.QUANTUM)


def cnot(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply CNOT (controlled-X) gate."""
    _cx_cz(builder, True, control, target, classically_controlled)


def cy(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply controlled-Y gate."""
    s_dag(builder, target)
    cnot(builder, control, target, classically_controlled)
    s(builder, target)


def cz(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply controlled-Z gate."""
    _cx_cz(builder, False, control, target, classically_controlled)


def swap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply SWAP gate."""
    ensure_lane(builder, qubit1)
    ensure_lane(builder, qubit2)

    v1 = builder.last_vertex[qubit1]
    v2 = builder.last_vertex[qubit2]
    builder.last_vertex[qubit1] = v2
    builder.last_vertex[qubit2] = v1

    # NOTE: both wire ends move to the later column first, so neither lands on
    # a position the other lane already uses.
    col = max(builder.graph.col(v1), builder.graph.col(v2))
    builder.graph.set_col(v1, col)
    builder.graph.set_col(v2, col)
    builder.graph.set_row(v1, qubit2)
    builder.graph.set_row(v2, qubit1)


def cxswap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply CX then SWAP."""
    cnot(builder, qubit1, qubit2)
    swap(builder, qubit1, qubit2)


def czswap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply CZ then SWAP."""
    cz(builder, qubit1, qubit2)
    swap(builder, qubit1, qubit2)


def swapcx(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply SWAP then CX."""
    swap(builder, qubit1, qubit2)
    cnot(builder, qubit1, qubit2)


def swapcz(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply SWAP then CZ."""
    swap(builder, qubit1, qubit2)
    cz(builder, qubit1, qubit2)


def iswap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Swap two qubits and phase the -1 eigenspace of the ZZ observable by i."""
    cnot(builder, qubit1, qubit2)
    s(builder, qubit2)
    cnot(builder, qubit1, qubit2)
    swap(builder, qubit1, qubit2)


def iswap_dag(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Swap two qubits and phase the -1 eigenspace of the ZZ observable by -i."""
    cnot(builder, qubit1, qubit2)
    s_dag(builder, qubit2)
    cnot(builder, qubit1, qubit2)
    swap(builder, qubit1, qubit2)


def sqrt_xx(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the XX observable by i."""
    cnot(builder, qubit1, qubit2)
    sqrt_x(builder, qubit1)
    cnot(builder, qubit1, qubit2)


def sqrt_xx_dag(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the XX observable by -i."""
    cnot(builder, qubit1, qubit2)
    sqrt_x_dag(builder, qubit1)
    cnot(builder, qubit1, qubit2)


def sqrt_yy(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the YY observable by i."""
    s(builder, qubit1)
    cnot(builder, qubit2, qubit1)
    z(builder, qubit1)
    h(builder, qubit2)
    cnot(builder, qubit2, qubit1)
    s(builder, qubit1)


def sqrt_yy_dag(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the YY observable by -i."""
    s_dag(builder, qubit1)
    cnot(builder, qubit2, qubit1)
    h(builder, qubit2)
    z(builder, qubit1)
    cnot(builder, qubit2, qubit1)
    s_dag(builder, qubit1)


def sqrt_zz(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the ZZ observable by i."""
    cnot(builder, qubit1, qubit2)
    s(builder, qubit2)
    cnot(builder, qubit1, qubit2)


def sqrt_zz_dag(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the ZZ observable by -i."""
    h(builder, qubit2)
    cnot(builder, qubit1, qubit2)
    h(builder, qubit2)
    s_dag(builder, qubit1)
    s_dag(builder, qubit2)


def xcx(builder: GraphBuilder, control: int, target: int) -> None:
    """X-controlled X gate. Applies X to target if control is in |-> state."""
    h(builder, control)
    cnot(builder, control, target)
    h(builder, control)


def xcy(builder: GraphBuilder, control: int, target: int) -> None:
    """X-controlled Y gate. Applies Y to target if control is in |-> state."""
    h(builder, control)
    cy(builder, control, target)
    h(builder, control)


def xcz(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """X-controlled Z gate. Applies Z to target if control is in |-> state."""
    cnot(
        builder,
        target,
        control,
        classically_controlled[::-1] if classically_controlled else None,
    )


def ycx(builder: GraphBuilder, control: int, target: int) -> None:
    """Y-controlled X gate. Applies X to target if control is in |-i> state."""
    h_yz(builder, control)
    cnot(builder, control, target)
    h_yz(builder, control)


def ycy(builder: GraphBuilder, control: int, target: int) -> None:
    """Y-controlled Y gate. Applies Y to target if control is in |-i> state."""
    h_yz(builder, control)
    cy(builder, control, target)
    h_yz(builder, control)


def ycz(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Y-controlled Z gate. Applies Z to target if control is in |-i> state."""
    cy(
        builder,
        target,
        control,
        classically_controlled[::-1] if classically_controlled else None,
    )


# =============================================================================
# Error Channels (stores probability arrays for channel sampling)
# =============================================================================


def _error(
    builder: GraphBuilder,
    qubit: int,
    vertex_type: VertexType,
    error_bit: int | None,
) -> Vertex:
    """Insert a noise spider; it carries ``error_bit`` (its ID) unless None."""
    phase = None if error_bit is None else _error_phase(error_bit)
    return _wire_spider(builder, qubit, vertex_type, phase)


def pauli_channel_1(
    builder: GraphBuilder, qubit: int, px: float, py: float, pz: float
) -> None:
    """Apply single-qubit Pauli channel with given X, Y, Z error probabilities."""
    builder.channel_probs.append(pauli_channel_1_probs(px, py, pz))
    _error(builder, qubit, VertexType.Z, builder.num_error_bits)
    _error(builder, qubit, VertexType.X, builder.num_error_bits + 1)
    builder.num_error_bits += 2


def pauli_channel_2(
    builder: GraphBuilder,
    qubit_i: int,
    qubit_j: int,
    pix: float,
    piy: float,
    piz: float,
    pxi: float,
    pxx: float,
    pxy: float,
    pxz: float,
    pyi: float,
    pyx: float,
    pyy: float,
    pyz: float,
    pzi: float,
    pzx: float,
    pzy: float,
    pzz: float,
) -> None:
    """Apply two-qubit Pauli channel with given error probabilities for all 15 Pauli pairs."""
    builder.channel_probs.append(
        pauli_channel_2_probs(
            pix, piy, piz, pxi, pxx, pxy, pxz, pyi, pyx, pyy, pyz, pzi, pzx, pzy, pzz
        )
    )
    _error(builder, qubit_i, VertexType.Z, builder.num_error_bits)
    _error(builder, qubit_i, VertexType.X, builder.num_error_bits + 1)
    _error(builder, qubit_j, VertexType.Z, builder.num_error_bits + 2)
    _error(builder, qubit_j, VertexType.X, builder.num_error_bits + 3)
    builder.num_error_bits += 4


def depolarize1(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply single-qubit depolarizing channel with total error probability p."""
    pauli_channel_1(builder, qubit, p / 3, p / 3, p / 3)


def depolarize2(builder: GraphBuilder, qubit_i: int, qubit_j: int, p: float) -> None:
    """Apply two-qubit depolarizing channel with total error probability p."""
    pauli_channel_2(
        builder,
        qubit_i,
        qubit_j,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
    )


def x_error(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply X error with probability p."""
    builder.channel_probs.append(error_probs(p))
    _error(builder, qubit, VertexType.X, builder.num_error_bits)
    builder.num_error_bits += 1


def y_error(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply Y error with probability p."""
    builder.channel_probs.append(error_probs(p))
    # Y = X·Z, so both vertices use the same error bit
    _error(builder, qubit, VertexType.Z, builder.num_error_bits)
    _error(builder, qubit, VertexType.X, builder.num_error_bits)
    builder.num_error_bits += 1


def z_error(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply Z error with probability p."""
    builder.channel_probs.append(error_probs(p))
    _error(builder, qubit, VertexType.Z, builder.num_error_bits)
    builder.num_error_bits += 1


def heralded_pauli_channel_1(
    builder: GraphBuilder,
    qubit: int,
    pi: float,
    px: float,
    py: float,
    pz: float,
) -> None:
    """Apply heralded single-qubit Pauli channel.

    Records a herald bit into the measurement record. When the channel fires
    (with total probability pi+px+py+pz), the herald is 1 and one of I/X/Y/Z
    is applied. When it doesn't fire, the herald is 0 and nothing happens.
    """
    builder.channel_probs.append(heralded_pauli_channel_1_probs(pi, px, py, pz))
    aux = HELPER_LANE
    r(builder, aux)
    _error(builder, aux, VertexType.X, builder.num_error_bits)  # herald bit flip
    m(builder, aux)
    _error(builder, qubit, VertexType.Z, builder.num_error_bits + 1)  # Z error bit
    _error(builder, qubit, VertexType.X, builder.num_error_bits + 2)  # X error bit
    builder.num_error_bits += 3


def heralded_erase(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply heralded erasure channel.

    Special case of heralded_pauli_channel_1 with equal probabilities p/4
    for each of I, X, Y, Z when the channel fires.
    """
    heralded_pauli_channel_1(builder, qubit, p / 4, p / 4, p / 4, p / 4)


def finalize_correlated_error(builder: GraphBuilder) -> None:
    """Finalize the current correlated error channel.

    1. Give each pending noise spider its error bit ``num_error_bits + i``
    2. Compute and append the 2^k probability array to channel_probs
    3. Increment num_error_bits by k
    4. Reset num_correlated_error_bits to 0 and correlated_error_probs to []
    """
    k = builder.num_correlated_error_bits
    if k == 0:
        return
    for v, bit in builder.pending_correlated:
        builder.graph.set_phase(v, _error_phase(builder.num_error_bits + bit))
    builder.pending_correlated = []
    builder.channel_probs.append(correlated_error_probs(builder.correlated_error_probs))
    builder.num_error_bits += k
    builder.num_correlated_error_bits = 0
    builder.correlated_error_probs = []


def correlated_error(
    builder: GraphBuilder,
    qubits: list[int],
    types: list[Literal["X", "Y", "Z"]],
    p: float,
) -> None:
    """Add a correlated error term affecting multiple qubits with given Pauli types."""
    for qubit, type_ in zip(qubits, types, strict=True):
        if type_ == "X" or type_ == "Y":
            builder.pending_correlated.append(
                (
                    _error(builder, qubit, VertexType.X, None),
                    builder.num_correlated_error_bits,
                )
            )
        if type_ == "Z" or type_ == "Y":
            builder.pending_correlated.append(
                (
                    _error(builder, qubit, VertexType.Z, None),
                    builder.num_correlated_error_bits,
                )
            )

    builder.correlated_error_probs.append(p)
    builder.num_correlated_error_bits += 1


# =============================================================================
# Collapsing gates
# =============================================================================


def _m(builder: GraphBuilder, qubit: int, p: float = 0, silent: bool = False) -> None:
    """Perform measurement on qubit with optional error probability."""
    error_bit = None
    if p > 0:
        builder.channel_probs.append(error_probs(p))
        error_bit = builder.num_error_bits
        _error(builder, qubit, VertexType.X, error_bit)
        builder.num_error_bits += 1
    ensure_lane(builder, qubit)
    v1 = builder.last_vertex[qubit]
    builder.graph.set_vertex_type(v1, VertexType.Z)
    builder.graph.merge_mirror_copies(v1)
    if not silent:
        builder.graph.set_vertex_label(v1, f"rec[{len(builder.rec)}]")
        builder.rec.append(v1)
    else:
        builder.graph.set_vertex_label(v1, f"m[{len(builder.silent_rec)}]")
        builder.silent_rec.append(v1)
    v2 = add_quantum_boundary(builder, qubit)
    builder.graph.add_edge(v1, v2)
    if error_bit is not None:
        _error(builder, qubit, VertexType.X, error_bit)


def _r(builder: GraphBuilder, qubit: int) -> None:
    """Perform reset on qubit.

    If the qubit does not yet have a lane, create one in the reset state.
    Otherwise, silently measure the existing lane to trace out the previous
    state before reconnecting the qubit in the reset state.
    """
    if qubit not in builder.last_vertex:
        builder.graph.plug_basis(add_lane(builder, qubit))
        return
    _m(builder, qubit, silent=True)
    col = last_col(builder, qubit)
    v1 = builder.last_vertex[qubit]
    (v2,) = builder.graph.neighbors(v1)
    builder.graph.remove_edge(v1, v2)
    builder.graph.plug_basis(v1)
    v3 = add_quantum_boundary(builder, qubit, col + 1)
    builder.graph.add_edge(v1, v3)


def m(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in Z basis with optional bit-flip error probability p."""
    if invert:
        x(builder, qubit)
    _m(builder, qubit, p, silent=False)
    if invert:
        x(builder, qubit)


def mpp(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    invert: bool = False,
    p: float = 0,
) -> None:
    """Measure a single Pauli product.

    Args:
        builder: The graph builder to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product.
        invert: Whether to invert the measurement result.
        p: Measurement flip error probability.

    """
    aux = HELPER_LANE
    r(builder, aux)
    h(builder, aux)
    _apply_pauli_controls(builder, aux, paulis)
    h(builder, aux)
    m(builder, aux, p=p, invert=invert)


def _apply_pauli_controls(
    builder: GraphBuilder,
    aux: int,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
) -> None:
    """Apply controlled-Pauli operations between aux and target qubits."""
    for pauli_type, qubit in paulis:
        if pauli_type == "X":
            cnot(builder, aux, qubit)
        elif pauli_type == "Z":
            cz(builder, aux, qubit)
        elif pauli_type == "Y":
            cy(builder, aux, qubit)
        else:
            raise ValueError(f"Invalid Pauli operator: {pauli_type}")


def _pauli_product_phase(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    phase_gate: Callable[[GraphBuilder, int], None],
    phase_gate_dag: Callable[[GraphBuilder, int], None],
    dagger: bool,
) -> None:
    """Apply exp(-i theta P) (up to global phase) for a Pauli product P.

    If `dagger` is True, apply exp(+i theta P) instead.
    Phases the -1 eigenspace of P by exp(2i theta). Shared implementation for
    SPP (theta=pi/4, phase by i) and TPP (theta=pi/8, phase by exp(i pi/4))
    and their daggers.
    """
    if len(paulis) == 0:
        return

    # Rotate each qubit so its Pauli eigenvalue maps to Z
    for pauli_type, qubit in paulis:
        if pauli_type == "X":
            h(builder, qubit)
        elif pauli_type == "Y":
            s_dag(builder, qubit)
            h(builder, qubit)

    # Accumulate Z-parity into the last qubit
    _, last_qubit = paulis[-1]
    for _, qubit in paulis[:-1]:
        cnot(builder, qubit, last_qubit)

    # Phase the parity
    if dagger:
        phase_gate_dag(builder, last_qubit)
    else:
        phase_gate(builder, last_qubit)

    # Uncompute parity accumulation
    for _, qubit in reversed(paulis[:-1]):
        cnot(builder, qubit, last_qubit)

    # Undo basis rotations
    for pauli_type, qubit in paulis:
        if pauli_type == "X":
            h(builder, qubit)
        elif pauli_type == "Y":
            h(builder, qubit)
            s(builder, qubit)


def spp(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    dagger: bool = False,
) -> None:
    """Apply exp(-i pi/4 P) (up to global phase) for a Pauli product P.

    Phases the -1 eigenspace of P by i (or -i if dagger). For a single qubit,
    ``SPP Z0`` is the S gate and ``SPP_DAG Z0`` is S_DAG.

    Args:
        builder: The graph builder to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        dagger: If True, apply exp(+i pi/4 P) (phase by -i) instead.

    """
    _pauli_product_phase(builder, paulis, s, s_dag, dagger)


def tpp(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    dagger: bool = False,
) -> None:
    """Apply exp(-i pi/8 P) (up to global phase) for a Pauli product P.

    Phases the -1 eigenspace of P by exp(i pi/4) (or exp(-i pi/4) if dagger).
    For a single qubit, ``TPP Z0`` is the T gate and ``TPP_DAG Z0`` is T_DAG.

    Args:
        builder: The graph builder to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        dagger: If True, apply exp(+i pi/8 P) (phase by exp(-i pi/4)) instead.

    """
    _pauli_product_phase(builder, paulis, t, t_dag, dagger)


def r_pauli(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    theta: Fraction,
    dagger: bool = False,
) -> None:
    """Apply exp(-i theta pi/2 P) for a Pauli product P, with theta in half-turns.

    Generalizes ``SPP`` (theta=1/2) and ``TPP`` (theta=1/4) to an arbitrary angle.
    For a single qubit, ``r_pauli`` reduces to the named single-qubit rotation in
    that Pauli's basis (e.g. ``R_PAULI Z0`` is ``R_Z``). If `dagger` is True, apply
    exp(+i theta pi/2 P) instead.

    Args:
        builder: The graph builder to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        theta: Rotation angle in half-turns (units of pi).
        dagger: If True, negate the rotation angle.

    """
    _pauli_product_phase(
        builder,
        paulis,
        lambda builder, qubit: r_z(builder, qubit, theta),
        lambda builder, qubit: r_z(builder, qubit, -theta),
        dagger,
    )


def mpad(builder: GraphBuilder, value: int, p: float = 0) -> None:
    """Pad measurement record with a fixed bit value.

    Args:
        builder: The graph builder to modify.
        value: The bit value to record (0 or 1).
        p: Error probability for the recorded bit.

    """
    aux = HELPER_LANE
    r(builder, aux)
    if value == 1:
        x(builder, aux)
    m(builder, aux, p=p)


def mr(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Z-basis demolition measurement (optionally noisy).

    Projects each target qubit into |0> or |1>, reports its value (false=|0>, true=|1>),
    then resets to |0>.
    """
    m(builder, qubit, p=p, invert=invert)
    _r(builder, qubit)


def mrx(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """X-basis demolition measurement (optionally noisy).

    Projects each target qubit into |+> or |->, reports its value (false=|+>, true=|->),
    then resets to |+>.
    """
    h(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    _r(builder, qubit)
    h(builder, qubit)


def mry(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Y-basis demolition measurement (optionally noisy).

    Projects each target qubit into |i> or |-i>, reports its value (false=|i>, true=|-i>),
    then resets to |i>.
    """
    h_yz(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    _r(builder, qubit)
    h_yz(builder, qubit)


def mx(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in X basis."""
    h(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    h(builder, qubit)


def my(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in Y basis."""
    h_yz(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    h_yz(builder, qubit)


def mxx(
    builder: GraphBuilder, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in XX basis."""
    mpp(builder, [("X", q0), ("X", q1)], invert, p=p)


def myy(
    builder: GraphBuilder, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in YY basis."""
    mpp(builder, [("Y", q0), ("Y", q1)], invert, p=p)


def mzz(
    builder: GraphBuilder, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in ZZ basis."""
    mpp(builder, [("Z", q0), ("Z", q1)], invert, p=p)


def r(builder: GraphBuilder, qubit: int) -> None:
    """Z-basis reset.

    Forces each target qubit into the |0> state by silently measuring it in the Z basis
    and applying an X gate if it ended up in the |1> state.
    """
    _r(builder, qubit)


def rx(builder: GraphBuilder, qubit: int) -> None:
    """X-basis reset.

    Forces the qubit into the |+> state: a Z-basis reset followed by H. The reset
    traces out the old state, which does not depend on the basis.
    """
    r(builder, qubit)
    h(builder, qubit)


def ry(builder: GraphBuilder, qubit: int) -> None:
    """Y-basis reset.

    Forces the qubit into the |i> state: a Z-basis reset followed by H_YZ. The
    reset traces out the old state, which does not depend on the basis.
    """
    r(builder, qubit)
    h_yz(builder, qubit)


# =============================================================================
# Annotations
# =============================================================================


def _annotation_col(builder: GraphBuilder, rec: list[int]) -> float:
    """Pick a fresh column for a detector/observable vertex.

    Empty `rec` is valid Stim and represents a deterministic-zero annotation;
    fall back to a column after existing annotations rather than `min()` on an
    empty set.
    """
    d_cols = {builder.graph.col(d) for d in builder.detectors + builder.observables}
    if rec:
        col: float = min(builder.graph.col(builder.rec[r]) for r in rec) - 0.5
    else:
        col = (max(d_cols) + 1) if d_cols else 0
    while col in d_cols:
        col += 1
    return col


def _xor_record(builder: GraphBuilder, annotation: Vertex, rec: int) -> None:
    """Add measurement record ``rec`` to the parity of ``annotation``.

    A record that is already in the parity cancels: its edge is removed rather
    than Hopf-cancelled by ``add_edge``, which would leave a scalar of 1/2 behind
    in graphs that drop the annotation.
    """
    record = builder.rec[rec]
    if builder.graph.has_edge(annotation, record):
        builder.graph.remove_edge(annotation, record)
    else:
        builder.graph.add_edge(annotation, record)


def detector(builder: GraphBuilder, rec: list[int], *args) -> None:
    """Add detector annotation that XORs the given measurement record bits."""
    v0 = builder.graph.add_vertex(
        VertexType.X,
        col=_annotation_col(builder, rec),
        row=ANNOTATION_ROW,
        domain=Domain.CLASSICAL,
        label=f"det[{len(builder.detectors)}]",
    )
    for rec_ in rec:
        _xor_record(builder, v0, rec_)
    builder.detectors.append(v0)


def observable_include(builder: GraphBuilder, rec: list[int], idx: int) -> None:
    """Add observable annotation that XORs the given measurement record bits."""
    idx = int(idx)

    if idx not in builder.observables_dict:
        v0 = builder.graph.add_vertex(
            VertexType.X,
            col=_annotation_col(builder, rec),
            row=ANNOTATION_ROW,
            domain=Domain.CLASSICAL,
            label=f"obs[{idx}]",
        )
        builder.observables_dict[idx] = v0

    v0 = builder.observables_dict[idx]
    for rec_ in rec:
        _xor_record(builder, v0, rec_)


def tick(builder: GraphBuilder) -> None:
    """Add a tick to the circuit (align all qubits to same column)."""
    if len(builder.last_vertex) == 0:
        return
    col = max(last_col(builder, q) for q in builder.last_vertex)
    for q in builder.last_vertex:
        builder.graph.set_col(builder.last_vertex[q], col)


# =============================================================================
# Gate Dispatch Table
# =============================================================================

GATE_TABLE: dict[str, tuple[Callable[..., None], int]] = {
    # ---- Pauli gates -----------------------------------------------------------
    "I": (i, 1),
    "I_ERROR": (i, 1),
    # QUBIT_COORDS is a visualization annotation; dispatched as identity to allocate
    # the qubit lane. Coordinate args are accepted and ignored by `i`.
    "QUBIT_COORDS": (i, 1),
    "II": (ii, 2),
    "II_ERROR": (ii, 2),
    "X": (x, 1),
    "Y": (y, 1),
    "Z": (z, 1),
    # ---- Non-Clifford gates ---------------------------------------------------
    "T": (t, 1),
    "T_DAG": (t_dag, 1),
    # ---- Single-qubit gates ---------------------------------------------------
    "C_NXYZ": (c_nxyz, 1),
    "C_NZYX": (c_nzyx, 1),
    "C_XNYZ": (c_xnyz, 1),
    "C_XYNZ": (c_xynz, 1),
    "C_XYZ": (c_xyz, 1),
    "C_ZNYX": (c_znyx, 1),
    "C_ZYNX": (c_zynx, 1),
    "C_ZYX": (c_zyx, 1),
    "H": (h, 1),
    "H_NXY": (h_nxy, 1),
    "H_NXZ": (h_nxz, 1),
    "H_NYZ": (h_nyz, 1),
    "H_XY": (h_xy, 1),
    "H_XZ": (h, 1),
    "H_YZ": (h_yz, 1),
    "S": (s, 1),
    "SQRT_X": (sqrt_x, 1),
    "SQRT_X_DAG": (sqrt_x_dag, 1),
    "SQRT_Y": (sqrt_y, 1),
    "SQRT_Y_DAG": (sqrt_y_dag, 1),
    "SQRT_Z": (s, 1),
    "SQRT_Z_DAG": (s_dag, 1),
    "S_DAG": (s_dag, 1),
    # ---- Two-qubit gates ------------------------------------------------------
    "CNOT": (cnot, 2),
    "CX": (cnot, 2),
    "CXSWAP": (cxswap, 2),
    "CZ": (cz, 2),
    "CZSWAP": (czswap, 2),
    "CY": (cy, 2),
    "ISWAP": (iswap, 2),
    "ISWAP_DAG": (iswap_dag, 2),
    "SQRT_XX": (sqrt_xx, 2),
    "SQRT_XX_DAG": (sqrt_xx_dag, 2),
    "SQRT_YY": (sqrt_yy, 2),
    "SQRT_YY_DAG": (sqrt_yy_dag, 2),
    "SQRT_ZZ": (sqrt_zz, 2),
    "SQRT_ZZ_DAG": (sqrt_zz_dag, 2),
    "SWAP": (swap, 2),
    "SWAPCX": (swapcx, 2),
    "SWAPCZ": (swapcz, 2),
    "XCX": (xcx, 2),
    "XCY": (xcy, 2),
    "XCZ": (xcz, 2),
    "YCX": (ycx, 2),
    "YCY": (ycy, 2),
    "YCZ": (ycz, 2),
    "ZCX": (cnot, 2),
    "ZCY": (cy, 2),
    "ZCZ": (cz, 2),
    # ---- Noise channels -------------------------------------------------------
    "DEPOLARIZE1": (depolarize1, 1),
    "DEPOLARIZE2": (depolarize2, 2),
    "PAULI_CHANNEL_1": (pauli_channel_1, 1),
    "PAULI_CHANNEL_2": (pauli_channel_2, 2),
    "HERALDED_ERASE": (heralded_erase, 1),
    "HERALDED_PAULI_CHANNEL_1": (heralded_pauli_channel_1, 1),
    "X_ERROR": (x_error, 1),
    "Y_ERROR": (y_error, 1),
    "Z_ERROR": (z_error, 1),
    # ---- Collapsing gates -----------------------------------------------------
    "M": (m, 1),
    # MPP, SPP, SPP_DAG, MPAD handled by parser
    "MR": (mr, 1),
    "MRX": (mrx, 1),
    "MRY": (mry, 1),
    "MRZ": (mr, 1),
    "MX": (mx, 1),
    "MY": (my, 1),
    "MZ": (m, 1),
    "MXX": (mxx, 2),
    "MYY": (myy, 2),
    "MZZ": (mzz, 2),
    "R": (r, 1),
    "RX": (rx, 1),
    "RY": (ry, 1),
    "RZ": (r, 1),
    # ---- Annotations handled by parser -----------------------------------------
}
