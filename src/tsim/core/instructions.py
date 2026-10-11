"""ZX graph representations of quantum gates and instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Literal

import numpy as np
from zxparam import Domain, EdgeType, Graph, Phase, Scalar, Vertex, VertexType

from tsim.noise.channels import (
    correlated_error_probs,
    error_probs,
    heralded_pauli_channel_1_probs,
    pauli_channel_1_probs,
    pauli_channel_2_probs,
)


@dataclass
class GraphRepresentation:
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


def _scale(b: GraphRepresentation, sqrt2_power: int) -> None:
    """Multiply both copies of the doubled diagram by ``√2^sqrt2_power``."""
    b.graph.mul_scalar(Scalar.one().add_power_sqrt2(sqrt2_power), Domain.QUANTUM)


def _error_phase(error_bit: int) -> Phase:
    """Return the phase of a noise spider for error bit ``error_bit`` (ID = bit)."""
    return Phase(Fraction(0), frozenset({error_bit}))


def _make_classical(b: GraphRepresentation, v: Vertex) -> None:
    """Make spider ``v`` classical, keeping one wire edge per copy.

    Between two classical spiders (e.g. two measurements in a row) a wire is one
    edge per copy, which ``add_edge_per_copy`` collapses correctly.
    """
    g = b.graph
    edges = [(n, g.edge_type(v, n)) for n in g.neighbors(v)]
    for n, _ in edges:
        g.remove_edge(v, n)
    g.set_vertex_domain(v, Domain.CLASSICAL)
    for n, edge_type in edges:
        g.add_edge_per_copy(v, n, edge_type)


def last_col(b: GraphRepresentation, qubit: int) -> float:
    """Get the column of the last vertex for a qubit."""
    return b.graph.col(b.last_vertex[qubit])


def last_edge(b: GraphRepresentation, qubit: int) -> tuple[Vertex, Vertex]:
    """Get the last edge for a qubit."""
    edges = b.graph.incident_edges(b.last_vertex[qubit])
    assert len(edges) == 1
    return edges[0]


def add_dummy(b: GraphRepresentation, qubit: int, col: float | None = None) -> Vertex:
    """Add a dummy quantum boundary vertex at the end of a qubit's wire."""
    if col is None:
        col = last_col(b, qubit) + 1
    v = b.graph.add_vertex(
        VertexType.BOUNDARY, col=col, row=qubit, domain=Domain.QUANTUM
    )
    b.last_vertex[qubit] = v
    return v


def add_lane(b: GraphRepresentation, qubit: int) -> Vertex:
    """Initialize a qubit lane: two quantum boundaries joined by a wire."""
    v1 = b.graph.add_vertex(
        VertexType.BOUNDARY, col=0, row=qubit, domain=Domain.QUANTUM
    )
    v2 = b.graph.add_vertex(
        VertexType.BOUNDARY, col=1, row=qubit, domain=Domain.QUANTUM
    )
    b.graph.add_edge(v1, v2)
    b.first_vertex[qubit] = v1
    b.last_vertex[qubit] = v2
    return v1


def ensure_lane(b: GraphRepresentation, qubit: int) -> None:
    """Ensure qubit lane exists."""
    if qubit not in b.last_vertex:
        add_lane(b, qubit)


# =============================================================================
# Non-Clifford Gates
# =============================================================================


def x_phase(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """Apply X-axis rotation to qubit. This is equivalent to `r_x` up to a phase."""
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_vertex_type(v1, VertexType.X)
    b.graph.set_phase(v1, phase)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge(v1, v2)


def z_phase(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """Apply Z-axis phase rotation to qubit. This is equivalent to `r_z` up to a phase."""
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_vertex_type(v1, VertexType.Z)
    b.graph.set_phase(v1, phase)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge(v1, v2)


def t(b: GraphRepresentation, qubit: int) -> None:
    """Apply T gate (π/4 Z rotation)."""
    z_phase(b, qubit, Fraction(1, 4))


def t_dag(b: GraphRepresentation, qubit: int) -> None:
    """Apply T† gate (-π/4 Z rotation)."""
    z_phase(b, qubit, Fraction(-1, 4))


def r_z(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """Apply R_Z rotation gate with given phase (in units of π)."""
    z_phase(b, qubit, phase)


def r_x(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """Apply R_X rotation gate with given phase (in units of π)."""
    x_phase(b, qubit, phase)


def r_y(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    """Apply R_Y rotation gate with given phase (in units of π)."""
    h_yz(b, qubit)
    r_z(b, qubit, phase)
    h_yz(b, qubit)


def u3(
    b: GraphRepresentation,
    qubit: int,
    theta: Fraction,
    phi: Fraction,
    lambda_: Fraction,
) -> None:
    """Apply U3 gate: U3(θ,φ,λ) = R_Z(φ)·R_Y(θ)·R_Z(λ)."""
    r_z(b, qubit, lambda_)
    r_y(b, qubit, theta)
    r_z(b, qubit, phi)


# =============================================================================
# Pauli Gates
# =============================================================================


def i(b: GraphRepresentation, qubit: int, *_args: float) -> None:
    """Apply identity (advances the column)."""
    ensure_lane(b, qubit)
    v = b.last_vertex[qubit]
    b.graph.set_col(v, last_col(b, qubit) + 1)


def ii(b: GraphRepresentation, qubit1: int, qubit2: int, *_args: float) -> None:
    """Apply two-qubit identity (advances the column on both qubits)."""
    i(b, qubit1)
    i(b, qubit2)


def x(b: GraphRepresentation, qubit: int) -> None:
    """Apply Pauli X gate."""
    x_phase(b, qubit, Fraction(1, 1))


def y(b: GraphRepresentation, qubit: int) -> None:
    """Apply Pauli Y gate."""
    z(b, qubit)
    x(b, qubit)


def z(b: GraphRepresentation, qubit: int) -> None:
    """Apply Pauli Z gate."""
    z_phase(b, qubit, Fraction(1, 1))


# =============================================================================
# Single-Qubit Clifford gates
# =============================================================================


def c_xyz(b: GraphRepresentation, qubit: int) -> None:
    """Right handed period 3 axis cycling gate, sending X -> Y -> Z -> X."""
    s_dag(b, qubit)
    h(b, qubit)


def c_nxyz(b: GraphRepresentation, qubit: int) -> None:
    """Period 3 axis cycling gate, sending -X -> Y -> Z -> -X."""
    s_dag(b, qubit)
    sqrt_y_dag(b, qubit)


def c_xnyz(b: GraphRepresentation, qubit: int) -> None:
    """Period 3 axis cycling gate, sending X -> -Y -> Z -> X."""
    s(b, qubit)
    h(b, qubit)


def c_xynz(b: GraphRepresentation, qubit: int) -> None:
    """Period 3 axis cycling gate, sending X -> Y -> -Z -> X."""
    s(b, qubit)
    sqrt_y_dag(b, qubit)


def c_zyx(b: GraphRepresentation, qubit: int) -> None:
    """Left handed period 3 axis cycling gate, sending Z -> Y -> X -> Z."""
    h(b, qubit)
    s(b, qubit)


def c_nzyx(b: GraphRepresentation, qubit: int) -> None:
    """Period 3 axis cycling gate, sending -Z -> Y -> X -> -Z."""
    s_dag(b, qubit)
    sqrt_x(b, qubit)


def c_znyx(b: GraphRepresentation, qubit: int) -> None:
    """Period 3 axis cycling gate, sending Z -> -Y -> X -> Z."""
    s(b, qubit)
    sqrt_x(b, qubit)


def c_zynx(b: GraphRepresentation, qubit: int) -> None:
    """Period 3 axis cycling gate, sending Z -> Y -> -X -> Z."""
    s(b, qubit)
    sqrt_x_dag(b, qubit)


def h(b: GraphRepresentation, qubit: int) -> None:
    """Apply Hadamard gate."""
    ensure_lane(b, qubit)
    b.graph.toggle_edge(*last_edge(b, qubit))


def h_xy(b: GraphRepresentation, qubit: int) -> None:
    """Apply variant of Hadamard gate that swaps the X and Y axes (instead of X and Z)."""
    x(b, qubit)
    s(b, qubit)


def h_nxy(b: GraphRepresentation, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -X <-> Y, Z -> -Z."""
    x(b, qubit)
    s_dag(b, qubit)


def h_nxz(b: GraphRepresentation, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -X <-> Z."""
    z(b, qubit)
    sqrt_y_dag(b, qubit)


def h_yz(b: GraphRepresentation, qubit: int) -> None:
    """Apply variant of Hadamard gate that swaps the Y and Z axes (instead of X and Z)."""
    sqrt_x(b, qubit)
    z(b, qubit)


def h_nyz(b: GraphRepresentation, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -Y <-> Z, X -> -X."""
    z(b, qubit)
    sqrt_x(b, qubit)


def s(b: GraphRepresentation, qubit: int) -> None:
    """Apply S gate (π/2 Z rotation)."""
    z_phase(b, qubit, Fraction(1, 2))


def sqrt_x(b: GraphRepresentation, qubit: int) -> None:
    """Apply √X gate (π/2 X rotation)."""
    x_phase(b, qubit, Fraction(1, 2))


def sqrt_x_dag(b: GraphRepresentation, qubit: int) -> None:
    """Apply √X† gate (-π/2 X rotation)."""
    x_phase(b, qubit, Fraction(-1, 2))


def sqrt_y(b: GraphRepresentation, qubit: int) -> None:
    """Apply √Y gate (π/2 Y rotation)."""
    z(b, qubit)
    h(b, qubit)


def sqrt_y_dag(b: GraphRepresentation, qubit: int) -> None:
    """Apply √Y† gate (-π/2 Y rotation)."""
    h(b, qubit)
    z(b, qubit)


def sqrt_z(b: GraphRepresentation, qubit: int) -> None:
    """Apply √Z gate (alias for S gate)."""
    s(b, qubit)


def sqrt_z_dag(b: GraphRepresentation, qubit: int) -> None:
    """Apply √Z† gate (alias for S† gate)."""
    s_dag(b, qubit)


def s_dag(b: GraphRepresentation, qubit: int) -> None:
    """Apply S† gate (-π/2 Z rotation)."""
    z_phase(b, qubit, Fraction(-1, 2))


# =============================================================================
# Two-Qubit Gates
# =============================================================================


def _cx_cz(
    b: GraphRepresentation,
    is_cx: bool,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Implement CNOT or CZ gate depending on is_cx flag."""
    edge_type = EdgeType.SIMPLE if is_cx else EdgeType.HADAMARD
    vertex_type = VertexType.X if is_cx else VertexType.Z

    m_vertex: Vertex | None = None
    if classically_controlled:
        assert len(classically_controlled) == 2
        if classically_controlled[1] and not is_cx:
            # Only control is allowed to be classically controlled, swap control and target for symmetric CZ gate
            classically_controlled = classically_controlled[::-1]
            control, target = target, control
        if classically_controlled[1]:
            raise ValueError("Measurement record editing is not supported.")
        m_vertex = b.rec[control]
        control = int(b.graph.row(m_vertex))
    ensure_lane(b, control)
    ensure_lane(b, target)

    col = max(last_col(b, control), last_col(b, target))

    v1 = b.last_vertex[control]
    b.graph.set_vertex_type(v1, VertexType.Z)
    b.graph.set_col(v1, col)
    v3 = add_dummy(b, control, int(col + 1))
    b.graph.add_edge(v1, v3)

    if control == target:
        col += 1

    v2 = b.last_vertex[target]
    b.graph.set_vertex_type(v2, vertex_type)
    b.graph.set_col(v2, col)
    v4 = add_dummy(b, target, int(col + 1))
    b.graph.add_edge(v2, v4)

    b.graph.add_edge(v1 if m_vertex is None else m_vertex, v2, edge_type)
    _scale(b, 1)


def cnot(
    b: GraphRepresentation,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply CNOT (controlled-X) gate."""
    _cx_cz(b, True, control, target, classically_controlled)


def cy(
    b: GraphRepresentation,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply controlled-Y gate."""
    s_dag(b, target)
    cnot(b, control, target, classically_controlled)
    s(b, target)


def cz(
    b: GraphRepresentation,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply controlled-Z gate."""
    _cx_cz(b, False, control, target, classically_controlled)


def swap(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Apply SWAP gate."""
    ensure_lane(b, qubit1)
    ensure_lane(b, qubit2)

    v1 = b.last_vertex[qubit1]
    v2 = b.last_vertex[qubit2]
    b.last_vertex[qubit1] = v2
    b.last_vertex[qubit2] = v1

    b.graph.set_row(v1, qubit2)
    b.graph.set_row(v2, qubit1)


def cxswap(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Apply CX then SWAP."""
    cnot(b, qubit1, qubit2)
    swap(b, qubit1, qubit2)


def czswap(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Apply CZ then SWAP."""
    cz(b, qubit1, qubit2)
    swap(b, qubit1, qubit2)


def swapcx(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Apply SWAP then CX."""
    swap(b, qubit1, qubit2)
    cnot(b, qubit1, qubit2)


def swapcz(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Apply SWAP then CZ."""
    swap(b, qubit1, qubit2)
    cz(b, qubit1, qubit2)


def iswap(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Swap two qubits and phase the -1 eigenspace of the ZZ observable by i."""
    cnot(b, qubit1, qubit2)
    s(b, qubit2)
    cnot(b, qubit1, qubit2)
    swap(b, qubit1, qubit2)


def iswap_dag(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Swap two qubits and phase the -1 eigenspace of the ZZ observable by -i."""
    cnot(b, qubit1, qubit2)
    s_dag(b, qubit2)
    cnot(b, qubit1, qubit2)
    swap(b, qubit1, qubit2)


def sqrt_xx(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the XX observable by i."""
    cnot(b, qubit1, qubit2)
    sqrt_x(b, qubit1)
    cnot(b, qubit1, qubit2)


def sqrt_xx_dag(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the XX observable by -i."""
    cnot(b, qubit1, qubit2)
    sqrt_x_dag(b, qubit1)
    cnot(b, qubit1, qubit2)


def sqrt_yy(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the YY observable by i."""
    s(b, qubit1)
    cnot(b, qubit2, qubit1)
    z(b, qubit1)
    h(b, qubit2)
    cnot(b, qubit2, qubit1)
    s(b, qubit1)


def sqrt_yy_dag(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the YY observable by -i."""
    s_dag(b, qubit1)
    cnot(b, qubit2, qubit1)
    h(b, qubit2)
    z(b, qubit1)
    cnot(b, qubit2, qubit1)
    s_dag(b, qubit1)


def sqrt_zz(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the ZZ observable by i."""
    cnot(b, qubit1, qubit2)
    s(b, qubit2)
    cnot(b, qubit1, qubit2)


def sqrt_zz_dag(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the ZZ observable by -i."""
    h(b, qubit2)
    cnot(b, qubit1, qubit2)
    h(b, qubit2)
    s_dag(b, qubit1)
    s_dag(b, qubit2)


def xcx(b: GraphRepresentation, control: int, target: int) -> None:
    """X-controlled X gate. Applies X to target if control is in |-> state."""
    h(b, control)
    cnot(b, control, target)
    h(b, control)


def xcy(b: GraphRepresentation, control: int, target: int) -> None:
    """X-controlled Y gate. Applies Y to target if control is in |-> state."""
    h(b, control)
    cy(b, control, target)
    h(b, control)


def xcz(
    b: GraphRepresentation,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """X-controlled Z gate. Applies Z to target if control is in |-> state."""
    cnot(
        b,
        target,
        control,
        classically_controlled[::-1] if classically_controlled else None,
    )


def ycx(b: GraphRepresentation, control: int, target: int) -> None:
    """Y-controlled X gate. Applies X to target if control is in |-i> state."""
    h_yz(b, control)
    cnot(b, control, target)
    h_yz(b, control)


def ycy(b: GraphRepresentation, control: int, target: int) -> None:
    """Y-controlled Y gate. Applies Y to target if control is in |-i> state."""
    h_yz(b, control)
    cy(b, control, target)
    h_yz(b, control)


def ycz(
    b: GraphRepresentation,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Y-controlled Z gate. Applies Z to target if control is in |-i> state."""
    cy(
        b,
        target,
        control,
        classically_controlled[::-1] if classically_controlled else None,
    )


# =============================================================================
# Error Channels (stores probability arrays for channel sampling)
# =============================================================================


def _error(
    b: GraphRepresentation,
    qubit: int,
    vertex_type: VertexType,
    error_bit: int | None,
) -> Vertex:
    """Insert a noise spider; it carries ``error_bit`` (its ID) unless None."""
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    v2 = add_dummy(b, qubit)
    b.graph.add_edge(v1, v2)
    b.graph.set_vertex_type(v1, vertex_type)
    if error_bit is not None:
        b.graph.set_phase(v1, _error_phase(error_bit))
    return v1


def pauli_channel_1(
    b: GraphRepresentation, qubit: int, px: float, py: float, pz: float
) -> None:
    """Apply single-qubit Pauli channel with given X, Y, Z error probabilities."""
    b.channel_probs.append(pauli_channel_1_probs(px, py, pz))
    _error(b, qubit, VertexType.Z, b.num_error_bits)
    _error(b, qubit, VertexType.X, b.num_error_bits + 1)
    b.num_error_bits += 2


def pauli_channel_2(
    b: GraphRepresentation,
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
    b.channel_probs.append(
        pauli_channel_2_probs(
            pix, piy, piz, pxi, pxx, pxy, pxz, pyi, pyx, pyy, pyz, pzi, pzx, pzy, pzz
        )
    )
    _error(b, qubit_i, VertexType.Z, b.num_error_bits)
    _error(b, qubit_i, VertexType.X, b.num_error_bits + 1)
    _error(b, qubit_j, VertexType.Z, b.num_error_bits + 2)
    _error(b, qubit_j, VertexType.X, b.num_error_bits + 3)
    b.num_error_bits += 4


def depolarize1(b: GraphRepresentation, qubit: int, p: float) -> None:
    """Apply single-qubit depolarizing channel with total error probability p."""
    pauli_channel_1(b, qubit, p / 3, p / 3, p / 3)


def depolarize2(b: GraphRepresentation, qubit_i: int, qubit_j: int, p: float) -> None:
    """Apply two-qubit depolarizing channel with total error probability p."""
    pauli_channel_2(
        b,
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


def x_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    """Apply X error with probability p."""
    b.channel_probs.append(error_probs(p))
    _error(b, qubit, VertexType.X, b.num_error_bits)
    b.num_error_bits += 1


def y_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    """Apply Y error with probability p."""
    b.channel_probs.append(error_probs(p))
    # Y = X·Z, so both vertices use the same error bit
    _error(b, qubit, VertexType.Z, b.num_error_bits)
    _error(b, qubit, VertexType.X, b.num_error_bits)
    b.num_error_bits += 1


def z_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    """Apply Z error with probability p."""
    b.channel_probs.append(error_probs(p))
    _error(b, qubit, VertexType.Z, b.num_error_bits)
    b.num_error_bits += 1


def heralded_pauli_channel_1(
    b: GraphRepresentation,
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
    b.channel_probs.append(heralded_pauli_channel_1_probs(pi, px, py, pz))
    aux = -2
    r(b, aux)
    _error(b, aux, VertexType.X, b.num_error_bits)  # herald bit flip
    m(b, aux)
    _error(b, qubit, VertexType.Z, b.num_error_bits + 1)  # Z error bit
    _error(b, qubit, VertexType.X, b.num_error_bits + 2)  # X error bit
    b.num_error_bits += 3


def heralded_erase(b: GraphRepresentation, qubit: int, p: float) -> None:
    """Apply heralded erasure channel.

    Special case of heralded_pauli_channel_1 with equal probabilities p/4
    for each of I, X, Y, Z when the channel fires.
    """
    heralded_pauli_channel_1(b, qubit, p / 4, p / 4, p / 4, p / 4)


def finalize_correlated_error(b: GraphRepresentation) -> None:
    """Finalize the current correlated error channel.

    1. Give each pending noise spider its error bit ``num_error_bits + i``
    2. Compute and append the 2^k probability array to channel_probs
    3. Increment num_error_bits by k
    4. Reset num_correlated_error_bits to 0 and correlated_error_probs to []
    """
    k = b.num_correlated_error_bits
    if k == 0:
        return
    for v, bit in b.pending_correlated:
        b.graph.set_phase(v, _error_phase(b.num_error_bits + bit))
    b.pending_correlated = []
    b.channel_probs.append(correlated_error_probs(b.correlated_error_probs))
    b.num_error_bits += k
    b.num_correlated_error_bits = 0
    b.correlated_error_probs = []


def correlated_error(
    b: GraphRepresentation,
    qubits: list[int],
    types: list[Literal["X", "Y", "Z"]],
    p: float,
) -> None:
    """Add a correlated error term affecting multiple qubits with given Pauli types."""
    for qubit, type_ in zip(qubits, types, strict=True):
        if type_ == "X" or type_ == "Y":
            b.pending_correlated.append(
                (_error(b, qubit, VertexType.X, None), b.num_correlated_error_bits)
            )
        if type_ == "Z" or type_ == "Y":
            b.pending_correlated.append(
                (_error(b, qubit, VertexType.Z, None), b.num_correlated_error_bits)
            )

    b.correlated_error_probs.append(p)
    b.num_correlated_error_bits += 1


# =============================================================================
# Collapsing gates
# =============================================================================


def _m(b: GraphRepresentation, qubit: int, p: float = 0, silent: bool = False) -> None:
    """Perform measurement on qubit with optional error probability."""
    error_bit = None
    if p > 0:
        b.channel_probs.append(error_probs(p))
        error_bit = b.num_error_bits
        _error(b, qubit, VertexType.X, error_bit)
        b.num_error_bits += 1
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_vertex_type(v1, VertexType.Z)
    _make_classical(b, v1)
    if not silent:
        b.graph.set_vertex_label(v1, f"rec[{len(b.rec)}]")
        b.rec.append(v1)
    else:
        b.graph.set_vertex_label(v1, f"m[{len(b.silent_rec)}]")
        b.silent_rec.append(v1)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge(v1, v2)
    if error_bit is not None:
        _error(b, qubit, VertexType.X, error_bit)


def _r(b: GraphRepresentation, qubit: int) -> None:
    """Perform reset on qubit.

    If the qubit does not yet have a lane, create one in the reset state.
    Otherwise, silently measure the existing lane to trace out the previous
    state before reconnecting the qubit in the reset state.
    """
    if qubit not in b.last_vertex:
        v1 = add_lane(b, qubit)
        b.graph.set_vertex_type(v1, VertexType.X)
        _scale(b, -1)
        return
    _m(b, qubit, silent=True)
    col = last_col(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_vertex_type(v1, VertexType.X)
    (v2,) = b.graph.neighbors(v1)
    b.graph.remove_edge(v1, v2)
    v3 = add_dummy(b, qubit, col + 1)
    b.graph.add_edge(v1, v3)
    _scale(b, -1)


def m(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in Z basis with optional bit-flip error probability p."""
    if invert:
        x(b, qubit)
    _m(b, qubit, p, silent=False)
    if invert:
        x(b, qubit)


def mpp(
    b: GraphRepresentation,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    invert: bool = False,
    p: float = 0,
) -> None:
    """Measure a single Pauli product.

    Args:
        b: The graph representation to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product.
        invert: Whether to invert the measurement result.
        p: Measurement flip error probability.

    """
    aux = -2
    r(b, aux)
    h(b, aux)
    _apply_pauli_controls(b, aux, paulis)
    h(b, aux)
    m(b, aux, p=p, invert=invert)


def _apply_pauli_controls(
    b: GraphRepresentation,
    aux: int,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
) -> None:
    """Apply controlled-Pauli operations between aux and target qubits."""
    for pauli_type, qubit in paulis:
        if pauli_type == "X":
            cnot(b, aux, qubit)
        elif pauli_type == "Z":
            cz(b, aux, qubit)
        elif pauli_type == "Y":
            cy(b, aux, qubit)
        else:
            raise ValueError(f"Invalid Pauli operator: {pauli_type}")


def _pauli_product_phase(
    b: GraphRepresentation,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    phase_gate: Callable[[GraphRepresentation, int], None],
    phase_gate_dag: Callable[[GraphRepresentation, int], None],
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
            h(b, qubit)
        elif pauli_type == "Y":
            s_dag(b, qubit)
            h(b, qubit)

    # Accumulate Z-parity into the last qubit
    _, last_qubit = paulis[-1]
    for _, qubit in paulis[:-1]:
        cnot(b, qubit, last_qubit)

    # Phase the parity
    if dagger:
        phase_gate_dag(b, last_qubit)
    else:
        phase_gate(b, last_qubit)

    # Uncompute parity accumulation
    for _, qubit in reversed(paulis[:-1]):
        cnot(b, qubit, last_qubit)

    # Undo basis rotations
    for pauli_type, qubit in paulis:
        if pauli_type == "X":
            h(b, qubit)
        elif pauli_type == "Y":
            h(b, qubit)
            s(b, qubit)


def spp(
    b: GraphRepresentation,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    dagger: bool = False,
) -> None:
    """Apply exp(-i pi/4 P) (up to global phase) for a Pauli product P.

    Phases the -1 eigenspace of P by i (or -i if dagger). For a single qubit,
    ``SPP Z0`` is the S gate and ``SPP_DAG Z0`` is S_DAG.

    Args:
        b: The graph representation to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        dagger: If True, apply exp(+i pi/4 P) (phase by -i) instead.

    """
    _pauli_product_phase(b, paulis, s, s_dag, dagger)


def tpp(
    b: GraphRepresentation,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    dagger: bool = False,
) -> None:
    """Apply exp(-i pi/8 P) (up to global phase) for a Pauli product P.

    Phases the -1 eigenspace of P by exp(i pi/4) (or exp(-i pi/4) if dagger).
    For a single qubit, ``TPP Z0`` is the T gate and ``TPP_DAG Z0`` is T_DAG.

    Args:
        b: The graph representation to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        dagger: If True, apply exp(+i pi/8 P) (phase by exp(-i pi/4)) instead.

    """
    _pauli_product_phase(b, paulis, t, t_dag, dagger)


def r_pauli(
    b: GraphRepresentation,
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
        b: The graph representation to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        theta: Rotation angle in half-turns (units of pi).
        dagger: If True, negate the rotation angle.

    """
    _pauli_product_phase(
        b,
        paulis,
        lambda b, qubit: r_z(b, qubit, theta),
        lambda b, qubit: r_z(b, qubit, -theta),
        dagger,
    )


def mpad(b: GraphRepresentation, value: int, p: float = 0) -> None:
    """Pad measurement record with a fixed bit value.

    Args:
        b: The graph representation to modify.
        value: The bit value to record (0 or 1).
        p: Error probability for the recorded bit.

    """
    aux = -2
    r(b, aux)
    if value == 1:
        x(b, aux)
    m(b, aux, p=p)


def mr(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Z-basis demolition measurement (optionally noisy).

    Projects each target qubit into |0> or |1>, reports its value (false=|0>, true=|1>),
    then resets to |0>.
    """
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit)


def mrx(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """X-basis demolition measurement (optionally noisy).

    Projects each target qubit into |+> or |->, reports its value (false=|+>, true=|->),
    then resets to |+>.
    """
    h(b, qubit)
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit)
    h(b, qubit)


def mry(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Y-basis demolition measurement (optionally noisy).

    Projects each target qubit into |i> or |-i>, reports its value (false=|i>, true=|-i>),
    then resets to |i>.
    """
    h_yz(b, qubit)
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit)
    h_yz(b, qubit)


def mx(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in X basis."""
    h(b, qubit)
    m(b, qubit, p=p, invert=invert)
    h(b, qubit)


def my(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in Y basis."""
    h_yz(b, qubit)
    m(b, qubit, p=p, invert=invert)
    h_yz(b, qubit)


def mxx(
    b: GraphRepresentation, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in XX basis."""
    mpp(b, [("X", q0), ("X", q1)], invert, p=p)


def myy(
    b: GraphRepresentation, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in YY basis."""
    mpp(b, [("Y", q0), ("Y", q1)], invert, p=p)


def mzz(
    b: GraphRepresentation, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in ZZ basis."""
    mpp(b, [("Z", q0), ("Z", q1)], invert, p=p)


def r(b: GraphRepresentation, qubit: int) -> None:
    """Z-basis reset.

    Forces each target qubit into the |0> state by silently measuring it in the Z basis
    and applying an X gate if it ended up in the |1> state.
    """
    _r(b, qubit)


def rx(b: GraphRepresentation, qubit: int) -> None:
    """X-basis reset.

    Forces each target qubit into the |+> state by silently measuring it in the X basis
    and applying a Z gate if it ended up in the |-> state.
    """
    if qubit in b.last_vertex:
        h(b, qubit)
    r(b, qubit)
    h(b, qubit)


def ry(b: GraphRepresentation, qubit: int) -> None:
    """Y-basis reset.

    Forces each target qubit into the |i> state by silently measuring it in the Y basis
    and applying an X gate if it ended up in the |-i> state.
    """
    if qubit in b.last_vertex:
        h_yz(b, qubit)
    r(b, qubit)
    h_yz(b, qubit)


# =============================================================================
# Annotations
# =============================================================================


def _annotation_col(b: GraphRepresentation, rec: list[int]) -> float:
    """Pick a fresh column for a detector/observable vertex.

    Empty `rec` is valid Stim and represents a deterministic-zero annotation;
    fall back to a column after existing annotations rather than `min()` on an
    empty set.
    """
    d_cols = {b.graph.col(d) for d in b.detectors + b.observables}
    if rec:
        col: float = min(b.graph.col(b.rec[r]) for r in rec) - 0.5
    else:
        col = (max(d_cols) + 1) if d_cols else 0
    while col in d_cols:
        col += 1
    return col


def _xor_record(b: GraphRepresentation, annotation: Vertex, rec: int) -> None:
    """Add measurement record ``rec`` to the parity of ``annotation``.

    A record that is already in the parity cancels: its edge is removed rather
    than Hopf-cancelled by ``add_edge``, which would leave a scalar of 1/2 behind
    in graphs that drop the annotation.
    """
    record = b.rec[rec]
    if b.graph.has_edge(annotation, record):
        b.graph.remove_edge(annotation, record)
    else:
        b.graph.add_edge(annotation, record)


def detector(b: GraphRepresentation, rec: list[int], *args) -> None:
    """Add detector annotation that XORs the given measurement record bits."""
    v0 = b.graph.add_vertex(
        VertexType.X,
        col=_annotation_col(b, rec),
        row=-1,
        domain=Domain.CLASSICAL,
        label=f"det[{len(b.detectors)}]",
    )
    for rec_ in rec:
        _xor_record(b, v0, rec_)
    b.detectors.append(v0)


def observable_include(b: GraphRepresentation, rec: list[int], idx: int) -> None:
    """Add observable annotation that XORs the given measurement record bits."""
    idx = int(idx)

    if idx not in b.observables_dict:
        v0 = b.graph.add_vertex(
            VertexType.X,
            col=_annotation_col(b, rec),
            row=-1,
            domain=Domain.CLASSICAL,
            label=f"obs[{idx}]",
        )
        b.observables_dict[idx] = v0

    v0 = b.observables_dict[idx]
    for rec_ in rec:
        _xor_record(b, v0, rec_)


def tick(b: GraphRepresentation) -> None:
    """Add a tick to the circuit (align all qubits to same column)."""
    if len(b.last_vertex) == 0:
        return
    col = max(last_col(b, q) for q in b.last_vertex)
    for q in b.last_vertex:
        b.graph.set_col(b.last_vertex[q], col)


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
