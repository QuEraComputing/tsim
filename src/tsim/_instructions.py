from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

from tsim.channels import (
    Depolarize1,
    Depolarize2,
    Error,
    ErrorSpec,
    PauliChannel1,
    PauliChannel2,
)
from tsim.external.pyzx.graph.graph_s import GraphS
from tsim.external.pyzx.utils import EdgeType, VertexType


@dataclass
class GraphRepresentation:
    """ZX graph built from a stim circuit.

    Contains the graph and all auxiliary data needed for sampling.
    """

    graph: GraphS = field(default_factory=GraphS)
    rec: list[int] = field(default_factory=list)
    silent_rec: list[int] = field(default_factory=list)
    detectors: list[int] = field(default_factory=list)
    observables_dict: dict[int, int] = field(default_factory=dict)
    first_vertex: dict[int, int] = field(default_factory=dict)
    last_vertex: dict[int, int] = field(default_factory=dict)
    error_specs: list[ErrorSpec] = field(default_factory=list)
    num_error_bits: int = 0

    @property
    def observables(self) -> list[int]:
        """Get list of observable vertices sorted by index."""
        return [self.observables_dict[i] for i in sorted(self.observables_dict)]


def last_row(b: GraphRepresentation, qubit: int) -> float:
    """Get the row of the last vertex for a qubit."""
    return b.graph.row(b.last_vertex[qubit])


def last_edge(b: GraphRepresentation, qubit: int):
    """Get the last edge for a qubit."""
    edges = b.graph.incident_edges(b.last_vertex[qubit])
    assert len(edges) == 1
    return edges[0]


def add_dummy(
    b: GraphRepresentation, qubit: int, row: float | int | None = None
) -> int:
    """Add a dummy boundary vertex for a qubit."""
    if row is None:
        row = last_row(b, qubit) + 1
    v1 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=row)
    b.last_vertex[qubit] = v1
    return v1


def add_lane(b: GraphRepresentation, qubit: int) -> int:
    """Initialize a qubit lane if it doesn't exist."""
    v1 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=0)
    v2 = b.graph.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=1)
    b.graph.add_edge((v1, v2))
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
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_type(v1, VertexType.X)
    b.graph.set_phase(v1, phase)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2))


def z_phase(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_type(v1, VertexType.Z)
    b.graph.set_phase(v1, phase)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2))


def t(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(1, 4))


def t_dag(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(-1, 4))


def r_z(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    z_phase(b, qubit, phase)
    b.graph.scalar.add_phase(-phase / 2)


def r_x(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    x_phase(b, qubit, phase)
    b.graph.scalar.add_phase(-phase / 2)


def r_y(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
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
    r_z(b, qubit, lambda_)
    r_y(b, qubit, theta)
    r_z(b, qubit, phi)
    b.graph.scalar.add_phase((phi + lambda_) / 2)


# =============================================================================
# Pauli Gates
# =============================================================================


def i(b: GraphRepresentation, qubit: int) -> None:
    """Apply identity (advances the row)."""
    ensure_lane(b, qubit)
    v = b.last_vertex[qubit]
    b.graph.set_row(v, last_row(b, qubit) + 1)


def x(b: GraphRepresentation, qubit: int) -> None:
    x_phase(b, qubit, Fraction(1, 1))


def y(b: GraphRepresentation, qubit: int) -> None:
    z(b, qubit)
    x(b, qubit)
    b.graph.scalar.add_phase(Fraction(1, 2))


def z(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(1, 1))


# =============================================================================
# Single-Qubit Clifford gates
# =============================================================================


def c_xyz(b: GraphRepresentation, qubit: int) -> None:
    """Right handed period 3 axis cycling gate, sending X -> Y -> Z -> X."""
    s_dag(b, qubit)
    h(b, qubit)
    b.graph.scalar.add_phase(Fraction(-1, 4))


def c_zyx(b: GraphRepresentation, qubit: int) -> None:
    """Left handed period 3 axis cycling gate, sending Z -> Y -> X -> Z."""
    h(b, qubit)
    s(b, qubit)
    b.graph.scalar.add_phase(Fraction(1, 4))


def h(b: GraphRepresentation, qubit: int) -> None:
    ensure_lane(b, qubit)
    e = last_edge(b, qubit)
    b.graph.set_edge_type(
        e,
        (
            EdgeType.HADAMARD
            if b.graph.edge_type(e) == EdgeType.SIMPLE
            else EdgeType.SIMPLE
        ),
    )


def h_xy(b: GraphRepresentation, qubit: int) -> None:
    """Variant of Hadamard gate that swaps the X and Y axes (instead of X and Z)."""
    x(b, qubit)
    s(b, qubit)
    b.graph.scalar.add_phase(Fraction(-1, 4))


def h_yz(b: GraphRepresentation, qubit: int) -> None:
    """Variant of Hadamard gate that swaps the Y and Z axes (instead of X and Z)."""
    sqrt_x(b, qubit)
    z(b, qubit)
    b.graph.scalar.add_phase(Fraction(-1, 4))


def s(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(1, 2))


def sqrt_x(b: GraphRepresentation, qubit: int) -> None:
    x_phase(b, qubit, Fraction(1, 2))


def sqrt_x_dag(b: GraphRepresentation, qubit: int) -> None:
    x_phase(b, qubit, Fraction(-1, 2))


def sqrt_y(b: GraphRepresentation, qubit: int) -> None:
    z(b, qubit)
    h(b, qubit)
    b.graph.scalar.add_phase(Fraction(1, 4))


def sqrt_y_dag(b: GraphRepresentation, qubit: int) -> None:
    h(b, qubit)
    z(b, qubit)
    b.graph.scalar.add_phase(Fraction(-1, 4))


def sqrt_z(b: GraphRepresentation, qubit: int) -> None:
    s(b, qubit)


def sqrt_z_dag(b: GraphRepresentation, qubit: int) -> None:
    s_dag(b, qubit)


def s_dag(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(-1, 2))


# =============================================================================
# Two-Qubit Gates
# =============================================================================


def _cx_cz(
    b: GraphRepresentation,
    is_cx: bool,
    control: int,
    target: int,
    classically_controlled: bool = False,
) -> None:
    edge_type = EdgeType.SIMPLE if is_cx else EdgeType.HADAMARD
    vertex_type = VertexType.X if is_cx else VertexType.Z

    m_vertex = 0
    if classically_controlled:
        m_vertex = b.rec[control]
        control = b.graph.qubit(m_vertex)
    ensure_lane(b, control)
    ensure_lane(b, target)

    lr1 = last_row(b, control)
    lr2 = last_row(b, target)
    row = max(lr1, lr2)

    v1 = b.last_vertex[control]
    b.graph.set_type(v1, VertexType.Z)
    b.graph.set_row(v1, row)
    v3 = add_dummy(b, control, int(row + 1))
    b.graph.add_edge((v1, v3))

    if control == target:
        row += 1

    v2 = b.last_vertex[target]
    b.graph.set_type(v2, vertex_type)
    b.graph.set_row(v2, row)
    v4 = add_dummy(b, target, int(row + 1))
    b.graph.add_edge((v2, v4))

    if classically_controlled:
        b.graph.add_edge((m_vertex, v2), edge_type)
    else:
        b.graph.add_edge((v1, v2), edge_type)
    b.graph.scalar.add_power(1)


def cnot(
    b: GraphRepresentation,
    control: int,
    target: int,
    classically_controlled: bool = False,
) -> None:
    _cx_cz(b, True, control, target, classically_controlled)


def cy(
    b: GraphRepresentation,
    control: int,
    target: int,
    classically_controlled: bool = False,
) -> None:
    s_dag(b, target)
    cnot(b, control, target, classically_controlled)
    s(b, target)


def cz(
    b: GraphRepresentation,
    control: int,
    target: int,
    classically_controlled: bool = False,
) -> None:
    _cx_cz(b, False, control, target, classically_controlled)


def swap(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    ensure_lane(b, qubit1)
    ensure_lane(b, qubit2)

    v1 = b.last_vertex[qubit1]
    v2 = b.last_vertex[qubit2]
    b.last_vertex[qubit1] = v2
    b.last_vertex[qubit2] = v1

    b.graph.set_qubit(v1, qubit2)
    b.graph.set_qubit(v2, qubit1)


def iswap(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Swaps two qubits and phases the -1 eigenspace of the ZZ observable by i."""
    cnot(b, qubit1, qubit2)
    s(b, qubit2)
    cnot(b, qubit1, qubit2)
    swap(b, qubit1, qubit2)


def iswap_dag(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Swaps two qubits and phases the -1 eigenspace of the ZZ observable by -i."""
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
    b.graph.scalar.add_phase(Fraction(1, 4))


def sqrt_yy_dag(b: GraphRepresentation, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the YY observable by -i."""
    s_dag(b, qubit1)
    cnot(b, qubit2, qubit1)
    h(b, qubit2)
    z(b, qubit1)
    cnot(b, qubit2, qubit1)
    s_dag(b, qubit1)
    b.graph.scalar.add_phase(Fraction(-1, 4))


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


def xcz(b: GraphRepresentation, control: int, target: int) -> None:
    """X-controlled Z gate. Applies Z to target if control is in |-> state."""
    cnot(b, target, control)


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


def ycz(b: GraphRepresentation, control: int, target: int) -> None:
    """Y-controlled Z gate. Applies Z to target if control is in |-i> state."""
    h_yz(b, control)
    cz(b, control, target)
    h_yz(b, control)


# =============================================================================
# Error Channels (creates ErrorSpecs, not actual channels)
# =============================================================================


def _error(b: GraphRepresentation, qubit: int, error_type: int, phase: str) -> None:
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2))

    b.graph.set_type(v1, error_type)
    b.graph.set_phase(v1, phase)


def depolarize1(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Depolarize1, (p,)))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits + 1}")
    b.num_error_bits += 2


def depolarize2(b: GraphRepresentation, qubit_i: int, qubit_j: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Depolarize2, (p,)))
    _error(b, qubit_i, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit_i, VertexType.X, f"e{b.num_error_bits + 1}")
    _error(b, qubit_j, VertexType.Z, f"e{b.num_error_bits + 2}")
    _error(b, qubit_j, VertexType.X, f"e{b.num_error_bits + 3}")
    b.num_error_bits += 4


def pauli_channel_1(
    b: GraphRepresentation, qubit: int, px: float, py: float, pz: float
) -> None:
    b.error_specs.append(ErrorSpec(PauliChannel1, (px, py, pz)))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits + 1}")
    b.num_error_bits += 2


def pauli_channel_2(
    b: GraphRepresentation,
    qubit_i: int,
    qubit_j: int,
    pix: float = 0,
    piy: float = 0,
    piz: float = 0,
    pxi: float = 0,
    pxx: float = 0,
    pxy: float = 0,
    pxz: float = 0,
    pyi: float = 0,
    pyx: float = 0,
    pyy: float = 0,
    pyz: float = 0,
    pzi: float = 0,
    pzx: float = 0,
    pzy: float = 0,
    pzz: float = 0,
) -> None:
    b.error_specs.append(
        ErrorSpec(
            PauliChannel2,
            (pix, piy, piz, pxi, pxx, pxy, pxz, pyi, pyx, pyy, pyz, pzi, pzx, pzy, pzz),
        )
    )
    _error(b, qubit_i, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit_i, VertexType.X, f"e{b.num_error_bits + 1}")
    _error(b, qubit_j, VertexType.Z, f"e{b.num_error_bits + 2}")
    _error(b, qubit_j, VertexType.X, f"e{b.num_error_bits + 3}")
    b.num_error_bits += 4


def x_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Error, (p,)))
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits}")
    b.num_error_bits += 1


def y_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Error, (p,)))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits}")
    b.num_error_bits += 1


def z_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Error, (p,)))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    b.num_error_bits += 1


# =============================================================================
# Collapsing gates
# =============================================================================


def _m(b: GraphRepresentation, qubit: int, p: float = 0, silent: bool = False) -> None:
    """Internal measurement implementation."""
    if p > 0:
        x_error(b, qubit, p)
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_type(v1, VertexType.Z)
    if not silent:
        b.graph.set_phase(v1, f"rec[{len(b.rec)}]")
        b.rec.append(v1)
    else:
        b.graph.set_phase(v1, f"m[{len(b.silent_rec)}]")
        b.silent_rec.append(v1)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2))
    b.graph.scalar.add_power(-1)


def _r(b: GraphRepresentation, qubit: int, perform_trace: bool) -> None:
    if qubit not in b.last_vertex:
        v1 = add_lane(b, qubit)
        b.graph.set_type(v1, VertexType.X)
        b.graph.scalar.add_power(-1)
    else:
        v = b.last_vertex[qubit]
        neighbors = list(b.graph.neighbors(v))
        assert len(neighbors) == 1
        if perform_trace:
            _m(b, qubit, silent=True)
        row = last_row(b, qubit)
        v1 = b.last_vertex[qubit]
        b.graph.set_type(v1, VertexType.X)
        v2 = list(b.graph.neighbors(v1))[0]
        b.graph.remove_edge((v1, v2))
        v3 = add_dummy(b, qubit, row + 1)
        b.graph.add_edge((v1, v3))
        b.graph.scalar.add_power(-1)


def m(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    if invert:
        x(b, qubit)
    _m(b, qubit, p, silent=False)
    if invert:
        x(b, qubit)


def mpp(b: GraphRepresentation, pp: str | list[str]) -> None:
    if isinstance(pp, list):
        for pp_ in pp:
            mpp(b, pp_)
        return

    aux = -2
    r(b, aux)
    h(b, aux)

    invert_rec = pp[0] == "!"
    if invert_rec:
        pp = pp[1:]

    components = pp.split("*")

    for comp in components:
        p, idx = comp[0].lower(), int(comp[1:])

        if p == "x":
            cnot(b, aux, idx)
        elif p == "z":
            cz(b, aux, idx)
        elif p == "y":
            cy(b, aux, idx)
        else:
            raise ValueError(f"Invalid Pauli operator: {p}")

    h(b, aux)
    m(b, aux, invert=invert_rec)


def mr(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    if p > 0:
        x_error(b, qubit, p)
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit, perform_trace=False)


def mrx(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    h(b, qubit)
    if p > 0:
        x_error(b, qubit, p)
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit, perform_trace=False)
    h(b, qubit)


def mry(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    h_yz(b, qubit)
    if p > 0:
        x_error(b, qubit, p)
    m(b, qubit, p=p, invert=invert)
    _r(b, qubit, perform_trace=False)
    h_yz(b, qubit)


def mx(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    h(b, qubit)
    m(b, qubit, p=p, invert=invert)
    h(b, qubit)


def my(b: GraphRepresentation, qubit: int, p: float = 0, invert: bool = False) -> None:
    h_yz(b, qubit)
    m(b, qubit, p=p, invert=invert)
    h_yz(b, qubit)


def r(b: GraphRepresentation, qubit: int) -> None:
    _r(b, qubit, perform_trace=True)


def rx(b: GraphRepresentation, qubit: int) -> None:
    if qubit in b.last_vertex:
        h(b, qubit)
    r(b, qubit)
    h(b, qubit)


def ry(b: GraphRepresentation, qubit: int) -> None:
    if qubit in b.last_vertex:
        h_yz(b, qubit)
    r(b, qubit)
    h_yz(b, qubit)


# =============================================================================
# Annotations
# =============================================================================


def detector(b: GraphRepresentation, rec: list[int], *args) -> None:
    row = min(set([b.graph.row(b.rec[r]) for r in rec])) - 0.5
    d_rows = set([b.graph.row(d) for d in b.detectors + b.observables])
    while row in d_rows:
        row += 1
    v0 = b.graph.add_vertex(
        VertexType.X, qubit=-1, row=row, phase=f"det[{len(b.detectors)}]"
    )
    for rec_ in rec:
        b.graph.add_edge((v0, b.rec[rec_]))
    b.detectors.append(v0)


def observable_include(b: GraphRepresentation, rec: list[int], idx: int) -> None:
    idx = int(idx)

    if idx not in b.observables_dict:
        row = min(set([b.graph.row(b.rec[r]) for r in rec])) - 0.5
        d_rows = set([b.graph.row(d) for d in b.detectors + b.observables])
        while row in d_rows:
            row += 1
        v0 = b.graph.add_vertex(VertexType.X, qubit=-1, row=row, phase=f"obs[{idx}]")
        b.observables_dict[idx] = v0

    v0 = b.observables_dict[idx]
    for rec_ in rec:
        b.graph.add_edge((v0, b.rec[rec_]))


def tick(b: GraphRepresentation) -> None:
    """Add a tick to the circuit (align all qubits to same row)."""
    if len(b.last_vertex) == 0:
        return
    row = max(last_row(b, q) for q in b.last_vertex)
    for q in b.last_vertex:
        b.graph.set_row(b.last_vertex[q], row)


# =============================================================================
# Gate Dispatch Table
# =============================================================================

GATE_TABLE: dict[str, tuple[Callable[..., None], int]] = {
    # ---- Pauli gates -----------------------------------------------------------
    "I": (i, 1),
    "X": (x, 1),
    "Y": (y, 1),
    "Z": (z, 1),
    # ---- Non-Clifford gates ---------------------------------------------------
    "T": (t, 1),
    "T_DAG": (t_dag, 1),
    # ---- Single-qubit gates ---------------------------------------------------
    "C_XYZ": (c_xyz, 1),
    "C_ZYX": (c_zyx, 1),
    "H": (h, 1),
    "H_XY": (h_xy, 1),
    "H_YZ": (h_yz, 1),
    "H_XZ": (h, 1),
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
    "CZ": (cz, 2),
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
    "X_ERROR": (x_error, 1),
    "Y_ERROR": (y_error, 1),
    "Z_ERROR": (z_error, 1),
    # ---- Collapsing gates -----------------------------------------------------
    "M": (m, 1),
    # MPP handled by parser
    "MR": (mr, 1),
    "MRX": (mrx, 1),
    "MRY": (mry, 1),
    "MRZ": (mr, 1),
    "MX": (mx, 1),
    "MY": (my, 1),
    "MZ": (m, 1),
    "R": (r, 1),
    "RX": (rx, 1),
    "RY": (ry, 1),
    "RZ": (r, 1),
    # ---- Annotations handled by parser -----------------------------------------
}
