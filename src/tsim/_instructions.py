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
# Single-Qubit Gates
# =============================================================================


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


def x_phase(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_type(v1, VertexType.X)
    b.graph.set_phase(v1, phase)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2))


def x(b: GraphRepresentation, qubit: int) -> None:
    x_phase(b, qubit, Fraction(1, 1))


def z_phase(b: GraphRepresentation, qubit: int, phase: Fraction) -> None:
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    b.graph.set_type(v1, VertexType.Z)
    b.graph.set_phase(v1, phase)
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2))


def z(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(1, 1))


def y(b: GraphRepresentation, qubit: int) -> None:
    z(b, qubit)
    x(b, qubit)
    b.graph.scalar.add_phase(Fraction(1, 2))


def s(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(1, 2))


def s_dag(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(-1, 2))


def t(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(1, 4))


def t_dag(b: GraphRepresentation, qubit: int) -> None:
    z_phase(b, qubit, Fraction(-1, 4))


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


def i(b: GraphRepresentation, qubit: int) -> None:
    """Apply identity (advances the row)."""
    ensure_lane(b, qubit)
    v = b.last_vertex[qubit]
    b.graph.set_row(v, last_row(b, qubit) + 1)


# =============================================================================
# Reset and Measurement
# =============================================================================


def r(b: GraphRepresentation, qubit: int) -> None:
    if qubit not in b.last_vertex:
        v1 = add_lane(b, qubit)
        b.graph.set_type(v1, VertexType.X)
    else:
        # If the last vertex is not a measurement, we need to perform silent measurement
        v = b.last_vertex[qubit]
        neighbors = list(b.graph.neighbors(v))
        assert len(neighbors) == 1
        n = neighbors[0]
        last_vertex_is_measurement = any("rec" in var for var in b.graph._phaseVars[n])
        if not last_vertex_is_measurement:
            _m(b, qubit, silent=True)
        row = last_row(b, qubit)
        v1 = b.last_vertex[qubit]
        b.graph.set_type(v1, VertexType.X)
        v2 = list(b.graph.neighbors(v1))[0]
        b.graph.remove_edge((v1, v2))
        v3 = add_dummy(b, qubit, row + 1)
        b.graph.add_edge((v1, v3))


def rx(b: GraphRepresentation, qubit: int) -> None:
    if qubit in b.last_vertex:
        h(b, qubit)
    r(b, qubit)
    h(b, qubit)


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


def m(b: GraphRepresentation, qubit: int, p: float = 0) -> None:
    _m(b, qubit, p, silent=False)


def mx(b: GraphRepresentation, qubit: int, p: float = 0) -> None:
    h(b, qubit)
    m(b, qubit, p=p)


def mr(b: GraphRepresentation, qubit: int, p: float = 0) -> None:
    if p > 0:
        x_error(b, qubit, p)
    m(b, qubit, p=p)
    r(b, qubit)


def mpp(b: GraphRepresentation, pp: str | list[str]) -> None:
    if isinstance(pp, list):
        for pp_ in pp:
            mpp(b, pp_)
        return

    aux = -2
    r(b, aux)
    h(b, aux)

    components = pp.split("*")

    for comp in components:
        p, idx = comp[0].lower(), int(comp[1:])

        if p == "x":
            cx(b, aux, idx)
        elif p == "z":
            cz(b, aux, idx)
        elif p == "y":
            cy(b, aux, idx)
        else:
            raise ValueError(f"Invalid Pauli operator: {p}")

    h(b, aux)
    m(b, aux)


# =============================================================================
# Two-Qubit Gates
# =============================================================================


def cx(b: GraphRepresentation, control: int, target: int) -> None:
    ensure_lane(b, control)
    ensure_lane(b, target)

    lr1 = last_row(b, control)
    lr2 = last_row(b, target)
    row = max(lr1, lr2)

    v1 = b.last_vertex[control]
    v2 = b.last_vertex[target]
    b.graph.set_type(v1, VertexType.Z)
    b.graph.set_type(v2, VertexType.X)
    b.graph.set_row(v1, row)
    b.graph.set_row(v2, row)
    b.graph.add_edge((v1, v2))

    v3 = add_dummy(b, control, int(row + 1))
    v4 = add_dummy(b, target, int(row + 1))
    b.graph.add_edge((v1, v3))
    b.graph.add_edge((v2, v4))

    b.graph.scalar.add_power(1)


def cz(b: GraphRepresentation, control: int, target: int) -> None:
    ensure_lane(b, control)
    ensure_lane(b, target)

    lr1 = last_row(b, control)
    lr2 = last_row(b, target)
    row = max(lr1, lr2)

    v1 = b.last_vertex[control]
    v2 = b.last_vertex[target]
    b.graph.set_type(v1, VertexType.Z)
    b.graph.set_type(v2, VertexType.Z)
    b.graph.set_row(v1, row)
    b.graph.set_row(v2, row)
    b.graph.add_edge((v1, v2), EdgeType.HADAMARD)

    v3 = add_dummy(b, control, int(row + 1))
    v4 = add_dummy(b, target, int(row + 1))
    b.graph.add_edge((v1, v3))
    b.graph.add_edge((v2, v4))

    b.graph.scalar.add_power(1)


def cy(b: GraphRepresentation, control: int, target: int) -> None:
    s_dag(b, target)
    cx(b, control, target)
    s(b, target)


# =============================================================================
# Error Channels (creates ErrorSpecs, not actual channels)
# =============================================================================


def _error(b: GraphRepresentation, qubit: int, error_type: int, phase: str) -> None:
    ensure_lane(b, qubit)
    v1 = b.last_vertex[qubit]
    v2 = add_dummy(b, qubit)
    b.graph.add_edge((v1, v2))

    b.graph.set_type(v1, error_type)  # type: ignore[arg-type]
    b.graph.set_phase(v1, phase)  # type: ignore[arg-type]


def x_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Error, (p,)))
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits}")
    b.num_error_bits += 1


def z_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Error, (p,)))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    b.num_error_bits += 1


def y_error(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Error, (p,)))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits}")
    b.num_error_bits += 1


def depolarize1(b: GraphRepresentation, qubit: int, p: float) -> None:
    b.error_specs.append(ErrorSpec(Depolarize1, (p,)))
    _error(b, qubit, VertexType.Z, f"e{b.num_error_bits}")
    _error(b, qubit, VertexType.X, f"e{b.num_error_bits + 1}")
    b.num_error_bits += 2


def pauli_channel_1(
    b: GraphRepresentation, qubit: int, px: float, py: float, pz: float
) -> None:
    b.error_specs.append(ErrorSpec(PauliChannel1, (px, py, pz)))
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


# =============================================================================
# Annotations
# =============================================================================


def tick(b: GraphRepresentation) -> None:
    """Add a tick to the circuit (align all qubits to same row)."""
    if len(b.last_vertex) == 0:
        return
    row = max(last_row(b, q) for q in b.last_vertex)
    for q in b.last_vertex:
        b.graph.set_row(b.last_vertex[q], row)


def detector(b: GraphRepresentation, rec: list[int], *args) -> None:
    row = min(set([b.graph.row(b.rec[r]) for r in rec])) - 0.5
    d_rows = set([b.graph.row(d) for d in b.detectors + b.observables])
    while row in d_rows:
        row += 1
    v0 = b.graph.add_vertex(
        VertexType.X, qubit=-1, row=row, phase=f"det[{len(b.detectors)}]"  # type: ignore[arg-type]
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
        v0 = b.graph.add_vertex(
            VertexType.X, qubit=-1, row=row, phase=f"obs[{idx}]"  # type: ignore[arg-type]
        )
        b.observables_dict[idx] = v0

    v0 = b.observables_dict[idx]
    for rec_ in rec:
        b.graph.add_edge((v0, b.rec[rec_]))


# =============================================================================
# Gate Dispatch Table
# =============================================================================

GATE_TABLE: dict[str, tuple[Callable[..., None], int]] = {
    # Single-qubit gates
    "H": (h, 1),
    "X": (x, 1),
    "Y": (y, 1),
    "Z": (z, 1),
    "S": (s, 1),
    "S_DAG": (s_dag, 1),
    "T": (t, 1),
    "T_DAG": (t_dag, 1),
    "SQRT_X": (sqrt_x, 1),
    "SQRT_X_DAG": (sqrt_x_dag, 1),
    "SQRT_Y": (sqrt_y, 1),
    "SQRT_Y_DAG": (sqrt_y_dag, 1),
    "SQRT_Z": (sqrt_z, 1),
    "SQRT_Z_DAG": (sqrt_z_dag, 1),
    "R": (r, 1),
    "RX": (rx, 1),
    "M": (m, 1),
    "MX": (mx, 1),
    "MR": (mr, 1),
    "I": (i, 1),
    # Two-qubit gates
    "CNOT": (cx, 2),
    "CX": (cx, 2),
    "CZ": (cz, 2),
    "CY": (cy, 2),
    # Noise channels
    "X_ERROR": (x_error, 1),
    "Z_ERROR": (z_error, 1),
    "Y_ERROR": (y_error, 1),
    "DEPOLARIZE1": (depolarize1, 1),
    "DEPOLARIZE2": (depolarize2, 2),
    "PAULI_CHANNEL_1": (pauli_channel_1, 1),
    "PAULI_CHANNEL_2": (pauli_channel_2, 2),
}
