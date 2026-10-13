"""ZX graph representations of quantum gates and instructions.

``GraphBuilder`` owns the graph and its bookkeeping: wire ends, measurement
records, error bits and annotations. The gate library below builds on its
methods and never adds vertices itself.

Qubit wires are quantum. Every wire vertex starts as a quantum boundary, and
gates turn a wire's trailing boundary into a spider in place (see
``GraphBuilder.spider``), so the spider stays quantum. Only measurements
(``merge_mirror_copies``) and detector and observable annotations create
classical vertices. NOTE: zxparam's ``Graph.add_vertex`` defaults to the
classical domain, so wire vertices are only ever added by ``GraphBuilder``.

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


def _error_phase(error_bit: int) -> Phase:
    """Return the phase of a noise spider for error bit ``error_bit`` (ID = bit)."""
    return Phase(Fraction(0), frozenset({error_bit}))


@dataclass
class GraphBuilder:
    """Builds the ZX graph of a stim circuit, with quantum wires and classical records.

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

    # -------------------------------------------------------------------------
    # Wires
    # -------------------------------------------------------------------------

    def _last_col(self, qubit: int) -> float:
        """Get the column of the last vertex for a qubit."""
        return self.graph.col(self.last_vertex[qubit])

    def _last_edge(self, qubit: int) -> tuple[Vertex, Vertex]:
        """Get the last edge for a qubit."""
        edges = self.graph.incident_edges(self.last_vertex[qubit])
        assert len(edges) == 1
        return edges[0]

    def _add_quantum_boundary(self, qubit: int, col: float | None = None) -> Vertex:
        """Add a dummy quantum boundary vertex at the end of a qubit's wire."""
        if col is None:
            col = self._last_col(qubit) + 1
        v = self.graph.add_vertex(
            VertexType.BOUNDARY, col=col, row=qubit, domain=Domain.QUANTUM
        )
        self.last_vertex[qubit] = v
        return v

    def _add_lane(self, qubit: int) -> Vertex:
        """Initialize a qubit lane: two quantum boundaries joined by a wire."""
        v1 = self.graph.add_vertex(
            VertexType.BOUNDARY, col=0, row=qubit, domain=Domain.QUANTUM
        )
        self.first_vertex[qubit] = v1
        self.graph.add_edge(v1, self._add_quantum_boundary(qubit, col=1))
        return v1

    def _ensure_lane(self, qubit: int) -> None:
        """Ensure qubit lane exists."""
        if qubit not in self.last_vertex:
            self._add_lane(qubit)

    def spider(
        self,
        qubit: int,
        vertex_type: VertexType,
        phase: Phase | Fraction | None = None,
    ) -> Vertex:
        """Turn the trailing boundary of a qubit's wire into a quantum spider.

        The spider keeps the boundary's quantum domain, and a new quantum
        boundary is appended after it. Returns the spider.
        """
        return self._spider(qubit, vertex_type, phase)

    def _spider(
        self,
        qubit: int,
        vertex_type: VertexType,
        phase: Phase | Fraction | None = None,
        col: float | None = None,
    ) -> Vertex:
        """Add a spider like ``spider``, moved to column ``col`` if given."""
        self._ensure_lane(qubit)
        v1 = self.last_vertex[qubit]
        self.graph.set_vertex_type(v1, vertex_type)
        if col is not None:
            self.graph.set_col(v1, col)
        if phase is not None:
            self.graph.set_phase(v1, phase)
        v2 = self._add_quantum_boundary(qubit)
        self.graph.add_edge(v1, v2)
        return v1

    def hadamard(self, qubit: int) -> None:
        """Apply a Hadamard gate by toggling the wire's trailing edge."""
        self._ensure_lane(qubit)
        self.graph.toggle_edge(*self._last_edge(qubit))

    def advance(self, qubit: int) -> None:
        """Move the wire's end one column (the identity gate)."""
        self._ensure_lane(qubit)
        v = self.last_vertex[qubit]
        self.graph.set_col(v, self._last_col(qubit) + 1)

    def swap(self, qubit1: int, qubit2: int) -> None:
        """Swap two wires by exchanging their ends (and rows)."""
        self._ensure_lane(qubit1)
        self._ensure_lane(qubit2)

        v1 = self.last_vertex[qubit1]
        v2 = self.last_vertex[qubit2]
        self.last_vertex[qubit1] = v2
        self.last_vertex[qubit2] = v1

        # NOTE: both wire ends move to the later column first, so neither lands on
        # a position the other lane already uses.
        col = max(self.graph.col(v1), self.graph.col(v2))
        self.graph.set_col(v1, col)
        self.graph.set_col(v2, col)
        self.graph.set_row(v1, qubit2)
        self.graph.set_row(v2, qubit1)

    def tick(self) -> None:
        """Align all wire ends to the same column."""
        if len(self.last_vertex) == 0:
            return
        col = max(self._last_col(q) for q in self.last_vertex)
        for q in self.last_vertex:
            self.graph.set_col(self.last_vertex[q], col)

    def cx(
        self,
        control: int,
        target: int,
        classically_controlled: list[bool] | None = None,
    ) -> None:
        """Apply CNOT; ``classically_controlled`` marks record targets (``rec[-k]``)."""
        self._cx_cz(True, control, target, classically_controlled)

    def cz(
        self,
        control: int,
        target: int,
        classically_controlled: list[bool] | None = None,
    ) -> None:
        """Apply CZ; ``classically_controlled`` marks record targets (``rec[-k]``)."""
        self._cx_cz(False, control, target, classically_controlled)

    def _cx_cz(
        self,
        is_cx: bool,
        control: int,
        target: int,
        classically_controlled: list[bool] | None,
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
            m_vertex = self.rec[control]
            control = int(self.graph.row(m_vertex))
        self._ensure_lane(control)
        self._ensure_lane(target)

        col = max(self._last_col(control), self._last_col(target))
        v1 = self._spider(control, VertexType.Z, col=col)
        if control == target:
            col += 1
        v2 = self._spider(target, vertex_type, col=col)
        self.graph.add_edge(v1 if m_vertex is None else m_vertex, v2, edge_type)
        self.graph.mul_sqrt2(1, Domain.QUANTUM)

    # -------------------------------------------------------------------------
    # Measurement and reset
    # -------------------------------------------------------------------------

    def measure(self, qubit: int) -> Vertex:
        """Measure a qubit in the Z basis and record it as ``rec[i]``.

        Returns the measurement's classical spider.
        """
        return self._measure(qubit, silent=False)

    def _measure(self, qubit: int, silent: bool) -> Vertex:
        """Measure a qubit; a silent measurement is labelled ``m[i]``, not recorded."""
        self._ensure_lane(qubit)
        v1 = self.last_vertex[qubit]
        self.graph.set_vertex_type(v1, VertexType.Z)
        self.graph.merge_mirror_copies(v1)
        if not silent:
            self.graph.set_vertex_label(v1, f"rec[{len(self.rec)}]")
            self.rec.append(v1)
        else:
            self.graph.set_vertex_label(v1, f"m[{len(self.silent_rec)}]")
            self.silent_rec.append(v1)
        v2 = self._add_quantum_boundary(qubit)
        self.graph.add_edge(v1, v2)
        return v1

    def reset(self, qubit: int) -> None:
        """Reset a qubit to ``|0⟩``.

        If the qubit does not yet have a lane, create one in the reset state.
        Otherwise, silently measure the existing lane to trace out the previous
        state before reconnecting the qubit in the reset state.
        """
        if qubit not in self.last_vertex:
            self.graph.plug_basis(self._add_lane(qubit))
            return
        self._measure(qubit, silent=True)
        col = self._last_col(qubit)
        v1 = self.last_vertex[qubit]
        (v2,) = self.graph.neighbors(v1)
        self.graph.remove_edge(v1, v2)
        self.graph.plug_basis(v1)
        v3 = self._add_quantum_boundary(qubit, col + 1)
        self.graph.add_edge(v1, v3)

    # -------------------------------------------------------------------------
    # Noise
    # -------------------------------------------------------------------------

    def add_channel(self, probs: np.ndarray) -> int:
        """Record a noise channel and allocate its error bits.

        ``probs`` is the channel's distribution over its ``k`` error bits, so it
        has ``2^k`` entries. Returns the first bit; the channel owns bits
        ``first … first + k − 1``.
        """
        num_bits = len(probs).bit_length() - 1
        assert len(probs) == 1 << num_bits, "a channel has 2^k outcomes"
        first = self.num_error_bits
        self.channel_probs.append(probs)
        self.num_error_bits += num_bits
        return first

    def correlated_error(
        self,
        qubits: list[int],
        types: list[Literal["X", "Y", "Z"]],
        p: float,
    ) -> None:
        """Add a correlated error term affecting multiple qubits with given Pauli types.

        Its noise spiders get their error bit in ``finalize_correlated_error``.
        """
        for qubit, type_ in zip(qubits, types, strict=True):
            if type_ == "X" or type_ == "Y":
                self.pending_correlated.append(
                    (self.spider(qubit, VertexType.X), self.num_correlated_error_bits)
                )
            if type_ == "Z" or type_ == "Y":
                self.pending_correlated.append(
                    (self.spider(qubit, VertexType.Z), self.num_correlated_error_bits)
                )

        self.correlated_error_probs.append(p)
        self.num_correlated_error_bits += 1

    def finalize_correlated_error(self) -> None:
        """Finalize the current correlated error channel.

        Adds the channel (``2^k`` outcomes for its ``k`` terms), gives each
        pending noise spider its term's error bit, and clears the pending state.
        """
        if self.num_correlated_error_bits == 0:
            return
        first = self.add_channel(correlated_error_probs(self.correlated_error_probs))
        for v, bit in self.pending_correlated:
            self.graph.set_phase(v, _error_phase(first + bit))
        self.pending_correlated = []
        self.num_correlated_error_bits = 0
        self.correlated_error_probs = []

    # -------------------------------------------------------------------------
    # Detectors and observables
    # -------------------------------------------------------------------------

    def add_detector(self, rec: list[int]) -> None:
        """Add a detector that XORs the given measurement records."""
        v0 = self.graph.add_vertex(
            VertexType.X,
            col=self._annotation_col(rec),
            row=ANNOTATION_ROW,
            domain=Domain.CLASSICAL,
            label=f"det[{len(self.detectors)}]",
        )
        for rec_ in rec:
            self._xor_record(v0, rec_)
        self.detectors.append(v0)

    def include_observable(self, rec: list[int], index: int) -> None:
        """XOR the given measurement records into observable ``index``.

        The observable is created on first use, as in Stim's
        ``OBSERVABLE_INCLUDE``; with no records, that is all this does.
        """
        index = int(index)

        if index not in self.observables_dict:
            v0 = self.graph.add_vertex(
                VertexType.X,
                col=self._annotation_col(rec),
                row=ANNOTATION_ROW,
                domain=Domain.CLASSICAL,
                label=f"obs[{index}]",
            )
            self.observables_dict[index] = v0

        v0 = self.observables_dict[index]
        for rec_ in rec:
            self._xor_record(v0, rec_)

    def _annotation_col(self, rec: list[int]) -> float:
        """Pick a fresh column for a detector/observable vertex.

        Empty `rec` is valid Stim and represents a deterministic-zero annotation;
        fall back to a column after existing annotations rather than `min()` on an
        empty set.
        """
        d_cols = {self.graph.col(d) for d in self.detectors + self.observables}
        if rec:
            col: float = min(self.graph.col(self.rec[r]) for r in rec) - 0.5
        else:
            col = (max(d_cols) + 1) if d_cols else 0
        while col in d_cols:
            col += 1
        return col

    def _xor_record(self, annotation: Vertex, rec: int) -> None:
        """Add measurement record ``rec`` to the parity of ``annotation``.

        A record that is already in the parity cancels: its edge is removed rather
        than Hopf-cancelled by ``add_edge``, which would leave a scalar of 1/2 behind
        in graphs that drop the annotation.
        """
        record = self.rec[rec]
        if self.graph.has_edge(annotation, record):
            self.graph.remove_edge(annotation, record)
        else:
            self.graph.add_edge(annotation, record)


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
    builder.spider(qubit, VertexType.Z, phase)


def r_x(builder: GraphBuilder, qubit: int, phase: Fraction) -> None:
    """Apply R_X rotation gate with given phase (in units of π)."""
    builder.spider(qubit, VertexType.X, phase)


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
    builder.advance(qubit)


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
    builder.hadamard(qubit)


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


def cnot(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply CNOT (controlled-X) gate."""
    builder.cx(control, target, classically_controlled)


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
    builder.cz(control, target, classically_controlled)


def swap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply SWAP gate."""
    builder.swap(qubit1, qubit2)


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


def pauli_channel_1(
    builder: GraphBuilder, qubit: int, px: float, py: float, pz: float
) -> None:
    """Apply single-qubit Pauli channel with given X, Y, Z error probabilities."""
    bit = builder.add_channel(pauli_channel_1_probs(px, py, pz))
    builder.spider(qubit, VertexType.Z, _error_phase(bit))
    builder.spider(qubit, VertexType.X, _error_phase(bit + 1))


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
    bit = builder.add_channel(
        pauli_channel_2_probs(
            pix, piy, piz, pxi, pxx, pxy, pxz, pyi, pyx, pyy, pyz, pzi, pzx, pzy, pzz
        )
    )
    builder.spider(qubit_i, VertexType.Z, _error_phase(bit))
    builder.spider(qubit_i, VertexType.X, _error_phase(bit + 1))
    builder.spider(qubit_j, VertexType.Z, _error_phase(bit + 2))
    builder.spider(qubit_j, VertexType.X, _error_phase(bit + 3))


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
    bit = builder.add_channel(error_probs(p))
    builder.spider(qubit, VertexType.X, _error_phase(bit))


def y_error(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply Y error with probability p."""
    bit = builder.add_channel(error_probs(p))
    # Y = X·Z, so both vertices use the same error bit
    builder.spider(qubit, VertexType.Z, _error_phase(bit))
    builder.spider(qubit, VertexType.X, _error_phase(bit))


def z_error(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply Z error with probability p."""
    bit = builder.add_channel(error_probs(p))
    builder.spider(qubit, VertexType.Z, _error_phase(bit))


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
    bit = builder.add_channel(heralded_pauli_channel_1_probs(pi, px, py, pz))
    aux = HELPER_LANE
    r(builder, aux)
    builder.spider(aux, VertexType.X, _error_phase(bit))  # herald bit flip
    m(builder, aux)
    builder.spider(qubit, VertexType.Z, _error_phase(bit + 1))  # Z error bit
    builder.spider(qubit, VertexType.X, _error_phase(bit + 2))  # X error bit


def heralded_erase(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply heralded erasure channel.

    Special case of heralded_pauli_channel_1 with equal probabilities p/4
    for each of I, X, Y, Z when the channel fires.
    """
    heralded_pauli_channel_1(builder, qubit, p / 4, p / 4, p / 4, p / 4)


# =============================================================================
# Collapsing gates
# =============================================================================


def m(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in Z basis with optional bit-flip error probability p."""
    if invert:
        x(builder, qubit)
    if p > 0:
        # NOTE: the same X error before and after the measurement flips the
        # record but not the state.
        flip = _error_phase(builder.add_channel(error_probs(p)))
        builder.spider(qubit, VertexType.X, flip)
        builder.measure(qubit)
        builder.spider(qubit, VertexType.X, flip)
    else:
        builder.measure(qubit)
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
    r(builder, qubit)


def mrx(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """X-basis demolition measurement (optionally noisy).

    Projects each target qubit into |+> or |->, reports its value (false=|+>, true=|->),
    then resets to |+>.
    """
    h(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    r(builder, qubit)
    h(builder, qubit)


def mry(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Y-basis demolition measurement (optionally noisy).

    Projects each target qubit into |i> or |-i>, reports its value (false=|i>, true=|-i>),
    then resets to |i>.
    """
    h_yz(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    r(builder, qubit)
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
    builder.reset(qubit)


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
