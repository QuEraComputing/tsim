"""``GraphBuilder``: the ZX graph of a stim circuit and its bookkeeping.

``GraphBuilder`` owns the graph while a circuit is parsed: wire ends,
measurement records, error bits, pending correlated errors, and detectors and
observables. The gate library (``tsim.core.instructions``) builds on its
methods and never adds vertices itself; ``build`` returns the finished open or
closed graph.

Qubit wires are quantum. Every wire vertex starts as a quantum boundary, and
gates turn a wire's trailing boundary into a spider in place (see
``GraphBuilder.spider``), so the spider stays quantum. Only measurements
(``merge_mirror_copies``) and detector and observable annotations create
classical vertices. NOTE: zxparam's ``Graph.add_vertex`` defaults to the
classical domain, so wire vertices are only ever added here.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, get_args

import numpy as np
from zxparam import Domain, EdgeType, Graph, Phase, Vertex, VertexType

from tsim.noise.channels import correlated_error_probs

HELPER_LANE = -1
"""Qubit index, and row, of the helper wire used by MPP, MPAD and heralded channels."""

ANNOTATION_ROW = -2.0
"""Row of the detector and observable spiders, above the helper wire."""

Outputs = Literal["detectors", "records", "none"]
"""The classical outputs of a built graph (see ``GraphBuilder.build``)."""


def error_phase(error_bit: int) -> Phase:
    """Return the phase of a noise spider for error bit ``error_bit`` (ID = bit)."""
    return Phase(Fraction(0), frozenset({error_bit}))


def _move_label(g: Graph, source: Vertex, target: Vertex) -> None:
    """Move the display label of ``source`` to ``target``."""
    g.set_vertex_label(target, g.vertex_label(source))
    g.set_vertex_label(source, None)


class GraphBuilder:
    """Builds the ZX graph of a stim circuit, with quantum wires and classical records.

    Qubit wires are quantum; measurement records, detectors and observables are
    classical spiders, read through the properties below (their labels are only
    for display). ``build`` returns the finished open or closed graph.
    """

    def __init__(self) -> None:
        """Start an empty circuit."""
        self._graph = Graph()
        self._records: list[Vertex] = []
        self._silent_records: list[Vertex] = []
        self._detectors: list[Vertex] = []
        self._observables: dict[int, Vertex] = {}
        self._first_vertex: dict[int, Vertex] = {}
        self._last_vertex: dict[int, Vertex] = {}
        self._channel_probs: list[np.ndarray] = []
        self._num_error_bits = 0
        self._correlated_error_probs: list[float] = []
        self._pending_correlated: list[tuple[Vertex, int]] = []

    @property
    def graph(self) -> Graph:
        """The live graph built so far, for inspection; ``build`` returns copies.

        Changing it directly bypasses the builder's bookkeeping.
        """
        return self._graph

    @property
    def records(self) -> list[Vertex]:
        """The measurement record spiders, in record order (``rec[i]``)."""
        return list(self._records)

    @property
    def detectors(self) -> list[Vertex]:
        """The detector spiders, in order (``det[i]``)."""
        return list(self._detectors)

    @property
    def observables(self) -> list[Vertex]:
        """The observable spiders, sorted by index (``obs[k]``)."""
        return [self._observables[i] for i in sorted(self._observables)]

    @property
    def channel_probs(self) -> list[np.ndarray]:
        """Each noise channel's distribution over its error bits, in bit order."""
        self._check_finalized()
        return list(self._channel_probs)

    @property
    def num_error_bits(self) -> int:
        """The number of error bits, which are the error parameter IDs."""
        self._check_finalized()
        return self._num_error_bits

    def _check_finalized(self) -> None:
        """Raise if a correlated error still waits for its error bits."""
        if self._correlated_error_probs:
            raise RuntimeError(
                "a correlated error is pending; call finalize_correlated_error() first"
            )

    # -------------------------------------------------------------------------
    # Wires
    # -------------------------------------------------------------------------

    def _last_col(self, qubit: int) -> float:
        """Get the column of the last vertex for a qubit."""
        return self._graph.col(self._last_vertex[qubit])

    def _last_edge(self, qubit: int) -> tuple[Vertex, Vertex]:
        """Get the last edge for a qubit."""
        edges = self._graph.incident_edges(self._last_vertex[qubit])
        assert len(edges) == 1
        return edges[0]

    def _add_quantum_boundary(self, qubit: int, col: float | None = None) -> Vertex:
        """Add a dummy quantum boundary vertex at the end of a qubit's wire."""
        if col is None:
            col = self._last_col(qubit) + 1
        v = self._graph.add_vertex(
            VertexType.BOUNDARY, col=col, row=qubit, domain=Domain.QUANTUM
        )
        self._last_vertex[qubit] = v
        return v

    def _add_lane(self, qubit: int) -> Vertex:
        """Initialize a qubit lane: two quantum boundaries joined by a wire."""
        v1 = self._graph.add_vertex(
            VertexType.BOUNDARY, col=0, row=qubit, domain=Domain.QUANTUM
        )
        self._first_vertex[qubit] = v1
        self._graph.add_edge(v1, self._add_quantum_boundary(qubit, col=1))
        return v1

    def _ensure_lane(self, qubit: int) -> None:
        """Ensure qubit lane exists."""
        if qubit not in self._last_vertex:
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
        v1 = self._last_vertex[qubit]
        self._graph.set_vertex_type(v1, vertex_type)
        if col is not None:
            self._graph.set_col(v1, col)
        if phase is not None:
            self._graph.set_phase(v1, phase)
        v2 = self._add_quantum_boundary(qubit)
        self._graph.add_edge(v1, v2)
        return v1

    def hadamard(self, qubit: int) -> None:
        """Apply a Hadamard gate by toggling the wire's trailing edge."""
        self._ensure_lane(qubit)
        self._graph.toggle_edge(*self._last_edge(qubit))

    def advance(self, qubit: int) -> None:
        """Move the wire's end one column (the identity gate)."""
        self._ensure_lane(qubit)
        v = self._last_vertex[qubit]
        self._graph.set_col(v, self._last_col(qubit) + 1)

    def swap(self, qubit1: int, qubit2: int) -> None:
        """Swap two wires by exchanging their ends (and rows)."""
        self._ensure_lane(qubit1)
        self._ensure_lane(qubit2)

        v1 = self._last_vertex[qubit1]
        v2 = self._last_vertex[qubit2]
        self._last_vertex[qubit1] = v2
        self._last_vertex[qubit2] = v1

        # NOTE: both wire ends move to the later column first, so neither lands on
        # a position the other lane already uses.
        col = max(self._graph.col(v1), self._graph.col(v2))
        self._graph.set_col(v1, col)
        self._graph.set_col(v2, col)
        self._graph.set_row(v1, qubit2)
        self._graph.set_row(v2, qubit1)

    def tick(self) -> None:
        """Align all wire ends to the same column."""
        if len(self._last_vertex) == 0:
            return
        col = max(self._last_col(q) for q in self._last_vertex)
        for q in self._last_vertex:
            self._graph.set_col(self._last_vertex[q], col)

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
            m_vertex = self._records[control]
            control = int(self._graph.row(m_vertex))
        self._ensure_lane(control)
        self._ensure_lane(target)

        col = max(self._last_col(control), self._last_col(target))
        v1 = self._spider(control, VertexType.Z, col=col)
        if control == target:
            col += 1
        v2 = self._spider(target, vertex_type, col=col)
        self._graph.add_edge(v1 if m_vertex is None else m_vertex, v2, edge_type)
        self._graph.mul_sqrt2(1, Domain.QUANTUM)

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
        v1 = self._last_vertex[qubit]
        self._graph.set_vertex_type(v1, VertexType.Z)
        self._graph.merge_mirror_copies(v1)
        if not silent:
            self._graph.set_vertex_label(v1, f"rec[{len(self._records)}]")
            self._records.append(v1)
        else:
            self._graph.set_vertex_label(v1, f"m[{len(self._silent_records)}]")
            self._silent_records.append(v1)
        v2 = self._add_quantum_boundary(qubit)
        self._graph.add_edge(v1, v2)
        return v1

    def reset(self, qubit: int) -> None:
        """Reset a qubit to ``|0⟩``.

        If the qubit does not yet have a lane, create one in the reset state.
        Otherwise, silently measure the existing lane to trace out the previous
        state before reconnecting the qubit in the reset state.
        """
        if qubit not in self._last_vertex:
            self._graph.plug_basis(self._add_lane(qubit))
            return
        self._measure(qubit, silent=True)
        col = self._last_col(qubit)
        v1 = self._last_vertex[qubit]
        (v2,) = self._graph.neighbors(v1)
        self._graph.remove_edge(v1, v2)
        self._graph.plug_basis(v1)
        v3 = self._add_quantum_boundary(qubit, col + 1)
        self._graph.add_edge(v1, v3)

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
        first = self._num_error_bits
        self._channel_probs.append(probs)
        self._num_error_bits += num_bits
        return first

    def correlated_error(
        self,
        qubits: list[int],
        types: list[Literal["X", "Y", "Z"]],
        p: float,
    ) -> None:
        """Add a correlated error term affecting multiple qubits with given Pauli types.

        Its noise spiders get their error bit in ``finalize_correlated_error``,
        which must run before the graph is built.
        """
        term = len(self._correlated_error_probs)
        for qubit, type_ in zip(qubits, types, strict=True):
            if type_ == "X" or type_ == "Y":
                self._pending_correlated.append(
                    (self.spider(qubit, VertexType.X), term)
                )
            if type_ == "Z" or type_ == "Y":
                self._pending_correlated.append(
                    (self.spider(qubit, VertexType.Z), term)
                )
        self._correlated_error_probs.append(p)

    def finalize_correlated_error(self) -> None:
        """Finalize the current correlated error channel.

        Adds the channel (``2^k`` outcomes for its ``k`` terms), gives each
        pending noise spider its term's error bit, and clears the pending state.
        """
        if not self._correlated_error_probs:
            return
        first = self.add_channel(correlated_error_probs(self._correlated_error_probs))
        for v, term in self._pending_correlated:
            self._graph.set_phase(v, error_phase(first + term))
        self._pending_correlated = []
        self._correlated_error_probs = []

    # -------------------------------------------------------------------------
    # Detectors and observables
    # -------------------------------------------------------------------------

    def add_detector(self, rec: list[int]) -> None:
        """Add a detector that XORs the given measurement records."""
        v0 = self._graph.add_vertex(
            VertexType.X,
            col=self._annotation_col(rec),
            row=ANNOTATION_ROW,
            domain=Domain.CLASSICAL,
            label=f"det[{len(self._detectors)}]",
        )
        for rec_ in rec:
            self._xor_record(v0, rec_)
        self._detectors.append(v0)

    def include_observable(self, rec: list[int], index: int) -> None:
        """XOR the given measurement records into observable ``index``.

        The observable is created on first use, as in Stim's
        ``OBSERVABLE_INCLUDE``; with no records, that is all this does.
        """
        index = int(index)

        if index not in self._observables:
            v0 = self._graph.add_vertex(
                VertexType.X,
                col=self._annotation_col(rec),
                row=ANNOTATION_ROW,
                domain=Domain.CLASSICAL,
                label=f"obs[{index}]",
            )
            self._observables[index] = v0

        v0 = self._observables[index]
        for rec_ in rec:
            self._xor_record(v0, rec_)

    def _annotation_col(self, rec: list[int]) -> float:
        """Pick a fresh column for a detector/observable vertex.

        Empty `rec` is valid Stim and represents a deterministic-zero annotation;
        fall back to a column after existing annotations rather than `min()` on an
        empty set.
        """
        d_cols = {self._graph.col(d) for d in self._detectors + self.observables}
        if rec:
            col: float = min(self._graph.col(self._records[r]) for r in rec) - 0.5
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
        record = self._records[rec]
        if self._graph.has_edge(annotation, record):
            self._graph.remove_edge(annotation, record)
        else:
            self._graph.add_edge(annotation, record)

    # -------------------------------------------------------------------------
    # Building the graph
    # -------------------------------------------------------------------------

    def build(self, *, closed: bool, outputs: Outputs) -> Graph:
        """Return the circuit's graph with its wires closed or given ports.

        With ``closed=False``, qubit ``q``'s wire start, if it is still a
        boundary, gets input port ``-(q + 1)`` and its end gets output port
        ``q``. With ``closed=True``, wire starts that are still boundaries become
        ``|0⟩`` (a quantum X(0) spider) and wire ends are discarded. Helper lanes
        (negative indices, used by MPP, MPAD and heralded channels) are always
        discarded at the end.

        ``outputs`` picks the classical output boundaries, whose ports follow the
        qubit outputs (from 0 when closed):

        - ``"detectors"``: one per detector, then one per observable; the
          ``det[i]`` / ``obs[k]`` labels move to them.
        - ``"records"``: one per measurement record; the ``rec[i]`` labels move
          to them.
        - ``"none"``: no classical outputs.

        Detector and observable spiders are removed unless they are the outputs.
        In open graphs, measurement spiders that are not outputs keep their
        ``rec[i]`` / ``m[i]`` labels; closed graphs are meant to be reduced, so
        only their output boundaries carry labels, which survive reduction.

        An open graph's tensor is the circuit's channel together with the output
        bits; for bound error parameters, a closed graph's tensor is the joint
        distribution of its outputs. The builder is unchanged, so it can build
        again.
        """
        if outputs not in get_args(Outputs):
            raise ValueError(
                f"outputs must be one of {get_args(Outputs)}, got {outputs!r}"
            )
        self._check_finalized()
        g = self._graph.copy()
        self._align_wire_ends(g)
        if closed:
            for v in self._first_vertex.values():
                if g.vertex_type(v) == VertexType.BOUNDARY:
                    g.plug_basis(v)
            for v in self._last_vertex.values():
                g.discard(v)
            first_port = 0
        else:
            for q, v in self._first_vertex.items():
                if q >= 0 and g.vertex_type(v) == VertexType.BOUNDARY:
                    g.set_port(v, -(q + 1))
            for q, v in self._last_vertex.items():
                if q >= 0:
                    g.set_port(v, q)
                else:
                    g.discard(v)
            first_port = max((q for q in self._last_vertex if q >= 0), default=-1) + 1
        if outputs == "detectors":
            self._attach_detector_outputs(g, first_port)
        else:
            for v in [*self._detectors, *self.observables]:
                g.remove_vertex(v)
            if outputs == "records":
                self._attach_record_outputs(g, first_port)
        if closed:
            for v in [*self._records, *self._silent_records]:
                g.set_vertex_label(v, None)
        return g

    def _align_wire_ends(self, g: Graph) -> None:
        """Put every wire's last vertex in the same column (for drawing)."""
        if self._last_vertex:
            end = max(g.col(v) for v in self._last_vertex.values())
            for v in self._last_vertex.values():
                g.set_col(v, end)

    def _attach_detector_outputs(self, g: Graph, first_port: int) -> None:
        """Give every detector and observable spider a classical output boundary.

        Ports are ``first_port, first_port + 1, …``: detectors first, then
        observables in index order. Each spider is scaled so that, with its output
        leg, its tensor is exactly the parity of its measurement records. The
        spider's ``det[i]`` / ``obs[k]`` label moves to its boundary.
        """
        annotations = [(v, ANNOTATION_ROW - 1) for v in self._detectors]
        annotations += [(v, ANNOTATION_ROW - 1.5) for v in self.observables]
        for port, (v, row) in enumerate(annotations, start=first_port):
            # NOTE: an X spider with n legs is 2^(1 - n/2) times a parity check;
            # with its output leg, an annotation has degree + 1 legs.
            g.mul_sqrt2(g.degree(v) - 1)
            b = g.add_vertex(
                VertexType.BOUNDARY, col=g.col(v), row=row, domain=Domain.CLASSICAL
            )
            g.set_port(b, port)
            g.add_edge(v, b)
            _move_label(g, v, b)

    def _attach_record_outputs(self, g: Graph, first_port: int) -> None:
        """Give every measurement record a classical output boundary.

        Ports are ``first_port, first_port + 1, …`` in record order. The
        record's ``rec[i]`` label moves to its boundary.
        """
        for port, v in enumerate(self._records, start=first_port):
            # NOTE: half a row below its record, where no wire vertex sits.
            b = g.add_vertex(
                VertexType.BOUNDARY,
                col=g.col(v),
                row=g.row(v) + 0.5,
                domain=Domain.CLASSICAL,
            )
            g.set_port(b, port)
            g.add_edge(v, b)
            _move_label(g, v, b)
