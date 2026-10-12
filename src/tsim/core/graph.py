"""ZX graph construction, manipulation, and preparation for sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zxparam
from zxparam import Domain, Graph, Phase, Scalar, Vertex, VertexType

from tsim.core.instructions import ANNOTATION_ROW, GraphRepresentation
from tsim.core.params import ParamLayout
from tsim.core.parse import parse_stim_circuit
from tsim.core.types import SamplingGraph
from tsim.utils.linalg import find_basis

if TYPE_CHECKING:
    from tsim.circuit import Circuit


def _align_wire_ends(g: Graph, built: GraphRepresentation) -> None:
    """Put every wire's last vertex in the same column (for drawing)."""
    if built.last_vertex:
        end = max(g.col(v) for v in built.last_vertex.values())
        for v in built.last_vertex.values():
            g.set_col(v, end)


def _move_label(g: Graph, source: Vertex, target: Vertex) -> None:
    """Move the display label of ``source`` to ``target``."""
    g.set_vertex_label(target, g.vertex_label(source))
    g.set_vertex_label(source, None)


def _clear_measurement_labels(g: Graph, built: GraphRepresentation) -> None:
    """Remove the ``rec[i]`` and ``m[i]`` labels from the measurement spiders."""
    for v in [*built.rec, *built.silent_rec]:
        g.set_vertex_label(v, None)


def _attach_annotation_outputs(
    g: Graph, built: GraphRepresentation, first_port: int
) -> None:
    """Give every detector and observable spider a classical output boundary.

    Ports are ``first_port, first_port + 1, …``: detectors first, then
    observables in index order. Each spider is scaled so that, with its output
    leg, its tensor is exactly the parity of its measurement records. The
    spider's ``det[i]`` / ``obs[k]`` label moves to its boundary.
    """
    annotations = [(v, ANNOTATION_ROW - 1) for v in built.detectors]
    annotations += [(v, ANNOTATION_ROW - 1.5) for v in built.observables]
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


def circuit_graph(built: GraphRepresentation, *, annotations: bool = True) -> Graph:
    """Return the circuit's graph with qubit ports.

    For qubit ``q``, a wire start that is still a boundary gets input port
    ``-(q + 1)`` and the wire end gets output port ``q``. Helper lanes (negative
    indices, used by MPP, MPAD and heralded channels) are discarded at the end.

    With ``annotations=True``, every detector and observable spider gets a
    classical output boundary, with ports after the qubit outputs (detectors,
    then observables), so the graph's tensor is the circuit's channel together
    with the detector and observable bits; the ``det[i]`` / ``obs[k]`` labels
    sit on those boundaries, and the ``rec[i]`` / ``m[i]`` labels on the
    measurement spiders. With ``annotations=False``, the detector and
    observable spiders are left out, and the graph's tensor is the circuit's
    channel.
    """
    g = built.graph.copy()
    _align_wire_ends(g, built)
    for q, v in built.first_vertex.items():
        if q >= 0 and g.vertex_type(v) == VertexType.BOUNDARY:
            g.set_port(v, -(q + 1))
    for q, v in built.last_vertex.items():
        if q >= 0:
            g.set_port(v, q)
        else:
            g.discard(v)
    if annotations:
        first_port = max((q for q in built.last_vertex if q >= 0), default=-1) + 1
        _attach_annotation_outputs(g, built, first_port)
    else:
        for v in [*built.detectors, *built.observables]:
            g.remove_vertex(v)
    return g


def build_sampling_graph(
    built: GraphRepresentation, sample_detectors: bool = False
) -> Graph:
    """Close the circuit graph into a sampling graph with classical outputs.

    Wire starts that are still boundaries become ``|0⟩`` (a quantum X(0) spider),
    wire ends are discarded, and classical output boundaries are attached: one
    per measurement record (port ``i``), or, with ``sample_detectors``, one per
    detector (ports ``0 … D−1``) followed by one per observable. In measurement
    mode the detector and observable spiders are removed. For bound error
    parameters, the graph's tensor is the joint distribution of the outputs.
    Only the output boundaries carry labels (``rec[i]``, or ``det[i]`` and
    ``obs[k]``), so the labels survive reduction.
    """
    g = built.graph.copy()
    _align_wire_ends(g, built)
    for v in built.first_vertex.values():
        if g.vertex_type(v) == VertexType.BOUNDARY:
            g.plug_basis(v)
    for v in built.last_vertex.values():
        g.discard(v)
    if sample_detectors:
        _attach_annotation_outputs(g, built, first_port=0)
        _clear_measurement_labels(g, built)
        return g
    for v in [*built.detectors, *built.observables]:
        g.remove_vertex(v)
    for port, v in enumerate(built.rec):
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
    _clear_measurement_labels(g, built)
    return g


def transform_error_basis(g: Graph, num_errors: int) -> tuple[Graph, np.ndarray]:
    """Rewrite error parameters in a smaller basis of error-basis bits.

    Finds a linearly independent basis over GF(2) for the parity sets on the
    graph's vertices and rewrites each vertex's error IDs as basis IDs
    (``ParamLayout(num_errors).basis(j)``). Output-connected spiders come first,
    so basis bits are numbered in output order where possible, which helps the
    direct path avoid a reindex.

    Args:
        g: A graph whose vertex parameters are error IDs ``0 … num_errors − 1``.
        num_errors: The number of error bits.

    Returns:
        The graph (mutated in place) and a binary matrix of shape
        ``(num_basis, num_errors)`` whose row ``j`` lists the errors whose XOR is
        basis bit ``j``.

    """
    layout = ParamLayout(num_errors)
    near_outputs = [n for b in g.outputs() for n in g.neighbors(b)]
    parametrized = list(
        dict.fromkeys(
            v for v in [*near_outputs, *g.vertices()] if g.phase(v).parameters
        )
    )
    error_matrix = np.zeros((len(parametrized), num_errors), dtype=np.uint8)
    for row, v in enumerate(parametrized):
        params = sorted(g.phase(v).parameters)
        if params[-1] >= num_errors:
            raise ValueError(f"vertex {v} carries a parameter that is not an error bit")
        error_matrix[row, params] = 1

    basis, transform = find_basis(error_matrix)
    for v, transform_row in zip(parametrized, transform, strict=True):
        ids = frozenset(layout.basis(int(j)) for j in np.nonzero(transform_row)[0])
        g.set_phase(v, Phase(g.phase(v).value, ids))
    return g, basis


def prepare_graph(circuit: Circuit, *, sample_detectors: bool) -> SamplingGraph:
    """Prepare a circuit for compilation.

    1. Parse the stim circuit into a quantum/classical ZX graph.
    2. Close it into a sampling graph with classical outputs.
    3. Reduce it with zxparam's ``full_reduce``.
    4. Rewrite error bits as error-basis bits (Gaussian elimination).
    5. Reset the scalar (normalization is computed separately).
    """
    built = parse_stim_circuit(circuit._stim_circ)
    graph = build_sampling_graph(built, sample_detectors=sample_detectors)
    zxparam.full_reduce(graph)
    graph, error_transform = transform_error_basis(graph, built.num_error_bits)
    graph.scalar = Scalar.one()
    return SamplingGraph(
        graph=graph,
        error_transform=error_transform,
        channel_probs=built.channel_probs,
        num_outputs=len(graph.outputs()),
        num_detectors=len(built.detectors),
        layout=ParamLayout(built.num_error_bits),
    )
