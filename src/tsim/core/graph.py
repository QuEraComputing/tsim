"""ZX graph construction, manipulation, and preparation for sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zx_param
from zx_param import Domain, Graph, Phase, Scalar, Vertex, VertexType

from tsim.core.instructions import GraphRepresentation
from tsim.core.params import ParamLayout
from tsim.core.parse import parse_stim_circuit
from tsim.core.types import SamplingGraph
from tsim.utils.linalg import find_basis

if TYPE_CHECKING:
    from tsim.circuit import Circuit


def _discard(g: Graph, v: Vertex) -> None:
    """Turn the wire-end boundary ``v`` into a classical Z(0) spider (a discard).

    The wire edge is re-added one per copy, so this stays right when the
    neighbour is classical (e.g. a measurement record).
    """
    (n,) = g.neighbors(v)
    edge_type = g.edge_type(v, n)
    g.remove_edge(v, n)
    g.set_vertex_type(v, VertexType.Z)
    g.set_vertex_domain(v, Domain.CLASSICAL)
    g.add_edge_per_copy(v, n, edge_type)


def _align_wire_ends(g: Graph, built: GraphRepresentation) -> None:
    """Put every wire's last vertex in the same column (for drawing)."""
    if built.last_vertex:
        end = max(g.col(v) for v in built.last_vertex.values())
        for v in built.last_vertex.values():
            g.set_col(v, end)


def circuit_graph(built: GraphRepresentation, *, annotations: bool = True) -> Graph:
    """Return the circuit's graph with qubit ports.

    For qubit ``q``, a wire start that is still a boundary gets input port
    ``-(q + 1)`` and the wire end gets output port ``q``. Helper lanes (negative
    indices, used by MPP, MPAD and heralded channels) are discarded at the end.
    Detector and observable spiders are annotations; with ``annotations=False``
    they are left out, so the graph's tensor is the circuit's channel.
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
            _discard(g, v)
    if not annotations:
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
    """
    g = built.graph.copy()
    _align_wire_ends(g, built)
    for v in built.first_vertex.values():
        if g.vertex_type(v) == VertexType.BOUNDARY:
            g.set_vertex_type(v, VertexType.X)
            g.mul_scalar(Scalar.one().add_power_sqrt2(-1), Domain.QUANTUM)
    for v in built.last_vertex.values():
        _discard(g, v)
    if sample_detectors:
        outputs = [*built.detectors, *built.observables]
        for v in outputs:
            # NOTE: an X spider with n legs is 2^(1 - n/2) times a parity check;
            # with its output leg, an annotation has degree + 1 legs.
            g.mul_scalar(Scalar.one().add_power_sqrt2(g.degree(v) - 1))
    else:
        for v in [*built.detectors, *built.observables]:
            g.remove_vertex(v)
        outputs = list(built.rec)
    col = max((g.col(v) for v in g.vertices()), default=0.0) + 1
    for port, v in enumerate(outputs):
        b = g.add_vertex(
            VertexType.BOUNDARY, col=col, row=g.row(v), domain=Domain.CLASSICAL
        )
        g.set_port(b, port)
        g.add_edge(v, b)
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
    layout = ParamLayout(num_errors=num_errors)
    output_spiders: list[Vertex] = []
    for b in g.outputs():
        nbrs = list(g.neighbors(b))
        if len(nbrs) == 1 and g.phase(nbrs[0]).parameters:
            output_spiders.append(nbrs[0])
    output_spiders = list(dict.fromkeys(output_spiders))
    first = set(output_spiders)
    parametrized = output_spiders + [
        v for v in g.vertices() if v not in first and g.phase(v).parameters
    ]
    if not parametrized:
        return g, np.zeros((0, num_errors), dtype=np.uint8)

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
    3. Reduce it with zx_param's ``full_reduce``.
    4. Rewrite error bits as error-basis bits (Gaussian elimination).
    5. Reset the scalar (normalization is computed separately).
    """
    built = parse_stim_circuit(circuit._stim_circ)
    graph = build_sampling_graph(built, sample_detectors=sample_detectors)
    zx_param.full_reduce(graph)
    graph, error_transform = transform_error_basis(graph, built.num_error_bits)
    graph.scalar = Scalar.one()
    layout = ParamLayout(
        num_errors=built.num_error_bits,
        num_basis=error_transform.shape[0],
        num_outputs=len(graph.outputs()),
    )
    return SamplingGraph(
        graph=graph,
        error_transform=error_transform,
        channel_probs=built.channel_probs,
        num_outputs=layout.num_outputs,
        num_detectors=len(built.detectors),
        layout=layout,
    )
