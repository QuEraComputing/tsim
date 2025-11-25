from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from tsim.external.pyzx.graph.base import BaseGraph
from tsim.external.pyzx.graph.graph import Graph
from tsim.external.pyzx.graph.scalar import Scalar
from tsim.util.linalg import find_basis


@dataclass
class ConnectedComponent:
    graph: BaseGraph
    output_indices: list[int]


def connected_components(g: BaseGraph) -> list[ConnectedComponent]:
    """Return each connected component of ``g`` as its own ZX subgraph.

    Each component is packaged inside a :class:`ConnectedComponent` that contains
    the subgraph and a list of output indices matching the original output indices.
    """

    components: list[ConnectedComponent] = []
    visited: set[Any] = set()
    outputs = tuple(g.outputs())
    output_indices = {vertex: idx for idx, vertex in enumerate(outputs)}

    for vertex in list(g.vertices()):
        if vertex in visited:
            continue

        component_vertices = _collect_vertices(g, vertex, visited)
        subgraph, vertex_map = _induced_subgraph(g, component_vertices)

        component_output_indices = [
            output_indices[v] for v in component_vertices if v in output_indices
        ]
        component_output_indices.sort()

        components.append(
            ConnectedComponent(
                graph=subgraph,
                output_indices=component_output_indices,
            )
        )

    return components


def _collect_vertices(
    g: BaseGraph,
    start: Any,
    visited: set[Any],
) -> list[Any]:
    """Breadth-first search to collect the connected component of ``start``."""

    queue: deque[Any] = deque([start])
    component: list[Any] = []

    while queue:
        vertex = queue.pop()
        if vertex in visited:
            continue

        visited.add(vertex)
        component.append(vertex)

        for neighbor in g.neighbors(vertex):
            if neighbor not in visited:
                queue.appendleft(neighbor)

    return component


def _induced_subgraph(
    g: BaseGraph,
    vertices: Sequence[Any],
) -> tuple[BaseGraph, Dict[Any, Any]]:
    """Build the subgraph that is induced by ``vertices``."""

    subgraph = Graph()
    subgraph.track_phases = g.track_phases
    subgraph.merge_vdata = g.merge_vdata

    vert_map: Dict[Any, Any] = {}
    phases = g.phases()
    qubits = g.qubits()
    rows = g.rows()
    types = g.types()
    get_params = getattr(g, "get_params", None)

    for vertex in vertices:
        params = None
        if get_params is not None:
            params = set(get_params(vertex))

        new_vertex = subgraph.add_vertex(
            types[vertex],
            qubit=qubits.get(vertex, -1),
            row=rows.get(vertex, -1),
            phase=phases.get(vertex, 0),
            phaseVars=params,
        )

        for key in g.vdata_keys(vertex):
            subgraph.set_vdata(new_vertex, key, g.vdata(vertex, key))

        vert_map[vertex] = new_vertex

    added_edges: set[Tuple[Any, Any]] = set()
    for vertex in vertices:
        for neighbor in g.neighbors(vertex):
            if neighbor not in vert_map:
                continue
            edge = g.edge(vertex, neighbor)
            if edge in added_edges:
                continue
            added_edges.add(edge)
            new_edge = subgraph.edge(vert_map[vertex], vert_map[neighbor])
            subgraph.add_edge(new_edge, g.edge_type(edge))

    component_inputs = tuple(vert_map[v] for v in g.inputs() if v in vert_map)
    component_outputs = tuple(vert_map[v] for v in g.outputs() if v in vert_map)
    subgraph.set_inputs(component_inputs)
    subgraph.set_outputs(component_outputs)

    return subgraph, vert_map


def transform_error_basis(g: BaseGraph) -> tuple[BaseGraph, dict[str, set[str]]]:
    """Transform phase variables from the original 'e' basis to a reduced 'f' basis.

    This function finds a linearly independent basis for the phase variables
    across all vertices and transforms them accordingly. The original variables
    (e0, e1, ...) are mapped to a smaller set (f0, f1, ...) where each f_i
    corresponds to a linear combination of original e variables.

    Args:
        g: A ZX graph with phase variables attached to vertices.

    Returns:
        A tuple containing:
            - The modified graph (same object, mutated in place)
            - A mapping from new basis variables to original variables,
              e.g. {"f0": {"e1", "e3"}, "f1": {"e2"}}
    """
    parametrized_vertices = [
        v for v in g.vertices() if v in g._phaseVars and g._phaseVars[v]
    ]

    if not parametrized_vertices:
        g.scalar = Scalar()
        return g, {}

    # Parse variable indices and find the dimension
    error_indices = [
        [int(var[1:]) for var in g._phaseVars[v]] for v in parametrized_vertices
    ]
    num_errors = max(max(indices) for indices in error_indices) + 1

    # Build binary matrix representation
    error_matrix = np.zeros((len(error_indices), num_errors), dtype=np.uint8)
    for row_idx, indices in enumerate(error_indices):
        error_matrix[row_idx, indices] = 1

    basis, transform = find_basis(error_matrix)
    # Now: error_matrix = transform @ basis

    for v, transform_row in zip(parametrized_vertices, transform):
        new_vars = {f"f{j}" for j in np.nonzero(transform_row)[0]}
        g._phaseVars[v] = new_vars

    error_transform = {
        f"f{i}": {f"e{j}" for j in np.nonzero(basis_vec)[0]}
        for i, basis_vec in enumerate(basis)
    }

    # Remove the scalar. Since we have not started the stabilizer rank decomposition,
    # it is safe to remove the overall scalar.
    # TODO: we can only remove the scalar if it is not 0
    g.scalar = Scalar()

    # clean the diagram up a bit
    for v in g.outputs():
        n = list(g.neighbors(v))[0]
        if len(list(g.neighbors(n))) == 1:
            g.set_qubit(n, g.qubit(v) - 1)
            g.set_row(n, g.row(v))

    return g, error_transform
