from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import tsim.external.pyzx as zx
from tsim.external.pyzx.graph.base import BaseGraph


@dataclass
class ConnectedComponent:
    graph: BaseGraph
    output_indices: list[int]


def connected_components(g: BaseGraph) -> list[ConnectedComponent]:
    """Return each connected component of ``g`` as its own ZX subgraph.

    The function mirrors the way ``Circuit`` builds and manipulates ``zx.Graph``
    instances. Each component is packaged inside a :class:`ConnectedComponent`
    so downstream consumers know both the subgraph and which original output
    indices still live inside it.
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

    subgraph = zx.Graph(backend=type(g).backend)
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


if __name__ == "__main__":
    from tsim.circuit import Circuit

    circuit = Circuit.random(3, 3)
    circuit.x_error(0, 0.1)
    circuit.g.normalize()
    zx.draw(circuit.g, labels=True)
    for component in connected_components(circuit.g):
        zx.draw(component.graph, labels=True)
        print(component.output_indices)
