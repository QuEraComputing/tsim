from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from tsim._instructions import GraphRepresentation
from tsim.external.pyzx import VertexType
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


def build_sampling_graph(
    built: GraphRepresentation, sample_detectors: bool = False
) -> BaseGraph:
    """Build a ZX graph for sampling from a BuiltGraph.

    This is the internal implementation of get_sampling_graph.
    """
    g = built.graph.copy()

    # Initialize un-initialized first vertices to the 0 state
    for v in built.first_vertex.values():
        if g.type(v) == VertexType.BOUNDARY:
            g.set_type(v, VertexType.X)

    # Clean up last row
    if built.last_vertex:
        max_row = max(g.row(v) for v in built.last_vertex.values())
        for q in built.last_vertex:
            g.set_row(built.last_vertex[q], max_row)

    num_measurements = len(built.rec)
    outputs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    g.set_outputs(tuple(outputs))

    g_adj = g.adjoint()
    g.compose(g_adj)

    g = g.copy()

    label_to_vertex: dict[str, list[int]] = defaultdict(list)
    annotation_to_vertex: dict[str, list[int]] = defaultdict(list)
    for v in g.vertices():
        phase_vars = g._phaseVars[v]
        if not len(phase_vars) == 1:
            continue
        phase = list(phase_vars)[0]
        if "det" in phase or "obs" in phase or "rec" in phase or "m" in phase:
            label_to_vertex[phase].append(v)
        if "det" in phase or "obs" in phase:
            annotation_to_vertex[phase].append(v)

    outputs = [0] * num_measurements if not sample_detectors else []

    # Connect all rec[i] vertices to each other and add red vertex with rec[i] label
    for i in range(num_measurements):
        label = f"rec[{i}]"
        vertices = label_to_vertex[label]

        assert len(vertices) == 2
        v0, v1 = vertices
        if not g.connected(v0, v1):
            g.add_edge((v0, v1))
        g.set_phase(v0, 0)
        g.set_phase(v1, 0)

        # Add outputs
        if not sample_detectors:
            v3 = g.add_vertex(VertexType.BOUNDARY, qubit=-1, row=i + 1, phase=0)
            outputs[i] = v3
            g.add_edge((v0, v3))

    # Connect all m[i] vertices to each other
    for i in range(len(built.silent_rec)):
        label = f"m[{i}]"
        vertices = label_to_vertex[label]

        assert len(vertices) == 2
        v0, v1 = vertices
        if not g.connected(v0, v1):
            g.add_edge((v0, v1))
        g.set_phase(v0, 0)
        g.set_phase(v1, 0)

    if not sample_detectors:
        # Sample measurements: remove detectors and observables
        for vertices in annotation_to_vertex.values():
            assert len(vertices) == 2
            for v in vertices:
                g.remove_vertex(v)
    else:
        # Sample detectors and observables:
        # Keep detector and observables but remove the adjoint (duplicated)
        # annotation nodes
        for vertices in annotation_to_vertex.values():
            assert len(vertices) == 2
            g.remove_vertex(vertices.pop())

        labels = [f"det[{i}]" for i in range(len(built.detectors))] + [
            f"obs[{i}]" for i in built.observables_dict.keys()
        ]
        for label in labels:
            vs = annotation_to_vertex[label]
            assert len(vs) == 1
            v = vs[0]
            row = g.row(v)
            vb = g.add_vertex(VertexType.BOUNDARY, qubit=-2, row=row)
            g.add_edge((v, vb))
            g.set_phase(v, 0)
            outputs.append(vb)

    g.set_outputs(tuple(outputs))

    return g


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

    return g, error_transform


def squash_graph(g: BaseGraph) -> None:
    """Compact the graph by placing vertices underneath their output connections.

    Starting from output vertices, each vertex is placed directly underneath
    (same row, qubit - 1) its already-placed neighbor. Positions are assigned
    via BFS traversal from outputs, ensuring no (qubit, row) collisions.
    """
    outputs = list(g.outputs())
    if not outputs:
        return

    # Normalize output positions: consecutive rows at qubit = num_outputs
    num_outputs = len(outputs)
    for row, v in enumerate(outputs):
        g.set_qubit(v, num_outputs)
        g.set_row(v, row)

    # Track occupied positions and placed vertices
    occupied: set[tuple[int, int]] = {(num_outputs, row) for row in range(num_outputs)}
    placed: set[Any] = set(outputs)

    # BFS from outputs
    queue: deque[Any] = deque(outputs)

    while queue:
        current = queue.popleft()
        current_qubit = int(g.qubit(current))
        current_row = int(g.row(current))

        for neighbor in g.neighbors(current):
            if neighbor in placed:
                continue

            # Try to place directly underneath: same row, qubit - 1
            target_qubit = current_qubit - 1
            target_row = current_row

            # If spot is taken, search for nearest free spot at same qubit level
            if (target_qubit, target_row) in occupied:
                # Search outward from target_row
                for offset in range(1, 1000):
                    if (target_qubit, target_row + offset) not in occupied:
                        target_row = target_row + offset
                        break
                    if (
                        target_qubit,
                        target_row - offset,
                    ) not in occupied and target_row - offset >= 0:
                        target_row = target_row - offset
                        break

            g.set_qubit(neighbor, target_qubit)
            g.set_row(neighbor, target_row)
            occupied.add((target_qubit, target_row))
            placed.add(neighbor)
            queue.append(neighbor)

    for v in g.outputs():
        neighbors = list(g.neighbors(v))
        if neighbors and len(list(g.neighbors(neighbors[0]))) == 1:
            g.set_qubit(neighbors[0], g.qubit(v) + 1)
            g.set_row(neighbors[0], g.row(v))
