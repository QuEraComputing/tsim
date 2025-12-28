from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import pyzx as zx
from pyzx.graph.base import BaseGraph

from tsim.compile import CompiledScalarGraphs, compile_scalar_graphs
from tsim.graph_util import ConnectedComponent, connected_components, get_params
from tsim.stabrank import find_stab
from tsim.types import CompiledComponent, CompiledProgram, SamplingGraph

DecompositionMode = Literal["sequential", "joint"]


def compile_program(
    prepared: SamplingGraph,
    *,
    mode: DecompositionMode,
) -> CompiledProgram:
    """Compile a prepared graph into an executable sampling program.

    This function performs the second phase of compilation:
    1. Split the graph into connected components
    2. For each component:
       - Plug outputs according to mode (sequential or joint)
       - Reduce each plugged graph
       - Perform stabilizer rank decomposition
       - Compile into CompiledScalarGraphs objects
    3. Assemble into CompiledProgram with output ordering

    Args:
        prepared: The prepared graph from prepare_graph().
        mode: Decomposition mode:
            - "sequential": For sampling - creates [0, 1, 2, ..., n] circuits
            - "joint": For probability estimation - creates [0, n] circuits

    Returns:
        A CompiledProgram ready for sampling.
    """
    components = connected_components(prepared.graph)

    # Determine global f-indices (numerically sorted) from the prepared graph
    f_indices_global = _get_f_indices(prepared.graph)
    num_outputs = prepared.num_outputs

    compiled_components: list[CompiledComponent] = []
    output_order: list[int] = []

    sorted_components = sorted(components, key=lambda c: len(c.output_indices))

    for component in sorted_components:
        compiled = _compile_component(
            component=component,
            f_indices_global=f_indices_global,
            mode=mode,
        )
        compiled_components.append(compiled)
        output_order.extend(component.output_indices)

    return CompiledProgram(
        components=tuple(compiled_components),
        output_order=jnp.array(output_order, dtype=jnp.int32),
        num_outputs=num_outputs,
        num_f_params=len(f_indices_global),
        num_detectors=prepared.num_detectors,
    )


def _get_f_indices(graph: BaseGraph) -> list[int]:
    """Extract numerically sorted list of f-parameter indices from the graph."""
    all_params = get_params(graph)
    f_indices = sorted([int(p[1:]) for p in all_params if p.startswith("f")])
    return f_indices


def _remove_phase_terms(graph: BaseGraph) -> None:
    """Remove phase terms from the graph."""
    graph.scalar.phasevars_halfpi = dict()
    graph.scalar.phasevars_pi_pair = []
    # TODO: clear additional phase terms


def _compile_component(
    component: ConnectedComponent,
    f_indices_global: list[int],
    mode: DecompositionMode,
) -> CompiledComponent:
    """Compile a single connected component.

    Args:
        component: The connected component to compile.
        f_indices_global: Global list of all f-parameter indices (numerically sorted).
        mode: Decomposition mode (sequential or joint).

    Returns:
        A CompiledComponent ready for sampling.
    """
    graph = component.graph
    output_indices = component.output_indices
    num_component_outputs = len(graph.outputs())

    # f_selection: subset of global f-indices used by this component
    component_f_set = set(_get_f_indices(graph))
    f_selection = [i for i in f_indices_global if i in component_f_set]

    if mode == "sequential":
        outputs_to_plug = list(range(num_component_outputs + 1))
    else:  # joint
        outputs_to_plug = [0, num_component_outputs]

    # Plug outputs and compile each graph
    compiled_graphs: list[CompiledScalarGraphs] = []

    component_m_chars = [f"m{i}" for i in output_indices]
    plugged_graphs = _plug_outputs(graph, component_m_chars, outputs_to_plug)

    # Track power2 for balancing scalar magnitudes
    power2_base: int | None = None

    for num_m_plugged, plugged_graph in zip(outputs_to_plug, plugged_graphs):
        g_copy = plugged_graph.copy()
        zx.full_reduce(g_copy, paramSafe=True)
        g_copy.normalize()

        # Balance power2 across graphs to avoid overflow/underflow
        if power2_base is None:
            power2_base = g_copy.scalar.power2
        g_copy.scalar.add_power(-power2_base)

        # Remove parametrized global phase terms. Global phases only matter once we
        # have started stabilizer rank decomposition.
        _remove_phase_terms(g_copy)

        # Parameter names: all f-params + m-params plugged so far
        param_names = [f"f{i}" for i in f_selection]
        param_names += [f"m{output_indices[j]}" for j in range(num_m_plugged)]

        # Perform stabilizer rank decomposition and compile
        g_list = find_stab(g_copy)

        if len(g_list) == 1:
            # This is a Clifford graph, we can clear the global phase terms
            _remove_phase_terms(g_list[0])

        compiled = compile_scalar_graphs(g_list, param_names)
        compiled_graphs.append(compiled)

    return CompiledComponent(
        output_indices=tuple(output_indices),
        f_selection=jnp.array(f_selection, dtype=jnp.int32),
        compiled_scalar_graphs=tuple(compiled_graphs),
    )


def _plug_outputs(
    graph: BaseGraph,
    m_chars: list[str],
    outputs_to_plug: list[int],
) -> list[BaseGraph]:
    """Create graphs with specified numbers of outputs plugged.

    Args:
        graph: The component graph.
        m_chars: The m-parameter names for this component's outputs.
        outputs_to_plug: List of integers specifying how many outputs to plug
            for each graph. E.g., [0, 1, 2, 3] creates 4 graphs.

    Returns:
        List of graphs with outputs plugged according to outputs_to_plug.
    """
    graphs: list[BaseGraph] = []
    num_outputs = len(graph.outputs())

    for num_plugged in outputs_to_plug:
        g = graph.copy()
        output_vertices = list(g.outputs())

        # Plug outputs either with an X vertex with phase m_char[i]
        # or with a Z vertex. Z vertices are equal to |0> + |1> and therefore
        # implement a trace.
        effect = "0" * num_plugged + "+" * (num_outputs - num_plugged)
        g.apply_effect(effect)
        for i, v in enumerate(output_vertices[:num_plugged]):
            g.set_phase(v, m_chars[i])  # type: ignore[arg-type]

        # Compensate power for trace of unplugged outputs
        g.scalar.add_power(num_outputs - num_plugged)

        graphs.append(g)

    return graphs
