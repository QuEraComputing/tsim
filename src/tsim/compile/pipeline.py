"""Compilation pipeline from prepared graphs to executable programs."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

import jax.numpy as jnp
import numpy as np
from zxparam import (
    DecompositionStrategy,
    Domain,
    Dyadic,
    EdgeType,
    Graph,
    Phase,
    Scalar,
    ScalarSum,
    Vertex,
    VertexType,
    decompose,
)

from tsim.compile.compile import CompiledScalarGraphs, compile_scalars
from tsim.core.params import ParamLayout
from tsim.core.types import CompiledComponent, CompiledProgram, SamplingGraph

DecompositionMode = Literal["sequential", "joint"]


def compile_program(
    prepared: SamplingGraph,
    *,
    mode: DecompositionMode,
    strategy: DecompositionStrategy = "cat5",
) -> CompiledProgram:
    """Compile a prepared graph into an executable sampling program.

    The graph is split into connected components. A component whose single
    output equals one error-basis bit takes the direct path; every other
    component has its outputs plugged (all prefixes in ``sequential`` mode, none
    and all in ``joint`` mode), each plugged graph is decomposed into stabilizer
    terms, and the terms are compiled into JAX arrays.

    Args:
        prepared: The prepared graph from ``prepare_graph``.
        mode: ``"sequential"`` for sampling, ``"joint"`` for probabilities.
        strategy: Stabilizer-rank decomposition strategy.

    Returns:
        A CompiledProgram ready for sampling.

    """
    graph = prepared.graph
    layout = prepared.layout
    basis_global = _basis_indices(graph, layout)

    direct_entries: list[tuple[int, int, bool]] = []
    compiled_components: list[CompiledComponent] = []
    compiled_output_order: list[int] = []

    components = sorted(graph.connected_components(), key=lambda c: len(c.outputs()))
    for component in components:
        output_indices = [_port(component, v) for v in component.outputs()]
        direct = classify_direct(component, layout)
        if direct is not None:
            basis_index, flip = direct
            direct_entries.append((output_indices[0], basis_index, flip))
            continue
        compiled_components.append(
            _compile_component(component, basis_global, layout, mode, strategy)
        )
        compiled_output_order.extend(output_indices)

    # Sort direct entries by output index so that — together with the output
    # prioritisation in transform_error_basis — the concatenated layout often
    # matches the original output order, sparing a reindex at sample time.
    direct_entries.sort()
    output_order = np.array(
        [e[0] for e in direct_entries] + compiled_output_order, dtype=np.int32
    )
    reindex = np.argsort(output_order)
    is_identity = np.array_equal(reindex, np.arange(len(output_order)))

    return CompiledProgram(
        components=tuple(compiled_components),
        direct_f_indices=jnp.array([e[1] for e in direct_entries], dtype=jnp.int32),
        direct_flips=jnp.array([e[2] for e in direct_entries], dtype=jnp.bool_),
        output_order=jnp.asarray(output_order),
        output_reindex=None if is_identity else jnp.asarray(reindex),
        num_outputs=prepared.num_outputs,
        num_detectors=prepared.num_detectors,
    )


def classify_direct(component: Graph, layout: ParamLayout) -> tuple[int, bool] | None:
    """Return ``(basis index, flip)`` if the component's only output is one basis bit.

    That is the case when the component is a classical output boundary joined by
    a Hadamard edge to a classical Z spider whose phase is 0 or π plus one
    error-basis parameter.
    """
    outputs = component.outputs()
    if len(outputs) != 1 or component.num_vertices() != 2:
        return None
    (out,) = outputs
    (spider,) = component.neighbors(out)
    if Domain.QUANTUM in (
        component.vertex_domain(out),
        component.vertex_domain(spider),
    ):
        return None
    if component.vertex_type(spider) != VertexType.Z:
        return None
    if component.edge_type(out, spider) != EdgeType.HADAMARD:
        return None
    phase = component.phase(spider)
    if len(phase.parameters) != 1 or phase.value not in (0, 1):
        return None
    (param,) = phase.parameters
    if not layout.num_errors <= param < layout.num_errors + layout.num_basis:
        return None
    return layout.basis_index(param), phase.value == 1


def _port(g: Graph, v: Vertex) -> int:
    port = g.port(v)
    assert port is not None
    return port


def _basis_indices(g: Graph, layout: ParamLayout) -> list[int]:
    """Return the sorted basis indices of ``g``'s parameters (its scalar is 1)."""
    return sorted(layout.basis_index(p) for p in g.parameters())


def _plugged(component: Graph, layout: ParamLayout, k: int) -> Graph:
    """Plug the first ``k`` outputs with their output bits and trace out the rest.

    Output ``i`` gets the normalized effect ``⟨m_i|`` for its output parameter
    ``m_i``, so all of a component's plugged graphs share one normalization.
    """
    g = component.copy()
    for j, v in enumerate(component.outputs()):
        if j < k:
            bit = Phase(Fraction(0), frozenset({layout.output(_port(component, v))}))
            g.plug_basis(v, bit)
        else:
            g.discard(v)
    return g


def _without_unit_factors(s: Scalar) -> Scalar:
    """Return ``s`` without its modulus-1 factors (global and parametric phases).

    The sampler only uses ``|amplitude|``, so factors of modulus 1 shared by all
    terms change nothing.
    """
    out = s.copy()
    out.phase = Fraction(0)
    out.phases = []
    out.pi_pairs = []
    return out


def _terms(ssum: ScalarSum, shift: int) -> list[Scalar]:
    """Return the decomposition's terms, ready to compile.

    ``common`` loses its modulus-1 factors (and so does the term if it is the
    only one), is multiplied into every term, and every term is divided by
    ``2^shift`` to keep the exact arithmetic in range.
    """
    common = _without_unit_factors(ssum.common)
    terms = ssum.terms
    if len(terms) == 1:
        terms = [_without_unit_factors(terms[0])]
    scale = Scalar(base=Dyadic(1, 0, 0, 0, -shift))
    return [common * t * scale for t in terms]


def _compile_component(
    component: Graph,
    basis_global: list[int],
    layout: ParamLayout,
    mode: DecompositionMode,
    strategy: DecompositionStrategy,
) -> CompiledComponent:
    """Compile one connected component."""
    output_indices = [_port(component, v) for v in component.outputs()]
    basis_here = set(_basis_indices(component, layout))
    f_selection = [i for i in basis_global if i in basis_here]
    counts = (
        list(range(len(output_indices) + 1))
        if mode == "sequential"
        else [0, len(output_indices)]
    )

    sums = [decompose(_plugged(component, layout, k), strategy) for k in counts]
    shift = sums[0].common.base.power

    compiled: list[CompiledScalarGraphs] = []
    for k, ssum in zip(counts, sums, strict=True):
        param_ids = [layout.basis(i) for i in f_selection]
        param_ids += [layout.output(output_indices[j]) for j in range(k)]
        compiled.append(compile_scalars(_terms(ssum, shift), param_ids))

    return CompiledComponent(
        output_indices=tuple(output_indices),
        f_selection=jnp.array(f_selection, dtype=jnp.int32),
        compiled_scalar_graphs=tuple(compiled),
    )
