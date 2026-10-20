"""Compilation pipeline from prepared graphs to executable programs."""

from __future__ import annotations

from fractions import Fraction
from functools import partial
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
from tsim.core.types import (
    CompiledComponent,
    CompiledProgram,
    SamplingGraph,
    SamplingStrategy,
)
from tsim.statevector.program import (
    STABILIZER_RANK_HINT,
    StatevectorComponent,
    compile_statevector_component,
)

DecompositionMode = Literal["sequential", "joint"]


def compile_program(
    prepared: SamplingGraph,
    *,
    mode: DecompositionMode,
    strategy: SamplingStrategy,
) -> CompiledProgram:
    """Compile a prepared graph into an executable sampling program.

    The graph is split into connected components. A component whose single
    output equals one error-basis bit takes the direct path. With
    ``strategy="active_statevector"`` every other component becomes an action tape for
    the statevector executor. Otherwise its outputs are plugged (all prefixes
    in ``sequential`` mode, none and all in ``joint`` mode), each plugged graph
    is decomposed into stabilizer terms, and the terms are compiled into JAX
    arrays.

    Args:
        prepared: The prepared graph from ``prepare_graph``.
        mode: ``"sequential"`` for sampling, ``"joint"`` for probabilities.
        strategy: ``"active_statevector"``, or a stabilizer-rank decomposition
            strategy.

    Returns:
        A CompiledProgram ready for sampling.

    Raises:
        ValueError: If ``strategy="active_statevector"`` is combined with ``mode="joint"``.

    """
    if strategy == "active_statevector":
        if mode == "joint":
            raise ValueError(
                "strategy='active_statevector' samples outcomes and cannot compute "
                f"probabilities. {STABILIZER_RANK_HINT}"
            )
        compile_component = compile_statevector_component
    else:
        compile_component = partial(_compile_component, mode=mode, strategy=strategy)
    graph = prepared.graph
    layout = prepared.layout

    direct_entries: list[tuple[int, int, bool]] = []
    compiled_components: list[CompiledComponent | StatevectorComponent] = []
    compiled_output_order: list[int] = []

    components = sorted(graph.connected_components(), key=lambda c: len(c.outputs()))
    for component in components:
        output_indices = [_port(component, v) for v in component.outputs()]
        direct = classify_direct(component, layout)
        if direct is not None:
            basis_index, flip = direct
            direct_entries.append((output_indices[0], basis_index, flip))
            continue
        compiled_components.append(compile_component(component, output_indices, layout))
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
        direct_f_indices=np.array([e[1] for e in direct_entries], dtype=np.int32),
        direct_flips=np.array([e[2] for e in direct_entries], dtype=np.bool_),
        output_order=output_order,
        output_reindex=None if is_identity else reindex,
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
    # NOTE: zxparam never lets a classical boundary touch a quantum vertex, so a
    # classical output implies a classical spider.
    if component.vertex_domain(out) == Domain.QUANTUM:
        return None
    if component.vertex_type(spider) != VertexType.Z:
        return None
    if component.edge_type(out, spider) != EdgeType.HADAMARD:
        return None
    phase = component.phase(spider)
    if len(phase.parameters) != 1 or phase.value not in (0, 1):
        return None
    (param,) = phase.parameters
    kind, basis_index = layout.kind(param)
    if kind != "f":
        return None
    return basis_index, phase.value == 1


def _port(g: Graph, v: Vertex) -> int:
    port = g.port(v)
    assert port is not None
    return port


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

    The sampler only uses the magnitude of each value, so factors of modulus 1
    shared by all terms change nothing.
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
    output_indices: list[int],
    layout: ParamLayout,
    mode: DecompositionMode,
    strategy: DecompositionStrategy,
) -> CompiledComponent:
    """Compile one connected component, whose outputs are ``output_indices``.

    The component's scalar is 1, so its parameters are its vertices' basis bits.
    """
    f_selection = sorted(layout.basis_index(p) for p in component.parameters())
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
