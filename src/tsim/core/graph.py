"""Preparing a circuit's graph for sampling: reduction and the error basis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zxparam
from zxparam import Graph, Phase, Scalar

from tsim.core.params import ParamLayout
from tsim.core.parse import parse_stim_circuit
from tsim.core.types import SamplingGraph
from tsim.utils.linalg import find_basis

if TYPE_CHECKING:
    from tsim.circuit import Circuit


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
    builder = parse_stim_circuit(circuit._stim_circ)
    graph = builder.build(
        closed=True, outputs="detectors" if sample_detectors else "records"
    )
    zxparam.full_reduce(graph)
    graph, error_transform = transform_error_basis(graph, builder.num_error_bits)
    graph.scalar = Scalar.one()
    return SamplingGraph(
        graph=graph,
        error_transform=error_transform,
        channel_probs=builder.channel_probs,
        num_outputs=len(graph.outputs()),
        num_detectors=len(builder.detectors),
        layout=ParamLayout(builder.num_error_bits),
    )
