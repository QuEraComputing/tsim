"""Stabilizer rank decomposition for non-Clifford ZX graphs."""

from typing import Iterable, Sequence

import pyzx_param as zx
from pyzx_param.graph.base import BaseGraph
from pyzx_param.simulate import DecompositionStrategy


def _decompose(
    graphs: Sequence[BaseGraph],
    count_fn,
    replace_fn,
) -> list[BaseGraph]:
    """Recursively decompose graphs using stabilizer rank decomposition."""
    results: list[BaseGraph] = []
    for graph in graphs:
        if count_fn(graph) == 0:
            results.append(graph)
            continue

        gsum = replace_fn(graph.copy())
        for g in gsum.graphs:
            zx.full_reduce(g, paramSafe=True)
            if g.scalar.is_zero:
                continue
            results.extend(_decompose([g], count_fn, replace_fn))
    return results


def find_stab_magic(
    graphs: Iterable[BaseGraph], strategy: DecompositionStrategy
) -> list[BaseGraph]:
    """Recursively decompose ZX-graphs into stabilizer components via magic-state removal."""
    return _decompose(
        list(graphs),
        count_fn=zx.simplify.tcount,
        replace_fn=lambda g: zx.simulate.replace_magic_states(
            g, pick_random=False, strategy=strategy
        ),
    )


def find_stab_u3(
    graphs: Iterable[BaseGraph], strategy: DecompositionStrategy
) -> list[BaseGraph]:
    """Recursively decompose ZX-graphs by removing U3 phases."""
    return _decompose(
        list(graphs),
        count_fn=zx.simplify.u3_count,
        replace_fn=lambda g: zx.simulate.replace_u3_states(g, strategy=strategy),
    )


def find_stab(graph: BaseGraph, strategy: DecompositionStrategy) -> list[BaseGraph]:
    """Decompose a ZX-graph into a sum of stabilizer components.

    This is the main entry point for stabilizer rank decomposition. It first removes
    U3 phases, then decomposes T gates via BSS decompositions, producing a sum of
    scalar graphs.

    Args:
        graph: The ZX graph to decompose.
        strategy: Decomposition strategy. Must be one of "cat5", "bss", "cutting".

    Returns:
        A list of scalar graphs whose sum equals the original graph.

    """
    zx.full_reduce(graph, paramSafe=True)
    graphs = find_stab_u3([graph], strategy=strategy)
    return find_stab_magic(graphs, strategy=strategy)
