from typing import Iterable, Sequence

import pyzx as zx
from pyzx.graph.base import BaseGraph


def _decompose(
    graphs: Sequence[BaseGraph],
    count_fn,
    replace_fn,
) -> list[BaseGraph]:
    """Generic recursive stabilizer decomposition helper."""
    results: list[BaseGraph] = []
    for graph in graphs:
        if count_fn(graph) == 0:
            results.append(graph)
            continue

        gsum = replace_fn(graph.copy())
        for g in gsum.graphs:
            zx.full_reduce(g, paramSafe=True)
            if g.scalar.is_zero:
                if len(results) > 0:
                    # this ensures results is never empty
                    # TODO: improve edge case handling
                    continue
            results.extend(_decompose([g], count_fn, replace_fn))
    return results


def find_stab_magic(graphs: Iterable[BaseGraph]) -> list[BaseGraph]:
    """Recursively decompose ZX-graphs into stabilizer components via magic-state removal."""
    return _decompose(
        list(graphs),
        count_fn=zx.simplify.tcount,
        replace_fn=lambda g: zx.simulate.replace_magic_states(g, pick_random=False),
    )


def find_stab_u3(graphs: Iterable[BaseGraph]) -> list[BaseGraph]:
    """Recursively decompose ZX-graphs by removing U3 phases."""
    return _decompose(
        list(graphs),
        count_fn=zx.simplify.u3_count,
        replace_fn=zx.simulate.replace_u3_states,
    )


def find_stab(graph: BaseGraph) -> list[BaseGraph]:
    """Decompose a ZX-graph into a sum of stabilizer components.

    This is the main entry point for stabilizer rank decomposition. It first removes
    U3 phases, then decomposes T gates via BSS decompositions, producing a sum of
    scalar graphs.

    Args:
        graph: The ZX graph to decompose.

    Returns:
        A list of scalar graphs whose sum equals the original graph.
    """
    zx.full_reduce(graph, paramSafe=True)
    graphs = find_stab_u3([graph])
    return find_stab_magic(graphs)
