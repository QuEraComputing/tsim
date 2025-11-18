import pyzx as zx
from pyzx.graph.base import BaseGraph


def find_stab(graph: BaseGraph) -> list[BaseGraph]:
    """Recursively decompose ZX-graph into stabilizer components."""
    if zx.simplify.tcount(graph) == 0:
        return [graph]

    gsum = zx.simulate.replace_magic_states(graph, pick_random=False)

    results = []
    for g in gsum.graphs:
        zx.full_reduce(g, paramSafe=True)
        if g.scalar.is_zero:
            continue

        results.extend(find_stab(g))
    return results
