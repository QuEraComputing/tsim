from typing import Optional

import tsim.external.pyzx as zx


def int_cliff_simp(ggList, g, quiet: bool = False) -> int:
    """Keeps doing the simplifications ``id_simp``, ``spider_simp``,
    ``pivot_simp`` and ``lcomp_simp`` until none of them can be applied anymore."""
    zx.simplify.spider_simp(g, quiet=quiet, stats=None)
    zx.simplify.to_gh(g)
    i = 0
    while True:
        i1 = 0
        i2 = 0
        i3 = 0
        i4 = 0
        #        i1 = zx.simplify.id_simp(g, quiet=quiet, stats=None) #TEMP******
        i2 = zx.simplify.spider_simp(g, quiet=quiet, stats=None)
        i3 = zx.simplify.pivot_simp(g, quiet=quiet, stats=None)
        i4 = zx.simplify.lcomp_simp(g, quiet=quiet, stats=None)
        if i1 + i2 + i3 + i4 == 0:
            break
        i += 1
    return i


def cliff_simp(ggList, g, quiet: bool = True) -> int:
    """Keeps doing rounds of :func:`interior_clifford_simp` and
    :func:`pivot_boundary_simp` until they can't be applied anymore."""
    i = 0
    while True:
        i += int_cliff_simp(ggList, g, quiet=quiet)
        i2 = 0
        i2 = zx.simplify.pivot_boundary_simp(g, quiet=quiet, stats=None)
        if i2 == 0:
            break
    return i


def sim_full_red(
    ggList, g, quiet: bool = True, paramSafe: Optional[bool] = False
) -> None:
    """The main simplification routine of PyZX. It uses a combination of :func:`clifford_simp` and
    the gadgetization strategies :func:`pivot_gadget_simp` and :func:`gadget_simp`."""
    int_cliff_simp(ggList, g, quiet=quiet)
    zx.simplify.pivot_gadget_simp(g, quiet=quiet, stats=None)
    while True:
        cliff_simp(ggList, g, quiet=quiet)
        i = 0
        j = 0
        #        if (not paramSafe): i = zx.simplify.gadget_simp(g, quiet=quiet, stats=None) #TEMP******
        int_cliff_simp(ggList, g, quiet=quiet)
        j = zx.simplify.pivot_gadget_simp(g, quiet=quiet, stats=None)
        if i + j == 0:
            break


def full_red(g_list: zx.simulate.SumGraph, quiet: bool = True) -> None:
    terms = []
    for i in range(len(g_list.graphs)):
        gg = g_list.graphs[i]

        sim_full_red(g_list.graphs, gg, quiet=True)

        if not gg.scalar.is_zero:
            terms.append(gg)

    g_list.graphs = terms
