"""Upper bounds on the fault distance by random restriction (prototype).

The exact search of ``faults`` runs on random subsets of the faults. Every
logical error it finds there is a logical error of the whole circuit, so it
bounds the distance from above; a miss proves nothing. A subset keeps each
logical error of ``w`` faults with probability ``fraction^w``, so this finds the
distance fast when the shortest errors are many, as in Prange's
information-set decoding.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from tsim.noise.faults import LogicalError, _Search

if TYPE_CHECKING:
    from tsim.circuit import Circuit

_FAULT_ARRAYS = (
    "_bits",
    "_d",
    "_s",
    "_o",
    "_hash",
    "_fault_probability",
    "_single_channel",
)


def restricted(search: _Search, keep: np.ndarray) -> _Search:
    """Return ``search`` with only the faults that ``keep`` marks."""
    sub = copy.copy(search)
    index = np.flatnonzero(keep)
    sub._events = [search._events[i] for i in index]
    for name in _FAULT_ARRAYS:
        setattr(sub, name, getattr(search, name)[index])
    sub._explained = {}
    return sub


def random_restriction(
    circuit: Circuit,
    *,
    weight: int,
    fraction: float,
    trials: int,
    seed: int = 0,
    max_errors: int = 10,
) -> list[LogicalError]:
    """Return logical errors of ``weight`` faults found in random subsets of faults.

    Returns as soon as one subset has any; an empty list proves nothing.
    """
    search = _Search(circuit)
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        keep = rng.random(len(search._events)) < fraction
        found = restricted(search, keep).logical_errors(weight, max_errors=max_errors)
        if found:
            return found
    return []
