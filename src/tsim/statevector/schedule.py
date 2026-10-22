"""Reordering of Pauli operations to keep the dense array small.

Two operations must keep their relative order exactly when they anticommute;
every order that respects these pairs gives the same distribution. The active
width k = n − dim S, as in Clifft (arXiv:2604.27058), is tracked with the Pauli
group S that stabilizes every trajectory up to sign. ``schedule`` runs
measurements as early as possible and widening rotations as late as possible:
when nothing else can run, it runs the pending measurement whose unrun
ancestors reach the smallest width, the earliest one on a tie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsim.statevector.pbc import MEASUREMENT, PauliOps, anticommute


class StabilizerGroup:
    """Generators, up to sign, of the Pauli group stabilizing every trajectory."""

    def __init__(self, num_qubits: int) -> None:
        """Start from ``|0…0⟩``: the generators are ``Z_0 … Z_{n-1}``."""
        self.x = np.zeros((num_qubits, num_qubits), dtype=bool)
        self.z = np.eye(num_qubits, dtype=bool)

    @property
    def width(self) -> int:
        """Return the active width: qubits minus independent generators."""
        return self.x.shape[1] - self.x.shape[0]

    def copy(self) -> StabilizerGroup:
        """Return an independent copy of the group."""
        group = StabilizerGroup(0)
        group.x, group.z = self.x.copy(), self.z.copy()
        return group

    def anticommuting(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return which generators anticommute with the Pauli ``(x, z)``."""
        return anticommute(self.x, self.z, x, z)

    def _eliminate(self, x: np.ndarray, z: np.ndarray) -> int | None:
        """Leave one generator anticommuting with ``(x, z)``; return it, if any."""
        anti = np.flatnonzero(self.anticommuting(x, z))
        if len(anti) == 0:
            return None
        first, rest = anti[0], anti[1:]
        self.x[rest] ^= self.x[first]
        self.z[rest] ^= self.z[first]
        return int(first)

    def rotate(self, x: np.ndarray, z: np.ndarray) -> None:
        """Keep the generators that a rotation about ``(x, z)`` preserves."""
        first = self._eliminate(x, z)
        if first is not None:
            self.x = np.delete(self.x, first, axis=0)
            self.z = np.delete(self.z, first, axis=0)

    def measure(self, x: np.ndarray, z: np.ndarray) -> None:
        """Update the generators after measuring the Pauli ``(x, z)``."""
        first = self._eliminate(x, z)
        if first is not None:
            self.x[first] = x
            self.z[first] = z
        elif not self.contains(x, z):
            self.x = np.vstack([self.x, x[None]])
            self.z = np.vstack([self.z, z[None]])

    def contains(self, x: np.ndarray, z: np.ndarray) -> bool:
        """Return whether ``(x, z)`` is a product of the generators, up to sign.

        The generators commute and are independent, so their rank is their number.
        """
        matrix = np.hstack([self.x, self.z])
        extended = np.vstack([matrix, np.concatenate([x, z])[None]])
        return _rank(extended) == len(matrix)


def _rank(matrix: np.ndarray) -> int:
    """Return the GF(2) rank of a boolean matrix."""
    m = matrix.copy()
    rank = 0
    for column in range(m.shape[1]):
        if rank == m.shape[0]:
            break
        rows = np.flatnonzero(m[rank:, column])
        if len(rows) == 0:
            continue
        pivot = rank + rows[0]
        m[[rank, pivot]] = m[[pivot, rank]]
        others = np.flatnonzero(m[:, column])
        others = others[others != rank]
        m[others] ^= m[rank]
        rank += 1
    return rank


def anticommutation(ops: PauliOps) -> np.ndarray:
    """Return the symmetric matrix of which pairs of operations anticommute."""
    return anticommute(ops.x, ops.z, ops.x, ops.z)


def _ancestors(before: np.ndarray) -> np.ndarray:
    """Return ``a[j, i]``: op ``i`` must run before op ``j`` (transitively)."""
    # NOTE: the rows are OR-ed as bit-packed 64-bit words, which is several times
    # faster than as booleans.
    num_ops = before.shape[0]
    preds = np.ascontiguousarray(before.T)
    packed = np.zeros((num_ops, 8 * -(-num_ops // 64)), dtype=np.uint8)
    packed[:, : -(-num_ops // 8)] = np.packbits(preds, axis=1, bitorder="little")
    direct = packed.view(np.uint64)
    ancestors = np.zeros_like(direct)
    for j in range(num_ops):
        rows = np.flatnonzero(preds[j])
        if len(rows):
            ancestors[j] = np.bitwise_or.reduce(ancestors[rows], axis=0) | direct[j]
    bits = np.unpackbits(
        ancestors.view(np.uint8), axis=1, count=num_ops, bitorder="little"
    )
    return bits.astype(bool)


@dataclass(frozen=True)
class Schedule:
    """An order of the operations and the largest active width it reaches.

    Attributes:
        order: Indices into the ``PauliOps``, in execution order.
        width: The largest structural active width along ``order`` (W).

    """

    order: np.ndarray
    width: int


def schedule(ops: PauliOps) -> Schedule:
    """Return an order of ``ops`` that keeps the active width small."""
    num_ops = ops.num_ops
    before = np.triu(anticommutation(ops), 1)
    ancestors = _ancestors(before)
    successors = [np.flatnonzero(before[i]) for i in range(num_ops)]
    waiting = before.sum(axis=0)
    is_measurement = ops.kind == MEASUREMENT
    done = np.zeros(num_ops, dtype=bool)
    group = StabilizerGroup(ops.num_lines)
    order: list[int] = []
    width = 0

    def advance(stabilizers: StabilizerGroup, j: int) -> None:
        apply = stabilizers.measure if is_measurement[j] else stabilizers.rotate
        apply(ops.x[j], ops.z[j])

    def peak(chain: np.ndarray, bound: float) -> float:
        trial = group.copy()
        highest = trial.width
        for j in chain:
            advance(trial, int(j))
            highest = max(highest, trial.width)
            if highest >= bound:
                return float("inf")
        return highest

    def run(j: int) -> None:
        nonlocal width
        advance(group, j)
        done[j] = True
        order.append(j)
        width = max(width, group.width)
        waiting[successors[j]] -= 1

    while not done.all():
        progressed = True
        while progressed:
            progressed = False
            for j in np.flatnonzero(~done):
                if waiting[j]:
                    continue
                keeps_width = not group.anticommuting(ops.x[j], ops.z[j]).any()
                if is_measurement[j] or keeps_width:
                    run(int(j))
                    progressed = True
        pending = np.flatnonzero(~done & is_measurement)
        if len(pending) == 0:
            for j in np.flatnonzero(~done):
                run(int(j))
            break
        chains = [np.flatnonzero(ancestors[m] & ~done) for m in pending]
        # NOTE: a chain that reaches the best peak so far cannot win, as ties
        # go to the earliest, so its simulation stops there.
        best, lowest = 0, float("inf")
        for k, chain in enumerate(chains):
            if (highest := peak(chain, lowest)) < lowest:
                best, lowest = k, highest
        for j in chains[best]:
            run(int(j))
        run(int(pending[best]))
    return Schedule(order=np.asarray(order, dtype=np.int32), width=width)
