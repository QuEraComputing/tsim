"""Exact outcome probabilities of a statevector component, for given f-bits.

The sampler draws each measurement of a component's tape. ``Branches`` follows
both outcomes instead, for many rows of f-bits at once, and drops a branch as
soon as one of its detectors differs from the noiseless value. What reaches the
end of the tape gives, for each row, the exact probability that no detector of
the component flips, and that no detector and no observable flips.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property

import numpy as np

from tsim.statevector.program import MEASURE_DORMANT, ROTATE, StatevectorComponent

MAX_BYTES = 1 << 27
"""Bytes of amplitudes that a walk aims to hold at once."""

MAX_NOISELESS_BRANCHES = 1 << 12
"""Most measurement branches of the noiseless circuit that a component may have."""

IMPOSSIBLE = 1e-12
"""Weight below which a measurement outcome cannot happen."""

_I_POWERS = (1, 1j, -1, -1j)


def _parity(values: np.ndarray) -> np.ndarray:
    """Return the parity of the bits of each integer below 2^32."""
    values = values ^ (values >> 16)
    values = values ^ (values >> 8)
    values = values ^ (values >> 4)
    values = values ^ (values >> 2)
    return (values ^ (values >> 1)) & 1


def _weights(amps: np.ndarray) -> np.ndarray:
    """Return the squared norm of each row of amplitudes."""
    return (np.abs(amps) ** 2).sum(axis=1)


def _bit_parity(bits: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return the parity of ``bits & row`` for each row of ``bits`` and of ``rows``."""
    # NOTE: uint8 sums wrap modulo 256, which keeps their parity.
    return (bits @ rows.T) & 1


def unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group the equal rows of a uint8 matrix.

    Returns:
        The index of one row of each group, and the group of each row.

    """
    # NOTE: sorting 64-bit words is much faster than ``np.unique(rows, axis=0)``.
    words = np.zeros((len(rows), max(8, -(-rows.shape[1] // 8) * 8)), dtype=np.uint8)
    words[:, : rows.shape[1]] = rows
    words = words.view(np.uint64)
    order = np.lexsort(words.T)
    first = np.ones(len(rows), dtype=bool)
    first[1:] = (words[order[1:]] != words[order[:-1]]).any(axis=1)
    group = np.empty(len(rows), dtype=np.int64)
    group[order] = np.cumsum(first) - 1
    return order[first], group


class Branches:
    """The measurement branches of one component's tape.

    Attributes:
        f_selection: Global f-bit index of each component f-bit.

    """

    def __init__(self, component: StatevectorComponent, num_detectors: int) -> None:
        """Read the tape of ``component``, whose outputs from ``num_detectors`` on are observables."""
        tape = component.tape
        num_f = len(tape.f_selection)
        self.f_selection = tape.f_selection
        self._tape = tape
        self._size = 1 << component.width
        self._index = np.arange(self._size)
        self._lanes = max(2, MAX_BYTES // (16 * self._size))
        self._sign_f = tape.sign_row[:, :num_f].astype(np.float32)
        self._sign_b = tape.sign_row[:, num_f:].astype(np.uint8)
        self._out_f = tape.output_row[:, :num_f].astype(np.float32)
        self._out_b = tape.output_row[:, num_f:].astype(np.uint8)
        self._branch = tape.branch - num_f
        # NOTE: an output is known once the last measurement that it reads ran.
        measured = np.flatnonzero(tape.branch >= 0)
        setter = np.full(self._out_b.shape[1], -1)
        setter[self._branch[measured]] = measured
        ready = np.max(np.where(self._out_b, setter[None, :], -1), axis=1, initial=-1)
        self._outputs = np.asarray(component.output_indices, dtype=np.int64)
        self._num_detectors = num_detectors
        self._observable = self._outputs >= num_detectors
        self._early = np.flatnonzero((ready < 0) & ~self._observable)
        self._ready = [
            np.flatnonzero((ready == a) & ~self._observable)
            for a in range(len(tape.kind))
        ]
        self._cache: dict[bytes, tuple[float, float]] = {}

    @cached_property
    def reference(self) -> np.ndarray:
        """The noiseless value of each output slot.

        Raises:
            ValueError: If a detector or an observable of the component is
                random without noise, or the noiseless tape has more than
                ``MAX_NOISELESS_BRANCHES`` measurement branches.

        """
        noiseless = np.zeros((1, len(self.f_selection)), dtype=np.uint8)
        leaves = list(self._leaves(noiseless, prune=False))
        values = np.concatenate([outputs for _, _, outputs in leaves])
        random = np.flatnonzero((values != values[0]).any(axis=0))
        if len(random):
            slot = int(random[0])
            index = int(self._outputs[slot])
            name = (
                f"observable {index - self._num_detectors}"
                if self._observable[slot]
                else f"detector {index}"
            )
            raise ValueError(
                f"{name} is random without noise, so no error can flip it; "
                "every detector and observable must be deterministic"
            )
        return values[0]

    def probabilities(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return each row's chance to pass the detectors, and to pass unflipped.

        Args:
            f: The component's f-bits, shape ``(rows, len(f_selection))``.

        Returns:
            The probability that no detector of the component flips, and the
            probability that no detector and no observable flips, per row.

        """
        f = np.asarray(f, dtype=np.uint8)
        packed = np.packbits(f, axis=1)
        one, group = unique_rows(packed)
        keys = [row.tobytes() for row in packed[one]]
        missing = [i for i, key in enumerate(keys) if key not in self._cache]
        if missing:
            passed = np.zeros(len(missing))
            unflipped = np.zeros(len(missing))
            for row, weight, outputs in self._leaves(f[one[missing]], prune=True):
                same = (outputs == self.reference)[:, self._observable].all(axis=1)
                np.add.at(passed, row, weight)
                np.add.at(unflipped, row[same], weight[same])
            for i, p, u in zip(missing, passed, unflipped, strict=True):
                self._cache[keys[i]] = (float(p), float(u))
        result = np.array([self._cache[key] for key in keys]).reshape(-1, 2)[group]
        return result[:, 0], result[:, 1]

    def _leaves(
        self, f: np.ndarray, *, prune: bool
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield the branches that reach the end of the tape, in batches.

        A batch is the row of ``f`` of each branch, its probability and its
        output values. With ``prune``, only branches whose detectors have their
        reference values are followed.
        """
        tape = self._tape
        sign = (f @ self._sign_f.T).astype(np.int64) & 1
        sign ^= tape.sign_const.astype(np.int64)
        out = ((f @ self._out_f.T).astype(np.int64) & 1).astype(np.uint8)
        out ^= tape.output_const.astype(np.uint8)
        rows = np.arange(len(f))
        if prune:
            early = self._early
            rows = rows[(out[:, early] == self.reference[early]).all(axis=1)]
        step = max(1, self._lanes // 4)
        for start in range(0, len(rows), step):
            row = rows[start : start + step]
            amps = np.zeros((len(row), self._size), dtype=np.complex128)
            amps[:, 0] = 1.0
            bits = np.zeros((len(row), self._out_b.shape[1]), dtype=np.uint8)
            yield from self._walk(0, amps, row, bits, sign, out, prune)

    def _walk(
        self,
        start: int,
        amps: np.ndarray,
        row: np.ndarray,
        bits: np.ndarray,
        sign: np.ndarray,
        out: np.ndarray,
        prune: bool,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run the actions from ``start`` on lanes and yield those that finish.

        Lane ``i`` has amplitudes ``amps[i]``, whose squared norm is its
        probability, belongs to row ``row[i]`` and has branch bits ``bits[i]``.
        ``sign`` and ``out`` hold the rows' f-bit parts of the rotation signs
        and of the outputs.
        """
        tape = self._tape
        for a in range(start, len(tape.kind)):
            if len(row) == 0:
                return
            x, z = int(tape.x_mask[a]), int(tape.z_mask[a])
            # NOTE: the Pauli P = i^|x&z| X^x Z^z takes amplitudes ``amps`` to
            # ``amps[:, source] * phase``.
            source = self._index ^ x
            phase = _I_POWERS[(x & z).bit_count() & 3] * (1 - 2 * _parity(source & z))
            if tape.kind[a] <= ROTATE:
                flip = sign[row, a] ^ _bit_parity(bits, self._sign_b[a])
                coupling = (2.0 * flip - 1.0) * (1j * tape.sin[a])
                amps = tape.cos[a] * amps + coupling[:, None] * (
                    amps[:, source] * phase
                )
                continue
            children = []
            for outcome in (0, 1):
                if tape.kind[a] == MEASURE_DORMANT:
                    kept, child = np.arange(len(row)), amps * np.sqrt(0.5)
                else:
                    moved = amps[:, source] * phase
                    projected = (amps - moved if outcome else amps + moved) / 2
                    kept = np.flatnonzero(_weights(projected) > IMPOSSIBLE)
                    projected = projected[kept]
                    # NOTE: the planner moved the measured Pauli onto the pivot,
                    # which ends in |0>, as in ``shared._project_range``.
                    pivot = 1 << int(tape.pivot[a])
                    low = self._index[self._index & pivot == 0]
                    child = np.zeros_like(projected)
                    if x == 0:
                        child[:, low] = projected[:, low] + projected[:, low | pivot]
                    else:
                        child[:, low] = np.sqrt(2.0) * projected[:, low]
                child_bits = bits[kept]
                child_bits[:, self._branch[a]] = outcome
                children.append((child, row[kept], child_bits))
            amps = np.concatenate([child[0] for child in children])
            row = np.concatenate([child[1] for child in children])
            bits = np.concatenate([child[2] for child in children])
            slots = self._ready[a]
            if prune and len(slots):
                values = out[row][:, slots] ^ _bit_parity(bits, self._out_b[slots])
                keep = (values == self.reference[slots]).all(axis=1)
                amps, row, bits = amps[keep], row[keep], bits[keep]
            if not prune and len(row) > MAX_NOISELESS_BRANCHES:
                raise ValueError(
                    "the noiseless circuit has more than "
                    f"{MAX_NOISELESS_BRANCHES} measurement branches in one component"
                )
            if len(row) > self._lanes:
                half = len(row) // 2
                for part in (slice(0, half), slice(half, None)):
                    yield from self._walk(
                        a + 1, amps[part], row[part], bits[part], sign, out, prune
                    )
                return
        yield row, _weights(amps), out[row] ^ _bit_parity(bits, self._out_b)
