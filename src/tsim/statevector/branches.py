"""Exact outcome distributions of a statevector component, for given f-bits.

The sampler draws each measurement of a component's tape. ``Branches`` follows
both outcomes instead, for many rows of f-bits at once, and returns the exact
distribution of the component's outputs.

The f-bits enter the tape in two ways: they set the signs of its rotations, and
they flip its outputs by a parity of their own. Only the first needs the walk.
``outcomes`` therefore reports how the outputs deviate from their noiseless
values apart from the second, so one walk serves every row with the same
rotation signs.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property

import numpy as np

from tsim.statevector.program import MEASURE_DORMANT, ROTATE, StatevectorComponent

MAX_BYTES = 1 << 27
"""Bytes of amplitudes that a walk aims to hold at once."""

MAX_BRANCHES = 1 << 20
"""Most measurement branches that one row of f-bits may have."""

IMPOSSIBLE = 1e-12
"""Probability below which a measurement outcome cannot happen."""

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


def gf2_product(bits: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return ``bits @ rows.T`` modulo 2, as uint8."""
    product = bits.astype(np.float32) @ rows.T.astype(np.float32)
    return (product.astype(np.int64) & 1).astype(np.uint8)


class Branches:
    """The measurement branches of one component's tape.

    Attributes:
        f_selection: Global f-bit index of each component f-bit.
        sign_rows: The component f-bits that each rotation's sign reads.
        output_rows: The component f-bits that each output slot reads.
        observable: Whether each output slot is an observable.
        coin_rows: The output slots that each fair coin flips, for the coins
            that ``support`` fixes at 0: dormant measurements that no
            rotation reads.

    """

    def __init__(self, component: StatevectorComponent, num_detectors: int) -> None:
        """Read the tape of ``component``; outputs from ``num_detectors`` on are observables."""
        tape = component.tape
        num_f = len(tape.f_selection)
        self.f_selection = tape.f_selection
        self._tape = tape
        self._size = 1 << component.width
        self._index = np.arange(self._size)
        self._lanes = max(2, MAX_BYTES // (16 * self._size))
        rotation = tape.kind <= ROTATE
        self._rotations = np.flatnonzero(rotation)
        self.sign_rows = tape.sign_row[rotation, :num_f].astype(np.uint8)
        self.output_rows = tape.output_row[:, :num_f].astype(np.uint8)
        self._sign_b = tape.sign_row[:, num_f:].astype(np.uint8)
        self._out_b = tape.output_row[:, num_f:].astype(np.uint8)
        self._branch = tape.branch - num_f
        self._outputs = np.asarray(component.output_indices, dtype=np.int64)
        self._num_detectors = num_detectors
        self.observable = self._outputs >= num_detectors
        # NOTE: a dormant measurement leaves the amplitudes alone; if no rotation
        # reads it either, it only flips outputs, independently of the rest.
        dormant = np.flatnonzero(tape.kind == MEASURE_DORMANT)
        unread = ~np.asarray(self._sign_b.any(axis=0))[self._branch[dormant]]
        self._coins = np.zeros(len(tape.kind), dtype=bool)
        self._coins[dormant[unread]] = True
        self.coin_rows = self._out_b[:, self._branch[dormant[unread]]].T

    @cached_property
    def reference(self) -> np.ndarray:
        """The noiseless value of each output slot.

        Raises:
            ValueError: If a detector or an observable of the component is
                random without noise.

        """
        noiseless = np.zeros((1, len(self.f_selection)), dtype=np.uint8)
        values = np.concatenate([out for _, _, out in self._leaves(noiseless)])
        random = np.flatnonzero((values != values[0]).any(axis=0))
        if len(random):
            slot = int(random[0])
            index = int(self._outputs[slot])
            name = (
                f"observable {index - self._num_detectors}"
                if self.observable[slot]
                else f"detector {index}"
            )
            raise ValueError(
                f"{name} is random without noise, so no error can flip it; "
                "every detector and observable must be deterministic"
            )
        return values[0]

    def outcomes(self, f: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return each row's distribution of output deviations.

        A deviation is the outputs XOR their noiseless values XOR
        ``gf2_product(f, output_rows)``, so it depends on ``f`` only through
        ``gf2_product(f, sign_rows)``.

        Args:
            f: The component's f-bits, shape ``(rows, len(f_selection))``.

        Returns:
            For each row, the deviations that can happen, shape
            ``(deviations, output slots)``, and their probabilities.

        Raises:
            ValueError: If the walk has more than ``MAX_BRANCHES`` branches
                per row.

        """
        f = np.asarray(f, dtype=np.uint8)
        flips = gf2_product(f, self.output_rows) ^ self.reference
        row, weight, outputs = self._collect(f, skip_coins=False)
        deviation = outputs ^ flips[row]
        bounds = np.searchsorted(row, np.arange(len(f) + 1))
        result = []
        for r in range(len(f)):
            mine = slice(bounds[r], bounds[r + 1])
            values, group = np.unique(deviation[mine], axis=0, return_inverse=True)
            probability = np.zeros(len(values))
            np.add.at(probability, group.ravel(), weight[mine])
            result.append((values, probability))
        return result

    def support(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the output values that each row of f-bits can give.

        The walk fixes the coins of ``coin_rows`` at 0: each of them flips the
        outputs by its row of ``coin_rows`` with probability 1/2, whatever
        else happens.

        Args:
            f: The component's f-bits, shape ``(rows, len(f_selection))``.

        Returns:
            The row of ``f`` of each possible outcome, in increasing order,
            and its output values, shape ``(outcomes, output slots)``.

        Raises:
            ValueError: If the walk has more than ``MAX_BRANCHES`` branches
                per row.

        """
        row, _, outputs = self._collect(np.asarray(f, dtype=np.uint8), skip_coins=True)
        return row, outputs

    def _collect(
        self, f: np.ndarray, *, skip_coins: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return every branch's row, probability and outputs, sorted by row."""
        rows, weights, outputs = [], [], []
        leaves = 0
        for row, weight, out in self._leaves(f, skip_coins=skip_coins):
            rows.append(row)
            weights.append(weight)
            outputs.append(out)
            leaves += len(row)
            if leaves > MAX_BRANCHES * len(f):
                raise ValueError(
                    f"the tape has more than {MAX_BRANCHES} measurement branches"
                )
        row = np.concatenate(rows)
        order = np.argsort(row, kind="stable")
        return (
            row[order],
            np.concatenate(weights)[order],
            np.concatenate(outputs)[order],
        )

    def _leaves(
        self, f: np.ndarray, *, skip_coins: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield the branches that reach the end of the tape, in batches.

        A batch is the row of ``f`` of each branch, its probability and its
        output values. With ``skip_coins``, the coins of ``coin_rows`` land 0.
        """
        skip = self._coins if skip_coins else np.zeros_like(self._coins)
        tape = self._tape
        sign = np.zeros((len(f), len(tape.kind)), dtype=np.uint8)
        sign[:, self._rotations] = gf2_product(f, self.sign_rows)
        sign ^= tape.sign_const.astype(np.uint8)
        out = gf2_product(f, self.output_rows) ^ tape.output_const.astype(np.uint8)
        step = max(1, self._lanes // 4)
        for start in range(0, len(f), step):
            row = np.arange(start, min(start + step, len(f)))
            amps = np.zeros((len(row), self._size), dtype=np.complex128)
            amps[:, 0] = 1.0
            bits = np.zeros((len(row), self._out_b.shape[1]), dtype=np.uint8)
            yield from self._walk(0, amps, row, bits, sign, out, skip)

    def _walk(
        self,
        start: int,
        amps: np.ndarray,
        row: np.ndarray,
        bits: np.ndarray,
        sign: np.ndarray,
        out: np.ndarray,
        skip: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run the actions from ``start`` on lanes and yield those that finish.

        Lane ``i`` has amplitudes ``amps[i]``, whose squared norm is its
        probability, belongs to row ``row[i]`` and has branch bits ``bits[i]``.
        ``sign`` and ``out`` hold the rows' f-bit parts of the action signs and
        of the outputs. The actions of ``skip`` do not run.
        """
        tape = self._tape
        for a in range(start, len(tape.kind)):
            if skip[a]:
                continue
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
            weight = _weights(amps)
            for outcome in (0, 1):
                if tape.kind[a] == MEASURE_DORMANT:
                    kept, child = np.arange(len(row)), amps * np.sqrt(0.5)
                else:
                    moved = amps[:, source] * phase
                    projected = (amps - moved if outcome else amps + moved) / 2
                    # NOTE: an outcome is impossible given the lane, whatever
                    # the lane's own weight.
                    possible = _weights(projected) > IMPOSSIBLE * weight
                    kept = np.flatnonzero(possible)
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
            if len(row) > self._lanes:
                half = len(row) // 2
                for part in (slice(0, half), slice(half, None)):
                    yield from self._walk(
                        a + 1, amps[part], row[part], bits[part], sign, out, skip
                    )
                return
        yield row, _weights(amps), out[row] ^ _bit_parity(bits, self._out_b)
