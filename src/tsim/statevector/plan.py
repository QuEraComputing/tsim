"""Fixed-width sampling plans.

``plan`` walks scheduled Heisenberg ops with a Clifford frame ``F``: a
``stim.Tableau`` that maps Paulis of the ``|0…0⟩`` frame to virtual
coordinates. Virtual coordinates ``0 … W-1`` are the axes of a dense array of
``2^W`` amplitudes, each active or free (in ``|0⟩``); coordinates ``W … n-1``
stay dormant (in ``|0⟩``). Each op becomes at most one action, and every sign an
``AffineRow`` over the symbols: the component f-bits, then the branch bits of
random measurements. A random measurement keeps the reference state on its
``b = 0`` branch; its ``b = 1`` branch differs by a correction Pauli, which
flips the sign of every later op it anticommutes with. Each output is an
``AffineRow`` too: the measurement's sign, XOR its branch bit if it is random.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import stim

from tsim.statevector.heisenberg import ROTATION, HeisenbergOps, anticommute
from tsim.statevector.schedule import Schedule


@dataclass(frozen=True)
class AffineRow:
    """The parity ``constant ⊕ (XOR of the symbols in terms)``."""

    constant: bool
    terms: frozenset[int]

    def __xor__(self, other: AffineRow) -> AffineRow:
        """Return the XOR of two parities."""
        return AffineRow(self.constant ^ other.constant, self.terms ^ other.terms)

    def evaluate(self, symbols: Sequence[int]) -> int:
        """Return the parity's value for the given symbol bits."""
        return int(self.constant) ^ (sum(symbols[t] for t in self.terms) & 1)


@dataclass(frozen=True)
class Rotate:
    """Apply ``exp(-iθ (-1)^sign P)`` for the Pauli with W-bit masks ``x``, ``z``."""

    x: int
    z: int
    angle: float
    sign: AffineRow


@dataclass(frozen=True)
class MeasureActive:
    """Measure the Pauli ``x``, ``z`` on the array, then reset ``pivot`` to ``|0⟩``.

    The sampled eigenvalue bit becomes symbol ``branch``.
    """

    x: int
    z: int
    pivot: int
    branch: int


@dataclass(frozen=True)
class MeasureDormant:
    """Sample a fair bit into symbol ``branch``."""

    branch: int


Action = Rotate | MeasureActive | MeasureDormant


@dataclass(frozen=True)
class Plan:
    """Actions on ``2^width`` amplitudes, and outputs over the symbols.

    Attributes:
        width: Number of array axes (W).
        num_f: Number of f-bit symbols (symbols ``0 … num_f-1``); the branch
            symbols follow, one per measurement action.
        actions: The actions, in execution order.
        outputs: The value of each output slot.

    """

    width: int
    num_f: int
    actions: tuple[Action, ...]
    outputs: tuple[AffineRow, ...]

    @property
    def num_branches(self) -> int:
        """Return the number of branch symbols."""
        return sum(not isinstance(action, Rotate) for action in self.actions)

    @property
    def num_symbols(self) -> int:
        """Return the number of f-bit and branch symbols."""
        return self.num_f + self.num_branches


def plan(ops: HeisenbergOps, schedule: Schedule) -> Plan:
    """Return the fixed-width plan of ``ops`` in schedule order.

    Raises:
        RuntimeError: If the ops need more than ``schedule.width`` active axes.

    """
    planner = _Planner(ops.take(schedule.order), schedule.width)
    for j in range(ops.num_ops):
        planner.step(j)
    return Plan(
        width=schedule.width,
        num_f=ops.num_f,
        actions=tuple(planner.actions),
        outputs=tuple(planner.outputs[slot] for slot in range(ops.num_outputs)),
    )


def _mask(bits: np.ndarray, axes: Iterable[int]) -> int:
    return sum(1 << int(q) for q in axes if bits[q])


_CONTROLLED_PAULI = {(True, False): "CX", (False, True): "CZ", (True, True): "CY"}


class _Planner:
    """The frame, the active axes and the branch corrections during planning."""

    def __init__(self, ops: HeisenbergOps, width: int) -> None:
        """Start with the identity frame and every coordinate dormant."""
        self.ops = ops
        self.width = width
        self.frame = stim.Tableau(ops.num_lines)
        self.active: set[int] = set()
        self.correction_x = np.zeros((0, ops.num_lines), dtype=bool)
        self.correction_z = np.zeros((0, ops.num_lines), dtype=bool)
        self.correction_condition: list[AffineRow] = []
        self.actions: list[Action] = []
        self.outputs: dict[int, AffineRow] = {}

    def _apply(self, gate: str, *targets: int) -> None:
        """Apply a virtual Clifford: ``F ← gate ∘ F``."""
        self.frame.append(stim.Tableau.from_named_gate(gate), list(targets))

    def _virtual(self, j: int) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return ``F(P_j)`` as bit rows and a sign flag."""
        pauli = stim.PauliString.from_numpy(xs=self.ops.x[j], zs=self.ops.z[j])
        image = self.frame(pauli)
        x, z = image.to_numpy()
        return x, z, bool(image.sign == -1)

    def _sign(self, j: int, negative: bool) -> AffineRow:
        """Return op ``j``'s sign: noise row, frame sign and branch corrections."""
        row = self.ops.sign[j]
        terms = frozenset(int(t) for t in np.flatnonzero(row[1:]))
        sign = AffineRow(bool(row[0]) ^ negative, terms)
        anti = anticommute(
            self.correction_x, self.correction_z, self.ops.x[j], self.ops.z[j]
        )
        for m in np.flatnonzero(anti):
            sign = sign ^ self.correction_condition[m]
        return sign

    def _correction(self, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``F⁻¹(X_axis)``, the correction Pauli in the ``|0…0⟩`` frame."""
        x, z = self.frame.inverse_x_output(axis).to_numpy()
        return x, z

    def _branch(self, correction: tuple[np.ndarray, np.ndarray], flip: bool) -> int:
        """Return a new branch symbol ``b``; ``correction`` applies when ``b ⊕ flip``."""
        branch = self.ops.num_f + len(self.correction_condition)
        x, z = correction
        self.correction_x = np.vstack([self.correction_x, x[None]])
        self.correction_z = np.vstack([self.correction_z, z[None]])
        self.correction_condition.append(AffineRow(flip, frozenset({branch})))
        return branch

    def step(self, j: int) -> None:
        """Plan op ``j``."""
        x, z, negative = self._virtual(j)
        dormant_x = [int(q) for q in np.flatnonzero(x) if int(q) not in self.active]
        on_active = any(x[q] or z[q] for q in self.active)
        if self.ops.kind[j] == ROTATION:
            if dormant_x:
                self._promote(j, dormant_x)
                x, z, negative = self._virtual(j)
            elif not on_active:
                return
            axes = sorted(self.active)
            angle = float(self.ops.angle[j])
            sign = self._sign(j, negative)
            self.actions.append(Rotate(_mask(x, axes), _mask(z, axes), angle, sign))
            return
        output = self._sign(j, negative)
        if dormant_x:
            branch = self._measure_dormant(j, dormant_x, negative)
        elif on_active:
            branch = self._measure_active(j, x, z, negative)
        else:
            branch = None
        if branch is not None:
            output ^= AffineRow(False, frozenset({branch}))
        slot = int(self.ops.slot[j])
        if slot >= 0:
            self.outputs[slot] = output

    def _promote(self, j: int, dormant_x: list[int]) -> None:
        """Move the rotation's dormant X support onto one free axis and activate it."""
        free = [q for q in range(self.width) if q not in self.active]
        if not free:
            raise RuntimeError(
                f"the plan needs more active axes than the schedule's width {self.width}"
            )
        in_array = [q for q in dormant_x if q < self.width]
        if in_array:
            axis = in_array[0]
        else:
            axis = free[0]
            self._apply("SWAP", axis, dormant_x[0])
        x, _, _ = self._virtual(j)
        for q in np.flatnonzero(x):
            if int(q) != axis and int(q) not in self.active:
                self._apply("CX", axis, int(q))
        self.active.add(axis)

    def _measure_dormant(
        self, j: int, dormant_x: list[int], negative: bool
    ) -> int | None:
        """Turn the measured Pauli into ``±X_d`` without touching the state, then H.

        Return the branch symbol, or None when no output or later op needs it.
        """
        d = dormant_x[0]
        for q in dormant_x[1:]:
            self._apply("CX", d, q)
        _, z, _ = self._virtual(j)
        if z[d]:
            self._apply("S_DAG", d)
        x, z, _ = self._virtual(j)
        for q in range(self.ops.num_lines):
            if q != d and (x[q] or z[q]):
                self._apply(_CONTROLLED_PAULI[(bool(x[q]), bool(z[q]))], d, q)
        x, z, after = self._virtual(j)
        assert x[d] and x.sum() == 1 and not z.any()
        self._apply("H", d)
        correction = self._correction(d)
        later = slice(j + 1, None)
        anti = anticommute(self.ops.x[later], self.ops.z[later], *correction)
        if self.ops.slot[j] < 0 and not anti.any():
            return None
        branch = self._branch(correction, after ^ negative)
        self.actions.append(MeasureDormant(branch))
        return branch

    def _measure_active(
        self, j: int, x: np.ndarray, z: np.ndarray, negative: bool
    ) -> int:
        """Measure on the array, then turn the Pauli into ``±Z_pivot`` and free it."""
        axes = sorted(self.active)
        x_mask, z_mask = _mask(x, axes), _mask(z, axes)
        x_support = [q for q in axes if x[q]]
        if x_support:
            pivot = x_support[0]
            for q in x_support[1:]:
                self._apply("CX", pivot, q)
            _, z_now, _ = self._virtual(j)
            for q in axes:
                if q != pivot and z_now[q]:
                    self._apply("CZ", pivot, q)
            _, z_now, _ = self._virtual(j)
            if z_now[pivot]:
                self._apply("S_DAG", pivot)
            self._apply("H", pivot)
        else:
            z_support = [q for q in axes if z[q]]
            pivot = z_support[0]
            for q in z_support[1:]:
                self._apply("CX", q, pivot)
        _, _, after = self._virtual(j)
        assert x_support or after == negative
        branch = self._branch(self._correction(pivot), after ^ negative)
        self.actions.append(MeasureActive(x_mask, z_mask, pivot, branch))
        self.active.remove(pivot)
        return branch
