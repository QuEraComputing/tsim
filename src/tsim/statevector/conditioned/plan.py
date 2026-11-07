"""The static step plan of a component.

With the measurement order fixed (the first measurement whose anticommuting
predecessors are done), which rotations a measurement applies and which it
measures through depend only on which ops anticommute and which angles are
T-like, not on signs, noise or outcomes. So the plan is computed once per
component, by the rules of the conditioned executor:

- a rotation becomes *pending* once every earlier op it anticommutes with is done;
- a measurement ``M`` measures through the largest set of pending T-like
  rotations that anticommute with ``M``, commute with each other and need no
  newer pending rotation to be applied first (the *layer*); every other pending
  rotation that ``M`` (or the layer) depends on is applied, oldest first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsim.statevector.pbc import MEASUREMENT, ROTATION, PauliOps, anticommute

# NOTE: the Gauss sums index a layer's subsets with one uint64.
MAX_LAYER = 64


@dataclass(frozen=True)
class StepPlan:
    """The steps of a component, one per measurement, in execution order.

    Attributes:
        measure: The measured op of each step, ``(S,)``.
        apply_start: Step ``s`` applies the rotation ops
            ``apply_ops[apply_start[s]:apply_start[s + 1]]``, in order, ``(S + 1,)``.
        apply_ops: The applied rotation ops.
        layer_start: Step ``s`` measures through ``layer_ops[layer_start[s]:layer_start[s + 1]]``.
        layer_ops: The layers' rotation ops.

    """

    measure: np.ndarray
    apply_start: np.ndarray
    apply_ops: np.ndarray
    layer_start: np.ndarray
    layer_ops: np.ndarray

    @property
    def num_steps(self) -> int:
        """Return the number of steps."""
        return len(self.measure)

    def layer(self, step: int) -> np.ndarray:
        """Return the layer of ``step``."""
        return self.layer_ops[self.layer_start[step] : self.layer_start[step + 1]]

    def applied(self, step: int) -> np.ndarray:
        """Return the rotations ``step`` applies."""
        return self.apply_ops[self.apply_start[step] : self.apply_start[step + 1]]


def is_t_like(theta: float) -> bool:
    """Return whether ``exp(-i theta P)`` is T-like: ``theta`` an odd multiple of ``pi/8``."""
    m = theta / (np.pi / 8)
    r = round(m)
    return abs(m - r) < 1e-9 and r % 2 == 1


def _plan_layer(anti: np.ndarray, t_like: np.ndarray, pending: list[int], m: int) -> tuple[list[int], list[int]]:
    """Return (rotations to apply, oldest first; the layer) for measuring ``m``.

    A pending rotation that anticommutes with ``m`` is in the layer if it is T-like
    and commutes with every newer layer member; otherwise it is applied, and so is
    every pending rotation that an applied one, or a layer member, needs first
    (an older one that anticommutes with it). Layer members that would need an
    applied rotation first are dropped from the layer, until nothing changes.
    """
    pos = {r: i for i, r in enumerate(pending)}
    hits = [r for r in pending if anti[r, m]]
    layer: list[int] = []
    for r in reversed(hits):
        if t_like[r] and not any(anti[r, t] for t in layer) and len(layer) < MAX_LAYER:
            layer.append(r)
    while True:
        in_layer = set(layer)
        apply = {r for r in hits if r not in in_layer}
        for r in pending:
            if r not in in_layer and r not in apply and any(anti[r, l] and pos[l] > pos[r] for l in layer):
                apply.add(r)
        grown = True
        while grown:
            grown = False
            for r in pending:
                if r in apply or r in in_layer:
                    continue
                if any(anti[r, t] and pos[t] > pos[r] for t in apply):
                    apply.add(r)
                    grown = True
        bad = {l for l in layer if any(anti[l, a] and pos[a] > pos[l] for a in apply)}
        if not bad:
            return [r for r in pending if r in apply], layer
        layer = [l for l in layer if l not in bad]


def step_plan(ops: PauliOps, *, measure_through: bool = True) -> StepPlan:
    """Return the step plan of ``ops``."""
    num_ops = ops.num_ops
    anti = anticommute(ops.x, ops.z, ops.x, ops.z)
    np.fill_diagonal(anti, False)
    earlier = [np.flatnonzero(anti[j, :j]) for j in range(num_ops)]
    t_like = np.array([ops.kind[j] == ROTATION and is_t_like(float(ops.angle[j])) for j in range(num_ops)])
    done = np.zeros(num_ops, dtype=bool)
    pending: list[int] = []
    measure, apply_ops, layer_ops = [], [], []
    apply_start, layer_start = [0], [0]
    while True:
        for j in range(num_ops):  # NOTE: rotations ready in op order, as the executor
            if not done[j] and ops.kind[j] == ROTATION and done[earlier[j]].all():
                done[j] = True
                pending.append(j)
        ready = [j for j in range(num_ops) if not done[j] and ops.kind[j] == MEASUREMENT and done[earlier[j]].all()]
        if not ready:
            break
        m = ready[0]
        if measure_through:
            apply, layer = _plan_layer(anti, t_like, pending, m)
        else:
            need: list[int] = []
            for r in reversed(pending):
                if anti[r, m] or any(anti[r, t] for t in need):
                    need.append(r)
            apply, layer = need[::-1], []
        done[m] = True
        applied = set(apply)
        pending = [r for r in pending if r not in applied]
        measure.append(m)
        apply_ops += apply
        layer_ops += layer
        apply_start.append(len(apply_ops))
        layer_start.append(len(layer_ops))
    assert done[ops.kind == MEASUREMENT].all()
    return StepPlan(
        measure=np.array(measure, dtype=np.int64),
        apply_start=np.array(apply_start, dtype=np.int64),
        apply_ops=np.array(apply_ops, dtype=np.int64),
        layer_start=np.array(layer_start, dtype=np.int64),
        layer_ops=np.array(layer_ops, dtype=np.int64),
    )
