"""Exact outcome distributions of a conditioned component, one noise key at a time.

``key_leaves`` walks every outcome branch of the component's step plan for a
key, with its probability: a leaf is the raw outcomes of the recorded
measurements (per output slot) and the probability of reaching it. The number
of leaves grows as ``2^(random outcomes)``, so this is for analysis and tests,
not sampling. ``allowed`` prunes branches early: a branch survives while some
row of ``allowed`` agrees with every outcome it has fixed in the constrained
slots. ``config_distribution`` does the same for one configuration in numba,
keeping only the observable bits.
"""

from __future__ import annotations

import numpy as np

from tsim.statevector.conditioned import lane
from tsim.statevector.conditioned.component import MAX_WIDTH, ConditionedComponent
from tsim.statevector.conditioned.through import (
    finish_through,
    prepare_through,
    root_outcome,
)
from tsim.utils.kernels import jit


def key_flips(component: ConditionedComponent, key_bits: np.ndarray) -> np.ndarray:
    """Return which ops' signs differ from the reference key's, for key bits ``key_bits``."""
    coords = component.key.sign_coords.astype(np.int64)
    if coords.shape[1] == 0:
        return np.zeros(component.ops.num_ops, dtype=bool)
    return ((coords @ np.asarray(key_bits, dtype=np.int64)) & 1).astype(bool)


def key_leaves(
    component: ConditionedComponent,
    flipped: np.ndarray,
    *,
    allowed: np.ndarray | None = None,
    constrained: np.ndarray | None = None,
    max_leaves: int = 1 << 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the leaves of a key: raw outcome bits per slot ``(leaves, slots)`` and probabilities.

    Args:
        component: The component.
        flipped: Which ops' signs differ from the reference key's (``key_flips``).
        allowed: Rows of required raw outcomes over the slots; a branch whose fixed
            outcomes in the ``constrained`` slots match no row is dropped.
        constrained: The slots that ``allowed`` constrains.
        max_leaves: Raise beyond this many leaves.

    """
    ops, plan = component.ops, component.plan
    n, num_ops = ops.num_lines, ops.num_ops
    W = -(-n // 64)
    ref = np.where(ops.sign[:, 0] & 1, -ops.angle, ops.angle).astype(np.float64)
    theta = np.where(flipped, -ref, ref)
    num_slots = component.num_outputs
    if allowed is None:
        allowed = np.zeros((1, num_slots), dtype=bool)
        constrained = np.zeros(num_slots, dtype=bool)
    v = np.empty(2 * W + 1, dtype=np.uint64)
    leaves_bits: list[np.ndarray] = []
    leaves_prob: list[float] = []
    root = lane.new_state(n, W, num_ops)
    stack = [
        (
            0,
            1.0,
            np.zeros(num_slots, dtype=bool),
            np.ones(len(allowed), dtype=bool),
            root,
        )
    ]
    while stack:
        step, prob, bits, rows, state = stack.pop()
        frame, active, pos, amps, cache, cver = state
        alive = True
        while step < plan.num_steps:
            for r in plan.applied(step):
                amps = lane.rotate(
                    frame,
                    active,
                    pos,
                    amps,
                    cache,
                    cver,
                    int(r),
                    component.op_x,
                    component.op_z,
                    component.op_e,
                    float(theta[r]),
                    W,
                    MAX_WIDTH,
                )
            m = int(plan.measure[step])
            layer = plan.layer(step)
            if len(layer):
                p1, keep, phi, kv = prepare_through(
                    frame,
                    active,
                    pos,
                    amps,
                    cache,
                    cver,
                    component.op_x,
                    component.op_z,
                    component.op_e,
                    layer,
                    theta[layer].copy(),
                    m,
                    W,
                    MAX_WIDTH,
                )
                certain = len(phi) == 0
            else:
                kind, p1, d = lane.prepare_measure(
                    frame,
                    active,
                    pos,
                    amps,
                    cache,
                    cver,
                    m,
                    component.op_x,
                    component.op_z,
                    component.op_e,
                    W,
                    v,
                )
                certain = p1 < lane.CERTAIN or p1 > 1 - lane.CERTAIN
            slot = int(ops.slot[m])
            outcomes = [int(p1 > 0.5)] if certain else [0, 1]
            branches = []
            for bit in outcomes:
                keep_rows = rows
                if slot >= 0 and constrained[slot]:
                    keep_rows = rows & (allowed[:, slot] == bool(bit))
                    if not keep_rows.any():
                        continue
                pb = 1.0 if certain else (p1 if bit else 1.0 - p1)
                branches.append((bit, pb, keep_rows))
            if not branches:
                alive = False
                break
            # The outcome-1 branch (if both survive) goes on the stack with a copy.
            for bit, pb, keep_rows in branches[1:]:
                f2, a2, q2, c2, cv2 = (
                    frame.copy(),
                    active.copy(),
                    pos.copy(),
                    cache.copy(),
                    cver.copy(),
                )
                if len(layer):
                    amps2 = finish_through(a2, q2, keep, phi, kv, bit)
                else:
                    amps2 = lane.finish_measure(
                        f2, a2, q2, amps.copy(), kind, d, bit, v.copy(), W
                    )
                amps2 = lane.compress(f2, a2, q2, amps2, W)
                bits2 = bits.copy()
                if slot >= 0:
                    bits2[slot] = bit
                stack.append(
                    (
                        step + 1,
                        prob * pb,
                        bits2,
                        keep_rows,
                        (f2, a2, q2, amps2, c2, cv2),
                    )
                )
            bit, pb, rows = branches[0]
            if len(layer):
                if not certain:
                    amps = finish_through(active, pos, keep, phi, kv, bit)
            else:
                amps = lane.finish_measure(frame, active, pos, amps, kind, d, bit, v, W)
            amps = lane.compress(frame, active, pos, amps, W)
            if slot >= 0:
                bits[slot] = bit
            prob *= pb
            step += 1
        if alive:
            leaves_bits.append(bits)
            leaves_prob.append(prob)
            if len(leaves_bits) > max_leaves:
                raise ValueError(f"more than {max_leaves} leaves")
    if not leaves_bits:
        return np.zeros((0, num_slots), dtype=bool), np.zeros(0)
    return np.array(leaves_bits), np.array(leaves_prob)


@jit
def config_distribution(
    num_lines: int,
    op_x,
    op_z,
    op_e,
    theta,
    flipped,
    ref_theta,
    slot,
    required,
    obs_bit,
    measure,
    apply_start,
    apply_ops,
    layer_start,
    layer_ops,
    root_amp,
    num_obs: int,
    max_width: int,
):
    """Return the exact distribution of the observable bits of one noise configuration.

    The branches walk the step plan as ``runner.run_chunk`` does, but carry their
    probability instead of shots. A measurement whose slot has a ``required`` raw
    outcome (0 or 1; -1 for none) keeps only that branch, so the total of the
    returned distribution is the configuration's acceptance probability. A slot
    with ``obs_bit >= 0`` writes its raw outcome to that bit of the observable code.

    Returns ``2^num_obs`` probabilities indexed by the raw observable code.
    """
    num_words = (num_lines + 63) >> 6
    num_ops = theta.shape[0]
    num_steps = measure.shape[0]
    dist = np.zeros(1 << num_obs, dtype=np.float64)
    v = np.empty(2 * num_words + 1, dtype=np.uint64)
    no_amps = np.empty(0, dtype=np.complex128)
    frame, active, pos, amps, cache, cache_ver = lane.new_state(
        num_lines, num_words, num_ops
    )
    stack = [(0, 1.0, 0, frame, active, pos, amps, cache, cache_ver)]
    while len(stack):
        step, prob, code, frame, active, pos, amps, cache, cache_ver = stack.pop()
        while step < num_steps:
            for i in range(apply_start[step], apply_start[step + 1]):
                r = apply_ops[i]
                amps = lane.rotate(
                    frame,
                    active,
                    pos,
                    amps,
                    cache,
                    cache_ver,
                    r,
                    op_x,
                    op_z,
                    op_e,
                    theta[r],
                    num_words,
                    max_width,
                )
            m = measure[step]
            l0, l1 = layer_start[step], layer_start[step + 1]
            through = l1 > l0
            kind, d = 0, -1
            keep = active[:0].copy()
            phi, kv = no_amps, no_amps
            shortcut = -1
            if (
                through
                and root_amp[step] != 0
                and apply_start[step] == apply_start[step + 1]
                and amps.shape[0] == 1
                and frame[frame.shape[0] - 1, 0] == np.uint64(1)
            ):
                shortcut = root_outcome(
                    layer_ops[l0:l1],
                    flipped,
                    ref_theta,
                    op_x,
                    op_z,
                    op_e,
                    root_amp[step],
                    num_words,
                )
            if shortcut >= 0:
                p1 = float(shortcut)
                certain = True
            elif through:
                layer = layer_ops[l0:l1]
                p1, keep, phi, kv = prepare_through(
                    frame,
                    active,
                    pos,
                    amps,
                    cache,
                    cache_ver,
                    op_x,
                    op_z,
                    op_e,
                    layer,
                    theta[layer],
                    m,
                    num_words,
                    max_width,
                )
                certain = phi.shape[0] == 0
            else:
                kind, p1, d = lane.prepare_measure(
                    frame,
                    active,
                    pos,
                    amps,
                    cache,
                    cache_ver,
                    m,
                    op_x,
                    op_z,
                    op_e,
                    num_words,
                    v,
                )
                certain = p1 < lane.CERTAIN or p1 > 1.0 - lane.CERTAIN
            s = slot[m]
            need = required[s] if s >= 0 else -1
            lo_bit, hi_bit = 0, 1
            if certain:
                lo_bit = hi_bit = 1 if p1 > 0.5 else 0
            if need >= 0:
                if need < lo_bit or need > hi_bit:
                    prob = 0.0
                    break
                lo_bit = hi_bit = need
            ob = obs_bit[s] if s >= 0 else -1
            if hi_bit > lo_bit:
                # Both outcomes: outcome 1 on a copy, onto the stack.
                f1, a1, q1, c1, cv1 = (
                    frame.copy(),
                    active.copy(),
                    pos.copy(),
                    cache.copy(),
                    cache_ver.copy(),
                )
                if through:
                    amps1 = finish_through(a1, q1, keep, phi, kv, 1)
                else:
                    amps1 = lane.finish_measure(
                        f1, a1, q1, amps.copy(), kind, d, 1, v.copy(), num_words
                    )
                amps1 = lane.compress(f1, a1, q1, amps1, num_words)
                code1 = code | (1 << ob) if ob >= 0 else code
                stack.append((step + 1, prob * p1, code1, f1, a1, q1, amps1, c1, cv1))
                prob *= 1.0 - p1
                bit = 0
            else:
                bit = lo_bit
                if not certain:
                    prob *= p1 if bit else 1.0 - p1
            if through:
                if not certain:
                    amps = finish_through(active, pos, keep, phi, kv, bit)
            else:
                amps = lane.finish_measure(
                    frame, active, pos, amps, kind, d, bit, v, num_words
                )
            amps = lane.compress(frame, active, pos, amps, num_words)
            if ob >= 0 and bit:
                code |= 1 << ob
            step += 1
        if prob > 0.0:
            dist[code] += prob
    return dist
