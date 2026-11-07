"""Runs a chunk of shots through a conditioned component.

Shots with the same noise key share a lane: one state that runs the component's
steps for all of them. At a random outcome every shot of the lane draws its own
uniform number (from a counter-based stream by the seed, its index and the step,
as in ``shared``), and the lane splits into one lane per outcome its shots drew.
A shot's outcomes so depend only on the seed, its index and its key.

Lanes run depth first: the outcome-1 lane is pushed on a stack with a copy of
the state, and the outcome-0 lane continues. Each shot XORs its outcomes into
its output bits; a shot whose postselected output reads 1 dies there.
"""

from __future__ import annotations

import numpy as np

from tsim.statevector.conditioned.lane import (
    CERTAIN,
    compress,
    finish_measure,
    new_state,
    prepare_measure,
    rotate,
    width_of,
)
from tsim.statevector.conditioned.paulis import popcount
from tsim.statevector.conditioned.through import (
    finish_through,
    prepare_through,
    root_outcome,
)
from tsim.utils.kernels import jit
from tsim.utils.rng import GOLDEN, splitmix64

_ONE = np.uint64(1)
_STEP = np.uint64(0xD1B54A32D192ED03)

# Indices of the stats array that run_chunk fills.
STAT_WIDTH, STAT_LANES, STAT_SPLITS, STAT_THROUGH, STAT_STEPS = 0, 1, 2, 3, 4
NUM_STATS = 5


@jit
def uniform(seed: np.uint64, shot: int, step: int) -> float:
    """Return shot ``shot``'s uniform number in [0, 1) at step ``step``."""
    state = seed ^ (np.uint64(shot) * GOLDEN) ^ (np.uint64(step + 1) * _STEP)
    _, z = splitmix64(state)
    return float(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@jit
def _record(
    words, alive, order, lo: int, hi: int, out_bit: int, post: bool, bit: int, bits
) -> int:
    """XOR each shot's outcome into its output and drop shots that postselection kills.

    ``bits`` holds per-shot outcomes (indexed like ``order``), or is empty when every
    shot has ``bit``. Shots of ``order[lo:hi]`` that stay alive move to the front
    of the range; returns its new end.
    """
    end = lo
    for i in range(lo, hi):
        s = order[i]
        b = bit if bits.shape[0] == 0 else bits[i]
        keep = True
        if out_bit >= 0:
            w, sh = out_bit >> 6, np.uint64(out_bit & 63)
            words[s, w] ^= np.uint64(b) << sh
            if post and (words[s, w] >> sh) & _ONE:
                alive[s] = False
                keep = False
        if keep:
            order[end] = s
            if bits.shape[0]:
                bits[end] = b
            end += 1
    return end


@jit
def _split(order, bits, lo: int, hi: int) -> int:
    """Move the shots of ``order[lo:hi]`` with outcome 0 to the front; return the first 1."""
    i, j = lo, hi - 1
    while i <= j:
        if bits[i] == 0:
            i += 1
        else:
            order[i], order[j] = order[j], order[i]
            bits[i], bits[j] = bits[j], bits[i]
            j -= 1
    return i


@jit
def group_by_key(words, alive, key_start: int, key_words: int):
    """Return the alive shots sorted so that equal keys are adjacent, and the group starts.

    The shots with the all-zero key (usually most of them: no fault reaches the
    component) form the first group, unsorted. The others are sorted by a hash of
    their key; a run of equal hashes is checked word by word, and a shot whose key
    differs from its run's first (a collision) gets a group of its own.
    """
    num_zero = 0
    num_other = 0
    for s in range(words.shape[0]):
        if not alive[s]:
            continue
        zero = True
        for w in range(key_words):
            if words[s, key_start + w] != 0:
                zero = False
                break
        if zero:
            num_zero += 1
        else:
            num_other += 1
    order = np.empty(num_zero + num_other, dtype=np.int64)
    others = np.empty(num_other, dtype=np.int64)
    hashes = np.empty(num_other, dtype=np.uint64)
    iz, io = 0, 0
    for s in range(words.shape[0]):
        if not alive[s]:
            continue
        h = np.uint64(0x243F6A8885A308D3)
        zero = True
        for w in range(key_words):
            word = words[s, key_start + w]
            if word != 0:
                zero = False
            _, h = splitmix64(h ^ word)
        if zero:
            order[iz] = s
            iz += 1
        else:
            others[io] = s
            hashes[io] = h
            io += 1
    perm = np.argsort(hashes, kind="mergesort")
    starts = np.empty(num_other + 2, dtype=np.int64)
    num = 0
    if num_zero:
        starts[0] = 0
        num = 1
    first = -1
    for i in range(num_other):
        s = others[perm[i]]
        order[num_zero + i] = s
        new = i == 0 or hashes[perm[i]] != hashes[perm[i - 1]]
        if not new:
            for w in range(key_words):
                if words[s, key_start + w] != words[first, key_start + w]:
                    new = True
                    break
        if new:
            starts[num] = num_zero + i
            num += 1
            first = s
    starts[num] = num_zero + num_other
    return order, starts[: num + 1].copy()


@jit
def run_chunk(
    words,
    alive,
    first_shot: int,
    seed: np.uint64,
    order,
    group_start,
    key_start: int,
    key_words: int,
    num_lines: int,
    op_x,
    op_z,
    op_e,
    angle,
    sign_const,
    sign_key,
    out_bit,
    post,
    measure,
    apply_start,
    apply_ops,
    layer_start,
    layer_ops,
    root_amp,
    max_width: int,
    stats,
    track,
    path_prob,
):
    """Run the shots ``order`` (grouped by key: ``group_start``) through every step.

    Args:
        words: The chunk's shot words; outputs are XOR-ed in.
        alive: Whether each shot is alive; postselection clears it.
        first_shot: Index of the chunk's first shot in the call.
        seed: The component's seed for this call.
        order: The alive shots, grouped by key.
        group_start: Group ``g`` is ``order[group_start[g]:group_start[g + 1]]``.
        key_start: First shot word of the key.
        key_words: Number of words of the key.
        num_lines: Number of qubit lines n.
        op_x: Each op's X part as bit words.
        op_z: Each op's Z part as bit words.
        op_e: Each op's phase exponent (its number of Y), mod 4.
        angle: Each rotation's angle (``exp(-i angle P)`` for sign +).
        sign_const: Each op's sign constant.
        sign_key: Each op's sign as key bits, ``(N, key_words)``.
        out_bit: Each measurement's output bit in the words, or -1.
        post: Whether each measurement's output is postselected.
        measure: The measured op of each step (``plan.StepPlan``).
        apply_start: Offsets of each step's applied rotations.
        apply_ops: The applied rotations.
        layer_start: Offsets of each step's measure-through layer.
        layer_ops: The layers' rotations.
        root_amp: Per step, the reference key's certain outcome (+1 or -1) at the
            root state, or 0 (see ``reference_prefix``).
        max_width: Most active qubits a lane may hold.
        stats: Receives the widest lane, lane count, splits, measure-throughs, steps.
        track: Whether each op's outcome probability goes into ``path_prob``
            (empty: none). For analysis.
        path_prob: Per shot of ``words``, multiplied by the probability of each
            tracked outcome it draws (empty: none).

    """
    num_words = (num_lines + 63) >> 6
    num_ops = angle.shape[0]
    num_steps = measure.shape[0]
    theta = np.empty(num_ops, dtype=np.float64)
    ref_theta = np.empty(num_ops, dtype=np.float64)
    flipped = np.zeros(num_ops, dtype=np.bool_)
    for r in range(num_ops):
        ref_theta[r] = -angle[r] if sign_const[r] & 1 else angle[r]
    bits = np.empty(order.shape[0], dtype=np.int64)
    no_bits = np.empty(0, dtype=np.int64)
    no_amps = np.empty(0, dtype=np.complex128)
    v = np.empty(2 * num_words + 1, dtype=np.uint64)
    for g in range(group_start.shape[0] - 1):
        lo, hi = group_start[g], group_start[g + 1]
        if hi == lo:
            continue
        # Concrete rotation angles from the group's key.
        s0 = order[lo]
        for r in range(num_ops):
            parity = 0
            for w in range(key_words):
                parity += popcount(sign_key[r, w] & words[s0, key_start + w])
            flipped[r] = parity & 1
            theta[r] = -ref_theta[r] if flipped[r] else ref_theta[r]
        frame, active, pos, amps, cache, cache_ver = new_state(
            num_lines, num_words, num_ops
        )
        stack = [(0, lo, hi, frame, active, pos, amps, cache, cache_ver)]
        stats[STAT_LANES] += 1
        while len(stack):
            step, lo, hi, frame, active, pos, amps, cache, cache_ver = stack.pop()
            while step < num_steps and hi > lo:
                for i in range(apply_start[step], apply_start[step + 1]):
                    r = apply_ops[i]
                    amps = rotate(
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
                kv = no_amps
                shortcut = -1
                if (
                    through
                    and root_amp[step] != 0
                    and apply_start[step] == apply_start[step + 1]
                    and amps.shape[0] == 1
                    and frame[frame.shape[0] - 1, 0] == _ONE
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
                    phi = no_amps
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
                    stats[STAT_THROUGH] += 1
                    certain = phi.shape[0] == 0
                if not through:
                    kind, p1, d = prepare_measure(
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
                    certain = p1 < CERTAIN or p1 > 1.0 - CERTAIN
                stats[STAT_STEPS] += 1
                if certain:
                    bit = 1 if p1 > 0.5 else 0
                    if not through:
                        amps = finish_measure(
                            frame, active, pos, amps, kind, d, bit, v, num_words
                        )
                    hi = _record(
                        words, alive, order, lo, hi, out_bit[m], post[m], bit, no_bits
                    )
                else:
                    for i in range(lo, hi):
                        bits[i] = (
                            1 if uniform(seed, first_shot + order[i], step) < p1 else 0
                        )
                    mid = _split(order, bits, lo, hi)
                    if path_prob.shape[0] and track[m]:
                        for i in range(lo, mid):
                            path_prob[order[i]] *= 1.0 - p1
                        for i in range(mid, hi):
                            path_prob[order[i]] *= p1
                    if mid < hi:
                        # The outcome-1 lane, on a copy of the state.
                        f1, a1, q1 = frame.copy(), active.copy(), pos.copy()
                        c1, cv1 = cache.copy(), cache_ver.copy()
                        if through:
                            amps1 = finish_through(a1, q1, keep, phi, kv, 1)
                        else:
                            v1 = v.copy()
                            amps1 = finish_measure(
                                f1, a1, q1, amps.copy(), kind, d, 1, v1, num_words
                            )
                        end1 = _record(
                            words,
                            alive,
                            order,
                            mid,
                            hi,
                            out_bit[m],
                            post[m],
                            1,
                            no_bits,
                        )
                        if end1 > mid:
                            amps1 = compress(f1, a1, q1, amps1, num_words)
                            if width_of(amps1) > stats[STAT_WIDTH]:
                                stats[STAT_WIDTH] = width_of(amps1)
                            if mid > lo:
                                stats[STAT_LANES] += 1
                                stats[STAT_SPLITS] += 1
                                stack.append(
                                    (step + 1, mid, end1, f1, a1, q1, amps1, c1, cv1)
                                )
                            else:
                                frame, active, pos, amps, cache, cache_ver = (
                                    f1,
                                    a1,
                                    q1,
                                    amps1,
                                    c1,
                                    cv1,
                                )
                                lo, hi = mid, end1
                                step += 1
                                continue
                    if mid == lo:
                        hi = lo
                        break
                    if through:
                        amps = finish_through(active, pos, keep, phi, kv, 0)
                    else:
                        amps = finish_measure(
                            frame, active, pos, amps, kind, d, 0, v, num_words
                        )
                    hi = _record(
                        words, alive, order, lo, mid, out_bit[m], post[m], 0, no_bits
                    )
                amps = compress(frame, active, pos, amps, num_words)
                if width_of(amps) > stats[STAT_WIDTH]:
                    stats[STAT_WIDTH] = width_of(amps)
                step += 1


@jit
def reference_prefix(
    num_lines: int,
    op_x,
    op_z,
    op_e,
    angle,
    sign_const,
    measure,
    apply_start,
    layer_start,
    layer_ops,
):
    """Return per step the reference key's outcome amplitude at the root state, or 0.

    The reference key (all key bits 0) is run from ``|0...0>`` while it stays at the
    root: no active qubit and an untouched frame. Each measure-through step there
    with a certain outcome records ``<0|K0|0> = +-1``; the rest record 0.
    """
    num_words = (num_lines + 63) >> 6
    num_ops = angle.shape[0]
    out = np.zeros(measure.shape[0], dtype=np.float64)
    theta = np.empty(num_ops, dtype=np.float64)
    for r in range(num_ops):
        theta[r] = -angle[r] if sign_const[r] & 1 else angle[r]
    frame, active, pos, amps, cache, cache_ver = new_state(
        num_lines, num_words, num_ops
    )
    for step in range(measure.shape[0]):
        l0, l1 = layer_start[step], layer_start[step + 1]
        if l1 == l0 or apply_start[step] != apply_start[step + 1]:
            break
        layer = layer_ops[l0:l1]
        p1, _, phi, _ = prepare_through(
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
            measure[step],
            num_words,
            64,
        )
        if phi.shape[0] != 0 or frame[frame.shape[0] - 1, 0] != _ONE:
            break
        out[step] = 1.0 if p1 < 0.5 else -1.0
    return out
