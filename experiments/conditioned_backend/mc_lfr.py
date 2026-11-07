"""Postselected logical error of a real-CCZ factory, estimated at sampling speed.

For an accepted shot with noise configuration c and observables o, the backend
knows D_c(o) exactly: the product of the probabilities of the observable
outcomes along the shot's path (every detector step comes before the
observable steps, so it is conditioned on acceptance). With D_0 the ideal
observable distribution (noiseless circuit, exact), the weight

    w = max(0, 1 - D_0(o) / D_c(o))

has mean TV(D_c, D_0) over o ~ D_c, so the mean of w over accepted shots is the
postselected logical error E[TV(D_c, D_0) | accepted]. For a deterministic
ideal (the Clifford proxy) w is 1 exactly when the observables are flipped,
which is the usual postselected logical failure rate.

Usage: mc_lfr.py CIRCUIT SHOTS [BATCH]
"""
import sys
import time

import numba
import numpy as np

import tsim
import tsim.statevector.conditioned.runner as R
import tsim.statevector.shots as S
from tsim.statevector.conditioned import exact
from tsim.statevector.conditioned.component import MAX_WIDTH
from tsim.utils.kernels import run_tasks

path, total = sys.argv[1], int(sys.argv[2])
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 22
c = tsim.Circuit(open(path).read())
nd, no = c.num_detectors, c.num_observables
mask = np.ones(nd, bool)
s = c.compile_detector_sampler(seed=int(time.time()) % 100000, strategy="conditioned_statevector")
sh = s._shots
layout = sh.layout
plan = sh.plan(mask)
(comp,), (program,) = layout.components, plan.programs
ops, cp = comp.ops, comp.plan
slot_of_output = {int(g): i for i, g in enumerate(comp.output_indices)}
obs_slots = np.array([slot_of_output.get(nd + o, -1) for o in range(no)])
assert (obs_slots >= 0).all(), "every observable must come from the component"
kinds = ["O" if comp.output_indices[ops.slot[m]] >= nd else "d" for m in cp.measure if ops.slot[m] >= 0]
assert "".join(kinds).rstrip("O").count("O") == 0, "detector steps must come before observable steps"
track = np.zeros(ops.num_ops, dtype=np.bool_)
for m in range(ops.num_ops):
    if ops.slot[m] >= 0 and comp.output_indices[ops.slot[m]] >= nd:
        track[m] = True

# Ideal distribution in final codes.
ref = np.where(ops.sign[:, 0] & 1, -ops.angle, ops.angle).astype(np.float64)
const = sh._constant(None)
const_bits = np.unpackbits(const.view(np.uint8), bitorder="little").astype(bool)
required = np.full(comp.num_outputs, -1, dtype=np.int64)
obs_bit = np.full(comp.num_outputs, -1, dtype=np.int64)
for i, g in enumerate(comp.output_indices):
    if g < nd:
        required[i] = int(const_bits[g])  # final 0 needs raw = constant
    else:
        obs_bit[i] = g - nd
raw0 = exact.config_distribution(
    ops.num_lines, comp.op_x, comp.op_z, comp.op_e, ref, np.zeros(ops.num_ops, bool), ref,
    ops.slot.astype(np.int64), required, obs_bit, cp.measure, cp.apply_start, cp.apply_ops,
    cp.layer_start, cp.layer_ops, comp.root_amp, no, MAX_WIDTH,
)
assert abs(raw0.sum() - 1) < 1e-9, raw0.sum()
const_obs = int((const_bits[nd : nd + no] * (1 << np.arange(no))).sum())
ideal = np.zeros(1 << no)
for code in range(1 << no):
    ideal[code ^ const_obs] = raw0[code]
print(f"ideal distribution: {int((ideal > 0).sum())} patterns, min {ideal[ideal > 0].min():.4g}, max {ideal.max():.4g}", flush=True)

threads = numba.get_num_threads()
done = accepted = 0
w_sum = w_sq = 0.0
events = 0
t0 = time.time()
while done < total:
    n = min(batch, total - done)
    words = np.empty((n, layout.num_words), dtype=np.uint64)
    blocks = sh._channels.block_seeds(n)
    sh._channels.sample_words(words, layout.noise, blocks)
    alive = np.empty(n, bool)
    S._screen(words, const, plan.screen, alive)
    order, starts = R.group_by_key(words, alive, program.key_start, program.key_words)
    path_prob = np.ones(n)
    pieces = []
    for g in range(len(starts) - 1):
        lo, hi = int(starts[g]), int(starts[g + 1])
        k = -(-(hi - lo) // (1 << 17))
        pieces += [(lo + (hi - lo) * i // k, lo + (hi - lo) * (i + 1) // k) for i in range(k)]
    seed = np.uint64(np.random.default_rng().integers(0, 2**63))
    tasks = [pieces[i :: 8 * threads] for i in range(min(len(pieces), 8 * threads))]

    def task(part):
        st = np.zeros(R.NUM_STATS, dtype=np.int64)
        sub = np.concatenate([order[a:b] for a, b in part])
        sub_starts = np.concatenate([[0], np.cumsum([b - a for a, b in part])]).astype(np.int64)
        R.run_chunk(
            words, alive, 0, seed, sub, sub_starts, program.key_start, program.key_words,
            ops.num_lines, comp.op_x, comp.op_z, comp.op_e, ops.angle.astype(np.float64),
            ops.sign[:, 0].astype(np.int64), program.sign_key, program.out_bit, program.post,
            cp.measure, cp.apply_start, cp.apply_ops, cp.layer_start, cp.layer_ops,
            comp.root_amp, MAX_WIDTH, st, track, path_prob,
        )

    run_tasks(task, tasks, threads)
    acc = np.flatnonzero(alive)
    bits = np.unpackbits(words[acc, : layout.output_words].view(np.uint8), axis=1, bitorder="little")[:, nd : nd + no]
    codes = (bits.astype(np.int64) * (1 << np.arange(no))).sum(axis=1)
    w = np.maximum(0.0, 1.0 - ideal[codes] / path_prob[acc])
    w[w < 1e-12] = 0.0
    accepted += len(acc)
    w_sum += w.sum()
    w_sq += (w * w).sum()
    events += int((w > 0).sum())
    done += n
    if done % (batch * 25) == 0 or done == total:
        lfr = w_sum / accepted
        err = np.sqrt(max(w_sq / accepted - lfr**2, 0) / accepted)
        print(f"  {done} shots ({time.time()-t0:.0f}s): acceptance {accepted/done:.5f}, logical error {lfr:.3e} +- {err:.1e} ({events} shots with w > 0)", flush=True)
lfr = w_sum / accepted
err = np.sqrt(max(w_sq / accepted - lfr**2, 0) / accepted)
acc_rate = accepted / done
print(f"RESULT {path}: shots {done} | acceptance {acc_rate:.5f} +- {np.sqrt(acc_rate*(1-acc_rate)/done):.5f} | "
      f"postselected logical error {lfr:.3e} +- {err:.1e} ({events} shots with w > 0) | normalized 1-(1-L)^(1/{no}) {1-(1-lfr)**(1/no):.3e} | {time.time()-t0:.0f}s", flush=True)
