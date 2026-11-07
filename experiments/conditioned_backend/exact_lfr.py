"""Exact-per-configuration acceptance and logical error of the real-CCZ BLP factory.

A shot's noise reaches the outputs as (a) the component key (rotation signs) and
(b) the output bits it flips (f-parts, after the constants). Shots are sampled
with Tsim's noise sampler and grouped by (key, output bits). For each key the
conditioned backend enumerates every outcome branch exactly (pruned to the
detector patterns that its configurations allow), which gives each
configuration's acceptance probability and its exact observable distribution
given acceptance.

Logical error: the total variation distance between a configuration's
observable distribution and the ideal one (the noiseless circuit's), averaged
over accepted shots. For a deterministic ideal (the Clifford proxy of the paper)
this is exactly the postselected logical failure rate.

Usage: exact_lfr.py CIRCUIT SHOTS [BATCH]
"""
import sys
import time
from collections import defaultdict

import numpy as np

import tsim
import tsim.statevector.shots as S
from tsim.noise.channels import _BLOCK_SHOTS
from tsim.statevector.conditioned import exact

path, total = sys.argv[1], int(sys.argv[2])
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
c = tsim.Circuit(open(path).read())
nd, no = c.num_detectors, c.num_observables
mask = np.ones(nd, bool)
s = c.compile_detector_sampler(seed=2026, strategy="conditioned_statevector")
sh = s._shots
layout = sh.layout
plan = sh.plan(mask)
(comp,) = layout.components
ks, kw = layout.key_starts[0], -(-len(layout.keys[0].rows) // 64)
ow = layout.output_words
slots_of_output = {int(g): i for i, g in enumerate(comp.output_indices)}
det_slots = np.array([slots_of_output.get(d, -1) for d in range(nd)])
obs_slots = np.array([slots_of_output.get(nd + o, -1) for o in range(no)])
assert (det_slots >= 0).sum() + 0 <= nd

# -- sample configurations ------------------------------------------------------
t0 = time.time()
configs = defaultdict(lambda: defaultdict(int))  # key bytes -> output bytes -> count
seeds = sh._channels.block_seeds(total)
done = 0
while done < total:
    n = min(batch, total - done)
    words = np.empty((n, layout.num_words), dtype=np.uint64)
    first_block = done // _BLOCK_SHOTS
    sh._channels.sample_words(words, layout.noise, seeds[first_block : first_block + -(-n // _BLOCK_SHOTS)])
    alive = np.empty(n, bool)
    S._screen(words, sh._constant(None), plan.screen, alive)
    rows = np.ascontiguousarray(np.concatenate([words[alive, ks : ks + kw], words[alive, :ow]], axis=1))
    u, counts = np.unique(rows, axis=0, return_counts=True)
    for r, cnt in zip(u, counts):
        configs[r[:kw].tobytes()][r[kw:].tobytes()] += int(cnt)
    done += n
num_configs = sum(len(v) for v in configs.values())
print(f"sampled {total} shots in {time.time()-t0:.0f}s: {sum(sum(v.values()) for v in configs.values())} pass the direct screen, "
      f"{len(configs)} keys, {num_configs} configurations", flush=True)


def out_bits(ob: bytes) -> np.ndarray:
    w = np.frombuffer(ob, dtype=np.uint64)
    return np.unpackbits(w.view(np.uint8), bitorder="little").astype(bool)


def key_bits(kb: bytes) -> np.ndarray:
    w = np.frombuffer(kb, dtype=np.uint64)
    return np.unpackbits(w.view(np.uint8), bitorder="little")[: len(layout.keys[0].rows)].astype(np.int64)


ref_theta = np.where(comp.ops.sign[:, 0] & 1, -comp.ops.angle, comp.ops.angle).astype(np.float64)
slot_of_op = comp.ops.slot.astype(np.int64)
obs_bit_of_slot = np.full(comp.num_outputs, -1, dtype=np.int64)
for o in range(no):
    if obs_slots[o] >= 0:
        obs_bit_of_slot[obs_slots[o]] = o
plan_c = comp.plan


def distribution(kb: bytes, outs: list[bytes]):
    """Per configuration: acceptance probability and {final observable code: prob}."""
    flipped = exact.key_flips(comp, key_bits(kb))
    theta = np.where(flipped, -ref_theta, ref_theta)
    result = []
    for ob in outs:
        b = out_bits(ob)
        required = np.full(comp.num_outputs, -1, dtype=np.int64)
        for d in range(nd):
            if det_slots[d] >= 0:
                required[det_slots[d]] = int(b[d])
        raw = exact.config_distribution(
            comp.num_lines, comp.op_x, comp.op_z, comp.op_e, theta, flipped, ref_theta,
            slot_of_op, required, obs_bit_of_slot, plan_c.measure, plan_c.apply_start,
            plan_c.apply_ops, plan_c.layer_start, plan_c.layer_ops, comp.root_amp, no, 26,
        )
        p_acc = float(raw.sum())
        shift = int((b[nd : nd + no] * (1 << np.arange(no))).sum())
        dist = {}
        if p_acc > 0:
            for code in np.flatnonzero(raw > 1e-15):
                dist[int(code) ^ shift] = raw[code] / p_acc
        result.append((p_acc, dist))
    return result


# Ideal: the noiseless circuit (key 0, outputs = constants only).
const_words = sh._constant(None)
zero_key = np.zeros(kw, dtype=np.uint64).tobytes()
pad = np.zeros(ow, dtype=np.uint64); pad[: len(const_words)] = const_words
(p_ideal, ideal), = distribution(zero_key, [pad.tobytes()])
assert abs(p_ideal - 1) < 1e-9, p_ideal
print(f"ideal observable distribution: {len(ideal)} patterns, max prob {max(ideal.values()):.4f}", flush=True)

t0 = time.time()
items = [(kb, dict(outs)) for kb, outs in configs.items()]


def shifted_tv(shift: int) -> float:
    """TV between the ideal distribution and the ideal shifted by an observable pattern."""
    keys = set(ideal) | {k ^ shift for k in ideal}
    return 0.5 * sum(abs(ideal.get(k ^ shift, 0.0) - ideal.get(k, 0.0)) for k in keys)


const_bits = out_bits(pad.tobytes())
const_det = const_bits[:nd]
const_obs_code = int((const_bits[nd : nd + no] * (1 << np.arange(no))).sum())


def work(part):
    acc = err = 0.0
    fails = []
    for kb, outs in part:
        if kb == zero_key:
            # The fault-free key: raw outcomes as in the ideal run, so a configuration
            # is accepted iff its detector bits are the constants, and its
            # observables are the ideal ones shifted by its flips.
            for ob, cnt in outs.items():
                b = out_bits(ob)
                if (b[:nd] != const_det).any():
                    continue
                shift = int((b[nd : nd + no] * (1 << np.arange(no))).sum()) ^ const_obs_code
                tv = shifted_tv(shift) if shift else 0.0
                acc += cnt
                err += cnt * tv
                if tv > 1e-9:
                    fails.append((cnt, tv, cnt))
            continue
        for ob, cnt in outs.items():
            ((p_acc, dist),) = distribution(kb, [ob])
            if p_acc == 0:
                continue
            tv = 0.5 * sum(abs(dist.get(k, 0.0) - ideal.get(k, 0.0)) for k in set(dist) | set(ideal))
            acc += cnt * p_acc
            err += cnt * p_acc * tv
            if tv > 1e-9:
                fails.append((cnt * p_acc, tv, cnt))
    return acc, err, fails


import multiprocessing as mp
procs = int(sys.argv[4]) if len(sys.argv) > 4 else 24
# big keys first, interleaved across workers
items.sort(key=lambda it: -sum(it[1].values()))
parts = [items[i::procs * 8] for i in range(procs * 8)]
with mp.get_context("fork").Pool(procs) as pool:
    results = pool.map(work, parts, chunksize=1)
acc_mass = sum(r[0] for r in results)
err_mass = sum(r[1] for r in results)
per_key = [f for r in results for f in r[2]]
acceptance = acc_mass / total
lfr = err_mass / acc_mass
n_fail_events = len(per_key)
print(f"RESULT {path}: shots {total} | acceptance {acceptance:.5f} +- {np.sqrt(acceptance*(1-acceptance)/total):.5f} | "
      f"postselected logical error (TV) {lfr:.3e} from {n_fail_events} failing configurations "
      f"(accepted mass {sum(f[0] for f in per_key):.1f} shots) | normalized 1-(1-L)^(1/9) {1-(1-lfr)**(1/9):.3e} | {time.time()-t0:.0f}s", flush=True)
