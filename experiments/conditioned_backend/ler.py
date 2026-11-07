"""Postselected / decoded logical error rates of one circuit under one strategy.

Usage: ler.py NAME STRATEGY SHOTS MODE [SEED]
MODE: post (postselect every detector, failure = any observable 1),
      mask (postselect the circuit's mask, decode the rest),
      decode (no postselection, decode every detector),
      dist (postselect every detector, observables random ideally: histogram).
Prints one JSON line.
"""
import json
import sys
import time

import numpy as np
import stim

import tsim

C = "/home/rafaelha/quera/tsim-conditioned/experiments/conditioned/circuits/"
T = "/home/rafaelha/quera/circs/tricycle_factories/"
CASES = {
    "cultivation_d3": C + "clifft-bench/msc_d3_inject_cultivate_p1e-3.stim",
    "cultivation_d5": C + "clifft-bench/msc_d5_inject_cultivate_p1e-3.stim",
    "cultivation_d7": C + "symft/MSC_circuit_d7_p0.0005.stim",
    "end2end_d5": C + "clifft-paper/msc_d5_end2end_p1e-3.stim",
    "coherent_d3_r1": C + "clifft-bench/coherent_d3_r1.stim",
    "coherent_d3_r3": C + "clifft-bench/coherent_d3_r3.stim",
    "coherent_d5_r1": C + "clifft-bench/coherent_d5_r1.stim",
    "distillation": C + "clifft-bench/distillation.stim",
    "tricycle_6": T + "tricycle_6_3_2.stim",
    "tricycle_9": T + "tricycle_9_3_2.stim",
    "tricycle_12": T + "tricycle_12_3_2.stim",
}
MASKS = {"end2end_d5": C + "clifft-paper/msc_d5_end2end_p1e-3.postselection.npy"}

name, strategy, shots, mode = sys.argv[1], sys.argv[2], int(float(sys.argv[3])), sys.argv[4]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
text = open(CASES[name]).read()
circuit = tsim.Circuit(text)
nd, no = circuit.num_detectors, circuit.num_observables


def clifford_dem() -> stim.DetectorErrorModel:
    """DEM of the circuit with T -> S and other non-Clifford rotations dropped (decoder only)."""
    keep = []
    for line in text.splitlines():
        word = line.strip().split("(")[0].split(" ")[0]
        if word in ("R_Z", "R_X", "R_Y", "U3"):
            continue
        if word in ("T", "T_DAG"):
            # NOTE: the S proxy of a T gate keeps cultivation's checks deterministic.
            line = line.replace("T_DAG", "S_DAG", 1) if word == "T_DAG" else line.replace("T", "S", 1)
        keep.append(line)
    c = stim.Circuit("\n".join(keep))
    return c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, allow_gauge_detectors=True, ignore_decomposition_failures=True)


mask = None
if mode in ("post", "dist"):
    mask = np.ones(nd, bool)
elif mode in ("mask", "maskdist"):
    mask = np.load(MASKS[name]).astype(bool).reshape(-1)[:nd]
decoder = None
decode_cols = None
if mode in ("mask", "decode"):
    import pymatching

    dem = clifford_dem()
    decode_cols = np.flatnonzero(~mask) if mask is not None else np.arange(nd)
    decoder = pymatching.Matching.from_detector_error_model(dem)

sampler = circuit.compile_detector_sampler(seed=seed, strategy=strategy)
batch = 10_000_000 if mode in ("post", "dist", "maskdist") else 2_000_000
accepted = failures = 0
hist = np.zeros(1 << min(no, 12), dtype=np.int64)
t0 = time.time()
done = 0
while done < shots:
    n = min(batch, shots - done)
    dets, obs = sampler.sample(n, postselection_mask=mask, separate_observables=True)
    dets = np.asarray(dets, bool)
    obs = np.asarray(obs, bool)
    accepted += len(obs)
    if mode == "post":
        failures += int(obs.any(axis=1).sum())
    elif mode in ("dist", "maskdist"):
        codes = (obs[:, :12] * (1 << np.arange(min(no, 12)))).sum(axis=1)
        hist += np.bincount(codes, minlength=len(hist))
    else:
        full = np.zeros((len(dets), nd), dtype=np.uint8)
        full[:, decode_cols] = dets[:, decode_cols]
        pred = decoder.decode_batch(full)
        failures += int((pred.astype(bool) != obs).any(axis=1).sum())
    done += n
out = {"case": name, "strategy": strategy, "mode": mode, "shots": shots, "accepted": accepted,
       "failures": failures, "seconds": round(time.time() - t0, 1)}
if mode in ("dist", "maskdist"):
    out["hist"] = hist.tolist()
print(json.dumps(out), flush=True)
