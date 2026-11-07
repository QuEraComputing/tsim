"""Clifft counterpart of ler.py (run in the tsim-fuzz venv, which has clifft 0.12).

Usage: clifft_ler.py NAME SHOTS MODE [SEED]   (MODE as in ler.py)
Raw detector/observable parities, as Tsim's detector sampler reports them.
"""
import json
import sys
import time

import numpy as np

import clifft

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
    "blp": T + "blp_27_3_3.stim",
}
MASKS = {"end2end_d5": C + "clifft-paper/msc_d5_end2end_p1e-3.postselection.npy"}

name, shots, mode = sys.argv[1], int(float(sys.argv[2])), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
text = open(CASES[name]).read()
MAX_WIDTH = 20  # NOTE: 2^20 amplitudes per state; wider programs are not sampled
probe = clifft.compile(text)
if probe.peak_active_width > MAX_WIDTH:
    print(json.dumps({"case": name, "strategy": "clifft", "skipped": f"peak_active_width {probe.peak_active_width}"}), flush=True)
    sys.exit(0)
first = clifft.sample(probe, 1, seed=0, threads=1)
nd, no = len(first.detectors[0]), len(first.observables[0])
mask = None
if mode in ("post", "dist"):
    mask = [1] * nd
elif mode in ("mask", "maskdist"):
    mask = [int(b) for b in np.load(MASKS[name]).astype(bool).reshape(-1)[:nd]]
prog = clifft.compile(text, postselection_mask=mask) if mask is not None else probe
decoder = None
if mode in ("mask", "decode"):
    import pymatching
    import stim

    keep = []
    for line in text.splitlines():
        word = line.strip().split("(")[0].split(" ")[0]
        if word in ("R_Z", "R_X", "R_Y", "U3"):
            continue
        if word in ("T", "T_DAG"):
            line = line.replace("T_DAG", "S_DAG", 1) if word == "T_DAG" else line.replace("T", "S", 1)
        keep.append(line)
    dem = stim.Circuit("\n".join(keep)).detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True, allow_gauge_detectors=True,
        ignore_decomposition_failures=True,
    )
    decoder = pymatching.Matching.from_detector_error_model(dem)

batch = 1_000_000
THREADS = 12
accepted = failures = 0
hist = np.zeros(1 << min(no, 12), dtype=np.int64)
t0 = time.time()
done = 0
rng = np.random.default_rng(seed)
while done < shots:
    n = min(batch, shots - done)
    s = int(rng.integers(2**62))
    if mask is not None:
        r = clifft.sample_survivors(prog, n, seed=s, keep_records=True, threads=THREADS)
    else:
        r = clifft.sample(prog, n, seed=s, threads=THREADS)
    dets = np.asarray(r.detectors, dtype=bool)
    obs = np.asarray(r.observables, dtype=bool).reshape(len(dets), no)
    accepted += len(obs)
    if mode == "post":
        failures += int(obs.any(axis=1).sum())
    elif mode in ("dist", "maskdist"):
        codes = (obs[:, :12] * (1 << np.arange(min(no, 12)))).sum(axis=1)
        hist += np.bincount(codes, minlength=len(hist))
    else:
        pred = decoder.decode_batch(dets.astype(np.uint8))
        failures += int((pred.astype(bool) != obs).any(axis=1).sum())
    done += n
out = {"case": name, "strategy": "clifft", "mode": mode, "shots": shots, "accepted": accepted,
       "failures": failures, "seconds": round(time.time() - t0, 1)}
if mode in ("dist", "maskdist"):
    out["hist"] = hist.tolist()
print(json.dumps(out), flush=True)
