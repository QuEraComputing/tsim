"""tricycle 12, all detectors postselected: clifft (width-scheduled) acceptance and observables, chunk by chunk."""
import json, sys, time, clifft, numpy as np
text = open("/home/rafaelha/quera/circs/tricycle_factories/tricycle_12_3_2.stim").read()
first = clifft.sample(clifft.compile(text), 1, seed=0, threads=1)
nd, no = len(first.detectors[0]), len(first.observables[0])
pm = clifft.default_hir_pass_manager(); pm.add(clifft.ActiveWidthSchedulePass(beam_width=8))
prog = clifft.compile(text, postselection_mask=[1] * nd, hir_passes=pm)
assert prog.peak_active_width <= 20, prog.peak_active_width
chunk, chunks, base = int(float(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
for c in range(chunks):
    t = time.time(); passed = 0; hist = np.zeros(1 << no, dtype=np.int64)
    for i in range(chunk // 2_000_000):
        r = clifft.sample_survivors(prog, 2_000_000, seed=base * 100000 + c * 1000 + i, threads=12)
        passed += r.passed_shots
        obs = np.asarray(r.observables, dtype=bool).reshape(-1, no)
        hist += np.bincount((obs * (1 << np.arange(no))).sum(axis=1), minlength=1 << no)
    print(json.dumps({"chunk": c, "shots": chunk, "accepted": int(passed), "hist": hist.tolist(), "seconds": round(time.time() - t)}), flush=True)
