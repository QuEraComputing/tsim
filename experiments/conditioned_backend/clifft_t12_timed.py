"""tricycle 12, all detectors postselected: clifft (width-scheduled) chunks until a deadline."""
import json, sys, time, clifft, numpy as np
text = open("/home/rafaelha/quera/circs/tricycle_factories/tricycle_12_3_2.stim").read()
first = clifft.sample(clifft.compile(text), 1, seed=0, threads=1)
nd, no = len(first.detectors[0]), len(first.observables[0])
pm = clifft.default_hir_pass_manager(); pm.add(clifft.ActiveWidthSchedulePass(beam_width=8))
prog = clifft.compile(text, postselection_mask=[1] * nd, hir_passes=pm)
assert prog.peak_active_width <= 20, prog.peak_active_width
threads, hours, base = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + 3600 * hours
c = 0
while time.time() < deadline:
    t = time.time(); passed = shots = 0; hist = np.zeros(1 << no, dtype=np.int64)
    while time.time() - t < 600 and time.time() < deadline:
        r = clifft.sample_survivors(prog, 2_000_000, seed=base * 10**7 + c * 10**4 + shots // 2_000_000,
                                    keep_records=True, threads=threads)
        passed += r.passed_shots
        shots += 2_000_000
        obs = np.asarray(r.observables, dtype=bool).reshape(-1, no)
        hist += np.bincount((obs * (1 << np.arange(no))).sum(axis=1), minlength=1 << no)
    print(json.dumps({"chunk": c, "shots": shots, "accepted": int(passed), "hist": hist.tolist(), "seconds": round(time.time() - t)}), flush=True)
    c += 1
