"""Acceptance of a Clifford circuit (all detectors postselected): tsim or stim."""
import sys, time, numpy as np
path, tool, shots, seed = sys.argv[1], sys.argv[2], int(float(sys.argv[3])), int(sys.argv[4])
text = open(path).read()
acc = 0; done = 0; t = time.time()
if tool == "stim":
    import stim
    s = stim.Circuit(text).compile_detector_sampler(seed=seed)
    while done < shots:
        n = min(10_000_000, shots - done)
        d = s.sample(n, bit_packed=True); acc += int((~d.any(axis=1)).sum()); done += n
else:
    import tsim
    c = tsim.Circuit(text)
    s = c.compile_detector_sampler(seed=seed)
    mask = np.ones(c.num_detectors, bool)
    while done < shots:
        n = min(10_000_000, shots - done)
        acc += len(s.sample(n, postselection_mask=mask)); done += n
print(f"{tool:6s} {done} shots accept {acc/done:.6f} +- {np.sqrt(acc/done*(1-acc/done)/done):.6f} ({time.time()-t:.0f}s)", flush=True)
