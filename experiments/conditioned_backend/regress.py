"""Bit-exact regression: hashes of postselected samples at a fixed seed."""
import sys, time, hashlib, numpy as np, tsim
sys.path.insert(0, "/tmp/cb")
from harness import CASES
for name in sys.argv[1:]:
    c = tsim.Circuit(open(CASES[name]).read())
    mask = np.ones(c.num_detectors, bool)
    s = c.compile_detector_sampler(seed=123, strategy="conditioned_statevector")
    t = time.time(); a = s.sample(300000, postselection_mask=mask); dt = time.time() - t
    print(name, len(a), hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12], f"{dt:.2f}s", flush=True)
