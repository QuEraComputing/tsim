import sys, time, numpy as np, tsim, numba
sys.path.insert(0, "/tmp/cb")
from harness import CASES
import tsim.statevector.conditioned.component as comp
import tsim.statevector.shots as shots_mod
c = tsim.Circuit(open(CASES["blp"]).read())
mask = np.ones(c.num_detectors, bool)
s = c.compile_detector_sampler(seed=3, strategy="conditioned_statevector")
s.sample(4000, postselection_mask=mask)
N = int(sys.argv[1])
orig = comp.run_chunk
tg = []
import tsim.statevector.conditioned.runner as R
orig_numba = R.run_chunk
def timed(*a, **k):
    t0 = time.perf_counter(); r = orig(*a, **k); tg.append(time.perf_counter() - t0); return r
shots_mod.conditioned.run_chunk = timed
for th in (1, 2, 4, 8, 12, 24):
    numba.set_num_threads(th); tg.clear()
    t = time.time(); s.sample(N, postselection_mask=mask); dt = time.time() - t
    print(f"threads {th:2d}: {1e6*dt/N:.3f} us/shot wall, component CPU {1e6*sum(tg)/N:.2f} us/shot, chunks {len(tg)}", flush=True)
