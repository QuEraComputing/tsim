"""Two-sample comparison of conditioned_statevector against active_statevector."""
import sys, time, numpy as np, tsim
from scipy import stats as st
sys.path.insert(0, "/tmp/cb")
from harness import CASES
name, shots = sys.argv[1], int(sys.argv[2])
c = tsim.Circuit(open(CASES[name]).read())
out = {}
for strat, seed in (("active_statevector", 11), ("conditioned_statevector", 12)):
    s = c.compile_detector_sampler(seed=seed, strategy=strat)
    s.sample(64)
    t = time.time(); d, o = s.sample(shots, separate_observables=True); dt = time.time() - t
    out[strat] = (np.asarray(d, bool), np.asarray(o, bool), dt)
(d1, o1, t1), (d2, o2, t2) = out.values()
a1, a2 = ~d1.any(1), ~d2.any(1)
def z(p1, p2, n):
    p = (p1 + p2) / 2; se = np.sqrt(2 * p * (1 - p) / n) + 1e-12
    return (p1 - p2) / se
za = z(a1.mean(), a2.mean(), shots)
zd = np.abs(z(d1.mean(0), d2.mean(0), shots)).max() if d1.shape[1] else 0
# observable patterns among accepted shots (first 12 observables)
k = min(o1.shape[1], 12)
def hist(o, a):
    codes = (o[a][:, :k] * (1 << np.arange(k))).sum(1) if k else np.zeros(a.sum(), int)
    return np.bincount(codes, minlength=1 << k)
h1, h2 = hist(o1, a1), hist(o2, a2)
keep = (h1 + h2) >= 10
chi = st.chi2_contingency(np.vstack([h1[keep], h2[keep]]))[1] if keep.sum() > 1 else 1.0
print(f"{name:16s} shots {shots} | accept {a1.mean():.4f} vs {a2.mean():.4f} (z {za:+.2f}) | max |z| detector marginals {zd:.2f} over {d1.shape[1]} | obs-pattern chi2 p={chi:.3f} ({keep.sum()} bins) | time {1e6*t1/shots:.1f} vs {1e6*t2/shots:.1f} us/shot", flush=True)
