"""Li-Preskill-Xu BLP d3 Clifford proxy (CCZ omitted, CCZ noise as weight-5 Z errors), with stim."""
import sys, numpy as np, stim, multiprocessing as mp
PATH, SHOTS, PROCS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def work(seed):
    c = stim.Circuit(open(PATH).read())
    s = c.compile_detector_sampler(seed=seed)
    acc = fail = 0
    per = SHOTS // PROCS
    for _ in range(per // 1_000_000):
        d, o = s.sample(1_000_000, separate_observables=True)
        ok = ~d.any(axis=1)
        acc += int(ok.sum()); fail += int(o[ok].any(axis=1).sum())
    return acc, fail
if __name__ == "__main__":
    with mp.Pool(PROCS) as p:
        res = p.map(work, range(1000, 1000 + PROCS))
    acc = sum(a for a, _ in res); fail = sum(f for _, f in res); n = (SHOTS // PROCS // 1_000_000) * 1_000_000 * PROCS
    lfr = fail / acc
    print(f"{PATH}: shots {n} accept {acc/n:.5f} +- {np.sqrt(acc/n*(1-acc/n)/n):.5f} | postselected LFR {fail}/{acc} = {lfr:.3e} +- {np.sqrt(lfr*(1-lfr)/acc):.1e} | normalized 1-(1-LFR)^(1/9) = {1-(1-lfr)**(1/9):.3e}", flush=True)
