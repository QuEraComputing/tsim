import json, numpy as np
from scipy import stats
cond = [json.loads(l) for l in open("/tmp/cb/cond_t12_more.jsonl")]
cond += [json.loads(l) for l in open("/tmp/cb/ler_all.jsonl") if '"tricycle_12"' in l and "conditioned" in l]
ca = sum(r["accepted"] for r in cond) + 78509706; cn = sum(r["shots"] for r in cond) + 100_000_000
ch = sum(np.array(r["hist"]) for r in cond)
old = [json.loads(l) for f in ("/tmp/cb/clifft_all.jsonl", "/tmp/cb/clifft_t12b.jsonl", "/tmp/cb/clifft_t12c.jsonl") for l in open(f) if '"tricycle_12"' in l]
new = [json.loads(l) for l in open("/tmp/cb/clifft_t12_more.jsonl")]
timed = [json.loads(l) for l in open("/tmp/cb/clifft_t12_timed.jsonl")]
def show(label, rows):
    ka = sum(r["accepted"] for r in rows); kn = sum(r["shots"] for r in rows)
    p1, p2 = ca / cn, ka / kn
    se = np.sqrt(p1 * (1 - p1) / cn + p2 * (1 - p2) / kn)
    with_hist = [r for r in rows if sum(r["hist"])]  # NOTE: chunks without keep_records have none
    chi = float("nan")
    if with_hist:
        kh = sum(np.array(r["hist"]) for r in with_hist)
        keep = (ch + kh) > 0
        chi = stats.chi2_contingency(np.vstack([ch[keep], kh[keep]]))[1]
    print(f"{label:28s} clifft {kn:.2e} shots accept {p2:.6f} | conditioned {cn:.2e} accept {p1:.6f} | diff {p1-p2:+.2e} +- {se:.1e} (z {(p1-p2)/se:+.2f}) | obs chi2 p {chi:.3f}")
print("chunks:", [round(r["accepted"] / r["shots"], 6) for r in new])
show("new clifft chunks only", new)
show("2-hour run only", timed)
show("all clifft", old + new + timed)
