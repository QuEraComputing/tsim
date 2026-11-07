import json, numpy as np
from scipy import stats
rows = [json.loads(l) for f in ("/tmp/cb/ler_all.jsonl", "/tmp/cb/clifft_all.jsonl") for l in open(f) if l.startswith("{")]
by = {}
for r in rows:
    if "skipped" not in r:
        by.setdefault(r["case"], {})[r["strategy"]] = r
def z2(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (k1 / n1 - k2 / n2) / se if se > 0 else 0.0
for case, d in by.items():
    if "clifft" not in d:
        continue
    c, k = d["conditioned_statevector"], d["clifft"]
    line = f"{case:15s} accept {c['accepted']/c['shots']:.5f} vs clifft {k['accepted']/k['shots']:.5f} (z {z2(c['accepted'], c['shots'], k['accepted'], k['shots']):+.2f})"
    if c["mode"] in ("post", "decode"):
        line += (f" | LER {c['failures']/c['accepted']:.3e} vs {k['failures']/k['accepted']:.3e} "
                 f"({c['failures']} vs {k['failures']}; z {z2(c['failures'], c['accepted'], k['failures'], k['accepted']):+.2f})")
    else:
        h1, h2 = np.array(c["hist"]), np.array(k["hist"])
        keep = (h1 + h2) > 0
        p = stats.chi2_contingency(np.vstack([h1[keep], h2[keep]]))[1]
        tv = 0.5 * np.abs(h1 / h1.sum() - h2 / h2.sum()).sum()
        line += f" | observables: chi2 p {p:.3f} over {keep.sum()} patterns, TV {tv:.1e}"
    print(line)
