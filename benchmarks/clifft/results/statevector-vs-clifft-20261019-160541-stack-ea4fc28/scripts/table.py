import json
import sys
from collections import defaultdict

D = sys.argv[1]


def load(name):
    try:
        return [json.loads(line) for line in open(f"{D}/{name}")]
    except FileNotFoundError:
        return []


best = defaultdict(float)
width = {}
k_max = {}
compile_s = {}
for r in load("tsim-all.jsonl") + load("tsim-all-large.jsonl") + load("tsim-1.jsonl"):
    kind = "one" if r["configuration"] == "one lane per shot" else "shared"
    key = (r["workload"], "tsim", kind, r["threads"])
    best[key] = max(best[key], r["shots_per_second"])
    width[r["workload"]] = r["width"]
for passes, files in (("default", ("clifft.jsonl",)), ("all", ("clifft-allpasses.jsonl", "clifft-all-large.jsonl"))):
    for f in files:
        for r in load(f):
            key = (r["workload"], "clifft", passes, r["threads"])
            best[key] = max(best[key], r["shots_per_second"])
            if "width" in r:
                k_max[(r["workload"], passes)] = r["width"]
            if "compile_seconds" in r:
                compile_s[(r["workload"], passes)] = r["compile_seconds"]

names = list(dict.fromkeys(r["workload"] for r in load("tsim-all.jsonl")))


def fmt(x):
    if not x:
        return "—"
    for unit, scale in (("M", 1e6), ("k", 1e3)):
        if x >= scale:
            return f"{x / scale:.3g}{unit}"
    return f"{x:.3g}"


for threads in ("1", "all"):
    print(f"\n### {threads} thread{'s' if threads == 'all' else ''}\n")
    print("| circuit | Tsim width | Clifft k_max default → all | Tsim one lane | Tsim shared | Clifft default | Clifft all passes | Tsim best / Clifft all |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for n in names:
        t1, t2 = best[(n, "tsim", "one", threads)], best[(n, "tsim", "shared", threads)]
        c1, c2 = best[(n, "clifft", "default", threads)], best[(n, "clifft", "all", threads)]
        ratio = f"{max(t1, t2) / c2:.3g}x" if c2 else "—"
        print(f"| {n} | {width[n]} | {k_max.get((n, 'default'))} → {k_max.get((n, 'all'))} | {fmt(t1)} | {fmt(t2)} | {fmt(c1)} | {fmt(c2)} | {ratio} |")
print("\ncompile s (default → all):", {n: f"{compile_s.get((n, 'default'), 0):.2f} → {compile_s.get((n, 'all'), 0):.2f}" for n in names})
