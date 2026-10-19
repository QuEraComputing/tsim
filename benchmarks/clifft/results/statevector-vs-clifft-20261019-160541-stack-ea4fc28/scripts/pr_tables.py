"""Print PR 199's performance section: a collapsible table per setting.

Usage: pr_tables.py RESULTS_DIR PR_DIR, where RESULTS_DIR holds the Clifft runs
and the base runs, and PR_DIR the runs of this PR's final code.
"""

import json
import sys
from collections import defaultdict

RESULTS, PR = sys.argv[1], sys.argv[2]
LABELS = {
    "surface_d7_r7": "surface d7r7",
    "coherent_d3_r1": "coherent d3r1",
    "coherent_d5_r1": "coherent d5r1",
    "distillation": "distillation",
    "cultivation_d3": "cultivation d3",
    "coherent_d3_r3": "coherent d3r3",
    "cultivation_d5": "cultivation d5",
    "coherent_d5_r5": "coherent d5r5",
    "qv_q10": "quantum volume, 10 qubits",
    "qv_q20": "quantum volume, 20 qubits",
    "symft_msc_d7": "SymFT d=7, no postselection",
    "symft_msc_d7_postselected": "SymFT d=7, postselected",
}


def load(directory, name):
    try:
        return [json.loads(line) for line in open(f"{directory}/{name}")]
    except FileNotFoundError:
        return []


best = defaultdict(float)
width, k_max = {}, {}
for source, directory, files in (
    ("base", RESULTS, ("base-all.jsonl", "base-1.jsonl")),
    ("pr", PR, ("tsim-all.jsonl", "tsim-all-large.jsonl", "tsim-1.jsonl")),
):
    for f in files:
        for r in load(directory, f):
            sharing = r["configuration"] != "one lane per shot"
            key = (source, r["workload"], sharing, r["threads"])
            best[key] = max(best[key], r["shots_per_second"])
            if source == "pr":
                width[r["workload"]] = r["width"]
for passes, files in (
    ("default", ("clifft.jsonl",)),
    ("all", ("clifft-allpasses.jsonl", "clifft-all-large.jsonl")),
):
    for f in files:
        for r in load(RESULTS, f):
            key = ("clifft-" + passes, r["workload"], None, r["threads"])
            best[key] = max(best[key], r["shots_per_second"])
            if "width" in r:
                k_max[(r["workload"], passes)] = r["width"]


def fmt(x):
    if not x:
        return "—"
    for unit, scale in (("M", 1e6), ("k", 1e3)):
        if x >= scale:
            return f"{x / scale:.3g}{unit}"
    return f"{x:.2g}" if x < 10 else f"{x:.0f}"


def ratio(a, b):
    if not a or not b:
        return "—"
    r = a / b
    text = f"{r:.2g}×" if r < 10 else f"{r:,.0f}×"
    return f"**{text}**" if r >= 1 else text


def tsim(source, name, sharing, threads):
    value = best[(source, name, sharing, threads)]
    if sharing and not value:
        # NOTE: no statevector component, so sharing does not apply.
        return best[(source, name, False, threads)], "†"
    return value, ""


SETTINGS = [
    ("all", True, "All threads · trajectory sharing on (default)", True),
    ("all", False, "All threads · trajectory sharing off", False),
    ("1", True, "1 thread · trajectory sharing on (default)", False),
    ("1", False, "1 thread · trajectory sharing off", False),
]
out = []
for threads, sharing, title, is_open in SETTINGS:
    out.append(f"<details{' open' if is_open else ''}>")
    out.append(f"<summary><b>{title}</b></summary>\n")
    out.append(
        "| circuit | Tsim width | Clifft k_max | base | this PR | Clifft "
        "| this PR / base | this PR / Clifft |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in LABELS.items():
        base, mark = tsim("base", name, sharing, threads)
        pr, _ = tsim("pr", name, sharing, threads)
        clifft = best[("clifft-all", name, None, threads)]
        default_k, all_k = k_max.get((name, "default")), k_max.get((name, "all"))
        k = f"{all_k}" if default_k == all_k else f"{all_k} ({default_k})"
        out.append(
            f"| {label}{mark} | {width.get(name, '—')} | {k} | {fmt(base)} | "
            f"{fmt(pr)} | {fmt(clifft)} | {ratio(pr, base)} | {ratio(pr, clifft)} |"
        )
    out.append("\n</details>\n")
print("\n".join(out))
