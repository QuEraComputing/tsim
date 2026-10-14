"""Merge the Tsim and Clifft JSON lines of a comparison run into Markdown tables.

Run with any Python 3 from the repository root:

    uv run python scripts/clifft_comparison/report.py RUN_DIR

``RUN_DIR`` holds ``tsim-all.jsonl``, ``tsim-1.jsonl`` and ``clifft.jsonl``. The
speed table takes each simulator's fastest configuration per thread setting.
The check table pools every run of a simulator and compares Tsim with Clifft
by a two-proportion z-test, on acceptance and on the observable-0 rate among
accepted shots; |z| > 5 is flagged.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from workloads import WORKLOADS


def _load(run: Path) -> list[dict]:
    records = []
    for name in ("tsim-all.jsonl", "tsim-1.jsonl", "clifft.jsonl"):
        path = run / name
        if path.exists():
            records += [json.loads(line) for line in path.read_text().splitlines()]
    return records


def _z(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = (k1 + k2) / (n1 + n2)
    variance = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    return 0.0 if variance == 0 else (k1 / n1 - k2 / n2) / math.sqrt(variance)


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}"


def main() -> None:
    """Print the speed, check and detail tables of one run."""
    run = Path(sys.argv[1])
    records = _load(run)
    best: dict[tuple, dict] = {}
    pooled: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    width: dict[tuple, int] = {}
    for r in records:
        variant = r["simulator"]
        if variant == "tsim":
            variant += " shared" if r["configuration"] == "shared" else " one lane"
        key = (r["workload"], variant, r["threads"])
        if key not in best or r["shots_per_second"] > best[key]["shots_per_second"]:
            best[key] = r
        totals = pooled[(r["workload"], r["simulator"])]
        totals[0] += r["attempted"]
        totals[1] += r["accepted"]
        totals[2] += r["observable_ones"]
        width[(r["workload"], r["simulator"])] = r["width"]

    names = [n for n in WORKLOADS if any(r["workload"] == n for r in records)]
    print("## Speed (attempted shots/s; best configuration per thread setting)\n")
    print(
        "| circuit | postselect | Tsim width | Clifft k_max "
        "| Tsim one lane, 1 thread | Tsim shared, 1 thread | Clifft, 1 thread "
        "| Tsim best / Clifft, 1 thread "
        "| Tsim one lane, all | Tsim shared, all | Clifft, all "
        "| Tsim best / Clifft, all |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in names:
        cells = []
        for threads in ("1", "all"):
            rates = {
                variant: best.get((name, variant, threads), {}).get("shots_per_second")
                for variant in ("tsim one lane", "tsim shared", "clifft")
            }
            tsim_best = max(
                (rates[v] for v in ("tsim one lane", "tsim shared") if rates[v]),
                default=None,
            )
            ratio = (
                f"{tsim_best / rates['clifft']:.2f}x"
                if tsim_best and rates["clifft"]
                else "—"
            )
            cells += [_rate(rates[v]) for v in rates] + [ratio]
        print(
            f"| {name} | {'yes' if WORKLOADS[name].postselect else 'no'} "
            f"| {width.get((name, 'tsim'), '—')} | {width.get((name, 'clifft'), '—')} | "
            + " | ".join(cells)
            + " |"
        )

    print("\n## Check (every run pooled; z = Tsim vs Clifft)\n")
    print(
        "| circuit | Tsim acceptance | Clifft acceptance | z | Tsim obs-0 / accepted "
        "| Clifft obs-0 / accepted | z | check |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for name in names:
        t = pooled.get((name, "tsim"), [0, 0, 0])
        c = pooled.get((name, "clifft"), [0, 0, 0])
        z_accept = _z(t[1], t[0], c[1], c[0])
        z_obs = _z(t[2], t[1], c[2], c[1])
        flagged = any(abs(z) > 5 for z in (z_accept, z_obs) if not math.isnan(z))
        print(
            f"| {name} | {t[1] / max(t[0], 1):.4f} | {c[1] / max(c[0], 1):.4f} "
            f"| {z_accept:+.1f} | {t[2]} / {t[1]} | {c[2]} / {c[1]} | {z_obs:+.1f} "
            f"| {'MISMATCH' if flagged else 'ok'} |"
        )

    print("\n## Every measurement\n")
    print(
        "| circuit | simulator | configuration | threads | batch | shots/call "
        "| compile s | warmup s | shots/s | cores busy | calls |"
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in names:
        for r in records:
            if r["workload"] != name:
                continue
            print(
                f"| {name} | {r['simulator']} | {r.get('configuration', '—')} "
                f"| {r['threads']} | {r['batch']:,} | {r['shots_per_call']:,} "
                f"| {r['compile_seconds']:.2f} | {r.get('warmup_seconds', float('nan')):.2f} "
                f"| {r['shots_per_second']:,.0f} | {r['cores_busy']:.1f} | {r['calls']} |"
            )


if __name__ == "__main__":
    main()
