"""Plot QuZX duration vs TSIM best duration from enriched benchmark JSON.

This script has no CLI by design. Edit the constants below and run:

    uv run python benchmarks/plot_quizx_vs_tsim.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration (edit these paths as needed)
# ---------------------------------------------------------------------------

INPUT_JSON = Path(
    "/Users/rafaelhaenel/Documents/quera/tsim/benchmarks/quizx/results_tsim.json"
)
OUTPUT_PNG = Path("benchmarks/quizx/q50_d10_quizx_vs_tsim.png")


def _load_records(path: Path) -> list[dict]:
    """Load records from enriched benchmark JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(
            "Expected a JSON list or an object containing a 'records' list."
        )
    if not isinstance(records, list):
        raise ValueError("'records' must be a list.")
    return records


def _extract_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract seed, quizx duration, and tsim best duration arrays."""
    seeds: list[int] = []
    quizx: list[float] = []
    tsim_best: list[float] = []

    for i, rec in enumerate(records):
        bench = rec.get("tsim_benchmark", {})
        best = bench.get("best_duration_per_shot_secs")
        quizx_dur = rec.get("simulation_duration_secs")
        status = bench.get("status")

        if status != "ok" or best is None or quizx_dur is None:
            continue

        seed = rec.get("seed", i)
        seeds.append(int(seed))
        quizx.append(float(quizx_dur))
        tsim_best.append(float(best))

    if not seeds:
        raise ValueError("No valid records found with both QuZX and TSIM durations.")

    seed_arr = np.array(seeds, dtype=int)
    quizx_arr = np.array(quizx, dtype=float)
    tsim_arr = np.array(tsim_best, dtype=float)

    order = np.argsort(seed_arr)
    return seed_arr[order], quizx_arr[order], tsim_arr[order]


def main() -> None:
    """Read JSON, build comparison plot, and save as PNG."""
    records = _load_records(INPUT_JSON)
    seeds, quizx_durations, tsim_best_durations = _extract_arrays(records)
    speedup = quizx_durations / tsim_best_durations

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(seeds, quizx_durations, "o-", label="QuZX duration (s)")
    axes[0].plot(seeds, tsim_best_durations, "s-", label="TSIM best duration (s)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Duration per sample (s)")
    axes[0].set_title("QuZX vs TSIM performance by circuit seed")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(seeds, speedup, "d-", color="tab:green")
    axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Circuit seed")
    axes[1].set_ylabel("Speedup = QuZX / TSIM")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    # OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(OUTPUT_PNG, dpi=150)
    plt.show()

    # print(f"Saved plot to: {OUTPUT_PNG}")
    print(f"Records plotted: {len(seeds)}")
    print(f"Median speedup (QuZX / TSIM): {np.median(speedup):.2f}x")
    print(f"Min speedup: {np.min(speedup):.2f}x")
    print(f"Max speedup: {np.max(speedup):.2f}x")


if __name__ == "__main__":
    main()
