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

INPUT_JSONS = (
    Path("/Users/rafaelhaenel/Documents/quera/tsim/benchmarks/quizx/results_tsim.json"),
    Path(
        "/Users/rafaelhaenel/Documents/quera/tsim/benchmarks/quizx/results_tsim_gpu.json"
    ),
)
INPUT_LABELS = ("CPU (M4 Pro)", "GPU (GH200)")
OUTPUT_PNG = Path("benchmarks/quizx/q50_d10_quizx_vs_tsim.png")
TARGET_QUBITS = 20
plt.rcParams["text.usetex"] = True


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
        rec_qubits = rec.get("qubits")
        if rec_qubits is None or int(rec_qubits) != TARGET_QUBITS:
            continue

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
    if len(INPUT_JSONS) != len(INPUT_LABELS):
        raise ValueError("INPUT_JSONS and INPUT_LABELS must have the same length.")

    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for path, label in zip(INPUT_JSONS, INPUT_LABELS, strict=True):
        records = _load_records(path)
        seeds_i, quizx_i, tsim_i = _extract_arrays(records)
        datasets.append((label, seeds_i, quizx_i, tsim_i))

    primary_label, seeds, quizx_durations, tsim_best_durations = datasets[0]
    speedup = quizx_durations / tsim_best_durations

    # fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)

    # axes[0].plot(seeds, quizx_durations, "o-", label=f"QuZX duration (s) [{primary_label}]")
    # axes[0].plot(
    #     seeds, tsim_best_durations, "s-", label=f"TSIM best duration (s) [{primary_label}]"
    # )
    # axes[0].set_yscale("log")
    # axes[0].set_ylabel("Duration per sample (s)")
    # axes[0].set_title("QuZX vs TSIM performance by circuit seed")
    # axes[0].grid(True, alpha=0.3)
    # axes[0].legend()

    # axes[1].plot(seeds, speedup, "d-", color="tab:green")
    # axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1)
    # axes[1].set_yscale("log")
    # axes[1].set_xlabel("Circuit seed")
    # axes[1].set_ylabel("Speedup = QuZX / TSIM")
    # axes[1].grid(True, alpha=0.3)

    # fig.tight_layout()

    # Correlation scatter: QuZX duration vs TSIM best duration.
    scatter_fig, scatter_ax = plt.subplots(figsize=(4.2, 3.5))
    dataset_colors = ["tab:blue", "#76ba00"]
    for idx, (label, _, quizx_i, tsim_i) in enumerate(datasets):
        color = dataset_colors[idx] if idx < len(dataset_colors) else None
        scatter_ax.scatter(quizx_i, tsim_i, alpha=0.8, label=label, color=color)
    scatter_ax.set_xscale("log")
    scatter_ax.set_yscale("log")
    scatter_ax.set_xlabel("Quizx time per shot (s)")
    scatter_ax.set_ylabel("Tsim time per shot (s)")

    scatter_ax.grid(True, alpha=0.3)

    all_quizx = np.concatenate([quizx_i for _, _, quizx_i, _ in datasets])
    all_tsim = np.concatenate([tsim_i for _, _, _, tsim_i in datasets])
    x_min, x_max = float(np.min(all_quizx)), float(np.max(all_quizx))
    y_min, y_max = float(np.min(all_tsim)), float(np.max(all_tsim))
    scatter_ax.set_xlim(5e-4, 4e-1)
    scatter_ax.set_ylim(4e-9, 5e-3)

    x_left, x_right = scatter_ax.get_xlim()
    x_line = np.array([x_left, x_right], dtype=float)
    for factor in (100, 1000, 10000, 100000):
        y_line = x_line / factor
        scatter_ax.plot(x_line, y_line, "-", color="black", linewidth=1.0, alpha=0.7)
        y_text = x_right / factor
        scatter_ax.text(
            1.03,
            y_text,
            f"{factor}x",
            color="black",
            fontsize=9,
            transform=scatter_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            clip_on=False,
        )
    scatter_ax.legend()
    scatter_fig.tight_layout(rect=(0, 0, 0.9, 1))

    corr = np.corrcoef(quizx_durations, tsim_best_durations)[0, 1]
    # OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    scatter_fig.savefig("quizx_vs_tsim_scatter.svg", transparent=True)
    plt.show()

    # print(f"Saved plot to: {OUTPUT_PNG}")
    print(f"Records plotted: {len(seeds)}")
    print(f"Median speedup (QuZX / TSIM): {np.median(speedup):.2f}x")
    print(f"Min speedup: {np.min(speedup):.2f}x")
    print(f"Max speedup: {np.max(speedup):.2f}x")
    print(f"Pearson correlation (durations): {corr:.3f}")
    for label, seeds_i, quizx_i, tsim_i in datasets:
        speedup_i = quizx_i / tsim_i
        corr_i = np.corrcoef(quizx_i, tsim_i)[0, 1]
        print(f"[{label}] Records plotted: {len(seeds_i)}")
        print(f"[{label}] Median speedup (QuZX / TSIM): {np.median(speedup_i):.2f}x")
        print(f"[{label}] Pearson correlation (durations): {corr_i:.3f}")


if __name__ == "__main__":
    main()
