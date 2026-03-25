"""Plot tsim vs stim detector-sampling throughput from benchmark JSONs.

Reads one or more result JSONs produced by ``run_vs_stim.py`` (e.g. one from
a CPU run and one from a GPU run) and produces a scatter plot with stim
time-per-shot on the x-axis and tsim time-per-shot on the y-axis.

Edit the constants below and run::

    uv run python benchmarks/stim/plot_tsim_vs_stim.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration — edit these paths / labels as needed
# ---------------------------------------------------------------------------

INPUT_JSONS: tuple[Path, ...] = (
    Path("benchmarks/stim/results_cpu.json"),
    # Path("benchmarks/stim/results_gpu.json"),
)

INPUT_LABELS: tuple[str, ...] = (
    "CPU",
    # "GPU",
)

OUTPUT_FILE = Path("benchmarks/stim/tsim_vs_stim.pdf")

plt.rcParams["text.usetex"] = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected a JSON list or an object containing a 'records' list.")


def _extract_label(path: Path) -> str:
    """Derive a fallback label from the run_metadata machine_label."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        meta = payload.get("run_metadata", {})
        machine = meta.get("machine", {})
        return machine.get("machine_label", path.stem)
    return path.stem


def _extract_arrays(
    records: list[dict],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels: list[str] = []
    stim_times: list[float] = []
    tsim_times: list[float] = []

    for rec in records:
        tsim_bench = rec.get("tsim", {})
        stim_bench = rec.get("stim", {})

        if tsim_bench.get("status") != "ok" or stim_bench.get("status") != "ok":
            continue

        tsim_dur = tsim_bench.get("best_duration_per_shot_secs")
        stim_dur = stim_bench.get("best_duration_per_shot_secs")
        if tsim_dur is None or stim_dur is None:
            continue

        labels.append(rec.get("label", rec.get("file", "?")))
        stim_times.append(float(stim_dur))
        tsim_times.append(float(tsim_dur))

    if not labels:
        raise ValueError("No valid records found with both tsim and stim durations.")

    return labels, np.array(stim_times), np.array(tsim_times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Read JSONs, build comparison scatter plot, and save to file."""
    if len(INPUT_JSONS) != len(INPUT_LABELS):
        raise ValueError("INPUT_JSONS and INPUT_LABELS must have the same length.")

    datasets: list[tuple[str, list[str], np.ndarray, np.ndarray]] = []
    for path, dataset_label in zip(INPUT_JSONS, INPUT_LABELS, strict=True):
        records = _load_records(path)
        circuit_labels, stim_arr, tsim_arr = _extract_arrays(records)
        datasets.append((dataset_label, circuit_labels, stim_arr, tsim_arr))

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    dataset_colors = ["tab:blue", "#76ba00", "tab:orange", "tab:red"]
    markers = ["o", "s", "^", "D"]

    for idx, (dataset_label, circuit_labels, stim_arr, tsim_arr) in enumerate(datasets):
        color = dataset_colors[idx % len(dataset_colors)]
        marker = markers[idx % len(markers)]
        ax.scatter(
            stim_arr,
            tsim_arr,
            alpha=0.85,
            label=dataset_label,
            color=color,
            marker=marker,
            zorder=3,
            s=50,
        )
        for lbl, sx, ty in zip(circuit_labels, stim_arr, tsim_arr):
            ax.annotate(
                lbl,
                (sx, ty),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=6,
                alpha=0.7,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Stim time per shot (s)")
    ax.set_ylabel("Tsim time per shot (s)")

    all_stim = np.concatenate([s for _, _, s, _ in datasets])
    all_tsim = np.concatenate([t for _, _, _, t in datasets])
    lo = min(float(np.min(all_stim)), float(np.min(all_tsim))) * 0.3
    hi = max(float(np.max(all_stim)), float(np.max(all_tsim))) * 3.0

    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, zorder=1, label="1x")

    for factor in (10, 100, 1000):
        ax.plot(
            [lo, hi],
            [lo / factor, hi / factor],
            "-",
            color="#4a4a4a",
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )
        x_text = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else hi
        y_text = x_text / factor
        ax.annotate(
            f"tsim {factor}x faster",
            xy=(hi, hi / factor),
            fontsize=6,
            color="#4a4a4a",
            alpha=0.7,
            ha="right",
            va="bottom",
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_FILE), transparent=True)
    plt.show()

    print(f"Saved plot to {OUTPUT_FILE}")
    for dataset_label, circuit_labels, stim_arr, tsim_arr in datasets:
        speedup = stim_arr / tsim_arr
        print(f"\n[{dataset_label}]")
        for lbl, s, t, sp in zip(circuit_labels, stim_arr, tsim_arr, speedup):
            print(f"  {lbl}: stim={s:.3e}  tsim={t:.3e}  speedup={sp:.1f}x")
        print(f"  Median speedup: {np.median(speedup):.1f}x")


if __name__ == "__main__":
    main()
