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
    Path("benchmarks/stim/results_m4-sweep-num-graphs.json"),
    # Path("benchmarks/stim/results_gh200.json"),
)

INPUT_LABELS: tuple[str, ...] = (
    "M4",
    # "GH200",
)

normalize_by_num_graphs = False

OUTPUT_FILE = Path("benchmarks/stim/tsim_vs_stim.pdf")

plt.rcParams["text.usetex"] = False

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
    normalize_by_num_graphs: bool = False,
) -> tuple[list[str], np.ndarray, np.ndarray, list[float | None]]:
    labels: list[str] = []
    stim_times: list[float] = []
    tsim_times: list[float] = []
    rescale_factors: list[float | None] = []

    for rec in records:
        tsim_bench = rec.get("tsim", {})
        stim_bench = rec.get("stim", {})

        if tsim_bench.get("status") != "ok" or stim_bench.get("status") != "ok":
            continue

        tsim_dur = tsim_bench.get("best_duration_per_shot_secs")
        stim_dur = stim_bench.get("best_duration_per_shot_secs")
        if tsim_dur is None or stim_dur is None:
            continue

        if normalize_by_num_graphs:
            ng = tsim_bench.get("num_graphs_of_largest_component")
            if ng is None:
                continue
            tsim_dur = float(tsim_dur) / max(ng)

        labels.append(rec.get("label", rec.get("file", "?")))
        stim_times.append(float(stim_dur))
        tsim_times.append(float(tsim_dur))
        rescale_factors.append(rec.get("rescale_factor"))

    if not labels:
        raise ValueError("No valid records found with both tsim and stim durations.")

    return labels, np.array(stim_times), np.array(tsim_times), rescale_factors


def _rescale_to_marker_size(
    rescale_factors: list[float | None],
    size_min: float = 5,
    size_max: float = 150,
) -> np.ndarray:
    """Map rescale factors to marker sizes on a log-log scale."""
    vals = np.array([f if f is not None else 1.0 for f in rescale_factors])
    # log_vals = np.log10(np.clip(vals, 1e-30, None))
    log_vals = np.clip(vals, 1e-30, None)
    lo, hi = float(np.min(log_vals)), float(np.max(log_vals))
    if hi - lo < 1e-12:
        return np.full(len(vals), (size_min + size_max) / 2)
    normed = (log_vals - lo) / (hi - lo)
    return size_min + normed * (size_max - size_min)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Read JSONs, build comparison scatter plot, and save to file."""
    if len(INPUT_JSONS) != len(INPUT_LABELS):
        raise ValueError("INPUT_JSONS and INPUT_LABELS must have the same length.")

    datasets: list[tuple[str, list[str], np.ndarray, np.ndarray, np.ndarray]] = []
    for path, dataset_label in zip(INPUT_JSONS, INPUT_LABELS, strict=True):
        records = _load_records(path)
        circuit_labels, stim_arr, tsim_arr, rescale_factors = _extract_arrays(
            records, normalize_by_num_graphs=normalize_by_num_graphs
        )
        sizes = _rescale_to_marker_size(rescale_factors)
        datasets.append((dataset_label, circuit_labels, stim_arr, tsim_arr, sizes))

    fig, ax = plt.subplots(figsize=(4.2, 3.5))

    all_circuit_labels: list[str] = []
    for _, circuit_labels, _, _, _ in datasets:
        for lbl in circuit_labels:
            if lbl not in all_circuit_labels:
                all_circuit_labels.append(lbl)

    cmap = plt.get_cmap("tab10")
    label_colors = {lbl: cmap(i % 10) for i, lbl in enumerate(all_circuit_labels)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for idx, (dataset_label, circuit_labels, stim_arr, tsim_arr, sizes) in enumerate(
        datasets
    ):
        marker = markers[idx % len(markers)]
        plotted_labels: set[str] = set()
        for lbl, sx, ty, sz in zip(circuit_labels, stim_arr, tsim_arr, sizes):
            legend_label = lbl if lbl not in plotted_labels else None
            plotted_labels.add(lbl)
            ax.scatter(
                sx,
                ty,
                alpha=0.85,
                label=legend_label,
                color=label_colors[lbl],
                marker=marker,
                zorder=3,
                s=sz,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Stim time per shot (s)")
    ax.set_ylabel("Tsim time per shot (s)")

    all_stim = np.concatenate([s for _, _, s, _, _ in datasets])
    all_tsim = np.concatenate([t for _, _, _, t, _ in datasets])
    lo = min(float(np.min(all_stim)), float(np.min(all_tsim))) * 0.3
    hi = max(float(np.max(all_stim)), float(np.max(all_tsim))) * 3.0

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    x_right = ax.get_xlim()[1]
    y_top = ax.get_ylim()[1]
    x_line = np.array([lo * 0.1, hi * 10], dtype=float)
    for factor in (0.01, 0.1, 1, 10, 100):
        y_line = x_line / factor
        ax.plot(
            x_line, y_line, "-", color="#4a4a4a", linewidth=1.0, alpha=0.7, zorder=1
        )
        if factor <= 0.1:
            x_text = y_top * factor
            ax.text(
                x_text,
                1.03,
                f"{factor:g}x",
                color="#4a4a4a",
                fontsize=9,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                clip_on=False,
            )
        else:
            y_text = x_right / factor
            ax.text(
                1.03,
                y_text,
                f"{factor:g}x",
                color="#4a4a4a",
                fontsize=9,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                clip_on=False,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_FILE), transparent=True)
    plt.show()

    print(f"Saved plot to {OUTPUT_FILE}")
    for dataset_label, circuit_labels, stim_arr, tsim_arr, _ in datasets:
        speedup = stim_arr / tsim_arr
        print(f"\n[{dataset_label}]")
        for lbl, s, t, sp in zip(circuit_labels, stim_arr, tsim_arr, speedup):
            print(f"  {lbl}: stim={s:.3e}  tsim={t:.3e}  speedup={sp:.1f}x")
        print(f"  Median speedup: {np.median(speedup):.1f}x")


if __name__ == "__main__":
    main()
