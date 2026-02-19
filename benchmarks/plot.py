"""Plot benchmark results from saved JSON files.

Usage::

    # Plot all results in the default directory
    uv run python -m benchmarks.plot

    # Plot specific files or directories, save to PNG
    uv run python -m benchmarks.plot results/run1.json results/run2.json -o comparison.png

    # Omit reference literature data
    uv run python -m benchmarks.plot --no-reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Reference data from published papers (for overlay comparison)
# ---------------------------------------------------------------------------

REFERENCE_DATA: dict[str, dict] = {
    "arXiv:2202.09202": {
        "t_count": [1, 4, 7, 10, 13, 16, 19],  # , 22, 25, 28, 31, 34, 37, 40, 43],
        "duration_per_shot": [
            1.9751864976758715e-06,
            2.5961750794608702e-05,
            1.6943603953294232e-04,
            6.05338183931951e-04,
            2.308449951931215e-03,
            5.903900045168939e-03,
            2.4125742877835272e-02,
            # 5.650977161878403e-02,
            # 2.028575236911587e-01,
            # 4.9443814581116835e-01,
            # 1.713865884482384,
            # 4.047989545803132,
            # 13.888779807398137,
            # 32.16862014131525,
            # 91.69701795218768,
        ],
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(paths: list[Path]) -> list[dict]:
    """Load benchmark results from JSON files or directories.

    Directories are scanned recursively for ``*.json`` files.
    """
    all_results: list[dict] = []
    for path in paths:
        if path.is_dir():
            for f in sorted(path.rglob("*.json")):
                with open(f) as fh:
                    all_results.append(json.load(fh))
        elif path.suffix == ".json":
            with open(path) as fh:
                all_results.append(json.load(fh))
        else:
            print(f"Skipping non-JSON file: {path}")
    return all_results


def _compute_stats(results: list[dict]) -> dict[str, np.ndarray]:
    """Aggregate per-t_count statistics from a single run's results list.

    Returns arrays keyed by ``t_count``, ``duration_mean``, ``duration_std``,
    ``batch_mean``, ``batch_std``.
    """
    t_counts_arr = np.array([r["t_count"] for r in results])
    durations_arr = np.array([r["duration_per_shot"] for r in results])
    batches_arr = np.array([r["best_batch_size"] for r in results])

    unique_t = np.unique(t_counts_arr)
    stats: dict[str, list] = {
        "t_count": [],
        "duration_mean": [],
        "duration_std": [],
        "batch_mean": [],
        "batch_std": [],
    }
    for t in unique_t:
        mask = t_counts_arr == t
        stats["t_count"].append(t)
        stats["duration_mean"].append(np.mean(durations_arr[mask]))
        stats["duration_std"].append(np.std(durations_arr[mask]))
        stats["batch_mean"].append(np.mean(batches_arr[mask]))
        stats["batch_std"].append(np.std(batches_arr[mask]))

    return {k: np.array(v) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_t_count_scaling(
    runs: list[dict],
    *,
    show_reference: bool = True,
    output: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot T-count scaling benchmark results.

    Each entry in *runs* is the full JSON payload produced by ``run.py``.
    Runs from different machines are overlaid with distinct markers.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for i, run in enumerate(runs):
        label = run["metadata"]["machine_label"]
        stats = _compute_stats(run["results"])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]

        axes[0].errorbar(
            stats["t_count"],
            stats["duration_mean"],
            # yerr=stats["duration_std"],
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            markersize=5,
            label=f"tsim ({label})",
        )
        axes[1].errorbar(
            stats["t_count"],
            stats["batch_mean"],
            # yerr=stats["batch_std"],
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            markersize=5,
            label=f"tsim ({label})",
        )

        # Fit log2(duration) = alpha * T + beta to the second half of the data
        n_pts = len(stats["t_count"])
        half = n_pts // 2
        t_fit = stats["t_count"][half:]
        log2_dur = np.log2(stats["duration_mean"][half:])
        alpha, beta = np.polyfit(t_fit, log2_dur, 1)
        t_line = np.linspace(t_fit[0], t_fit[-1], 100)
        axes[0].plot(
            t_line,
            2 ** (alpha * t_line + beta),
            "--",
            color=color,
            alpha=0.7,
            label=rf"$2^{{{alpha/2:.2f} \cdot 2 T}}$ ({label})",
        )

    # Reference data
    if show_reference:
        for name, data in REFERENCE_DATA.items():
            axes[0].plot(
                data["t_count"],
                data["duration_per_shot"],
                "x--",
                color="gray",
                alpha=0.7,
                label=name,
            )

            # Fit log2(duration) = alpha * T + beta to the second half
            t_ref = np.array(data["t_count"])
            dur_ref = np.array(data["duration_per_shot"])
            n_ref = len(t_ref)
            half_ref = n_ref // 2
            t_ref_fit = t_ref[half_ref:]
            log2_dur_ref = np.log2(dur_ref[half_ref:])
            alpha_ref, beta_ref = np.polyfit(t_ref_fit, log2_dur_ref, 1)
            t_ref_line = np.linspace(t_ref_fit[0], t_ref_fit[-1], 100)
            axes[0].plot(
                t_ref_line,
                2 ** (alpha_ref * t_ref_line + beta_ref),
                ":",
                color="gray",
                alpha=0.7,
                label=rf"$2^{{{alpha_ref/2:.2f} \cdot 2 T}}$ ({name})",
            )

    # Axes formatting
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Duration per shot (s)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    if ylim:
        axes[0].set_ylim(*ylim)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("T-count")
    axes[1].set_ylabel("Optimal batch size")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    if xlim:
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)

    fig.suptitle("T-count scaling benchmark", fontsize=13)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot tsim benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "paths",
        nargs="*",
        default=["benchmarks/results"],
        help="Paths to result JSON files or directories",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Save plot to file (e.g. comparison.png)",
    )
    p.add_argument(
        "--no-reference",
        action="store_true",
        help="Omit reference literature data from the plot",
    )
    p.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"))
    p.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"))
    return p


def main(argv: list[str] | None = None) -> None:
    """Load results and produce plots."""
    args = _build_parser().parse_args(argv)

    paths = [Path(p) for p in args.paths]
    runs = load_results(paths)

    if not runs:
        print("No result files found. Run benchmarks first:")
        print("  uv run python -m benchmarks.run --machine 'My Machine'")
        return

    print(f"Loaded {len(runs)} result file(s):")
    for run in runs:
        label = run["metadata"]["machine_label"]
        ts = run["metadata"].get("timestamp", "?")
        n = len(run["results"])
        print(f"  - {label} ({ts}, {n} data points)")
    print()

    plot_t_count_scaling(
        runs,
        show_reference=not args.no_reference,
        output=args.output,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
    )


if __name__ == "__main__":
    main()
