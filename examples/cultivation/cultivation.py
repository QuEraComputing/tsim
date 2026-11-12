"""Helpers of the cultivation notebook: the circuits, the shot budget and the figure.

The figure follows "Comparing cultivation of T|+>, S|+> and Z|+> states" of Gidney,
Shutty and Jones (arXiv:2409.17595), with the paper's own samples (``paper_data.csv``,
from its ``stats.csv``) drawn as open black markers.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent

GATES = ["T", "S", "Z"]
"""The gate that each S of the authors' Clifford proxy becomes."""

NOISE = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
"""Uniform depolarizing noise strengths of the figure."""

BATCH = 1 << 26
"""Shots of one ``sample`` call."""

_SHOTS = {0.001: 1 << 30, 0.002: 1 << 29}
"""Shots per point where logical errors are rarest; the others take 2^28."""

_STYLE = {"T": ("tab:orange", "v"), "S": ("tab:blue", "o"), "Z": ("tab:green", "*")}


@dataclass
class Point:
    """The tally of one gate at one noise strength."""

    gate: str
    p: float
    shots: int = 0
    kept: int = 0
    errors: int = 0


def points() -> list[Point]:
    """Return an empty tally for every gate and noise strength of the figure."""
    return [Point(gate, p) for gate in GATES for p in NOISE]


def circuit_path(point: Point) -> pathlib.Path:
    """Return the circuit file of ``point``."""
    return HERE / "circuits" / f"cultivate_d3_{point.gate}_p{point.p}.stim"


def batches(point: Point) -> range:
    """Return the ``sample`` calls of ``point``, ``BATCH`` shots each."""
    return range(_SHOTS.get(point.p, 1 << 28) // BATCH)


def _likely_range(
    kept: int, errors: int, factor: float = 1000.0
) -> tuple[float, float]:
    """Return the error rates whose binomial likelihood is within ``factor`` of the best."""
    best = errors / kept

    def loglike(q: float) -> float:
        return (errors * np.log(q) if errors else 0.0) + (kept - errors) * np.log1p(-q)

    floor = loglike(best) - np.log(factor)

    def edge(inside: float, outside: float) -> float:
        for _ in range(200):
            mid = np.sqrt(inside * outside) if outside > 0 else inside / 2
            inside, outside = (mid, outside) if loglike(mid) >= floor else (inside, mid)
        return inside

    return (edge(best, 0.0) if errors else 0.0), edge(max(best, 1e-300), 1.0)


def _paper() -> list[dict[str, str]]:
    with open(HERE / "paper_data.csv") as f:
        return list(csv.DictReader(f))


def plot(tally: list[Point], seconds: float) -> None:
    """Draw the figure from ``tally``, sampled in ``seconds``, with the paper's data."""
    paper = _paper()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    for gate in GATES:
        color, marker = _STYLE[gate]
        mine = [t for t in tally if t.gate == gate]
        x = np.array([t.shots / t.kept for t in mine])
        y = np.array([t.errors / t.kept for t in mine])
        low, high = np.array([_likely_range(t.kept, t.errors) for t in mine]).T
        ax.plot(x, y, marker=marker, color=color, label=f"gate={gate}")
        ax.fill_between(x, low, high, color=color, alpha=0.2)
        theirs = [r for r in paper if r["gate"] == gate]
        kept = [int(r["shots"]) - int(r["discards"]) for r in theirs]
        ax.plot(
            [int(r["shots"]) / k for r, k in zip(theirs, kept, strict=True)],
            [int(r["errors"]) / k for r, k in zip(theirs, kept, strict=True)],
            marker=marker,
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
        )
        if gate == "T":
            for t, xi, yi in zip(mine, x, y, strict=True):
                ax.annotate(
                    f"p={t.p}", (xi, yi), textcoords="offset points", xytext=(3, 3)
                )
    ax.plot(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor="black",
        label="paper's data (vector sim)",
    )
    shots = sum(t.shots for t in tally)
    paper_shots = sum(int(r["shots"]) for r in paper)
    paper_years = sum(float(r["cpu_seconds"]) for r in paper) / 86400 / 365
    ax.text(
        0.98,
        0.04,
        f"Tsim: {shots / 1e9:.1f} billion shots in {seconds:.0f} s\n"
        f"paper's vector sim: {paper_shots / 1e9:.1f} billion shots "
        f"in {paper_years:.1f} CPU-years",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray"},
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(1, 100),
        ylim=(1e-7, 1e-2),
        xlabel="Expected Attempts per Kept Shot",
        ylabel="Logical Error Rate (per kept shot)",
        title="Comparing Cultivation of T|+>, S|+>, and Z|+> States\n"
        "c=inject[unitary]+cultivate, d1=3, g=css, noise=uniform, q=15, r=4",
    )
    ax.grid(which="major", color="black", linewidth=0.8)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()
