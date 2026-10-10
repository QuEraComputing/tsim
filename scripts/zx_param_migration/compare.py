"""Compare the outputs of two Tsim versions and write a Markdown report.

Run from this checkout after ``probabilities.py``, ``rates.py`` and
``benchmark.py`` have written their JSON files for both versions:

    uv run python scripts/zx_param_migration/compare.py OLD_DIR NEW_DIR RESULTS_MD

OLD_DIR holds ``probabilities.json``, ``rates.json`` and one or more
``benchmark*.json`` files (one per benchmark run) of today's Tsim, NEW_DIR those
of the migrated Tsim; benchmark runs are paired by sorted file name. Every file
records the ``tsim.__file__`` it ran with, and the old and new paths must differ.
Probabilities must satisfy ``np.allclose(p_new, p_old, rtol=1e-4, atol=1e-7)``;
every detector and observable rate must satisfy
``|p_old - p_new| <= 5 * sigma + 1e-4``. Benchmarks are reported, not gated.
Exits with status 1 if a probability or rate row fails or two paths coincide.

The clifft-bench circuits are read from the directory in the
``CLIFFT_BENCH_CIRCUITS`` environment variable (see ``circuits.py``).
"""

import json
import sys
from pathlib import Path

import numpy as np
from circuits import SMALL, clifft, generated

RTOL = 1e-4
ATOL = 1e-7
SIGMAS = 5
SLACK = 1e-4
NOISY_SECONDS = 0.01

INTRO = """\
# zx_param migration: comparison with today's Tsim

- **Old:** today's Tsim, main `3dacc1a` (built on pyzx-param).
- **New:** the migrated Tsim, branch `zx-param-migration` (built on zx_param).

Each version ran `probabilities.py`, `rates.py` and `benchmark.py` from
`scripts/zx_param_migration/` in its own checkout and environment, on the JAX
CPU backend. This file is written by `compare.py`; the Tsim each run imported is
listed at the end.

Noisy circuits are not compared probability by probability: `ChannelSampler`
simplifies the noise channels with the error-basis transform, so seeded error
draws can differ between the versions. Exact probabilities are compared on
noiseless circuits, and noisy circuits through their detector and observable
rates.
"""

PROBABILITIES = f"""\
## Exact probabilities

Circuits: `SMALL`, `generated()` and `coherent_d3_r1`, each without noise.
States (written by `states.py` with the migrated Tsim, deduplicated): for
`SMALL`, all 2^n measurement records; for the others, 8 records drawn with
`compile_sampler(seed=0).sample(8, batch_size=8)`, the all-zeros record, and
the first 3 drawn records with one bit flipped in the support of detector 0, 1
and 2 respectively. Each version computes
`CompiledStateProbs(circuit, strategy="cat5", seed=0).probability_of(state, batch_size=1)`
(joint mode). A row passes if
`np.allclose(p_new, p_old, rtol={RTOL:g}, atol={ATOL:g})`, that is
`|p_new - p_old| <= {ATOL:g} + {RTOL:g} * |p_old|` for every state. "Values"
counts the distinct probabilities among the states (values closer than that
tolerance count once), and "Zero" counts the states with `p_old <= {ATOL:g}`.
The relative deviation is taken over states with `p_old > 0`.
"""

RATES = f"""\
## Detector and observable rates

Circuits: `generated()` (cat5) and `clifft()` (each with its strategy), with
noise, in detector mode. Each version samples
`compile_detector_sampler(strategy=..., seed=0).sample(shots, batch_size=b,
append_observables=True)` with 200,000 shots (20,000 for distillation and
msc_d3) and b = 8,192 (64 for msc_d3), and takes the mean of every column. A
column passes if `|p_old - p_new| <= {SIGMAS} * sigma + {SLACK:g}`, where
`sigma = sqrt(p_old * (1 - p_old) / n_old + p_new * (1 - p_new) / n_new)`; a row
passes if every column does. The z-score of a column is
`|p_old - p_new| / sigma`.

Sensitivity: the tolerance column gives the range of that per-column bound, so
a smaller deviation would go unnoticed in that row. A row whose maximum
deviation is 0 has identical column means in both versions: the seeded error
draws coincide there, so the row shows that both versions compute the same
outcomes from the same draws rather than testing the statistics.
"""

BENCHMARKS = f"""\
## Benchmarks

Circuits: `clifft()` with noise, detector mode. Compile time is one
`compile_detector_sampler(strategy=..., seed=0)` call. Terms is the sum of
`num_graphs` over every compiled scalar graph of every component (0 when all
outputs take the direct path). After one warm-up `sample(4096)` call, one
`sample(65536, batch_size=8192)` call is timed (`sample(4096, batch_size=64)`
for msc_d3). Ratios are new/old: above 1 means the new compile is slower, and
below 1 means the new sampling is slower. Each run benchmarks one version and
then the other on a shared machine (the load column is the 1-minute load
average after each circuit, old / new). A throughput ratio marked `*` comes
from a timed call under {NOISY_SECONDS * 1000:g} ms; it is noisy and not counted in
the verdict. Unlike the clifft-bench harness, no postselection mask is passed,
so absolute rates differ from that harness's. Benchmarks are reported, not
gated.

Not compared: the other three of clifft-bench's eight Tsim workloads,
`msc_d5_inject_cultivate_p1e-3` (cutting), `coherent_d3_r3` and `coherent_d5_r5`
(cat5). `clifft()` leaves them out because they exceed clifft-bench's 120 s
compile limit with today's Tsim, on clifft-bench's published H100 run and in
the local CPU calibration (`benchmark_instructions.md`).
"""


def load(path: Path) -> dict:
    """Return the parsed JSON file at ``path``."""
    return json.loads(path.read_text())


def missing_cell(name: str, old: dict, new: dict) -> str | None:
    """Return a note if ``name`` is missing from either version, else None."""
    sides = [side for side, data in (("old", old), ("new", new)) if name not in data]
    return f"not run ({', '.join(sides)})" if sides else None


def count_distinct(values: np.ndarray) -> int:
    """Return how many values differ from each other by more than the tolerance."""
    ordered = np.sort(values)
    gaps = np.diff(ordered) > ATOL + RTOL * np.abs(ordered[1:])
    return int(gaps.sum()) + 1 if ordered.size else 0


def probability_rows(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Return the probability table and the names of failing rows."""
    names = [*SMALL, *generated(), "coherent_d3_r1"]
    rows = [
        "| Circuit | States | Values | Zero | Max abs deviation | Max rel deviation "
        "| Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    failures = []
    for name in names:
        missing = missing_cell(name, old, new)
        if missing is not None:
            rows.append(f"| {name} | | | | | | FAIL: {missing} |")
            failures.append(name)
            continue
        p_old, p_new = np.array(old[name]), np.array(new[name])
        if p_old.shape != p_new.shape:
            rows.append(
                f"| {name} | | | | | | FAIL: {p_old.size} vs {p_new.size} states |"
            )
            failures.append(name)
            continue
        deviation = np.abs(p_new - p_old)
        positive = p_old > 0
        relative = (deviation[positive] / p_old[positive]).max(initial=0.0)
        passed = bool(np.allclose(p_new, p_old, rtol=RTOL, atol=ATOL))
        if not passed:
            failures.append(name)
        rows.append(
            f"| {name} | {p_old.size} | {count_distinct(p_old)} "
            f"| {int((p_old <= ATOL).sum())} | {deviation.max(initial=0.0):.2e} "
            f"| {relative:.2e} | {'pass' if passed else 'FAIL'} |"
        )
    return rows, failures


def rate_rows(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Return the detector-rate table and the names of failing rows."""
    names = [*generated(), *clifft()]
    rows = [
        "| Circuit | Columns | Shots (old / new) | Tolerance | Max abs deviation "
        "| Worst z | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    failures = []
    for name in names:
        missing = missing_cell(name, old, new)
        if missing is not None:
            rows.append(f"| {name} | | | | | | FAIL: {missing} |")
            failures.append(name)
            continue
        n_old, n_new = old[name]["shots"], new[name]["shots"]
        p_old, p_new = np.array(old[name]["rates"]), np.array(new[name]["rates"])
        if p_old.shape != p_new.shape:
            rows.append(
                f"| {name} | | | | | | FAIL: {p_old.size} vs {p_new.size} columns |"
            )
            failures.append(name)
            continue
        deviation = np.abs(p_old - p_new)
        sigma = np.sqrt(p_old * (1 - p_old) / n_old + p_new * (1 - p_new) / n_new)
        tolerance = SIGMAS * sigma + SLACK
        passed = bool(np.all(deviation <= tolerance))
        z_scores = np.divide(
            deviation,
            sigma,
            out=np.where(deviation > 0, np.inf, 0.0),
            where=sigma > 0,
        )
        if not passed:
            failures.append(name)
        rows.append(
            f"| {name} | {p_old.size} | {n_old:,} / {n_new:,} "
            f"| {tolerance.min():.1e} to {tolerance.max():.1e} "
            f"| {deviation.max(initial=0.0):.2e} | {z_scores.max(initial=0.0):.2f} "
            f"| {'pass' if passed else 'FAIL'} |"
        )
    return rows, failures


def benchmark_rows(
    old_runs: list[dict], new_runs: list[dict]
) -> tuple[list[str], list[str], list[str]]:
    """Return the benchmark table, circuits not run, and metrics worse than today.

    Each metric worse than today in at least one run is described with its
    ratio in every run, and whether it was worse in all of them. Noisy
    throughput ratios are marked and not counted; a footnote then ends the table.
    """
    rows = [
        "| Circuit | Run | Strategy | Compile s (old / new) | New/old "
        "| Terms (old / new) | Shots/s (old / new) | New/old | Load (old / new) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    not_run = []
    worse = []
    any_noisy = False
    for name, (_, strategy) in clifft().items():
        compile_ratios, rate_ratios = [], []
        for run, (old, new) in enumerate(zip(old_runs, new_runs, strict=True), 1):
            missing = missing_cell(name, old, new)
            if missing is not None:
                rows.append(f"| {name} | {run} | {strategy} | {missing} | | | | | |")
                not_run.append(f"{name} (run {run})")
                continue
            a, b = old[name], new[name]
            compile_ratio = b["compile_seconds"] / a["compile_seconds"]
            rate_ratio = b["shots_per_second"] / a["shots_per_second"]
            compile_ratios.append(compile_ratio)
            noisy = min(a["sample_seconds"], b["sample_seconds"]) < NOISY_SECONDS
            any_noisy = any_noisy or noisy
            if not noisy:
                rate_ratios.append(rate_ratio)
            marker = "\\*" if noisy else ""
            rows.append(
                f"| {name} | {run} | {strategy} "
                f"| {a['compile_seconds']:#.3g} / {b['compile_seconds']:#.3g} "
                f"| {compile_ratio:.2f} | {a['terms']:,} / {b['terms']:,} "
                f"| {a['shots_per_second']:,.0f} / {b['shots_per_second']:,.0f} "
                f"| {rate_ratio:.2f}{marker} "
                f"| {a['load_average']:.1f} / {b['load_average']:.1f} |"
            )
        for metric, ratios, is_worse in (
            ("compile time", compile_ratios, lambda ratio: ratio > 1),
            ("sampling rate", rate_ratios, lambda ratio: ratio < 1),
        ):
            num_worse = sum(map(is_worse, ratios))
            if num_worse:
                every = "every run" if num_worse == len(ratios) else "some runs"
                listed = ", ".join(f"{ratio:.2f}" for ratio in ratios)
                worse.append(
                    f"{name} {metric}, worse in {every} (new/old by run: {listed})"
                )
    if any_noisy:
        rows += ["", "\\* Timed call under 10 ms; noisy, not counted in the verdict."]
    return rows, not_run, worse


def main() -> None:
    """Compare the JSON files of both versions and write the Markdown report."""
    old_dir, new_dir, output = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    old_benchmarks = sorted(old_dir.glob("benchmark*.json"))
    new_benchmarks = sorted(new_dir.glob("benchmark*.json"))
    if not old_benchmarks or len(old_benchmarks) != len(new_benchmarks):
        sys.exit(
            f"error: {len(old_benchmarks)} old and {len(new_benchmarks)} new "
            "benchmark files; each side needs the same number, at least one"
        )
    pairs = [
        (old_dir / "probabilities.json", new_dir / "probabilities.json"),
        (old_dir / "rates.json", new_dir / "rates.json"),
        *zip(old_benchmarks, new_benchmarks, strict=True),
    ]
    loaded = [(load(old_path), load(new_path)) for old_path, new_path in pairs]
    provenance = [
        "| File (old / new) | Old `tsim.__file__` | New `tsim.__file__` |",
        "|---|---|---|",
    ]
    for (old_path, new_path), (old, new) in zip(pairs, loaded, strict=True):
        if old["tsim"] == new["tsim"]:
            sys.exit(
                f"error: {old_path} and {new_path} both ran with tsim at "
                f"{old['tsim']}; the two versions must come from different checkouts"
            )
        provenance.append(
            f"| {old_path.name} / {new_path.name} | `{old['tsim']}` "
            f"| `{new['tsim']}` |"
        )
    (old_probabilities, new_probabilities), (old_rates, new_rates) = loaded[:2]

    probabilities, probability_failures = probability_rows(
        old_probabilities["circuits"], new_probabilities["circuits"]
    )
    rates, rate_failures = rate_rows(old_rates["circuits"], new_rates["circuits"])
    benchmarks, not_benchmarked, worse = benchmark_rows(
        [old["circuits"] for old, _ in loaded[2:]],
        [new["circuits"] for _, new in loaded[2:]],
    )

    failed = bool(probability_failures or rate_failures)
    if failed:
        failing = ", ".join(
            [f"probabilities: {name}" for name in probability_failures]
            + [f"rates: {name}" for name in rate_failures]
        )
        verdict = f"**FAIL.** Failing rows: {failing}."
    else:
        verdict = (
            f"**PASS.** All {len(probabilities) - 2} exact-probability rows and all "
            f"{len(rates) - 2} detector-rate rows agree within tolerance."
        )
    num_runs = len(loaded) - 2
    verdict += f"\n\nBenchmarks ({num_runs} runs, reported, not gated): "
    if worse:
        verdict += "worse than today: " + "; ".join(worse) + ". "
    else:
        verdict += "nothing is worse than today. "
    verdict += "Every other counted ratio is the same as or better than today's."
    if not_benchmarked:
        verdict += f"\n\nNot benchmarked: {', '.join(not_benchmarked)}."
    elif not failed:
        verdict += (
            "\n\nEvery circuit in `circuits.py` ran in both versions; none was "
            "dropped. Three clifft-bench workloads that exceed its compile limit "
            "were never part of the comparison (see Benchmarks)."
        )
    sections = [
        INTRO,
        f"## Verdict\n\n{verdict}\n",
        PROBABILITIES + "\n" + "\n".join(probabilities) + "\n",
        RATES + "\n" + "\n".join(rates) + "\n",
        BENCHMARKS + "\n" + "\n".join(benchmarks) + "\n",
        "## Tsim imported by each run\n\n" + "\n".join(provenance) + "\n",
    ]
    Path(output).write_text("\n".join(sections))
    print(verdict)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
