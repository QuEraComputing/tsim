"""Compare the outputs of two Tsim versions and write a Markdown report.

Run from this checkout after ``probabilities.py``, ``rates.py`` and
``benchmark.py`` have written their JSON files for both versions:

    uv run python scripts/zx_param_migration/compare.py OLD_DIR NEW_DIR RESULTS_MD

OLD_DIR holds the ``probabilities.json``, ``rates.json`` and ``benchmark.json``
of today's Tsim, NEW_DIR those of the migrated Tsim. Probabilities must satisfy
``np.allclose(p_new, p_old, rtol=1e-4, atol=1e-7)``; every detector and
observable rate must satisfy ``|p_old - p_new| <= 5 * sigma + 1e-4``. Benchmarks
are reported, not gated. Exits with status 1 if a probability or rate row fails.
"""

import json
import runpy
import sys
from pathlib import Path

import numpy as np

# NOTE: circuits.py is loaded by path because deptry reports `import circuits`
# as a missing dependency.
circuits = runpy.run_path(str(Path(__file__).with_name("circuits.py")))

RTOL = 1e-4
ATOL = 1e-7
SIGMAS = 5
SLACK = 1e-4

INTRO = """\
# zx_param migration: comparison with today's Tsim

- **Old:** today's Tsim, main `3dacc1a` (built on pyzx-param).
- **New:** the migrated Tsim, branch `zx-param-migration` (built on zx_param).

Each version ran `probabilities.py`, `rates.py` and `benchmark.py` from
`scripts/zx_param_migration/` in its own checkout and environment, on the JAX
CPU backend. This file is written by `compare.py`.

Noisy circuits are not compared probability by probability: `ChannelSampler`
simplifies the noise channels with the error-basis transform, so seeded error
draws can differ between the versions. Exact probabilities are compared on
noiseless circuits, and noisy circuits through their detector and observable
rates.
"""

PROBABILITIES = f"""\
## Exact probabilities

Circuits: `SMALL`, `generated()` and `coherent_d3_r1`, each without noise.
States: 8 measurement records drawn by the migrated Tsim
(`compile_sampler(seed=0).sample(8, batch_size=8)`) plus the all-zeros record.
Each version computes
`CompiledStateProbs(circuit, strategy="cat5", seed=0).probability_of(state, batch_size=1)`
(joint mode). A row passes if
`np.allclose(p_new, p_old, rtol={RTOL:g}, atol={ATOL:g})`, that is
`|p_new - p_old| <= {ATOL:g} + {RTOL:g} * |p_old|` for every state. The relative
deviation is taken over states with `p_old > 0`.
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
"""

BENCHMARKS = """\
## Benchmarks

Circuits: `clifft()` with noise, detector mode. Compile time is one
`compile_detector_sampler(strategy=..., seed=0)` call. Terms is the sum of
`num_graphs` over every compiled scalar graph of every component (0 when all
outputs take the direct path). After one warm-up `sample(4096)` call, one
`sample(65536, batch_size=8192)` call is timed (`sample(4096, batch_size=64)`
for msc_d3). Ratios are new/old: above 1 means the new compile is slower, and
below 1 means the new sampling is slower. Single runs, one version after the
other, on a shared machine (the load column is the 1-minute load average after
each circuit, old / new). Unlike the clifft-bench harness, no postselection
mask is passed, so absolute rates differ from that harness's. Benchmarks are
reported, not gated.
"""


def load(directory: str, name: str) -> dict:
    """Return the parsed JSON file ``name`` in ``directory``."""
    return json.loads((Path(directory) / name).read_text())


def missing_cell(name: str, old: dict, new: dict) -> str | None:
    """Return a note if ``name`` is missing from either version, else None."""
    sides = [side for side, data in (("old", old), ("new", new)) if name not in data]
    return f"not run ({', '.join(sides)})" if sides else None


def probability_rows(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Return the probability table and the names of failing rows."""
    names = [*circuits["SMALL"], *circuits["generated"](), "coherent_d3_r1"]
    rows = [
        "| Circuit | States | Max abs deviation | Max rel deviation | Result |",
        "|---|---:|---:|---:|---|",
    ]
    failures = []
    for name in names:
        missing = missing_cell(name, old, new)
        if missing is not None:
            rows.append(f"| {name} | | | | FAIL: {missing} |")
            failures.append(name)
            continue
        p_old, p_new = np.array(old[name]), np.array(new[name])
        if p_old.shape != p_new.shape:
            rows.append(f"| {name} | | | | FAIL: {p_old.size} vs {p_new.size} states |")
            failures.append(name)
            continue
        deviation = np.abs(p_new - p_old)
        positive = p_old > 0
        relative = (deviation[positive] / p_old[positive]).max(initial=0.0)
        passed = bool(np.allclose(p_new, p_old, rtol=RTOL, atol=ATOL))
        if not passed:
            failures.append(name)
        rows.append(
            f"| {name} | {p_old.size} | {deviation.max():.2e} | {relative:.2e} "
            f"| {'pass' if passed else 'FAIL'} |"
        )
    return rows, failures


def rate_rows(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Return the detector-rate table and the names of failing rows."""
    names = [*circuits["generated"](), *circuits["clifft"]()]
    rows = [
        "| Circuit | Columns | Shots (old / new) | Max abs deviation | Worst z "
        "| Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    failures = []
    for name in names:
        missing = missing_cell(name, old, new)
        if missing is not None:
            rows.append(f"| {name} | | | | | FAIL: {missing} |")
            failures.append(name)
            continue
        n_old, n_new = old[name]["shots"], new[name]["shots"]
        p_old, p_new = np.array(old[name]["rates"]), np.array(new[name]["rates"])
        if p_old.shape != p_new.shape:
            rows.append(
                f"| {name} | | | | | FAIL: {p_old.size} vs {p_new.size} columns |"
            )
            failures.append(name)
            continue
        deviation = np.abs(p_old - p_new)
        sigma = np.sqrt(p_old * (1 - p_old) / n_old + p_new * (1 - p_new) / n_new)
        passed = bool(np.all(deviation <= SIGMAS * sigma + SLACK))
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
            f"| {deviation.max(initial=0.0):.2e} | {z_scores.max(initial=0.0):.2f} "
            f"| {'pass' if passed else 'FAIL'} |"
        )
    return rows, failures


def benchmark_rows(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Return the benchmark table and the names of circuits not run.

    A throughput ratio is marked with ``*`` if either timed call took under
    10 ms, which makes it noisy; a footnote then follows the table.
    """
    rows = [
        "| Circuit | Strategy | Compile s (old / new) | New/old | Terms (old / new) "
        "| Shots/s (old / new) | New/old | Load (old / new) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    not_run = []
    any_noisy = False
    for name, (_, strategy) in circuits["clifft"]().items():
        missing = missing_cell(name, old, new)
        if missing is not None:
            rows.append(f"| {name} | {strategy} | {missing} | | | | | |")
            not_run.append(name)
            continue
        a, b = old[name], new[name]
        is_noisy = min(a["sample_seconds"], b["sample_seconds"]) < 0.01
        any_noisy = any_noisy or is_noisy
        noisy = "\\*" if is_noisy else ""
        rows.append(
            f"| {name} | {strategy} "
            f"| {a['compile_seconds']:.3g} / {b['compile_seconds']:.3g} "
            f"| {b['compile_seconds'] / a['compile_seconds']:.2f} "
            f"| {a['terms']:,} / {b['terms']:,} "
            f"| {a['shots_per_second']:,.0f} / {b['shots_per_second']:,.0f} "
            f"| {b['shots_per_second'] / a['shots_per_second']:.2f}{noisy} "
            f"| {a['load_average']:.1f} / {b['load_average']:.1f} |"
        )
    if any_noisy:
        rows += ["", "\\* A timed call took under 10 ms, so this ratio is noisy."]
    return rows, not_run


def main() -> None:
    """Compare the JSON files of both versions and write the Markdown report."""
    old_dir, new_dir, output = sys.argv[1:4]
    probabilities, probability_failures = probability_rows(
        load(old_dir, "probabilities.json"), load(new_dir, "probabilities.json")
    )
    rates, rate_failures = rate_rows(
        load(old_dir, "rates.json"), load(new_dir, "rates.json")
    )
    old_benchmark = load(old_dir, "benchmark.json")
    new_benchmark = load(new_dir, "benchmark.json")
    benchmarks, not_benchmarked = benchmark_rows(
        old_benchmark["circuits"], new_benchmark["circuits"]
    )

    num_probability_rows = len(probabilities) - 2
    num_rate_rows = len(rates) - 2
    if probability_failures or rate_failures:
        failing = ", ".join(
            [f"probabilities: {name}" for name in probability_failures]
            + [f"rates: {name}" for name in rate_failures]
        )
        verdict = f"**FAIL.** Failing rows: {failing}."
    else:
        verdict = (
            f"**PASS.** All {num_probability_rows} exact-probability rows and all "
            f"{num_rate_rows} detector-rate rows agree within tolerance. "
            "Benchmarks are reported below, not gated."
        )
    if not_benchmarked:
        verdict += f" Not benchmarked: {', '.join(not_benchmarked)}."
    elif not (probability_failures or rate_failures):
        verdict += " Every circuit ran in both versions; none was dropped."
    sections = [
        INTRO,
        f"## Verdict\n\n{verdict}\n",
        PROBABILITIES + "\n" + "\n".join(probabilities) + "\n",
        RATES + "\n" + "\n".join(rates) + "\n",
        BENCHMARKS
        + "\n"
        + "\n".join(benchmarks)
        + "\n\nTsim imported by the benchmark runs: old "
        + f"`{old_benchmark['tsim']}`, new `{new_benchmark['tsim']}`.\n",
    ]
    Path(output).write_text("\n".join(sections))
    print(verdict)
    if probability_failures or rate_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
