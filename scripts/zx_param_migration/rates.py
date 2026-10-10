"""Sample detector and observable rates of the noisy circuits with one Tsim.

Run once per Tsim version, from inside that version's checkout so that
``uv run`` picks its environment, giving this script's absolute path:

    uv run python /path/to/scripts/zx_param_migration/rates.py OUT_JSON

For every circuit in ``generated()`` (cat5) and ``clifft()`` (its own strategy)
this compiles a detector sampler with seed 0, samples 200,000 shots (20,000 for
distillation and msc_d3) in batches of 8,192 (64 for msc_d3) with the
observables appended, and writes the mean of every column as
``{"tsim": tsim.__file__, "circuits": {name: {"shots": ..., "rates": [...]}}}``
to OUT_JSON. The file is rewritten after each circuit.

The clifft-bench circuits are read from the directory in the
``CLIFFT_BENCH_CIRCUITS`` environment variable (see ``circuits.py``).
"""

import json
import sys
import time
from pathlib import Path

from circuits import Strategy, clifft, generated

import tsim

SHOTS = {"distillation": 20_000, "msc_d3": 20_000}
BATCH_SIZES = {"msc_d3": 64}


def main() -> None:
    """Write the per-column detector and observable rates to the given JSON path."""
    print(f"tsim: {tsim.__file__}", flush=True)
    workloads: dict[str, tuple[str, Strategy]] = {
        name: (text, "cat5") for name, text in generated().items()
    }
    workloads.update(clifft())
    rates = {}
    for name, (text, strategy) in workloads.items():
        shots = SHOTS.get(name, 200_000)
        batch_size = BATCH_SIZES.get(name, 8192)
        start = time.perf_counter()
        sampler = tsim.Circuit(text).compile_detector_sampler(strategy=strategy, seed=0)
        samples = sampler.sample(shots, batch_size=batch_size, append_observables=True)
        rates[name] = {"shots": shots, "rates": samples.mean(axis=0).tolist()}
        output = {"tsim": tsim.__file__, "circuits": rates}
        Path(sys.argv[1]).write_text(json.dumps(output, indent=1))
        elapsed = time.perf_counter() - start
        print(
            f"{name}: {shots} shots, {samples.shape[1]} columns, {elapsed:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
