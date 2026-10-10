"""Time compilation and sampling of the clifft-bench circuits with one Tsim.

Run it for one Tsim version and then the other, on an otherwise idle machine,
from inside that version's checkout so that ``uv run`` picks its environment,
giving this script's absolute path:

    uv run python /path/to/scripts/zx_param_migration/benchmark.py OUT_JSON

For several runs, repeat that pair with OUT_JSON named ``benchmark-1.json``,
``benchmark-2.json``, ... in each version's output directory; ``compare.py``
reads every ``benchmark*.json`` there.

For every circuit in ``clifft()`` this times ``compile_detector_sampler`` (its
strategy, seed 0), counts the stabilizer terms (``num_graphs`` summed over all
compiled scalar graphs), makes one warm-up ``sample(4096)`` call, then times one
``sample(65536)`` call in batches of 8,192 (4,096 shots in batches of 64 for
msc_d3). It writes ``{"tsim": tsim.__file__, "circuits": {name: {...}}}`` to
OUT_JSON, rewriting the file after each circuit.
"""

import json
import os
import sys
import time
from pathlib import Path

from circuits import clifft

import tsim

BATCH_SIZES = {"msc_d3": 64}
SHOTS = {"msc_d3": 4096}


def main() -> None:
    """Write compile time, term count and sampling rate to the given JSON path."""
    print(f"tsim: {tsim.__file__}", flush=True)
    results = {}
    for name, (text, strategy) in clifft().items():
        batch_size = BATCH_SIZES.get(name, 8192)
        shots = SHOTS.get(name, 65536)
        circuit = tsim.Circuit(text)

        start = time.perf_counter()
        sampler = circuit.compile_detector_sampler(strategy=strategy, seed=0)
        compile_seconds = time.perf_counter() - start
        terms = sum(
            graphs.num_graphs
            for component in sampler._program.components
            for graphs in component.compiled_scalar_graphs
        )

        sampler.sample(4096, batch_size=batch_size)
        start = time.perf_counter()
        sampler.sample(shots, batch_size=batch_size)
        sample_seconds = time.perf_counter() - start

        results[name] = {
            "strategy": strategy,
            "batch_size": batch_size,
            "shots": shots,
            "compile_seconds": compile_seconds,
            "terms": terms,
            "sample_seconds": sample_seconds,
            "shots_per_second": shots / sample_seconds,
            "load_average": os.getloadavg()[0],
        }
        output = {"tsim": tsim.__file__, "circuits": results}
        Path(sys.argv[1]).write_text(json.dumps(output, indent=1))
        print(f"{name}: {results[name]}", flush=True)


if __name__ == "__main__":
    main()
