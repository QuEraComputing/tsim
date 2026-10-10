"""Draw the measurement records whose exact probabilities both Tsim versions compute.

Run once, with the migrated Tsim, from this checkout:

    uv run python scripts/zx_param_migration/states.py STATES_JSON

For every circuit in ``SMALL``, ``generated()`` and ``coherent_d3_r1`` this takes
the noiseless circuit, draws 8 measurement records with seed 0, adds the
all-zeros record, and writes
``{name: {"text": ..., "strategy": "cat5", "states": [[0/1, ...], ...]}}``
to STATES_JSON.
"""

import json
import runpy
import sys
import time
from pathlib import Path

import tsim

# NOTE: circuits.py is loaded by path because deptry reports `import circuits`
# as a missing dependency.
circuits = runpy.run_path(str(Path(__file__).with_name("circuits.py")))


def main() -> None:
    """Write each circuit with its measurement records to the given JSON path."""
    print(f"tsim: {tsim.__file__}", flush=True)
    texts = {
        **circuits["SMALL"],
        **circuits["generated"](),
        "coherent_d3_r1": circuits["clifft"]()["coherent_d3_r1"][0],
    }
    entries = {}
    for name, text in texts.items():
        start = time.perf_counter()
        circuit = tsim.Circuit(text).without_noise()
        records = circuit.compile_sampler(seed=0).sample(8, batch_size=8)
        states = [[int(bit) for bit in record] for record in records]
        states.append([0] * records.shape[1])
        entries[name] = {"text": text, "strategy": "cat5", "states": states}
        elapsed = time.perf_counter() - start
        print(f"{name}: {records.shape[1]} measurements, {elapsed:.1f} s", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(entries, indent=1))


if __name__ == "__main__":
    main()
