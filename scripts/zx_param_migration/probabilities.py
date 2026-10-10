"""Compute exact probabilities of the recorded measurement states with one Tsim.

Run once per Tsim version, from inside that version's checkout so that
``uv run`` picks its environment, giving this script's absolute path:

    uv run python /path/to/scripts/zx_param_migration/probabilities.py \
        STATES_JSON OUT_JSON

STATES_JSON is written by ``states.py``. For each circuit this builds
``CompiledStateProbs`` (joint mode) on the noiseless circuit with seed 0 and
writes ``{name: [probability of each state, ...]}`` to OUT_JSON.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import tsim
from tsim.sampler import CompiledStateProbs


def main() -> None:
    """Write the probability of every recorded state to the given JSON path."""
    print(f"tsim: {tsim.__file__}", flush=True)
    entries = json.loads(Path(sys.argv[1]).read_text())
    probabilities = {}
    for name, entry in entries.items():
        start = time.perf_counter()
        circuit = tsim.Circuit(entry["text"]).without_noise()
        state_probs = CompiledStateProbs(circuit, strategy=entry["strategy"], seed=0)
        probabilities[name] = [
            float(
                state_probs.probability_of(np.array(state, dtype=bool), batch_size=1)[0]
            )
            for state in entry["states"]
        ]
        elapsed = time.perf_counter() - start
        print(f"{name}: {probabilities[name]}, {elapsed:.1f} s", flush=True)
    Path(sys.argv[2]).write_text(json.dumps(probabilities, indent=1))


if __name__ == "__main__":
    main()
