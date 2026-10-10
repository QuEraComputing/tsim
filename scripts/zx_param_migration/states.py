"""Choose the measurement records whose exact probabilities both Tsim versions compute.

Run once, with the migrated Tsim, from this checkout:

    uv run python scripts/zx_param_migration/states.py STATES_JSON

Every circuit is taken without noise. For each circuit in ``SMALL`` (at most six
measurements) the states are all 2^n measurement records, including those of
probability 0. For ``generated()`` and ``coherent_d3_r1`` they are 8 records
drawn with seed 0, the all-zeros record, and the first 3 drawn records with one
bit flipped that lies in the support of detector 0, 1 and 2 respectively. The
states are deduplicated, and
``{name: {"text": ..., "strategy": "cat5", "states": [[0/1, ...], ...]}}`` is
written to STATES_JSON.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import stim
from circuits import SMALL, clifft, generated

import tsim

NUM_FLIPPED = 3


def detector_supports(circuit: stim.Circuit) -> list[int]:
    """Return, for each detector in order, the first measurement it refers to."""
    supports = []
    num_measurements = 0
    for instruction in circuit.flattened():
        assert isinstance(instruction, stim.CircuitInstruction)
        if instruction.name == "DETECTOR":
            offset = instruction.targets_copy()[0].value
            supports.append(num_measurements + offset)
        else:
            single = stim.Circuit()
            single.append(instruction)
            num_measurements += single.num_measurements
    return supports


def sampled_states(circuit: tsim.Circuit) -> list[list[int]]:
    """Return drawn records, the all-zeros record and detector-violating records."""
    records = circuit.compile_sampler(seed=0).sample(8, batch_size=8)
    states = [[int(bit) for bit in record] for record in records]
    states.append([0] * circuit.num_measurements)
    supports = detector_supports(circuit.stim_circuit)
    for state, index in zip(states[:NUM_FLIPPED], supports, strict=False):
        flipped = list(state)
        flipped[index] ^= 1
        states.append(flipped)
    return states


def main() -> None:
    """Write each circuit with its measurement records to the given JSON path."""
    print(f"tsim: {tsim.__file__}", flush=True)
    texts = {**SMALL, **generated(), "coherent_d3_r1": clifft()["coherent_d3_r1"][0]}
    entries = {}
    for name, text in texts.items():
        start = time.perf_counter()
        circuit = tsim.Circuit(text).without_noise()
        if name in SMALL:
            records = itertools.product((0, 1), repeat=circuit.num_measurements)
            states = [list(record) for record in records]
        else:
            states = sampled_states(circuit)
        unique = [list(state) for state in dict.fromkeys(map(tuple, states))]
        entries[name] = {"text": text, "strategy": "cat5", "states": unique}
        elapsed = time.perf_counter() - start
        print(
            f"{name}: {circuit.num_measurements} measurements, {len(unique)} states,"
            f" {elapsed:.1f} s",
            flush=True,
        )
    Path(sys.argv[1]).write_text(json.dumps(entries, indent=1))


if __name__ == "__main__":
    main()
