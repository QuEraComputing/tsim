"""Differential fuzz worker: random small circuits against a dense simulator.

Run from the repository root with the checkout's Python; it prints `RESULT <json>`.
Each circuit is sampled with `active_statevector` and with `VecSampler`, an
independent dense state-vector simulator, and the two sets of measurement records
are compared outcome by outcome. A mismatch is sampled again with more shots and
other seeds before it counts, so that a statistical fluke does not fail the run.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

# NOTE: the dense simulator has no PAULI_CHANNEL gates, so they are left out.
NOISE = ("p_x_error", "p_z_error", "p_y_error", "p_depolarize1", "p_depolarize2")
ROTATIONS = ("p_r_x", "p_r_y", "p_r_z", "p_u3")


def random_case(seed: int) -> dict:
    """Draw generator settings: small circuits with varied gates, noise and depth."""
    rng = random.Random(seed)
    settings: dict[str, float] = {
        "qubits": rng.randint(1, 5),
        "depth": rng.randint(5, 90),
        "p_t": rng.choice([0, 1, 3]),
        "seed": seed,
    }
    for name in rng.sample(NOISE, rng.randint(0, 3)):
        settings[name] = rng.choice([0.3, 1.0])
    for name in rng.sample(ROTATIONS, rng.randint(0, 2)):
        settings[name] = rng.choice([0.5, 1.0])
    return settings


def mismatch(stim_circuit, shots: int, seed: int) -> str | None:
    """Return a description if Tsim and the dense simulator disagree, else None."""
    from test.helpers.statevector import assert_counts_match

    from tsim.circuit import Circuit
    from tsim.external.vec_sim.vec_sampler import VecSampler

    expected, _, _ = VecSampler(stim_circuit).sample(shots)
    sampler = Circuit.from_stim_program(stim_circuit).compile_sampler(seed=seed)
    got = sampler.sample(shots)
    try:
        assert_counts_match(got, expected)
    except AssertionError:
        return "measurement distribution differs from the dense simulator"
    packed = Circuit.from_stim_program(stim_circuit).compile_sampler(seed=seed)
    import numpy as np

    unpacked = np.unpackbits(
        packed.sample(shots, bit_packed=True), axis=1, bitorder="little"
    )[:, : got.shape[1]]
    if not np.array_equal(unpacked.astype(bool), got.astype(bool)):
        return "bit-packed samples differ from unpacked samples with the same seed"
    return None


def main() -> None:
    """Fuzz for the given time and report circuits, failures and their files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from test.helpers.gen import gen_stim_circuit

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end = time.time() + args.seconds
    seed, failures = args.start, []
    while time.time() < end:
        settings = random_case(seed)
        stim_circuit = gen_stim_circuit(**settings)
        try:
            problem = mismatch(stim_circuit, args.shots, seed)
            if problem:
                # NOTE: confirm with ten times the shots and a fresh seed.
                problem = mismatch(stim_circuit, 10 * args.shots, seed + 1_000_003)
        except Exception as error:
            problem = f"{type(error).__name__}: {error}"[:300]
        if problem:
            path = out / f"fail-{seed}.stim"
            path.write_text(str(stim_circuit))
            failures.append({"seed": seed, "problem": problem, "circuit": str(path)})
            print(f"FAILURE seed={seed}: {problem} ({path})", flush=True)
        seed += 1
    result = {"circuits": seed - args.start, "start": args.start, "failures": failures}
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
