"""Split the time of a postselected sample call into its stages.

Run from the repository root:

    uv run python scripts/trajectory/breakdown.py [NAME ...]

For every clifft-bench workload this times, per ``shots`` attempted shots (the
benchmark's shots per call and batch size):

- noise: ``ChannelSampler.sample`` for every attempted shot;
- direct: the direct outputs and the postselection test for every attempted shot;
- statevector: ``sample_statevector_component`` on the surviving shots, in
  batches of the benchmark's batch size (the rest of ``sample_program`` is
  negligible);
- call: one full ``sampler.sample`` call, for reference.

Only the statevector share can shrink with trajectory sharing.
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tsim
from tsim.statevector.execute import sample_statevector_component
from tsim.statevector.program import StatevectorComponent

sys.path.insert(0, str(Path(__file__).parents[1] / "statevector"))
from benchmark import CIRCUITS, WORKLOADS


def best_of(function, repeats: int = 3) -> float:
    """Return the fastest of ``repeats`` calls of ``function``, in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def breakdown(name: str) -> str:
    """Return one Markdown row with the stage times of workload ``name``."""
    file, _, shots, batch_size = WORKLOADS[name]
    circuit = tsim.Circuit.from_file(str(CIRCUITS / file))
    sampler = circuit.compile_detector_sampler(strategy="statevector", seed=0)
    num_detectors = circuit.num_detectors
    mask = sampler._direct_detector_mask
    components = [
        c for c in sampler._program.components if isinstance(c, StatevectorComponent)
    ]

    f_params = sampler._channel_sampler.sample(shots)
    noise = best_of(lambda: sampler._channel_sampler.sample(shots))

    def direct_stage():
        direct = sampler._compute_direct_outputs(f_params)[:, :num_detectors]
        return ~(direct & mask).any(axis=1)

    direct = best_of(direct_stage)
    survivors = f_params[direct_stage()]
    num_batches = -(-len(survivors) // batch_size)
    padded = np.resize(survivors, (num_batches * batch_size, survivors.shape[1]))
    key = jax.random.key(0)

    def statevector_stage():
        for b in range(num_batches):
            batch = jnp.asarray(padded[b * batch_size : (b + 1) * batch_size])
            for component in components:
                samples, _, _ = sample_statevector_component(component, batch, key)
                samples.block_until_ready()

    statevector_stage()
    statevector = best_of(statevector_stage)
    postselection = np.ones(num_detectors, dtype=bool)

    def call():
        sampler.sample(
            shots,
            batch_size=batch_size,
            separate_observables=True,
            postselection_mask=postselection,
        )

    call()
    whole = best_of(call)
    total = noise + direct + statevector
    return (
        f"| {name} | {len(survivors) / shots:.4f} | {noise:.3f} | {direct:.3f} | "
        f"{statevector:.3f} | {total:.3f} | {whole:.3f} | {statevector / total:.0%} |"
    )


def main() -> None:
    """Print the stage times of the requested workloads as a Markdown table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    args = parser.parse_args()
    print(
        "| circuit | survival | noise s | direct s | statevector s | sum s | call s "
        "| statevector share |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in args.names:
        print(breakdown(name), flush=True)


if __name__ == "__main__":
    main()
