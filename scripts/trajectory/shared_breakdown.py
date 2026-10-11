"""Time the stages of the shared executor against the per-lane executor.

Run from the repository root:

    uv run python scripts/trajectory/shared_breakdown.py [--batch B] [NAME ...]

For every clifft-bench workload this samples one batch of shots that survive
postselection on every direct detector (the benchmark's batch size), then times
the per-lane executor, the whole shared executor and two of its stages
(grouping on the host, reconstruction), all with block_until_ready. The lane
loop column is the rest: the loop, its host checks and table resizes.
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tsim
from tsim.statevector import shared
from tsim.statevector.execute import pack_bits, sample_statevector_component
from tsim.statevector.program import StatevectorComponent

sys.path.insert(0, str(Path(__file__).parents[1] / "statevector"))
sys.path.insert(0, str(Path(__file__).parent))
from benchmark import CIRCUITS, WORKLOADS
from sharing_stats import survivor_f_params


def best_of(function, repeats: int = 5) -> tuple[float, object]:
    """Return the fastest of ``repeats`` calls in seconds, and the last result."""
    result, times = None, []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    return min(times), result


def breakdown(name: str, batch_size: int | None) -> str:
    """Return one Markdown row with the shared executor's stage times."""
    file, _, _, default_batch = WORKLOADS[name]
    batch_size = batch_size or default_batch
    circuit = tsim.Circuit.from_file(str(CIRCUITS / file))
    sampler = circuit.compile_detector_sampler(strategy="statevector", seed=0)
    component = max(
        (c for c in sampler._program.components if isinstance(c, StatevectorComponent)),
        key=lambda c: c.width,
    )
    f_params, _ = survivor_f_params(sampler, circuit.num_detectors, batch_size, True)
    f_params = jnp.asarray(f_params)
    postselected = np.array(
        [index < circuit.num_detectors for index in component.output_indices]
    )
    layout = shared.sharing_layout(component, postselected)
    key = jax.random.key(0)

    per_lane, _ = best_of(
        lambda: sample_statevector_component(component, f_params, key)[0]
    )
    shared.sample_shared_component(component, layout, f_params, key)
    total, (_, _, _, stats) = best_of(
        lambda: shared.sample_shared_component(component, layout, f_params, key)
    )

    f_bits = f_params[:, component.f_selection].astype(jnp.bool_)
    f_packed = pack_bits(f_bits, component.num_words)
    rng = np.random.default_rng(0)
    grouping, (order, start, count) = best_of(
        lambda: shared.group_on_host(
            np.asarray(shared.key_words(layout, f_packed)), rng
        )
    )
    capacity = stats["lane_capacity"]
    lanes = shared.initial_lanes(
        component,
        f_packed,
        jnp.asarray(order),
        jnp.asarray(np.resize(start, capacity)),
        jnp.asarray(np.resize(count, capacity)),
        jnp.asarray(min(len(start), capacity), dtype=jnp.int32),
        capacity=capacity,
        dead_capacity=stats["dead_capacity"],
    )
    device_order = jnp.asarray(order)
    rebuilding, _ = best_of(
        lambda: shared.reconstruct(
            component, layout, device_order, lanes, f_packed, key
        )
    )
    looping = total - grouping - rebuilding
    return (
        f"| {name} | {batch_size} | {layout.key_rows.shape[0]} | {stats['groups']} | "
        f"{stats['peak lanes']} / {stats['dead']} | {per_lane * 1e3:.1f} | "
        f"{grouping * 1e3:.1f} | {looping * 1e3:.1f} | {rebuilding * 1e3:.1f} | "
        f"{total * 1e3:.1f} | {per_lane / total:.1f}x |"
    )


def main() -> None:
    """Print the shared executor's stage times as a Markdown table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()
    print(
        "| circuit | shots | key bits | groups | peak live / dead lanes | per-lane ms "
        "| group ms | lanes and checks ms | reconstruct ms | shared ms | speedup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in args.names:
        print(breakdown(name, args.batch), flush=True)


if __name__ == "__main__":
    main()
