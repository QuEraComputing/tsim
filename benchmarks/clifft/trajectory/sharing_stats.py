"""Measure how much trajectory sharing the clifft-bench statevector components allow.

Run from the repository root:

    uv run python benchmarks/clifft/trajectory/sharing_stats.py [--lanes N ...] [--all-shots]
        [--mode detectors|measurements] [--noiseless] [NAME ...]

For every workload this samples f-bits, keeps the shots that survive
postselection on every direct detector (as the benchmark does, so these are the
shots that reach the statevector component), and runs the statevector executor
action by action on ``N`` lanes. After every action it counts

- histories: distinct sequences of rotation signs and active-measurement
  outcomes so far. Lanes with the same history have the same amplitudes, so this
  is the number of lanes a splitting executor needs;
- f-histories: the same, but only the f-bit part of the rotation signs. This is
  the grouping that depends on noise alone;
- states: distinct amplitude arrays (rounded to 2^-20), which also merges
  different histories that reach the same state.

Array work is counted as one unit per lane for every ``ROTATE*`` and
``MEASURE*`` action (dormant measurements touch no amplitudes).
The shared executor's work is the number of histories after each such action.

``--mode measurements`` samples raw measurement records instead of detectors
(there are no detectors to postselect on, so every shot is kept), and
``--noiseless`` zeroes every f-bit, which leaves only the branching that the
quantum randomness causes. For several statevector components the widest is
profiled.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tsim
from tsim.statevector.execute import _sign, initial_state, outputs, run
from tsim.statevector.program import (
    MEASURE,
    MEASURE_DIAG,
    ROTATE,
    ROTATE_DIAG,
    StatevectorComponent,
)
from tsim.statevector.shared import sharing_layout

sys.path.insert(0, str(Path(__file__).parents[1] / "statevector"))
from benchmark import CIRCUITS, WORKLOADS

ARRAY_KINDS = (ROTATE_DIAG, ROTATE, MEASURE_DIAG, MEASURE)


def survivor_f_params(sampler, num_detectors: int, lanes: int, postselect: bool):
    """Return f-params of ``lanes`` shots that survive every direct detector."""
    mask = sampler._direct_detector_mask
    kept, attempted, total = [], 0, 0
    while total < lanes:
        f_params = sampler._channel_sampler.sample(65536)
        direct = sampler._compute_direct_outputs(f_params)[:, :num_detectors]
        keep = (
            ~(direct & mask).any(axis=1) if postselect else np.ones(len(f_params), bool)
        )
        kept.append(f_params[keep])
        attempted += len(f_params)
        total += int(keep.sum())
    return np.concatenate(kept)[:lanes], total / attempted


@jax.jit
def _step(component, state, key, a):
    return run(component, state, key, a, a + 1)


@jax.jit
def _bits(component, before, after, a, f_words):
    """Return (full sign or outcome bit, f-part of the sign) of action ``a``."""
    sign = _sign(component, before, a)
    ones = jnp.sum(
        jax.lax.population_count(before.symbols & component.sign_row[a] & f_words),
        axis=1,
    )
    f_sign = (ones & 1) == 1
    branch = jnp.maximum(component.branch[a], 0)
    word = after.symbols[:, branch // 32]
    outcome = ((word >> (branch % 32).astype(jnp.uint32)) & 1) == 1
    rotation = (component.kind[a] == ROTATE) | (component.kind[a] == ROTATE_DIAG)
    return jnp.where(rotation, sign, outcome), jnp.where(rotation, f_sign, False)


@jax.jit
def _state_hash(amps, weights):
    quantized = jnp.round(
        jnp.stack([jnp.real(amps), jnp.imag(amps)], axis=-1) * 2.0**20
    ).astype(jnp.int32)
    flat = quantized.reshape(amps.shape[0], -1).astype(jnp.uint32)
    return jnp.stack(
        [jnp.sum(flat * w, axis=1, dtype=jnp.uint32) for w in weights], axis=1
    )


def f_word_mask(component: StatevectorComponent) -> np.ndarray:
    """Return the symbol words with every f-bit set."""
    num_f = len(component.f_selection)
    bits = np.zeros(component.num_words * 32, dtype=np.uint64)
    bits[:num_f] = 1
    words = bits.reshape(component.num_words, 32) << np.arange(32, dtype=np.uint64)
    return words.sum(axis=1).astype(np.uint32)


def mix(keys: np.ndarray, bits: np.ndarray, salt: int) -> np.ndarray:
    """Return history hashes extended by one bit per lane."""
    multiplier = np.uint64(0x9E3779B97F4A7C15)
    salted = np.uint64((salt * 0xBF58476D1CE4E5B9 + 1) % 2**64)
    return (keys ^ (bits.astype(np.uint64) * salted)) * multiplier


def profile_component(
    component, f_bits: np.ndarray, seed: int, postselected: np.ndarray
) -> dict:
    """Step the executor action by action and count distinct lanes.

    ``postselected[s]`` says whether output slot ``s`` is postselected; a lane
    whose postselected output is 1 is dead from the action that decides it on.
    """
    lanes = f_bits.shape[0]
    state = initial_state(component, jnp.asarray(f_bits))
    key = jax.random.key(seed)
    kinds = np.asarray(component.kind)
    tested = np.asarray(sharing_layout(component, postselected).prune_outputs)
    alive = np.ones(lanes, dtype=bool)
    rng = np.random.default_rng(seed)
    weights = [
        jnp.asarray(
            rng.integers(0, 2**32, 2 * 2**component.width, dtype=np.uint64)
        ).astype(jnp.uint32)
        for _ in range(2)
    ]
    f_words = jnp.asarray(f_word_mask(component))
    histories = np.zeros(lanes, dtype=np.uint64)
    f_histories = np.zeros(lanes, dtype=np.uint64)
    rows = []
    for a in range(component.num_actions):
        after = _step(component, state, key, a)
        bit, f_bit = (np.asarray(x) for x in _bits(component, state, after, a, f_words))
        state = after
        if tested[a].any():
            alive &= ~np.asarray(outputs(component, state))[:, tested[a]].any(axis=1)
        if kinds[a] not in ARRAY_KINDS:
            continue
        histories = mix(histories, bit, 2 * a + 1)
        f_histories = mix(f_histories, f_bit, 2 * a + 2)
        hashes = np.asarray(_state_hash(state.amps, weights)).view(np.uint64).ravel()
        _, group_sizes = np.unique(histories, return_counts=True)
        rows.append(
            {
                "action": a,
                "kind": int(kinds[a]),
                "histories": len(group_sizes),
                "f_histories": len(np.unique(f_histories)),
                "states": len(np.unique(hashes)),
                "largest": int(group_sizes.max()),
                "alive": int(alive.sum()),
                "alive_histories": len(np.unique(histories[alive])),
            }
        )
    return {"lanes": lanes, "rows": rows}


def summarize(name: str, result: dict) -> dict:
    """Return the final lane counts and work ratios of one profile."""
    rows, lanes = result["rows"], result["lanes"]
    work_lane = lanes * len(rows)
    work_history = sum(r["histories"] for r in rows)
    work_f = sum(r["f_histories"] for r in rows)
    work_state = sum(r["states"] for r in rows)
    work_pruned = sum(r["alive_histories"] for r in rows)
    return {
        "workload": name,
        "lanes": lanes,
        "array actions": len(rows),
        "final histories": rows[-1]["histories"],
        "final f-histories": rows[-1]["f_histories"],
        "final states": rows[-1]["states"],
        "largest final lane": rows[-1]["largest"] / lanes,
        "final alive": rows[-1]["alive"] / lanes,
        "final alive histories": rows[-1]["alive_histories"],
        "work ratio (histories)": work_lane / work_history,
        "work ratio (f only)": work_lane / work_f,
        "work ratio (states)": work_lane / work_state,
        "work ratio (histories, pruned)": work_lane / max(work_pruned, 1),
    }


def main() -> None:
    """Print one JSON summary per workload and lane count."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--lanes", nargs="+", type=int, default=[4096, 16384, 65536])
    parser.add_argument("--max-amplitudes", type=float, default=2**28)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--all-shots", action="store_true", help="skip the postselection filter"
    )
    parser.add_argument(
        "--mode",
        choices=["detectors", "measurements"],
        default="detectors",
        help="sample detectors (the benchmark) or raw measurement records",
    )
    parser.add_argument(
        "--noiseless", action="store_true", help="zero every f-bit (no noise)"
    )
    args = parser.parse_args()
    detectors = args.mode == "detectors"
    results = []
    for name in args.names:
        file = WORKLOADS[name][0]
        circuit = tsim.Circuit.from_file(str(CIRCUITS / file))
        compile_ = (
            circuit.compile_detector_sampler if detectors else circuit.compile_sampler
        )
        sampler = compile_(strategy="statevector", seed=0)
        components = [
            c
            for c in sampler._program.components
            if isinstance(c, StatevectorComponent)
        ]
        component = max(components, key=lambda c: c.width)
        for lanes in args.lanes:
            if lanes * 2**component.width > args.max_amplitudes:
                continue
            f_params, survival = survivor_f_params(
                sampler,
                circuit.num_detectors,
                lanes,
                postselect=detectors and not args.all_shots,
            )
            f_bits = f_params[:, np.asarray(component.f_selection)].astype(bool)
            if args.noiseless:
                f_bits[:] = False
            postselected = np.array(
                [
                    detectors and index < circuit.num_detectors
                    for index in component.output_indices
                ]
            )
            start = time.perf_counter()
            result = profile_component(
                component, f_bits, seed=lanes, postselected=postselected
            )
            summary = summarize(name, result)
            summary["mode"] = args.mode
            summary["components"] = [c.width for c in components]
            summary["outputs"] = component.num_outputs
            summary["actions"] = component.num_actions
            summary["survival"] = survival
            summary["seconds"] = round(time.perf_counter() - start, 1)
            results.append({"summary": summary, "rows": result["rows"]})
            print(json.dumps(summary), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
