"""Check a postselection plan for a Tsim circuit before sampling with it.

    python check_mask.py CIRCUIT.stim --summary
    python check_mask.py CIRCUIT.stim PLAN.json [--out mask.npy] [--reference] [--shots N]

``--summary`` lists the detectors in groups of consecutive detectors with the same time
coordinate, to help write the plan. The plan assigns every detector to a named stage with an
action:

    {"stages": [
        {"name": "injection", "detectors": "0-6", "action": "postselect"},
        {"name": "cultivation", "detectors": ["7-13"], "action": "postselect"},
        {"name": "final check", "detectors": "14-19", "action": "ignore"}
    ]}

``detectors`` is an index, an inclusive range "a-b", or a list of those. ``action`` is
"postselect" (masked), "decode" (sent to a decoder) or "ignore". The check fails (exit code 1)
if a detector is in no stage or in two, or if a postselected detector reads 1 in the noiseless
circuit, which would discard shots that have no error. It then reports the acceptance of each
stage from a short noisy run and, with ``--out``, writes the mask for
``sampler.sample(..., postselection_mask=np.load("mask.npy"))``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import tsim

ACTIONS = ("postselect", "decode", "ignore")
# Sampling a wider circuit for a quick check can take minutes; above this, pass --force.
MAX_QUICK_WIDTH = 16
# Cap on the boolean detector array of the noisy run (one byte per detector per shot).
MAX_SAMPLE_BYTES = 1 << 28


def parse_indices(spec, num_detectors: int) -> list[int]:
    """Return the detector indices of ``spec`` (an int, "a-b", or a list of those)."""
    items = spec if isinstance(spec, list) else [spec]
    out: list[int] = []
    for item in items:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and "-" in item:
            lo, hi = (int(x) for x in item.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(item))
    bad = [i for i in out if not 0 <= i < num_detectors]
    if bad:
        raise ValueError(
            f"detector indices out of range 0-{num_detectors - 1}: {fmt_indices(bad)}"
        )
    return out


def fmt_indices(indices) -> str:
    """Format sorted indices as compact ranges, e.g. "0-6, 9, 12-13"."""
    indices = sorted(set(int(i) for i in indices))
    parts, start = [], None
    for k, i in enumerate(indices):
        if start is None:
            start = i
        if k + 1 == len(indices) or indices[k + 1] != i + 1:
            parts.append(str(start) if start == i else f"{start}-{i}")
            start = None
    return ", ".join(parts) if parts else "none"


def summary(circuit: tsim.Circuit, time_index: int) -> None:
    """Print consecutive detectors grouped by their time coordinate."""
    coords = circuit.stim_circuit.get_detector_coordinates()
    print(f"{circuit.num_detectors} detectors, {circuit.num_observables} observables")
    groups: list[tuple[object, int, int]] = []
    extras: list[set] = []
    for i in range(circuit.num_detectors):
        c = coords.get(i, [])
        key = (len(c), c[time_index] if len(c) > time_index else None)
        if groups and groups[-1][0] == key:
            groups[-1] = (key, groups[-1][1], i)
        else:
            groups.append((key, i, i))
            extras.append(set())
        # Coordinates after the time often label the stage or the patch.
        extras[-1].add(tuple(c[time_index + 1 :]))
    rows = list(zip(groups, extras))
    if len(rows) > 80:
        print(f"{len(rows)} groups; showing the first and last 40")
        rows = rows[:40] + [None] + rows[-40:]
    for row in rows:
        if row is None:
            print("  ...")
            continue
        (key, lo, hi), extra = row
        n, t = key
        where = f"time coordinate {t:g}" if t is not None else f"{n} coordinates"
        tails = sorted(e for e in extra if e)
        tail = f"; coordinates after time {tails[:3]}" if tails else ""
        print(f"  D{lo}-D{hi} ({hi - lo + 1}): {where}{tail}")


def check(circuit: tsim.Circuit, plan: dict, args) -> int:
    """Validate ``plan`` against ``circuit``; return the number of errors."""
    n = circuit.num_detectors
    errors = 0
    owner: dict[int, str] = {}
    stages = []
    for stage in plan["stages"]:
        name, action = stage["name"], stage.get("action", "postselect")
        if action not in ACTIONS:
            print(
                f"ERROR: stage {name!r}: action must be one of {ACTIONS}, got {action!r}"
            )
            errors += 1
            continue
        indices = parse_indices(stage["detectors"], n)
        twice = [i for i in indices if i in owner]
        if twice:
            print(f"ERROR: stage {name!r} repeats detectors already in a stage: "
                  f"{fmt_indices(twice)}")  # fmt: skip
            errors += 1
        owner.update((i, name) for i in indices)
        stages.append((name, action, np.array(sorted(set(indices)), dtype=int)))
    missing = sorted(set(range(n)) - set(owner))
    if missing:
        print(f"ERROR: detectors in no stage: {fmt_indices(missing)}")
        errors += 1

    mask = np.zeros(n, dtype=bool)
    for _, action, idx in stages:
        if action == "postselect":
            mask[idx] = True

    print(f"{n} detectors; plan:")
    for name, action, idx in stages:
        print(f"  {action:10s} {name}: {len(idx)} detectors ({fmt_indices(idx)})")

    sampler = circuit.compile_detector_sampler(seed=args.seed)
    width = sampler.active_width
    print(f"active width {width}")
    quick = width is None or width <= MAX_QUICK_WIDTH or args.force

    ref = {
        "use_detector_reference_sample": args.reference,
        "use_observable_reference_sample": args.reference,
    }
    if quick:
        noiseless = circuit.without_noise().compile_detector_sampler(seed=args.seed)
        dets = noiseless.sample(shots=args.noiseless_shots, **ref)
        rate = dets.mean(axis=0)
        always, random = np.flatnonzero(rate == 1), np.flatnonzero(
            (rate > 0) & (rate < 1)
        )
        for label, idx, fix in (
            ("are 1 in every noiseless shot", always,
             "fix their definition, or pass --reference to sample with "
             "use_detector_reference_sample=True"),
            ("are random without noise", random,
             "postselecting on them discards shots that have no error"),
        ):  # fmt: skip
            bad = idx[mask[idx]]
            if len(bad):
                print(
                    f"ERROR: postselected detectors {fmt_indices(bad)} {label}; {fix}"
                )
                errors += 1
            other = idx[~mask[idx]]
            if len(other):
                print(
                    f"note: detectors {fmt_indices(other)} {label} (not postselected)"
                )
    else:
        print(f"skipping the sampled checks: width {width} > {MAX_QUICK_WIDTH} "
              "(pass --force to run them)")  # fmt: skip

    if quick and args.shots > 0:
        shots = int(min(args.shots, max(1, MAX_SAMPLE_BYTES // max(n, 1))))
        dets = sampler.sample(shots=shots, **ref)
        print(f"noisy run, {shots} shots: acceptance per postselected stage")
        for name, action, idx in stages:
            if action == "postselect" and len(idx):
                print(f"  {name}: {1 - dets[:, idx].any(axis=1).mean():.4f}")
        print(f"  all postselected stages: {1 - dets[:, mask].any(axis=1).mean():.4f}")

    if args.out and not errors:
        np.save(args.out, mask)
        print(f"wrote {args.out} ({int(mask.sum())} of {n} detectors masked)")
    print("OK" if not errors else f"{errors} error(s); fix the plan and run again")
    return errors


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("circuit", type=Path, help="circuit file in Stim/Tsim format")
    parser.add_argument("plan", type=Path, nargs="?", help="JSON plan (see module doc)")
    parser.add_argument("--summary", action="store_true", help="list detector groups")
    parser.add_argument("--time-index", type=int, default=2,
                        help="coordinate that holds time, for --summary (default 2)")  # fmt: skip
    parser.add_argument("--out", help="write the boolean mask to this .npy file")
    parser.add_argument("--reference", action="store_true",
                        help="check with use_detector_reference_sample=True")  # fmt: skip
    parser.add_argument(
        "--shots", type=int, default=100_000, help="noisy shots, 0 to skip"
    )
    parser.add_argument("--noiseless-shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="sample even if wide")
    args = parser.parse_args()

    circuit = tsim.Circuit.from_file(str(args.circuit))
    if args.summary or args.plan is None:
        summary(circuit, args.time_index)
        return
    plan = json.loads(args.plan.read_text())
    try:
        errors = check(circuit, plan, args)
    except (KeyError, ValueError) as error:
        print(f"ERROR: {error}")
        errors = 1
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
