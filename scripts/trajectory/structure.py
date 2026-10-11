"""Tabulate what Tsim's representation hands the shared executor, per workload.

Run from the repository root:

    uv run python scripts/trajectory/structure.py [NAME ...]

For detector sampling (the benchmark) and for raw measurement records this
prints, per clifft-bench workload:

- noise: raw error bits, error-basis bits after the Gaussian elimination, the
  widest quantum component's f-bits, the f-bits that reach a rotation sign, and
  the rank of the rotation signs' f-parts (the grouping key without pruning);
- outputs: all outputs, those on the direct (classical) path, and the quantum
  components with their widths;
- the widest component's tape: actions, rotations, active measurements, dormant
  measurements, and in detector mode where the postselected outputs fall (mean
  position as a fraction of the tape, and the share written in its first half).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import tsim
from tsim.core.graph import prepare_graph
from tsim.statevector.program import (
    MEASURE,
    MEASURE_DIAG,
    MEASURE_DORMANT,
    ROTATE,
    ROTATE_DIAG,
    StatevectorComponent,
)
from tsim.statevector.shared import _unpack, sharing_layout

sys.path.insert(0, str(Path(__file__).parents[1] / "statevector"))
from benchmark import CIRCUITS, WORKLOADS


def describe(name: str, detectors: bool) -> str:
    """Return one Markdown row describing workload ``name`` in one mode."""
    circuit = tsim.Circuit.from_file(str(CIRCUITS / WORKLOADS[name][0]))
    prepared = prepare_graph(circuit, sample_detectors=detectors)
    compile_ = (
        circuit.compile_detector_sampler if detectors else circuit.compile_sampler
    )
    sampler = compile_(strategy="statevector", seed=0)
    program = sampler._program
    components = [c for c in program.components if isinstance(c, StatevectorComponent)]
    widths = sorted((c.width for c in components), reverse=True)
    component = max(components, key=lambda c: c.width)
    num_f = len(component.f_selection)
    kind = np.asarray(component.kind)
    rotation = np.isin(kind, (ROTATE_DIAG, ROTATE))
    bits = _unpack(np.asarray(component.sign_row), component.num_words)[:, :num_f]
    rotation_f = int(bits[rotation].any(axis=0).sum())
    key_rank = sharing_layout(component).key_rows.shape[0]
    shown = ", ".join(str(w) for w in widths[:4]) + (", …" if len(widths) > 4 else "")
    row = (
        f"| {name} | {'detectors' if detectors else 'records'} | "
        f"{prepared.error_transform.shape[1]:,} | {prepared.error_transform.shape[0]:,} | "
        f"{num_f} | {rotation_f} | {key_rank} | {program.num_outputs} | "
        f"{len(program.direct_f_indices)} | {len(components)} ({shown}) | "
        f"{component.num_actions} | {int(rotation.sum())} | "
        f"{int(np.isin(kind, (MEASURE_DIAG, MEASURE)).sum())} | "
        f"{int((kind == MEASURE_DORMANT).sum())} |"
    )
    if detectors:
        slot = np.asarray(component.slot)
        written = np.flatnonzero(slot >= 0)
        postselected = [
            a
            for a in written
            if component.output_indices[slot[a]] < circuit.num_detectors
        ]
        position = np.asarray(postselected) / max(component.num_actions - 1, 1)
        if len(position):
            row += f" {position.mean():.2f} | {np.mean(position < 0.5):.0%} |"
        else:
            row += " none | none |"
    else:
        row += " | |"
    return row


def main() -> None:
    """Print the structure of the requested workloads as a Markdown table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    args = parser.parse_args()
    print(
        "| circuit | sampling | error bits | basis bits | component f-bits "
        "| f-bits in rotation signs | key rank | outputs | direct outputs "
        "| quantum components (widths) | actions | rotations | active measurements "
        "| dormant measurements | postselected position | postselected in first half |"
    )
    print(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"
    )
    for name in args.names:
        for detectors in (True, False):
            print(describe(name, detectors), flush=True)


if __name__ == "__main__":
    main()
