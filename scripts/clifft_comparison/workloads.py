"""Workloads of the Tsim/Clifft comparison: clifft-bench plus SymFT's d=7 circuit.

The clifft-bench circuits are read from ``CLIFFT_BENCH_CIRCUITS`` (default
``workloads/circuits`` of a clifft-bench checkout next to the main repository),
SymFT's from ``SYMFT_TEST_CIRCUITS`` (default ``benchmark/circuit`` of a
SymFT_Test checkout next to it, github.com/haoliri0/SymFT_Test). Every
clifft-bench workload follows its manifest (``manifests/workloads.v1.json``):
every detector is postselected except on Quantum Volume, and the logical count
is observable 0 among accepted shots. SymFT benchmarks its d=7 circuit without
postselection, so it runs both ways here.
"""

import os
from dataclasses import dataclass
from pathlib import Path


def _sibling(name: str) -> Path:
    """Return the checkout ``name`` in the nearest ancestor that holds one."""
    for parent in Path(__file__).resolve().parents:
        if (parent / name).is_dir():
            return parent / name
    return Path(name)


CLIFFT_BENCH = Path(
    os.environ.get(
        "CLIFFT_BENCH_CIRCUITS", _sibling("clifft-bench") / "workloads" / "circuits"
    )
)
SYMFT_TEST = Path(
    os.environ.get(
        "SYMFT_TEST_CIRCUITS", _sibling("SymFT_Test") / "benchmark" / "circuit"
    )
)


@dataclass(frozen=True)
class Workload:
    """One circuit and how it is sampled.

    Attributes:
        path: The circuit file.
        postselect: Whether every detector is postselected.
        shots: Shots per sample call.
        batch: Tsim's batch size with one lane per shot.
        shared: Tsim's (batch, shots per call) configurations with trajectory
            sharing; empty when the circuit has no statevector component.
        clifft_shots: Clifft's shots per call, if not ``shots`` (smaller where
            Clifft takes minutes per call).

    """

    path: Path
    postselect: bool
    shots: int
    batch: int
    shared: tuple[tuple[int, int], ...] = ()
    clifft_shots: int | None = None


_LARGE = ((65_536, 262_144),)

WORKLOADS = {
    "surface_d7_r7": Workload(
        CLIFFT_BENCH / "pure_surface_d7_r7_p1e-3.stim", True, 262_144, 65_536
    ),
    "coherent_d3_r1": Workload(
        CLIFFT_BENCH / "coherent_d3_r1.stim", True, 262_144, 65_536
    ),
    "coherent_d5_r1": Workload(
        CLIFFT_BENCH / "coherent_d5_r1.stim", True, 262_144, 65_536
    ),
    "distillation": Workload(
        CLIFFT_BENCH / "distillation.stim",
        True,
        262_144,
        65_536,
        ((65_536, 262_144),),
    ),
    "cultivation_d3": Workload(
        CLIFFT_BENCH / "msc_d3_inject_cultivate_p1e-3.stim",
        True,
        262_144,
        65_536,
        ((65_536, 262_144),),
    ),
    "coherent_d3_r3": Workload(
        CLIFFT_BENCH / "coherent_d3_r3.stim",
        True,
        262_144,
        65_536,
        ((65_536, 262_144),),
    ),
    "cultivation_d5": Workload(
        CLIFFT_BENCH / "msc_d5_inject_cultivate_p1e-3.stim",
        True,
        65_536,
        16_384,
        ((16_384, 65_536), *_LARGE),
    ),
    "coherent_d5_r5": Workload(
        CLIFFT_BENCH / "coherent_d5_r5.stim",
        True,
        16_384,
        4_096,
        ((4_096, 16_384), *_LARGE),
    ),
    "qv_q10": Workload(
        CLIFFT_BENCH / "quantum_volume_q10_seed42.stim",
        False,
        65_536,
        16_384,
        ((16_384, 65_536),),
    ),
    # NOTE: width 20 needs about 24 MB per lane, so batches stay small; a shared
    # lane table holds at most one lane per shot of the call.
    "qv_q20": Workload(
        CLIFFT_BENCH / "quantum_volume_q20_seed42.stim",
        False,
        8,
        8,
        ((128, 128),),
    ),
    "symft_msc_d7": Workload(
        SYMFT_TEST / "MSC_circuit_d7_p0.0005.stim",
        False,
        16_384,
        4_096,
        ((4_096, 16_384),),
        clifft_shots=512,
    ),
    "symft_msc_d7_postselected": Workload(
        SYMFT_TEST / "MSC_circuit_d7_p0.0005.stim",
        True,
        16_384,
        4_096,
        ((4_096, 16_384), *_LARGE),
        clifft_shots=2_048,
    ),
}
