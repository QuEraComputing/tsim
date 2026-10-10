"""Circuits for comparing two Tsim versions (works with both).

Imported by the other scripts in this directory; it is not run on its own.
"""

from pathlib import Path

import stim

CLIFFT = Path("/Users/rafaelhaenel/Documents/quera/clifft-bench/workloads/circuits")

SMALL = {
    "bell_t": "R 0 1\nH 0\nT 0\nCNOT 0 1\nT_DAG 1\nH 0\nM 0 1",
    "u3_rotations": "R 0 1\nU3(0.3, 0.24, 0.49) 0\nCNOT 0 1\nR_Z(0.13) 1\nR_X(0.21) 0\nH 1\nM 0 1",
    "feedforward": "R 0 1\nH 0\nT 0\nM 0\nCX rec[-1] 1\nH 1\nT 1\nH 1\nM 1",
    "mpp_and_resets": "R 0 1 2\nH 0\nT 0\nCNOT 0 1\nMPP X0*X1 Z1*Z2\nMR 0\nH 0\nT 0\nM 0 1 2",
    "toffoli": "R 0 1 2\nH 0 1\nCCX 0 1 2\nH 2\nM 0 1 2",
}


def generated() -> dict[str, str]:
    """Return noisy Stim-generated QEC circuits."""
    surface = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.002,
        before_measure_flip_probability=0.002,
    )
    repetition = stim.Circuit.generated(
        "repetition_code:memory",
        distance=5,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    return {"surface_d3_r2": str(surface), "repetition_d5_r3": str(repetition)}


def clifft() -> dict[str, tuple[str, str]]:
    """Return clifft-bench circuits that compile in time, with their strategy."""
    files = {
        "surface_d7_r7": ("pure_surface_d7_r7_p1e-3.stim", "cat5"),
        "coherent_d3_r1": ("coherent_d3_r1.stim", "cat5"),
        "coherent_d5_r1": ("coherent_d5_r1.stim", "cat5"),
        "distillation": ("distillation.stim", "cat5"),
        "msc_d3": ("msc_d3_inject_cultivate_p1e-3.stim", "cutting"),
    }
    return {name: ((CLIFFT / f).read_text(), s) for name, (f, s) in files.items()}
