"""The clifft-bench workloads shared by ``benchmark.py`` and ``benchmark_clifft.py``.

Circuits are read from the directory in ``CLIFFT_BENCH_CIRCUITS`` (default
``workloads/circuits`` of a clifft-bench checkout next to this repository).
"""

import math
import os
from pathlib import Path

_SIBLING_CHECKOUT = Path(__file__).resolve().parents[3] / "clifft-bench"
CIRCUITS = Path(
    os.environ.get(
        "CLIFFT_BENCH_CIRCUITS", _SIBLING_CHECKOUT / "workloads" / "circuits"
    )
)

# name: (file, Clifft 0.10.0 acceptance, shots per sample call, batch size)
WORKLOADS = {
    "distillation": ("distillation.stim", 0.0798, 262_144, 65_536),
    "cultivation_d3": ("msc_d3_inject_cultivate_p1e-3.stim", 0.6868, 262_144, 65_536),
    "coherent_d3_r3": ("coherent_d3_r3.stim", 0.3732, 262_144, 65_536),
    "cultivation_d5": ("msc_d5_inject_cultivate_p1e-3.stim", 0.1440, 65_536, 16_384),
    "coherent_d5_r5": ("coherent_d5_r5.stim", 0.0078, 16_384, 4_096),
}


def acceptance_check(acceptance: float, reference: float, attempted: int) -> str:
    """Return "ok" if ``acceptance`` is within 5σ of the reference, else "MISMATCH"."""
    sigma = math.sqrt(reference * (1 - reference) / attempted)
    return "ok" if abs(acceptance - reference) <= 5 * sigma else "MISMATCH"
