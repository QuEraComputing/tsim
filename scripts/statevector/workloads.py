"""The clifft-bench workloads shared by ``benchmark.py`` and ``benchmark_clifft.py``.

Circuits are read from the directory in ``CLIFFT_BENCH_CIRCUITS`` (default
``/Users/rafaelhaenel/Documents/quera/clifft-bench/workloads/circuits``).
"""

import os
import resource
from pathlib import Path

CIRCUITS = Path(
    os.environ.get(
        "CLIFFT_BENCH_CIRCUITS",
        "/Users/rafaelhaenel/Documents/quera/clifft-bench/workloads/circuits",
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


def cpu_seconds() -> float:
    """Return the CPU time of this process (user plus system) in seconds."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime
