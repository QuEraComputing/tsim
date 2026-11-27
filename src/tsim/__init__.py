"""ZX-calculus based circuit sampler for universal quantum circuits.

Tsim supports fast sampling from quantum circuits with few non-Clifford gates.
It exposes `Circuit`, `CompiledDetectorSampler`, and `CompiledMeasurementSampler`
objects, which follow the Stim API.

These three objects are Tsim's public API, which follows semantic versioning.
`tsim.experimental` holds helpers that may change in any release. Every other
module is internal and may change without notice, even where it is importable.
"""

__version__ = "1.1.0"

from tsim.circuit import Circuit as Circuit
from tsim.sampler import (
    CompiledDetectorSampler as CompiledDetectorSampler,
    CompiledMeasurementSampler as CompiledMeasurementSampler,
)

__all__ = ["Circuit", "CompiledDetectorSampler", "CompiledMeasurementSampler"]
