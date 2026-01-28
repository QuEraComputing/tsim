"""ZX-calculus based circuit sampler for universal quantum circuits.

tsim supports fast sampling from quantum circuits with few non-Clifford gates.
It exposes `Circuit`, `CompiledDetectorSampler`, and `CompiledMeasurementSampler`
objects, which follow the Stim API.

The package is organized as follows:

- `circuit.py`: The thin `Circuit` wrapper around `stim.Circuit`.
- `sampler.py`: Orchestrates compilation and evaluation to sample measurements or detectors.
- `core/`: ZX graph construction, parsing, and data types.
- `compile/`: JAX compilation pipeline including stabilizer rank decomposition.
- `noise/`: Pauli noise channels and detector error models.
- `utils/`: Utility functions for visualization and linear algebra.
"""

__version__ = "0.1.0rc1"

from tsim.circuit import Circuit as Circuit
from tsim.sampler import (
    CompiledDetectorSampler as CompiledDetectorSampler,
    CompiledMeasurementSampler as CompiledMeasurementSampler,
)
