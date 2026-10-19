"""ZX-calculus based circuit sampler for universal quantum circuits.

tsim supports fast sampling from quantum circuits with few non-Clifford gates.
It exposes `Circuit`, `CompiledDetectorSampler`, and `CompiledMeasurementSampler`
objects, which follow the Stim API.

The package is organized as follows:

- `circuit.py`: The thin `Circuit` wrapper around `stim.Circuit`.
- `sampler.py`: Orchestrates compilation and evaluation to sample measurements or detectors.
- `core/`: Builds quantum/classical ZX graphs with zxparam; parsing and data types.
- `compile/`: Compiles zxparam's stabilizer-rank decompositions to JAX.
- `statevector/`: Samples non-direct components as Pauli rotations and measurements on a dense state vector (`strategy="statevector"`).
- `noise/`: Pauli noise channels and detector error models.
- `utils/`: Utility functions for visualization and linear algebra.
"""

__version__ = "0.1.5"

from tsim.circuit import Circuit as Circuit
from tsim.sampler import (
    CompiledDetectorSampler as CompiledDetectorSampler,
    CompiledMeasurementSampler as CompiledMeasurementSampler,
)
