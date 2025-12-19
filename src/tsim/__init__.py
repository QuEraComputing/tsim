"""
tsim is a ZX-calculus based circuit sampler that supports fast sampling from universal
quantum circuits with few non-Clifford gates.
It `Circuit`, `CompiledDetectorSampler`, and `CompiledMeasurementSampler`, which follow the Stim API.

The repo is organized as follows:

1. `circuit.py` provides the thin `Circuit` wrapper around `stim.Circuit`.
2. `_instructions.py` represents every instruction as ZX diagrams that are used internally to transform a `Circuit` into a ZX graph.
3. `compile.py` compiles circuits together with Pauli noise models into contiguous `jax.Array` data structures using
   stabilizer rank decomposition.
4. `evaluate.py` evaluates compiled models to generate probability ampltiudes.
5. `sampler.py` orchestrates diagram construction, compilation, and evaluation to compute marginals and autoregressively build measurement or detector samples.
"""

__version__ = "0.1.0"

from tsim.circuit import Circuit as Circuit
from tsim.sampler import (
    CompiledDetectorSampler as CompiledDetectorSampler,
    CompiledMeasurementSampler as CompiledMeasurementSampler,
)
