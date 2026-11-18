"""Experimental helpers, outside of Tsim's stable API.

The objects here may change or be removed in any release, including patch
releases.

- `CompiledStateProbs` computes the probabilities of given measurement or
  detector outcomes with a stabilizer rank decomposition.
- `SteaneEncoder` and `ColorEncoder5` encode a circuit transversally into the
  [[7,1,3]] Steane code and the [[17,1,5]] color code.
- `shortest_errors` finds a circuit's fault distance: every smallest set of
  faults that flips an observable without flipping a detector, as `LogicalError`s
  of `Fault`s.
- `count_determined_measurements`, `missing_detectors` and
  `nondeterministic_detectors` find the parities of measurements that every
  noiseless shot gives the same value, as Stim's methods of those names do for
  Clifford circuits, and the declared detectors that are random.
"""

from tsim.determined import (
    count_determined_measurements as count_determined_measurements,
    missing_detectors as missing_detectors,
    nondeterministic_detectors as nondeterministic_detectors,
)
from tsim.noise.faults import (
    Fault as Fault,
    LogicalError as LogicalError,
    shortest_errors as shortest_errors,
)
from tsim.sampler import CompiledStateProbs as CompiledStateProbs
from tsim.utils.encoder import (
    ColorEncoder5 as ColorEncoder5,
    SteaneEncoder as SteaneEncoder,
)

__all__ = [
    "ColorEncoder5",
    "CompiledStateProbs",
    "Fault",
    "LogicalError",
    "SteaneEncoder",
    "count_determined_measurements",
    "missing_detectors",
    "nondeterministic_detectors",
    "shortest_errors",
]
