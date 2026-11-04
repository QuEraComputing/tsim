"""Experimental helpers, outside of Tsim's stable API.

The objects here may change or be removed in any release, including patch
releases.

- `CompiledStateProbs` computes the probabilities of given measurement or
  detector outcomes with a stabilizer rank decomposition.
- `SteaneEncoder` and `ColorEncoder5` encode a circuit transversally into the
  [[7,1,3]] Steane code and the [[17,1,5]] color code.
"""

from tsim.sampler import CompiledStateProbs as CompiledStateProbs
from tsim.utils.encoder import (
    ColorEncoder5 as ColorEncoder5,
    SteaneEncoder as SteaneEncoder,
)

__all__ = ["ColorEncoder5", "CompiledStateProbs", "SteaneEncoder"]
