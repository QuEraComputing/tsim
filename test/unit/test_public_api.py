import tsim
import tsim.experimental
from tsim.sampler import CompiledStateProbs
from tsim.utils.encoder import ColorEncoder5, SteaneEncoder


def test_public_api():
    assert sorted(tsim.__all__) == [
        "Circuit",
        "CompiledDetectorSampler",
        "CompiledMeasurementSampler",
    ]
    for name in tsim.__all__:
        assert getattr(tsim, name).__module__.startswith("tsim.")


def test_experimental_api():
    assert tsim.experimental.CompiledStateProbs is CompiledStateProbs
    assert tsim.experimental.ColorEncoder5 is ColorEncoder5
    assert tsim.experimental.SteaneEncoder is SteaneEncoder
    assert sorted(tsim.experimental.__all__) == [
        "ColorEncoder5",
        "CompiledStateProbs",
        "SteaneEncoder",
    ]
