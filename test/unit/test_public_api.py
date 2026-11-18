import tsim
import tsim.experimental
from tsim.noise.faults import Fault, LogicalError, shortest_errors
from tsim.sampler import CompiledStateProbs
from tsim.utils.encoder import ColorEncoder5, SteaneEncoder


def test_public_api():
    assert sorted(tsim.__all__) == [
        "Circuit",
        "CompiledDetectorSampler",
        "CompiledMeasurementSampler",
    ]
    assert all(hasattr(tsim, name) for name in tsim.__all__)


def test_experimental_api():
    assert tsim.experimental.CompiledStateProbs is CompiledStateProbs
    assert tsim.experimental.ColorEncoder5 is ColorEncoder5
    assert tsim.experimental.SteaneEncoder is SteaneEncoder
    assert tsim.experimental.shortest_errors is shortest_errors
    assert tsim.experimental.LogicalError is LogicalError
    assert tsim.experimental.Fault is Fault
    assert sorted(tsim.experimental.__all__) == [
        "ColorEncoder5",
        "CompiledStateProbs",
        "Fault",
        "LogicalError",
        "SteaneEncoder",
        "shortest_errors",
    ]
