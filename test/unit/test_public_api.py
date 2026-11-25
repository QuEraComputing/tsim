import tsim
import tsim.experimental
from tsim.determined import (
    count_determined_measurements,
    missing_detectors,
    nondeterministic_detectors,
)
from tsim.flows import flow_generators, has_all_flows, has_flow
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
    assert (
        tsim.experimental.count_determined_measurements is count_determined_measurements
    )
    assert tsim.experimental.missing_detectors is missing_detectors
    assert tsim.experimental.nondeterministic_detectors is nondeterministic_detectors
    assert tsim.experimental.flow_generators is flow_generators
    assert tsim.experimental.has_flow is has_flow
    assert tsim.experimental.has_all_flows is has_all_flows
    assert sorted(tsim.experimental.__all__) == [
        "ColorEncoder5",
        "CompiledStateProbs",
        "Fault",
        "LogicalError",
        "SteaneEncoder",
        "count_determined_measurements",
        "flow_generators",
        "has_all_flows",
        "has_flow",
        "missing_detectors",
        "nondeterministic_detectors",
        "shortest_errors",
    ]
