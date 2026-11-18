"""Tests for ``tsim.noise.fault_sat``: the MaxSAT form of the fault search."""

import pytest
import stim

pytest.importorskip("pysat")
pytest.importorskip("pycryptosat")

from test.unit.noise.test_faults import BELL, GHZ, NOISES

from tsim import Circuit
from tsim.experimental import shortest_errors
from tsim.noise.fault_sat import shortest_errors_sat


def _key(error):
    return (
        len(error),
        round(error.undetected_probability, 9),
        round(error.logical_flip_probability, 9),
        tuple(sorted(str(f.circuit_error_locations[0]) for f in error)),
    )


CIRCUITS = {
    "bell": BELL.format(gate="T"),
    "ghz": GHZ,
    "noises": NOISES,
    "surface": str(
        stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=2,
            after_clifford_depolarization=0.01,
            before_measure_flip_probability=0.01,
        )
    ),
}


@pytest.mark.parametrize("name", sorted(CIRCUITS))
@pytest.mark.parametrize("solver", ["cryptominisat", "cadical195"])
def test_sat_finds_the_errors_of_the_enumeration(name, solver):
    circuit = Circuit(CIRCUITS[name])
    expected = shortest_errors(circuit, max_weight=3)
    from tsim.noise.fault_sat import SatSearch

    got = SatSearch(circuit).shortest(3, every=True, solver=solver)
    assert sorted(map(_key, got)) == sorted(map(_key, expected))


def test_sat_stops_at_max_weight():
    circuit = Circuit(CIRCUITS["surface"])
    assert shortest_errors_sat(circuit, max_weight=2) == []
    (error,) = shortest_errors_sat(circuit, max_weight=3, every=False)
    assert len(error) == 3
