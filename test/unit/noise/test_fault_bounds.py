"""Tests for ``tsim.noise.fault_bounds``: upper bounds by random restriction."""

import stim

from tsim import Circuit
from tsim.experimental import shortest_errors
from tsim.noise.fault_bounds import random_restriction


def test_random_restriction_finds_true_logical_errors():
    circuit = Circuit(
        stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
            before_measure_flip_probability=0.01,
        )
    )
    every = {str(e) for e in shortest_errors(circuit, max_weight=3)}
    found = random_restriction(circuit, weight=3, fraction=0.5, trials=20)
    assert found
    assert {str(e) for e in found} <= every


def test_an_empty_subset_finds_nothing():
    circuit = Circuit("X_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    assert random_restriction(circuit, weight=1, fraction=0.0, trials=3) == []
    assert len(random_restriction(circuit, weight=1, fraction=1.0, trials=1)) == 1
