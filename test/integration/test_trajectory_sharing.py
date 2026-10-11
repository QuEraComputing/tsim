"""trajectory_sharing=True end to end, against one lane per shot."""

from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import numpy as np
import pytest

from tsim.circuit import Circuit


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_shared_detector_sampling_matches_one_lane_per_shot(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_detector_sampler(strategy="statevector", seed=1).sample(
        shots, batch_size=5_000, append_observables=True
    )
    sampler = circuit.compile_detector_sampler(
        strategy="statevector", seed=2, trajectory_sharing=True
    )
    got = sampler.sample(shots, batch_size=5_000, append_observables=True)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_shared_measurement_sampling_matches_one_lane_per_shot(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_sampler(strategy="statevector", seed=1).sample(
        shots, batch_size=5_000
    )
    sampler = circuit.compile_sampler(
        strategy="statevector", seed=2, trajectory_sharing=True
    )
    assert_counts_match(sampler.sample(shots, batch_size=5_000), expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_pruned_postselection_keeps_the_accepted_distribution(name):
    circuit = Circuit(CIRCUITS[name])
    if circuit.num_detectors == 0:
        pytest.skip("no detectors to postselect")
    shots = 20_000
    mask = np.ones(circuit.num_detectors, dtype=bool)

    def accepted(sampler):
        detectors, observables = sampler.sample(
            shots,
            batch_size=5_000,
            separate_observables=True,
            postselection_mask=mask,
        )
        keep = ~detectors.any(axis=1)
        return keep.mean(), np.concatenate([detectors, observables], axis=1)[keep]

    rate, expected = accepted(
        circuit.compile_detector_sampler(strategy="statevector", seed=1)
    )
    shared_rate, got = accepted(
        circuit.compile_detector_sampler(
            strategy="statevector", seed=2, trajectory_sharing=True
        )
    )
    sigma = np.sqrt(rate * (1 - rate) * 2 / shots)
    assert abs(shared_rate - rate) <= 5 * sigma + 3 / shots
    if len(got) and len(expected):
        assert_counts_match(got, expected)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_shared_sampling_of_random_noisy_circuits(seed):
    stim_circuit = gen_stim_circuit(3, 60, p_t=3, p_x_error=0.5, p_r_z=0.5, seed=seed)
    circuit = Circuit.from_stim_program(stim_circuit)
    shots = 4_096
    expected = circuit.compile_sampler(strategy="statevector", seed=seed).sample(
        shots, batch_size=shots
    )
    sampler = circuit.compile_sampler(
        strategy="statevector", seed=seed + 10, trajectory_sharing=True
    )
    assert_counts_match(sampler.sample(shots, batch_size=shots), expected)


def test_trajectory_sharing_needs_the_statevector_strategy():
    with pytest.raises(ValueError, match="trajectory_sharing"):
        Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(trajectory_sharing=True)
