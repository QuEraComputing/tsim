"""trajectory_sharing=True end to end, against one lane per shot."""

from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import numpy as np
import pytest

import tsim.sampler as sampler_module
import tsim.utils.kernels as kernels
from tsim.circuit import Circuit

# NOTE: t_ghz plus a direct detector: postselection only takes the pruning path
# when it masks a direct detector, and t_ghz's random detectors give reference
# bits of 1 for some seeds.
GHZ_WITH_DIRECT_DETECTOR = """
R 0 1 2 3
H 0
CNOT 0 1
CNOT 1 2
T 0 1 2
H 0 1 2
DEPOLARIZE1(0.05) 0 1 2
T 0 1
CZ 0 1
H 0 1
M 0 1 2
DETECTOR rec[-3] rec[-2]
DETECTOR rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
X_ERROR(0.2) 3
M 3
DETECTOR rec[-1]
"""


def _accepted(sampler, shots, mask, *, reference=False):
    detectors, observables = sampler.sample(
        shots,
        batch_size=5_000,
        separate_observables=True,
        postselection_mask=mask,
        use_detector_reference_sample=reference,
    )
    keep = ~detectors.any(axis=1)
    return keep.mean(), np.concatenate([detectors, observables], axis=1)[keep]


@pytest.mark.parametrize("reference", [False, True])
def test_pruning_through_the_sampler_keeps_the_accepted_distribution(reference):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    shots = 40_000
    seed = 1
    if reference:
        for seed in range(1, 40):
            sampler = circuit.compile_detector_sampler(
                strategy="active_statevector", seed=seed
            )
            if sampler._compute_reference_sample()[:2].any():
                break
    rate, expected = _accepted(
        circuit.compile_detector_sampler(strategy="active_statevector", seed=seed),
        shots,
        mask,
        reference=reference,
    )
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=seed, trajectory_sharing=True
    )
    shared_rate, got = _accepted(sampler, shots, mask, reference=reference)
    assert any(stats["dead"] > 0 for stats in sampler.sharing_stats)
    sigma = np.sqrt(rate * (1 - rate) * 2 / shots)
    assert abs(shared_rate - rate) <= 5 * sigma + 3 / shots
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_shared_detector_sampling_matches_one_lane_per_shot(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=1
    ).sample(shots, batch_size=5_000, append_observables=True)
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=2, trajectory_sharing=True
    )
    got = sampler.sample(shots, batch_size=5_000, append_observables=True)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_shared_measurement_sampling_matches_one_lane_per_shot(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_sampler(strategy="active_statevector", seed=1).sample(
        shots, batch_size=5_000
    )
    sampler = circuit.compile_sampler(
        strategy="active_statevector", seed=2, trajectory_sharing=True
    )
    assert_counts_match(sampler.sample(shots, batch_size=5_000), expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_pruned_postselection_keeps_the_accepted_distribution(name):
    circuit = Circuit(CIRCUITS[name])
    if circuit.num_detectors == 0:
        pytest.skip("no detectors to postselect")
    shots = 20_000
    mask = np.ones(circuit.num_detectors, dtype=bool)
    rate, expected = _accepted(
        circuit.compile_detector_sampler(strategy="active_statevector", seed=1),
        shots,
        mask,
    )
    shared_rate, got = _accepted(
        circuit.compile_detector_sampler(
            strategy="active_statevector", seed=2, trajectory_sharing=True
        ),
        shots,
        mask,
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
    expected = circuit.compile_sampler(strategy="active_statevector", seed=seed).sample(
        shots, batch_size=shots
    )
    sampler = circuit.compile_sampler(
        strategy="active_statevector", seed=seed + 10, trajectory_sharing=True
    )
    assert_counts_match(sampler.sample(shots, batch_size=shots), expected)


def test_trajectory_sharing_needs_the_statevector_strategy():
    with pytest.raises(ValueError, match="trajectory_sharing"):
        Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(
            strategy="cat5", trajectory_sharing=True
        )


@pytest.mark.parametrize("screen_shots", [None, 64])
@pytest.mark.parametrize("postselect", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_shared_samples_do_not_depend_on_the_thread_count(
    postselect, reference, screen_shots, monkeypatch
):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool) if postselect else None
    if screen_shots is not None:
        monkeypatch.setattr(sampler_module, "SCREEN_SHOTS", screen_shots)

    def sample(threads):
        sampler = circuit.compile_detector_sampler(
            strategy="active_statevector", seed=3, trajectory_sharing=True
        )
        sampler._threads = lambda: threads
        return [
            sampler.sample(
                5_003,
                batch_size=300,
                append_observables=True,
                postselection_mask=mask,
                use_detector_reference_sample=reference,
            )
            for _ in range(2)
        ]

    # NOTE: the serial baseline runs numba's parallel kernels, which the worker
    # threads replace by serial ones.
    with monkeypatch.context() as patch:
        patch.setattr(kernels, "PARALLEL_WORK", 0)
        expected = sample(1)
    for threads in (2, 5):
        for got, want in zip(sample(threads), expected, strict=True):
            np.testing.assert_array_equal(got, want)


def test_shared_measurement_samples_do_not_depend_on_the_thread_count():
    circuit = Circuit(CIRCUITS["midcircuit_feedforward"])

    def sample(threads):
        sampler = circuit.compile_sampler(
            strategy="active_statevector", seed=4, trajectory_sharing=True
        )
        sampler._threads = lambda: threads
        return sampler.sample(2_001, batch_size=250)

    np.testing.assert_array_equal(sample(4), sample(1))
