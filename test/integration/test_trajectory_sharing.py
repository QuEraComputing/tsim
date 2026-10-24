"""trajectory_sharing=True end to end, against the stabilizer-rank strategy."""

from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match, shot_sampler

import numba
import numpy as np
import pytest

import tsim.statevector.shots as shots_module
import tsim.utils.kernels as kernels
from tsim.circuit import Circuit

# NOTE: t_ghz plus a direct detector (detector 2); t_ghz's random detectors give
# reference bits of 1 for some seeds.
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


def _sharing_stats(sampler):
    return shot_sampler(sampler).sharing_stats


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
            if sampler._reference_sample()[:2].any():
                break
    # NOTE: the baseline needs this sampler's reference bits, so it only stops sharing.
    rate, expected = _accepted(
        circuit.compile_detector_sampler(
            strategy="active_statevector", seed=seed, trajectory_sharing=False
        ),
        shots,
        mask,
        reference=reference,
    )
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=seed, trajectory_sharing=True
    )
    shared_rate, got = _accepted(sampler, shots, mask, reference=reference)
    assert any(stats["dead"] > 0 for stats in _sharing_stats(sampler))
    sigma = np.sqrt(rate * (1 - rate) * 2 / shots)
    assert abs(shared_rate - rate) <= 5 * sigma + 3 / shots
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_shared_detector_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_detector_sampler(strategy="cat5", seed=1).sample(
        shots, batch_size=5_000, append_observables=True
    )
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=2, trajectory_sharing=True
    )
    got = sampler.sample(shots, batch_size=5_000, append_observables=True)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_shared_measurement_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_sampler(strategy="cat5", seed=1).sample(
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
        circuit.compile_detector_sampler(strategy="cat5", seed=1), shots, mask
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
    expected = circuit.compile_sampler(strategy="cat5", seed=seed).sample(
        shots, batch_size=shots
    )
    sampler = circuit.compile_sampler(
        strategy="active_statevector", seed=seed + 10, trajectory_sharing=True
    )
    assert_counts_match(sampler.sample(shots, batch_size=shots), expected)


# NOTE: two identical, disconnected components, each with two measurements.
TWO_COMPONENTS = """
H 0
T 0
H 0
M 0
H 0
T 0
H 0
M 0
H 1
T 1
H 1
M 1
H 1
T 1
H 1
M 1
"""


@pytest.mark.parametrize("sharing", [False, True])
def test_components_draw_independent_random_numbers(sharing):
    batch, batches = 64, 1_500
    sampler = Circuit(TWO_COMPONENTS).compile_sampler(
        strategy="active_statevector", seed=3, trajectory_sharing=sharing
    )
    assert len(sampler._program.components) == 2
    m = sampler.sample(batch * batches, batch_size=batch).astype(int)
    first, second = m[:, :2], m[:, 2:]
    shots = len(m)
    for i in range(2):
        for j in range(2):
            correlation = np.corrcoef(first[:, i], second[:, j])[0, 1]
            assert abs(correlation) < 5 / np.sqrt(shots)
    # NOTE: shots in one chunk must not share draws across components either.
    per_batch = m.reshape(batches, batch, 4).sum(axis=1)
    correlation = np.corrcoef(per_batch[:, :2].sum(1), per_batch[:, 2:].sum(1))[0, 1]
    assert abs(correlation) < 5 / np.sqrt(batches)


def test_trajectory_sharing_is_the_statevector_default():
    circuit = Circuit(CIRCUITS["t_ghz"])
    assert circuit.compile_detector_sampler()._trajectory_sharing
    assert circuit.compile_sampler()._trajectory_sharing
    assert not circuit.compile_detector_sampler(strategy="cat5")._trajectory_sharing


def test_trajectory_sharing_needs_the_statevector_strategy():
    with pytest.raises(ValueError, match="trajectory_sharing"):
        Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(
            strategy="cat5", trajectory_sharing=True
        )


@pytest.fixture
def threads():
    """Set numba's thread count for the test, then restore it."""
    original = numba.get_num_threads()
    yield numba.set_num_threads
    numba.set_num_threads(original)


@pytest.mark.parametrize("postselect", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_samples_depend_on_neither_threads_nor_batches_nor_sharing(
    postselect, reference, threads
):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool) if postselect else None

    def sample(count, batch_size, sharing):
        threads(count)
        sampler = circuit.compile_detector_sampler(
            strategy="active_statevector", seed=3, trajectory_sharing=sharing
        )
        return [
            sampler.sample(
                10_003,
                batch_size=batch_size,
                append_observables=True,
                postselection_mask=mask,
                use_detector_reference_sample=reference,
            )
            for _ in range(2)
        ]

    expected = sample(1, None, True)
    first, second = expected
    assert not np.array_equal(first, second)
    for count, batch_size, sharing in [
        (2, None, True),
        (5, 300, False),
        (3, 9_000, True),
    ]:
        for got, want in zip(sample(count, batch_size, sharing), expected, strict=True):
            np.testing.assert_array_equal(got, want)


def test_measurement_samples_depend_on_neither_threads_nor_batches(threads):
    circuit = Circuit(CIRCUITS["midcircuit_feedforward"])

    def sample(count, batch_size):
        threads(count)
        sampler = circuit.compile_sampler(strategy="active_statevector", seed=4)
        return sampler.sample(9_001, batch_size=batch_size)

    np.testing.assert_array_equal(sample(4, 250), sample(1, None))


@pytest.mark.parametrize("postselect", [False, True])
@pytest.mark.parametrize("name", ["t_ghz", "midcircuit_feedforward", "rotations"])
def test_parallel_amplitude_loops_give_the_serial_samples(
    name, postselect, threads, monkeypatch
):
    circuit = Circuit(CIRCUITS[name])
    mask = np.ones(circuit.num_detectors, dtype=bool) if postselect else None

    def sample(count):
        threads(count)
        sampler = circuit.compile_detector_sampler(seed=5)
        return sampler.sample(200, append_observables=True, postselection_mask=mask)

    serial = sample(1)
    # NOTE: a call of one chunk on several threads runs its amplitude loops in
    # parallel chunks once their work reaches PARALLEL_WORK.
    monkeypatch.setattr(kernels, "PARALLEL_WORK", 1)
    monkeypatch.setattr(shots_module, "PARALLEL_WORK", 1)
    np.testing.assert_array_equal(sample(4), serial)


@pytest.mark.parametrize(
    ("count", "shots", "batch_size", "expected"),
    [
        (4, 40_000, None, [10_000] * 4),
        (1, 70_000, None, [65_536, 4_464]),
        # NOTE: a chunk holds at least MIN_CHUNK_SHOTS shots.
        (10, 1_000, None, [256, 256, 256, 232]),
        (2, 20_000, 9_000, [9_000, 9_000, 2_000]),
        (2, 5_000, 100, [100] * 50),
    ],
)
def test_calls_split_into_a_chunk_per_thread(
    count, shots, batch_size, expected, threads, monkeypatch
):
    threads(count)
    sampler = Circuit(GHZ_WITH_DIRECT_DETECTOR).compile_detector_sampler(seed=7)
    rows = []
    finish = shot_sampler(sampler)._run_chunk

    def spy(chunk, *args):
        rows.append(len(chunk))
        return finish(chunk, *args)

    monkeypatch.setattr(shot_sampler(sampler), "_run_chunk", spy)
    assert sampler.sample(shots, batch_size=batch_size).shape[0] == shots
    assert sorted(rows, reverse=True) == expected


def test_masks_of_component_detectors_alone_prune_lanes():
    sampler = Circuit(GHZ_WITH_DIRECT_DETECTOR).compile_detector_sampler(
        strategy="active_statevector", seed=1
    )
    sampler.sample(20_000, postselection_mask=np.array([True, True, False]))
    assert any(stats["dead"] > 0 for stats in _sharing_stats(sampler))


@pytest.mark.parametrize("sharing", [False, True])
def test_reference_bits_leave_the_outputs_of_pruned_shots_false(sharing, monkeypatch):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool)

    def sample(reference):
        sampler = circuit.compile_detector_sampler(
            strategy="active_statevector", seed=4, trajectory_sharing=sharing
        )
        monkeypatch.setattr(sampler, "_reference_sample", lambda: reference)
        return sampler.sample(
            20_000,
            postselection_mask=mask,
            separate_observables=True,
            use_observable_reference_sample=True,
        )

    detectors, raw = sample(np.zeros(4, dtype=bool))
    _, flipped = sample(np.array([False, False, False, True]))
    raw, flipped = raw[:, 0], flipped[:, 0]
    kept = ~detectors.any(axis=1)
    np.testing.assert_array_equal(flipped[kept], ~raw[kept])
    # NOTE: a shot pruned before the observable is set reads False either way.
    cleared = ~raw & ~flipped
    assert cleared.any() and not cleared[kept].any()
    np.testing.assert_array_equal(cleared | (flipped == ~raw), np.ones_like(kept))
