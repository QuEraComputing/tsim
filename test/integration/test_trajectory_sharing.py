"""trajectory_sharing=True end to end, against the stabilizer-rank strategy."""

from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import numpy as np
import pytest

import tsim.sampler as sampler_module
import tsim.utils.kernels as kernels
from tsim.circuit import Circuit
from tsim.statevector.tables import OutcomeTables

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


# NOTE: flipped direct detectors and observables, unflipped ones, and two T
# components, so that the outputs are reordered.
DIRECT_FLIPS = """
R 0 1 2 3 4 5
H 0
T 0
H 0
X_ERROR(0.3) 1
X 2
X_ERROR(0.2) 2
M 2
DETECTOR rec[-1]
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
X 3
X_ERROR(0.1) 3
M 3
OBSERVABLE_INCLUDE(0) rec[-1]
X_ERROR(0.4) 4
M 4
OBSERVABLE_INCLUDE(1) rec[-1]
H 5
T 5
H 5
M 5
OBSERVABLE_INCLUDE(2) rec[-1]
"""


@pytest.fixture(params=["lanes", "tables", "mixed"])
def mode(request, monkeypatch):
    """Sample every shared component on lanes, from outcome tables, or both.

    ``mixed`` gives only the key of no fault a table, so that shots with other
    keys take lanes.
    """
    keys = {"tables": None, "mixed": np.zeros(1, dtype=np.uint64)}

    def tables(component, noise):
        if request.param == "lanes":
            return None
        return OutcomeTables(component, keys[request.param])

    monkeypatch.setattr(sampler_module, "outcome_tables", tables)
    return request.param


def _sharing_stats(sampler):
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    return backend.sharing_stats


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
def test_pruning_through_the_sampler_keeps_the_accepted_distribution(reference, mode):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    shots = 40_000
    seed, bits = 1, None
    if reference:
        for seed in range(1, 40):
            sampler = circuit.compile_detector_sampler(
                strategy="active_statevector", seed=seed
            )
            bits = sampler._compute_reference_sample()
            if bits[:2].any():
                break
    baseline = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=seed, trajectory_sharing=False
    )
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=seed, trajectory_sharing=True
    )
    if bits is not None:
        # NOTE: the random detectors make the reference bits a draw; both use
        # this one, so that the baseline only stops sharing.
        baseline._compute_reference_sample = lambda: bits
        sampler._compute_reference_sample = lambda: bits
    rate, expected = _accepted(baseline, shots, mask, reference=reference)
    shared_rate, got = _accepted(sampler, shots, mask, reference=reference)
    stats = _sharing_stats(sampler)
    if mode == "lanes":
        assert any(entry["dead"] > 0 for entry in stats)
    else:
        assert all(entry["tables"] > 0 for entry in stats)
    sigma = np.sqrt(rate * (1 - rate) * 2 / shots)
    assert abs(shared_rate - rate) <= 5 * sigma + 3 / shots
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_shared_detector_sampling_matches_stabilizer_rank(name, mode):
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
def test_shared_measurement_sampling_matches_stabilizer_rank(name, mode):
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
def test_pruned_postselection_keeps_the_accepted_distribution(name, mode):
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
def test_shared_sampling_of_random_noisy_circuits(seed, mode):
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
def test_components_draw_independent_random_numbers(sharing, mode):
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
    # NOTE: shots in one batch must not share draws across components either.
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


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("screen_shots", [None, 64])
@pytest.mark.parametrize("postselect", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_samples_do_not_depend_on_the_thread_count(
    postselect, reference, screen_shots, sharing, mode, monkeypatch
):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool) if postselect else None
    if screen_shots is not None:
        monkeypatch.setattr(sampler_module, "SCREEN_SHOTS", screen_shots)

    def sample(threads):
        sampler = circuit.compile_detector_sampler(
            strategy="active_statevector", seed=3, trajectory_sharing=sharing
        )
        sampler._threads = lambda batch_size: threads
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


def test_shared_measurement_samples_do_not_depend_on_the_thread_count(mode):
    circuit = Circuit(CIRCUITS["midcircuit_feedforward"])

    def sample(threads):
        sampler = circuit.compile_sampler(
            strategy="active_statevector", seed=4, trajectory_sharing=True
        )
        sampler._threads = lambda batch_size: threads
        return sampler.sample(2_001, batch_size=250)

    np.testing.assert_array_equal(sample(4), sample(1))


def _recording_rows(sampler, monkeypatch) -> list[int]:
    rows = []
    original = sampler_module.sample_statevector_program

    def spy(program, f_params, *args, **kwargs):
        assert isinstance(f_params, np.ndarray)
        rows.append(f_params.shape[0])
        return original(program, f_params, *args, **kwargs)

    monkeypatch.setattr(sampler_module, "sample_statevector_program", spy)
    return rows


def test_shared_postselection_samples_the_survivors_without_padding(monkeypatch):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=5, trajectory_sharing=True
    )
    sampler._threads = lambda batch_size: 1
    counts = _recording_rows(sampler, monkeypatch)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    detectors = sampler.sample(5_003, batch_size=300, postselection_mask=mask)

    assert counts[:-1] == [300] * (len(counts) - 1)
    assert 0 < counts[-1] < 300
    # NOTE: detector 2 is the circuit's only direct detector.
    assert sum(counts) == int((~detectors[:, 2]).sum())


@pytest.mark.parametrize(
    ("sharing", "shots", "expected"),
    [
        (False, 40_000, [10_000] * 4),
        (True, 40_000, [10_000] * 4),
        # NOTE: without sharing every shot needs a lane, so smaller calls split
        # too, down to MIN_LANE_BATCH shots per thread.
        (False, 12_000, [3_000] * 4),
        (False, 1_024, [256] * 4),
        (False, 1_000, [1_000]),
        # NOTE: the GHZ circuit needs at most 8 lanes however many shots it has.
        (True, 12_000, [12_000]),
        (True, 6_000, [6_000]),
    ],
)
def test_default_batches_give_every_thread_one(sharing, shots, expected, monkeypatch):
    sampler = Circuit(GHZ_WITH_DIRECT_DETECTOR).compile_detector_sampler(
        strategy="active_statevector", seed=7, trajectory_sharing=sharing
    )
    sampler._threads = lambda batch_size: 4
    rows = _recording_rows(sampler, monkeypatch)
    assert sampler.sample(shots).shape[0] == shots
    assert sorted(rows) == expected


@pytest.mark.parametrize("sharing", [False, True])
def test_one_thread_samples_a_call_as_one_batch(sharing, monkeypatch):
    sampler = Circuit(GHZ_WITH_DIRECT_DETECTOR).compile_detector_sampler(
        strategy="active_statevector", seed=7, trajectory_sharing=sharing
    )
    sampler._threads = lambda batch_size: 1
    rows = _recording_rows(sampler, monkeypatch)
    sampler.sample(12_000)
    assert rows == [12_000]


@pytest.mark.parametrize(
    ("max_lanes", "expected"), [(1_499, [6_000]), (1_500, [1_500] * 4)]
)
def test_shared_calls_split_when_a_batch_may_need_a_lane_per_shot(
    max_lanes, expected, monkeypatch
):
    sampler = Circuit(GHZ_WITH_DIRECT_DETECTOR).compile_detector_sampler(
        strategy="active_statevector", seed=7, trajectory_sharing=True
    )
    sampler._threads = lambda batch_size: 4
    monkeypatch.setattr(
        sampler._backend, "max_lanes", lambda mask, flip=None: max_lanes
    )
    rows = _recording_rows(sampler, monkeypatch)
    sampler.sample(6_000)
    assert sorted(rows) == expected


@pytest.mark.parametrize("sharing", [False, True])
def test_default_postselection_splits_the_survivors_over_the_threads(
    sharing, monkeypatch
):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=8, trajectory_sharing=sharing
    )
    sampler._threads = lambda batch_size: 4
    rows = _recording_rows(sampler, monkeypatch)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    detectors = sampler.sample(40_000, postselection_mask=mask)

    survivors = int((~detectors[:, 2]).sum())
    size = -(-survivors // 4)
    assert len(rows) == 4
    assert sum(rows) == survivors and max(rows) == size


@pytest.mark.parametrize("sharing", [False, True])
def test_small_postselected_calls_split_unless_sharing_bounds_the_lanes(
    sharing, monkeypatch
):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=8, trajectory_sharing=sharing
    )
    sampler._threads = lambda batch_size: 4
    rows = _recording_rows(sampler, monkeypatch)
    masks = []
    max_lanes = sampler._backend.max_lanes

    def spy(mask, flip=None):
        masks.append(mask)
        return max_lanes(mask, flip)

    monkeypatch.setattr(sampler._backend, "max_lanes", spy)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    detectors = sampler.sample(6_000, postselection_mask=mask)

    assert sum(rows) == int((~detectors[:, 2]).sum())
    # NOTE: with this mask, sharing keeps the GHZ circuit on at most 32 lanes.
    assert len(rows) == (1 if sharing else 4)
    assert masks and all(np.array_equal(m, mask) for m in masks)


def test_split_survivors_run_on_as_many_threads_as_their_size_allows(monkeypatch):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=8, trajectory_sharing=False
    )
    # NOTE: as for a wide component, only one batch of the largest size fits.
    sampler._threads = lambda batch_size: 1 if batch_size > 20_000 else 4
    calls = []
    original = sampler_module.run_tasks

    def spy(function, tasks, threads):
        tasks = list(tasks)
        calls.append((function.__name__, len(tasks), threads))
        return original(function, tasks, threads)

    monkeypatch.setattr(sampler_module, "run_tasks", spy)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    sampler.sample(40_000, postselection_mask=mask)
    assert [call for call in calls if call[0] == "dispatch"] == [("dispatch", 4, 4)]


@pytest.mark.parametrize("rows", [1, 300, 70_000])
@pytest.mark.parametrize("name", [*sorted(CIRCUITS), "direct_flips"])
def test_direct_outputs_land_in_their_columns(name, rows):
    text = DIRECT_FLIPS if name == "direct_flips" else CIRCUITS[name]
    sampler = Circuit(text).compile_detector_sampler(
        strategy="active_statevector", seed=1
    )
    program = sampler._program
    if name == "direct_flips":
        assert program.direct_flips.any()
        assert program.output_reindex is not None
    f_params = sampler._channel_sampler.sample(rows)
    samples = sampler._backend.sample(f_params, 2)
    columns = program.output_order[: len(program.direct_f_indices)]
    expected = np.broadcast_to(program.direct_flips, (rows, len(columns))).copy()
    has_f = program.direct_f_indices >= 0
    expected[:, has_f] ^= f_params[:, program.direct_f_indices[has_f]] != 0
    np.testing.assert_array_equal(samples[:, columns], expected)


def test_masks_of_component_detectors_alone_prune_lanes(monkeypatch):
    monkeypatch.setattr(sampler_module, "outcome_tables", lambda *args: None)
    sampler = Circuit(GHZ_WITH_DIRECT_DETECTOR).compile_detector_sampler(
        strategy="active_statevector", seed=1
    )
    sampler.sample(20_000, postselection_mask=np.array([True, True, False]))
    assert any(stats["dead"] > 0 for stats in _sharing_stats(sampler))


@pytest.mark.parametrize("sharing", [False, True])
def test_reference_bits_leave_the_outputs_of_pruned_shots_false(
    sharing, mode, monkeypatch
):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool)

    def sample(reference):
        sampler = circuit.compile_detector_sampler(
            strategy="active_statevector", seed=4, trajectory_sharing=sharing
        )
        monkeypatch.setattr(sampler, "_compute_reference_sample", lambda: reference)
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
