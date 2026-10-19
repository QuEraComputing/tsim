"""trajectory_sharing=True end to end, against one lane per shot."""

import warnings
from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import jax
import jax.numpy as jnp
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


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("screen_shots", [None, 64])
@pytest.mark.parametrize("postselect", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_samples_do_not_depend_on_the_thread_count(
    postselect, reference, screen_shots, sharing, monkeypatch
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


def test_shared_measurement_samples_do_not_depend_on_the_thread_count():
    circuit = Circuit(CIRCUITS["midcircuit_feedforward"])

    def sample(threads):
        sampler = circuit.compile_sampler(
            strategy="active_statevector", seed=4, trajectory_sharing=True
        )
        sampler._threads = lambda batch_size: threads
        return sampler.sample(2_001, batch_size=250)

    np.testing.assert_array_equal(sample(4), sample(1))


def test_shared_postselection_samples_the_survivors_without_padding(monkeypatch):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    sampler = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=5, trajectory_sharing=True
    )
    sampler._threads = lambda batch_size: 1
    rows = []
    original = sampler_module.sample_program

    def spy(program, f_params, key, **kwargs):
        rows.append((type(f_params), f_params.shape[0]))
        return original(program, f_params, key, **kwargs)

    monkeypatch.setattr(sampler_module, "sample_program", spy)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    detectors = sampler.sample(5_003, batch_size=300, postselection_mask=mask)

    counts = [count for _, count in rows]
    assert all(kind is np.ndarray for kind, _ in rows)
    assert counts[:-1] == [300] * (len(counts) - 1)
    assert 0 < counts[-1] < 300
    # NOTE: detector 2 is the circuit's only direct detector.
    assert sum(counts) == int((~detectors[:, 2]).sum())


def _recording_rows(sampler, monkeypatch) -> list[int]:
    rows = []
    original = sampler_module.sample_program

    def spy(program, f_params, key, **kwargs):
        rows.append(f_params.shape[0])
        return original(program, f_params, key, **kwargs)

    monkeypatch.setattr(sampler_module, "sample_program", spy)
    return rows


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize(
    ("shots", "expected"), [(40_000, [10_000] * 4), (6_000, [6_000])]
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
    if sharing:
        assert sum(rows) == survivors and max(rows) == size
    else:
        # NOTE: JAX pads every batch to a power of two.
        assert rows == [1 << (size - 1).bit_length()] * 4


@pytest.mark.parametrize("rows", [1, 300, 70_000])
@pytest.mark.parametrize("name", [*sorted(CIRCUITS), "direct_flips"])
def test_host_outputs_equal_the_jax_outputs(name, rows):
    text = DIRECT_FLIPS if name == "direct_flips" else CIRCUITS[name]
    sampler = Circuit(text).compile_detector_sampler(
        strategy="active_statevector", seed=1, trajectory_sharing=True
    )
    if name == "direct_flips":
        assert sampler._direct_flips.any()
        assert sampler._program.output_reindex is not None
    f_params = sampler._channel_sampler.sample(rows)
    key = jax.random.key(2)
    sharing = sampler._sharing_arguments(None)
    program = sampler._program
    on_host = sampler_module.sample_program(program, f_params, key, **sharing)
    on_device = sampler_module.sample_program(
        program, jnp.asarray(f_params), key, **sharing
    )
    assert isinstance(on_host, np.ndarray)
    np.testing.assert_array_equal(on_host, np.asarray(on_device))


@pytest.mark.parametrize("sharing", [False, True])
def test_workers_sample_in_the_callers_64_bit_mode(sharing):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)

    def sample(threads):
        sampler = circuit.compile_detector_sampler(
            strategy="active_statevector", seed=6, trajectory_sharing=sharing
        )
        sampler._threads = lambda batch_size: threads
        return sampler.sample(2_001, batch_size=250, append_observables=True)

    with jax.enable_x64(True), warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(sample(3), sample(1))
