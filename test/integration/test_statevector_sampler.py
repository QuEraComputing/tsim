"""strategy="active_statevector" end to end, against stabilizer rank and VecSampler."""

import warnings
from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match, shot_sampler

import jax.numpy as jnp
import numpy as np
import pytest

import tsim.sampler as sampler_module
import tsim.statevector.shared as shared
import tsim.statevector.shots as shots_module
from tsim.circuit import Circuit
from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.core.types import SamplingStrategy
from tsim.external.vec_sim.vec_sampler import VecSampler
from tsim.statevector.program import StatevectorComponent


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_detector_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_detector_sampler(strategy="cat5", seed=1).sample(
        shots, append_observables=True
    )
    got = circuit.compile_detector_sampler(
        strategy="active_statevector", seed=2
    ).sample(shots, batch_size=5_000, append_observables=True)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_measurement_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_sampler(strategy="cat5", seed=1).sample(shots)
    got = circuit.compile_sampler(strategy="active_statevector", seed=2).sample(
        shots, batch_size=5_000
    )
    assert_counts_match(got, expected)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_measurement_sampling_matches_vec_sampler(seed):
    stim_circuit = gen_stim_circuit(3, 60, p_t=3, p_x_error=0.5, p_r_z=0.5, seed=seed)
    shots = 512
    expected, _, _ = VecSampler(stim_circuit).sample(shots)
    sampler = Circuit.from_stim_program(stim_circuit).compile_sampler(
        strategy="active_statevector", seed=seed
    )
    assert_counts_match(sampler.sample(shots, batch_size=shots), expected)


def test_statevector_samplers_hold_statevector_components():
    sampler = Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(
        strategy="active_statevector"
    )
    components = sampler._program.components
    assert components
    assert all(isinstance(c, StatevectorComponent) for c in components)
    assert "statevector components" in repr(sampler)


def test_postselected_samples_agree_across_strategies():
    circuit = Circuit(CIRCUITS["two_components"])
    mask = np.ones(circuit.num_detectors, dtype=bool)
    shots = 30_000
    rates = []
    cases: tuple[tuple[SamplingStrategy, int], ...] = (
        ("cat5", 1),
        ("active_statevector", 2),
    )
    for strategy, seed in cases:
        sampler = circuit.compile_detector_sampler(strategy=strategy, seed=seed)
        detectors, observables = sampler.sample(
            shots, batch_size=7_000, separate_observables=True, postselection_mask=mask
        )
        assert detectors.shape == (shots, circuit.num_detectors)
        assert observables.shape == (shots, circuit.num_observables)
        keep = ~detectors.any(axis=1)
        rates.append((keep.mean(), observables[keep, 0].mean()))
    (accept_a, error_a), (accept_b, error_b) = rates
    accept_sigma = np.sqrt(accept_a * (1 - accept_a) * 2 / shots)
    assert abs(accept_a - accept_b) <= 5 * accept_sigma + 1e-3
    error_sigma = np.sqrt(max(error_a * (1 - error_a), 1e-4) * 2 / (accept_a * shots))
    assert abs(error_a - error_b) <= 5 * error_sigma + 1e-3


def test_reference_samples_work_with_statevector():
    circuit = Circuit(CIRCUITS["injection_check"])
    sampler = circuit.compile_detector_sampler(strategy="active_statevector", seed=0)
    samples = sampler.sample(
        1_000, use_detector_reference_sample=True, use_observable_reference_sample=True
    )
    assert samples.shape == (1_000, circuit.num_detectors)


def test_probabilities_reject_statevector():
    prepared = prepare_graph(Circuit(CIRCUITS["t_ghz"]), sample_detectors=False)
    with pytest.raises(ValueError, match="active_statevector"):
        compile_program(prepared, mode="joint", strategy="active_statevector")


def _norm_warnings(monkeypatch, strategy, deviation):
    sample = sampler_module.sample_component
    run_lanes = shared.run_lanes

    def drifting(component, f_params, key):
        samples, key, _ = sample(component, f_params, key)
        return samples, key, jnp.asarray(deviation)

    def drifting_lanes(*args):
        return (deviation, *run_lanes(*args)[1:])

    monkeypatch.setattr(sampler_module, "sample_component", drifting)
    monkeypatch.setattr(shared, "run_lanes", drifting_lanes)
    sampler = Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(
        strategy=strategy, seed=0
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler.sample(10)
    return [w for w in caught if "not normalized" in str(w.message)]


@pytest.mark.parametrize("deviation, warns", [(1e-10, False), (1e-8, True)])
def test_statevector_norm_warnings_have_a_complex128_threshold(
    monkeypatch, deviation, warns
):
    caught = _norm_warnings(monkeypatch, "active_statevector", deviation)
    assert bool(caught) == warns


@pytest.mark.parametrize("deviation, warns", [(5e-6, False), (2e-5, True)])
def test_stabilizer_rank_norm_warnings_keep_their_threshold(
    monkeypatch, deviation, warns
):
    assert bool(_norm_warnings(monkeypatch, "cat5", deviation)) == warns


def test_a_vanishing_statevector_norm_raises(monkeypatch):
    with pytest.raises(ValueError, match="vanishing"):
        _norm_warnings(monkeypatch, "active_statevector", 1.0)


def test_a_lane_must_fit_in_half_of_the_available_memory(monkeypatch):
    circuit = Circuit(CIRCUITS["t_ghz"])
    sampler = circuit.compile_detector_sampler(strategy="active_statevector")
    peak = shot_sampler(sampler).bytes_per_shot()
    monkeypatch.setattr(shots_module, "memory_budget", lambda: peak - 1)
    with pytest.raises(ValueError, match="cat5"):
        circuit.compile_detector_sampler(strategy="active_statevector")
    monkeypatch.setattr(shots_module, "memory_budget", lambda: peak)
    circuit.compile_detector_sampler(strategy="active_statevector")
