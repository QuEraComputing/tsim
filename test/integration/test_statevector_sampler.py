"""strategy="statevector" end to end, against stabilizer rank and VecSampler."""

import warnings
from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tsim.sampler as sampler_module
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
    got = circuit.compile_detector_sampler(strategy="statevector", seed=2).sample(
        shots, batch_size=5_000, append_observables=True
    )
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_measurement_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 20_000
    expected = circuit.compile_sampler(strategy="cat5", seed=1).sample(shots)
    got = circuit.compile_sampler(strategy="statevector", seed=2).sample(
        shots, batch_size=5_000
    )
    assert_counts_match(got, expected)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_measurement_sampling_matches_vec_sampler(seed):
    stim_circuit = gen_stim_circuit(3, 60, p_t=3, p_x_error=0.5, p_r_z=0.5, seed=seed)
    shots = 512
    expected, _, _ = VecSampler(stim_circuit).sample(shots)
    sampler = Circuit.from_stim_program(stim_circuit).compile_sampler(
        strategy="statevector", seed=seed
    )
    assert_counts_match(sampler.sample(shots, batch_size=shots), expected)


def test_statevector_samplers_hold_statevector_components():
    sampler = Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(
        strategy="statevector"
    )
    components = sampler._program.components
    assert components
    assert all(isinstance(c, StatevectorComponent) for c in components)
    assert "statevector components" in repr(sampler)
    widest = max(c.width for c in components if isinstance(c, StatevectorComponent))
    assert sampler._peak_bytes_per_sample() >= 3 * 2**widest * 8


def test_postselection_and_partial_batches():
    circuit = Circuit(CIRCUITS["two_components"])
    mask = np.ones(circuit.num_detectors, dtype=bool)
    shots = 30_000
    rates = []
    cases: tuple[tuple[SamplingStrategy, int], ...] = (("cat5", 1), ("statevector", 2))
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
    sampler = circuit.compile_detector_sampler(strategy="statevector", seed=0)
    samples = sampler.sample(
        1_000, use_detector_reference_sample=True, use_observable_reference_sample=True
    )
    assert samples.shape == (1_000, circuit.num_detectors)


def test_probabilities_reject_statevector():
    prepared = prepare_graph(Circuit(CIRCUITS["t_ghz"]), sample_detectors=False)
    with pytest.raises(ValueError, match="statevector"):
        compile_program(prepared, mode="joint", strategy="statevector")


def test_complex128_sampling_matches_stabilizer_rank():
    circuit = Circuit(CIRCUITS["t_ghz"])
    expected = circuit.compile_detector_sampler(strategy="cat5", seed=1).sample(
        10_000, append_observables=True
    )
    with jax.enable_x64(True):
        sampler = circuit.compile_detector_sampler(strategy="statevector", seed=2)
        assert all(
            isinstance(c, StatevectorComponent) and c.dtype == "complex128"
            for c in sampler._program.components
        )
        got = sampler.sample(10_000, append_observables=True)
    assert_counts_match(got, expected)


def test_a_lane_that_does_not_fit_in_memory_is_rejected(monkeypatch):
    monkeypatch.setattr(
        sampler_module._CompiledSamplerBase, "_available_memory", lambda self: 1
    )
    with pytest.raises(ValueError, match="cat5"):
        Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(strategy="statevector")


def _norm_warnings(monkeypatch, strategy, deviation, *, x64=False):
    name = (
        "sample_statevector_component"
        if strategy == "statevector"
        else "sample_component"
    )
    sample = getattr(sampler_module, name)

    def drifting(component, f_params, key):
        samples, key, _ = sample(component, f_params, key)
        return samples, key, jnp.asarray(deviation)

    monkeypatch.setattr(sampler_module, name, drifting)
    with jax.enable_x64(x64):
        sampler = Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(
            strategy=strategy, seed=0
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sampler.sample(10)
    return [w for w in caught if "not normalized" in str(w.message)]


@pytest.mark.parametrize(
    "x64, deviation, warns",
    [
        (False, 5e-4, False),
        (False, 2e-3, True),
        (True, 1e-10, False),
        (True, 1e-8, True),
    ],
)
def test_statevector_norm_warnings_depend_on_the_amplitude_dtype(
    monkeypatch, x64, deviation, warns
):
    caught = _norm_warnings(monkeypatch, "statevector", deviation, x64=x64)
    assert bool(caught) == warns


@pytest.mark.parametrize("deviation, warns", [(5e-6, False), (2e-5, True)])
def test_stabilizer_rank_norm_warnings_keep_their_threshold(
    monkeypatch, deviation, warns
):
    assert bool(_norm_warnings(monkeypatch, "cat5", deviation)) == warns


def test_a_vanishing_statevector_norm_raises(monkeypatch):
    with pytest.raises(ValueError, match="vanishing"):
        _norm_warnings(monkeypatch, "statevector", 1.0)


def test_the_automatic_batch_size_of_a_statevector_sampler_is_capped(monkeypatch):
    monkeypatch.setattr(
        sampler_module._CompiledSamplerBase, "_available_memory", lambda self: 2**50
    )
    circuit = Circuit(CIRCUITS["t_ghz"])
    sampler = circuit.compile_detector_sampler(strategy="statevector", seed=0)
    assert sampler._estimate_batch_size() == 2**16
    assert sampler._resolve_batch_size(10**7, None, compute_reference=False) <= 2**16
    lanes = []
    sample_program = sampler_module.sample_program

    def recording(program, f_params, key):
        lanes.append(f_params.shape[0])
        return sample_program(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", recording)
    sampler.sample(2**16 + 1)
    sampler.sample(
        2**16 + 1, postselection_mask=np.ones(circuit.num_detectors, dtype=bool)
    )
    assert lanes and max(lanes) <= 2**16
    stabilizer_rank = circuit.compile_detector_sampler(strategy="cat5")
    assert stabilizer_rank._estimate_batch_size() > 2**16


def test_the_per_shot_estimate_counts_f_params_and_records():
    sampler = Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(
        strategy="statevector"
    )
    widest = max(
        c.width
        for c in sampler._program.components
        if isinstance(c, StatevectorComponent)
    )
    num_f = sampler._channel_sampler.signature_matrix.shape[1]
    records = sampler._program.num_outputs
    amplitudes = 3 * 2**widest * 8
    assert sampler._peak_bytes_per_sample() >= amplitudes + 2 * num_f + 2 * records


def test_a_lane_must_fit_in_half_of_the_available_memory(monkeypatch):
    circuit = Circuit(CIRCUITS["t_ghz"])
    peak = circuit.compile_detector_sampler(
        strategy="statevector"
    )._peak_bytes_per_sample()
    base = sampler_module._CompiledSamplerBase
    monkeypatch.setattr(base, "_available_memory", lambda self: 2 * peak - 1)
    with pytest.raises(ValueError, match="cat5"):
        circuit.compile_detector_sampler(strategy="statevector")
    monkeypatch.setattr(base, "_available_memory", lambda self: 2 * peak)
    circuit.compile_detector_sampler(strategy="statevector")
