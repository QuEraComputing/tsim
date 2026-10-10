"""strategy="statevector" end to end, against stabilizer rank and VecSampler."""

from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import jax
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
