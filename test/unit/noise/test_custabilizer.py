"""Tests for the independent-Bernoulli decomposition and the cuStabilizer backend."""

import numpy as np
import pytest
import stim

import tsim
import tsim.sampler
from tsim.noise.channels import (
    ChannelSampler,
    error_probs,
    independent_bernoulli_decomposition,
    xor_convolve,
)
from tsim.utils.cuda_helpers import custabilizer_available

needs_custabilizer = pytest.mark.skipif(
    not custabilizer_available(), reason="cupy/cuquantum GPU stack not available"
)


def _forward(q: np.ndarray) -> np.ndarray:
    """XOR-convolve independent Bernoulli errors q[S] on pattern S."""
    n = q.size
    probs = np.zeros(n)
    probs[0] = 1.0
    for pattern in range(1, n):
        if q[pattern] == 0:
            continue
        single = np.zeros(n)
        single[0] = 1 - q[pattern]
        single[pattern] = q[pattern]
        probs = xor_convolve(probs, single)
    return probs


@pytest.mark.parametrize("p", [0.0, 1e-4, 0.1, 0.5, 0.74])
def test_depolarizing_decomposition(p):
    probs = np.array([1 - p, p / 3, p / 3, p / 3])
    q = independent_bernoulli_decomposition(probs)
    assert q is not None
    assert q[0] == 0
    np.testing.assert_allclose(_forward(q), probs, atol=1e-14)


def test_too_noisy_channel_has_no_decomposition():
    p = 0.76
    assert (
        independent_bernoulli_decomposition(np.array([1 - p, p / 3, p / 3, p / 3]))
        is None
    )
    assert independent_bernoulli_decomposition(np.array([0.0, 1.0])) is None


@pytest.mark.parametrize("seed", range(5))
def test_random_low_noise_channels(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 5))
    # Low total noise with no pattern much rarer than products of two
    # others (q_S ~ p_S - sum_{S1^S2=S} p_S1 p_S2 must stay positive).
    probs = (0.5 + rng.random(2**k)) * 0.005
    probs[0] = 0
    probs[0] = 1 - probs.sum()
    q = independent_bernoulli_decomposition(probs)
    assert q is not None
    assert np.all(q >= 0) and np.all(q < 0.5)
    np.testing.assert_allclose(_forward(q), probs, atol=1e-13)


def test_single_bit_channel():
    q = independent_bernoulli_decomposition(error_probs(0.3))
    assert q is not None
    np.testing.assert_allclose(q, [0.0, 0.3])
    trivial = independent_bernoulli_decomposition(np.array([1.0]))
    assert trivial is not None and trivial.tolist() == [0.0]


def test_independent_error_model_shapes():
    # Two channels sharing an f-variable: f0 = e0 ^ e1, f1 = e1 ^ e2.
    probs = [error_probs(0.1), np.array([0.8, 0.1, 0.05, 0.05])]
    transform = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    sampler = ChannelSampler(probs, transform, seed=0)
    model = sampler.independent_error_model()
    assert model is not None
    q, patterns = model
    assert patterns.shape == (q.shape[0], 2)
    assert patterns.dtype == np.uint8
    assert np.all((q > 0) & (q < 0.5))
    # Marginal P(f0 = 1) is the same in both models (Monte Carlo on the
    # numpy side, exact XOR-convolution on the Bernoulli side).
    f = sampler.sample(200_000)
    exact = np.zeros(4)
    exact[0] = 1.0
    for qi, row in zip(q, patterns, strict=True):
        single = np.zeros(4)
        idx = int(row[0]) | (int(row[1]) << 1)
        single[0], single[idx] = 1 - qi, qi
        exact = xor_convolve(exact, single)
    p_f0 = exact[1] + exact[3]
    p_f1 = exact[2] + exact[3]
    np.testing.assert_allclose(f.mean(axis=0), [p_f0, p_f1], atol=5e-3)


def test_independent_error_model_none_when_too_noisy():
    p = 0.9
    sampler = ChannelSampler(
        [np.array([1 - p, p / 3, p / 3, p / 3])], np.eye(2, dtype=np.uint8), seed=0
    )
    assert sampler.independent_error_model() is None


def test_numpy_backend_is_default_without_gpu_stack():
    circuit = tsim.Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    sampler = circuit.compile_detector_sampler(channel_backend="numpy")
    assert sampler.channel_backend == "numpy"
    if not custabilizer_available():
        with pytest.raises(RuntimeError):
            circuit.compile_detector_sampler(channel_backend="custabilizer")


def test_invalid_channel_backend():
    circuit = tsim.Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match="channel_backend"):
        circuit.compile_detector_sampler(channel_backend="cupy")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# GPU tests
# --------------------------------------------------------------------------


def _rep_code(distance=3, rounds=3, p=0.02):
    return tsim.Circuit.from_stim_program(
        stim.Circuit.generated(
            "repetition_code:memory",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
        )
    )


def _z_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    pa, pb = a.mean(axis=0), b.mean(axis=0)
    se = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / n) + 1e-9
    return np.abs(pa - pb) / se


@needs_custabilizer
def test_direct_path_matches_numpy():
    circuit = _rep_code()
    gpu = circuit.compile_detector_sampler(seed=1)
    cpu = circuit.compile_detector_sampler(seed=2, channel_backend="numpy")
    assert gpu.channel_backend == "custabilizer"
    assert not gpu._program.components  # fully direct
    n = 100_000
    a = gpu.sample(n, append_observables=True)
    b = cpu.sample(n, append_observables=True)
    assert a.shape == b.shape and a.dtype == np.bool_
    assert a.flags.writeable
    assert _z_scores(a, b).max() < 5
    # pairwise co-occurrence of the two most frequent detectors
    top = np.argsort(-b.mean(axis=0))[:2]
    ca = (a[:, top[0]] & a[:, top[1]]).sum()
    cb = (b[:, top[0]] & b[:, top[1]]).sum()
    assert abs(ca - cb) < 5 * np.sqrt(ca + cb + 1)


@needs_custabilizer
def test_mixed_path_matches_numpy():
    circuit = tsim.Circuit(
        str(_rep_code(rounds=2).stim_circuit) + "\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]"
    )
    gpu = circuit.compile_detector_sampler(seed=1)
    cpu = circuit.compile_detector_sampler(seed=2, channel_backend="numpy")
    assert gpu.channel_backend == "custabilizer"
    assert gpu._program.components  # has a compiled component
    n = 50_000
    a = gpu.sample(n, batch_size=20_000, append_observables=True)
    b = cpu.sample(n, batch_size=20_000, append_observables=True)
    assert a.shape == b.shape
    assert _z_scores(a, b).max() < 5
    ref_a, _obs_a = gpu.sample(
        3000, use_detector_reference_sample=True, separate_observables=True
    )
    assert ref_a.shape == (3000, gpu._num_detectors)
    mask = np.zeros(gpu._num_detectors, dtype=bool)
    mask[0] = True
    ps = gpu.sample(3000, postselection_mask=mask)
    assert ps.shape == (3000, gpu._num_detectors)
    assert "channel_backend=custabilizer" in repr(gpu)


@needs_custabilizer
def test_measurement_sampler_and_probs_on_gpu():
    # Errors must change the outcome distribution to survive graph reduction.
    circuit = tsim.Circuit(
        "H 0 1\nT 0\nH 0\nX_ERROR(0.1) 0\nDEPOLARIZE2(0.05) 0 1\nH 1\nM 0 1"
    )
    ms = circuit.compile_sampler(seed=0)
    assert ms.channel_backend == "custabilizer"
    assert ms.sample(1000).shape == (1000, 2)
    probs = tsim.sampler.CompiledStateProbs(circuit, seed=0).probability_of(
        np.array([0, 1], dtype=np.uint8), batch_size=16
    )
    assert probs.shape == (16,)
    assert np.all((probs >= -1e-6) & (probs <= 1 + 1e-6))


@needs_custabilizer
def test_explicit_backend_rejects_too_noisy_channels():
    circuit = tsim.Circuit("DEPOLARIZE1(0.9) 0\nM 0\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match="independent-Bernoulli"):
        circuit.compile_detector_sampler(channel_backend="custabilizer")
    auto = circuit.compile_detector_sampler()
    assert auto.channel_backend == "numpy"


@needs_custabilizer
def test_grow_only_capacity_and_zero_shots():
    from tsim.noise.custabilizer import CuStabilizerChannelSampler

    probs = np.array([0.1, 0.2])
    patterns = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    s = CuStabilizerChannelSampler(probs, patterns, seed=0)
    assert s.sample(0).shape == (0, 3)
    small = s.sample(10)
    cap = s._max_shots
    big = s.sample(1000)
    assert s._max_shots >= 1000 and cap < s._max_shots
    assert small.shape == (10, 3) and big.shape == (1000, 3)
    dev = s.sample_device(64)
    assert dev.shape == (64, 3) and str(dev.dtype) == "uint8"
    out = s.sample_outputs(500, np.array([2, 0]), np.array([1, 0], dtype=bool), None)
    assert out.shape == (500, 2) and out.dtype == np.bool_
    # column 0 is f2 flipped: mean should be 1 - P(f2 = 1) = 1 - (0.1 + 0.2 - 2*0.02)
    assert abs(out[:, 0].mean() - (1 - 0.26)) < 0.08
