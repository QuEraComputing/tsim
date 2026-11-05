"""The CUDA executor end to end, against the stabilizer-rank strategy and the CPU executor."""

import multiprocessing
import warnings
from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import numpy as np
import pytest

from tsim.circuit import Circuit
from tsim.statevector.cuda import available, codegen, runtime, sampler as cuda_sampler
from tsim.statevector.cuda.sampler import CudaShotSampler
from tsim.statevector.program import StatevectorComponent
from tsim.statevector.shots import ShotSampler

pytestmark = pytest.mark.skipif(not available(), reason="needs CuPy and a CUDA GPU")

# NOTE: t_ghz plus a direct detector (detector 2), as in test_trajectory_sharing.
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

# NOTE: a T state measured in X after a noisy idle: its observable's expectation
# is cos(π/4) = 0.707 before noise.
T_STATE_X = """
R 0
H 0
T 0
Z_ERROR(0.1) 0
H 0
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def _gpu(sampler):
    """Return ``sampler``, which must run on the GPU."""
    _gpu_shots(sampler)
    return sampler


def _gpu_shots(sampler) -> CudaShotSampler:
    """Return the GPU shot sampler of ``sampler``."""
    assert isinstance(sampler._shots, CudaShotSampler)
    return sampler._shots


def _cpu(compile, **kwargs):
    """Return ``compile(**kwargs)``, a sampler compiled as on a host without a GPU."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "available", lambda: False)
        sampler = compile(**kwargs)
    assert sampler._shots is None or type(sampler._shots) is ShotSampler
    return sampler


def _accepted(sampler, shots, mask, **kwargs):
    detectors, observables = sampler.sample(
        shots, separate_observables=True, postselection_mask=mask, **kwargs
    )
    # NOTE: sample returns only the shots that the mask keeps.
    assert mask is None or not (detectors & mask).any()
    return np.concatenate([detectors, observables], axis=1)


def _packed(sampler, shots, mask=None):
    """Return ``sample``'s bit-packed detectors then observables."""
    return sampler.sample(
        shots, append_observables=True, bit_packed=True, postselection_mask=mask
    )


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_cuda_detector_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 40_000
    expected = circuit.compile_detector_sampler(strategy="cat5", seed=1).sample(
        shots, batch_size=10_000, append_observables=True
    )
    got = _gpu(circuit.compile_detector_sampler(seed=2)).sample(
        shots, append_observables=True
    )
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_cuda_measurement_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 40_000
    expected = circuit.compile_sampler(strategy="cat5", seed=1).sample(
        shots, batch_size=10_000
    )
    got = _gpu(circuit.compile_sampler(seed=2)).sample(shots)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_cuda_postselection_keeps_the_accepted_distribution(name):
    circuit = Circuit(CIRCUITS[name])
    if circuit.num_detectors == 0:
        pytest.skip("no detectors to postselect")
    shots = 40_000
    mask = np.ones(circuit.num_detectors, dtype=bool)
    expected = _accepted(_cpu(circuit.compile_detector_sampler, seed=1), shots, mask)
    got = _accepted(_gpu(circuit.compile_detector_sampler(seed=2)), shots, mask)
    rate, cuda_rate = len(expected) / shots, len(got) / shots
    sigma = np.sqrt(rate * (1 - rate) * 2 / shots)
    assert abs(cuda_rate - rate) <= 5 * sigma + 3 / shots
    if len(got) and len(expected):
        assert_counts_match(got, expected)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cuda_sampling_of_random_noisy_circuits(seed):
    stim_circuit = gen_stim_circuit(3, 60, p_t=3, p_x_error=0.5, p_r_z=0.5, seed=seed)
    circuit = Circuit.from_stim_program(stim_circuit)
    shots = 8_192
    expected = circuit.compile_sampler(strategy="cat5", seed=seed).sample(
        shots, batch_size=shots
    )
    got = _gpu(circuit.compile_sampler(seed=seed + 10)).sample(shots)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("reference", [False, True])
def test_cuda_reference_samples_and_pruning(reference):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    shots = 40_000
    kwargs = {"use_detector_reference_sample": reference}
    expected = _accepted(
        _cpu(circuit.compile_detector_sampler, seed=3), shots, mask, **kwargs
    )
    got = _accepted(
        _gpu(circuit.compile_detector_sampler(seed=3)), shots, mask, **kwargs
    )
    rate, cuda_rate = len(expected) / shots, len(got) / shots
    assert abs(cuda_rate - rate) <= 5 * np.sqrt(rate * (1 - rate) * 2 / shots)
    assert_counts_match(got, expected)


def test_cuda_postselection_keeps_every_column():
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[[0, 2]] = True
    packed = _packed(_gpu(circuit.compile_detector_sampler(seed=4)), 20_000, mask)
    outputs = circuit.num_detectors + circuit.num_observables
    assert packed.shape[1] == -(-outputs // 8)
    bits = np.unpackbits(packed, axis=1, bitorder="little")
    assert not bits[:, outputs:].any()
    assert not bits[:, : circuit.num_detectors][:, mask].any()
    full = _accepted(_cpu(circuit.compile_detector_sampler, seed=5), 20_000, mask)
    assert abs(len(bits) - len(full)) <= 5 * np.sqrt(len(full))
    assert_counts_match(bits[:, :outputs].astype(bool), full)


@pytest.mark.parametrize("bit_packed", [False, True])
def test_cuda_arranges_the_columns_as_the_cpu_does(bit_packed):
    # NOTE: samplers of one seed draw the same shots, whichever columns they return.
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[0] = True
    num_detectors = circuit.num_detectors

    def sample(**kwargs):
        sampler = _gpu(circuit.compile_detector_sampler(seed=17))
        return sampler.sample(
            5_000, postselection_mask=mask, bit_packed=bit_packed, **kwargs
        )

    def bools(array, count):
        if not bit_packed:
            return array
        return np.unpackbits(array, axis=1, count=count, bitorder="little").astype(bool)

    appended = bools(sample(append_observables=True), num_detectors + 1)
    detectors = bools(sample(), num_detectors)
    prepended = bools(sample(prepend_observables=True), num_detectors + 1)
    separate = sample(separate_observables=True)
    np.testing.assert_array_equal(detectors, appended[:, :num_detectors])
    np.testing.assert_array_equal(prepended[:, 0], appended[:, -1])
    np.testing.assert_array_equal(prepended[:, 1:], detectors)
    np.testing.assert_array_equal(bools(separate[0], num_detectors), detectors)
    np.testing.assert_array_equal(bools(separate[1], 1), appended[:, -1:])
    cpu = _cpu(circuit.compile_detector_sampler, seed=17).sample(
        10, postselection_mask=mask, bit_packed=bit_packed, append_observables=True
    )
    gpu = sample(append_observables=True)
    assert gpu.dtype == cpu.dtype
    assert gpu.shape[1] == cpu.shape[1]


def test_cuda_samples_no_shots():
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    sampler = _gpu(circuit.compile_detector_sampler(seed=1))
    assert sampler.sample(0, append_observables=True).shape == (0, 4)
    assert sampler.sample(0, bit_packed=True).shape == (0, 1)


# NOTE: 150 noisy detectors, so the rows span three words, and two observables.
NOISY_DETECTORS = "\n".join(
    [
        "R " + " ".join(map(str, range(150))),
        "X_ERROR(0.3) " + " ".join(map(str, range(150))),
        "M " + " ".join(map(str, range(150))),
        *(f"DETECTOR rec[{i - 150}]" for i in range(150)),
        "OBSERVABLE_INCLUDE(0) rec[-1] rec[-7]",
        "OBSERVABLE_INCLUDE(1) rec[-150]",
    ]
)


def test_cuda_postselection_drops_the_rows_that_fail(monkeypatch):
    # NOTE: the masked run must draw the unmasked run's noise, so none is deferred.
    monkeypatch.setattr(cuda_sampler, "DEFER_SHARE", 2.0)
    circuit = Circuit(NOISY_DETECTORS)
    rng = np.random.default_rng(0)
    mask = np.zeros(150, dtype=bool)
    # NOTE: a few masked detectors, then unmasked runs across words, then a mix.
    mask[[0, 3, 4]] = True
    mask[100:] = rng.random(50) < 0.1
    full = _packed(_gpu(circuit.compile_detector_sampler(seed=11)), 5_000)
    rows = _packed(_gpu(circuit.compile_detector_sampler(seed=11)), 5_000, mask)
    full_bits = np.unpackbits(full, axis=1, count=152, bitorder="little")
    passing = ~full_bits[:, :150][:, mask].any(axis=1)
    assert 0 < passing.sum() < 5_000
    np.testing.assert_array_equal(rows, full[passing])


# NOTE: GHZ_WITH_DIRECT_DETECTOR's component and direct detector, then four
# noisy direct detectors that no mask below reads; one fault flips detectors 2
# and 3 together, and the observable also reads the last noisy qubit.
UNREAD_NOISE = """
R 0 1 2 3 4 5 6 7
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
E(0.1) X3 X4
X_ERROR(0.3) 4 5 6 7
M 3 4 5 6 7
DETECTOR rec[-5]
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def test_cuda_defers_the_noise_that_postselection_cannot_see():
    circuit = Circuit(UNREAD_NOISE)
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[[0, 2]] = True
    sampler = _gpu(circuit.compile_detector_sampler(seed=12))
    got = _accepted(sampler, 200_000, mask)
    assert _gpu_shots(sampler)._module(mask).splits[False] is not None
    full = _accepted(_cpu(circuit.compile_detector_sampler, seed=13), 200_000, mask)
    assert abs(len(got) - len(full)) <= 5 * np.sqrt(len(full))
    assert_counts_match(got, full)


def _expectations(sampler, shots, mask=None):
    """Return the passing rows and expectations of ``sample_accepted``."""
    rows, values, _ = _gpu_shots(sampler).sample_accepted(
        shots, mask=mask, expectations=True
    )
    assert values is not None
    return rows, values


def test_deferred_noise_leaves_the_expectations_unbiased():
    # NOTE: with expectations, the noise on the observable's direct qubit is not
    # deferred: the tape reads the observable's bit for its sign.
    circuit = Circuit(UNREAD_NOISE)
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[[0, 2]] = True
    sampler = _gpu(circuit.compile_detector_sampler(seed=14))
    _, values = _expectations(sampler, 200_000, mask)
    assert _gpu_shots(sampler)._module(mask).splits[True] is not None
    full = _accepted(_cpu(circuit.compile_detector_sampler, seed=15), 200_000, mask)
    expected = np.mean(1 - 2 * full[:, -1].astype(float))
    assert abs(values.mean() - expected) < 5 / np.sqrt(len(full))


@pytest.mark.parametrize("text", [NOISY_DETECTORS, UNREAD_NOISE])
def test_cuda_chunks_return_the_rows_of_one_pass(monkeypatch, text):
    # NOTE: 1000-shot chunks on both streams, and host rows that start with room
    # for the first chunk only; without deferral, a masked and an unmasked run on
    # the same chunks draw the same shots.
    monkeypatch.setattr(cuda_sampler, "DEFER_SHARE", 2.0)
    monkeypatch.setattr(
        cuda_sampler.CudaShotSampler, "_chunk_shots", lambda self, *args: 1000
    )
    monkeypatch.setattr(
        cuda_sampler._HostRows,
        "allocate",
        lambda self, passed, of, shots: self._resize(passed),
    )
    circuit = Circuit(text)
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[[0, 2]] = True
    outputs = circuit.num_detectors + circuit.num_observables
    full = _packed(_gpu(circuit.compile_detector_sampler(seed=16)), 10_500)
    rows = _packed(_gpu(circuit.compile_detector_sampler(seed=16)), 10_500, mask)
    full_bits = np.unpackbits(full, axis=1, count=outputs, bitorder="little")
    passing = ~full_bits[:, : circuit.num_detectors][:, mask].any(axis=1)
    assert 0 < passing.sum() < 10_500
    np.testing.assert_array_equal(rows, full[passing])


def test_observable_expectations_average_to_the_sampled_observable():
    circuit = Circuit(T_STATE_X)
    sampler = _gpu(circuit.compile_detector_sampler(seed=6))
    packed, values = _expectations(sampler, 200_000)
    assert values.shape == (len(packed), 1)
    # NOTE: given the shot's Z error, the expectation is ±cos(π/4).
    np.testing.assert_allclose(np.abs(values), np.sqrt(0.5), atol=1e-5)
    observed = 1 - 2 * np.unpackbits(packed, axis=1, bitorder="little")[:, 0].astype(
        float
    )
    sampled = _cpu(circuit.compile_detector_sampler, seed=7).sample(
        200_000, append_observables=True
    )[:, 0]
    expected = np.mean(1 - 2 * sampled.astype(float))
    assert abs(values.mean() - expected) < 5 * 1 / np.sqrt(200_000)
    # NOTE: the returned observable bit is sampled from the same state.
    assert abs(observed.mean() - values.mean()) < 5 / np.sqrt(200_000)


# NOTE: the T state's X outcome is copied onto qubit 1, whose postselected (and
# noisy) detector is settled after the observable's measurement.
T_STATE_X_THEN_DETECTOR = """
R 0 1
H 0
T 0
H 0
CNOT 0 1
X_ERROR(0.1) 1
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
M 1
DETECTOR rec[-1]
"""

# NOTE: a postselected T-state detector settled before the observable's measurement.
T_STATE_X_AFTER_DETECTOR = """
R 0 1
H 0 1
T 0 1
H 1
M 1
DETECTOR rec[-1]
Z_ERROR(0.1) 0
H 0
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def _postselected_expectations(text, shots):
    circuit = Circuit(text)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    sampler = _gpu(circuit.compile_detector_sampler(seed=9))
    _, values = _expectations(sampler, shots, mask)
    detectors, observables = _cpu(circuit.compile_detector_sampler, seed=10).sample(
        shots, separate_observables=True
    )
    kept = observables[~(detectors & mask).any(axis=1), 0]
    return values[:, 0], np.mean(1 - 2 * kept.astype(float)), len(kept)


def test_expectations_stay_unbiased_under_later_postselection():
    values, expected, kept = _postselected_expectations(
        T_STATE_X_THEN_DETECTOR, 200_000
    )
    # NOTE: the expectation before the measurement would average to cos(π/4).
    assert abs(values.mean() - expected) < 5 * np.sqrt(1 / len(values) + 1 / kept)
    np.testing.assert_array_equal(np.abs(values), 1)


def test_expectations_are_exact_after_earlier_postselection():
    values, expected, kept = _postselected_expectations(
        T_STATE_X_AFTER_DETECTOR, 200_000
    )
    np.testing.assert_allclose(np.abs(values), np.sqrt(0.5), atol=1e-5)
    assert abs(values.mean() - expected) < 5 * np.sqrt(1 / len(values) + 1 / kept)


def test_clifford_observables_have_expectations_of_plus_or_minus_one():
    circuit = Circuit(
        "R 0 1\nX_ERROR(0.3) 0\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    )
    sampler = _gpu(circuit.compile_detector_sampler(seed=8))
    packed, values = _expectations(sampler, 10_000)
    bits = np.unpackbits(packed, axis=1, bitorder="little")
    np.testing.assert_array_equal(values[:, 0], 1 - 2 * bits[:, 1].astype(np.float32))


@pytest.mark.parametrize("reference", [False, True])
def test_cuda_samples_programs_without_components(reference):
    circuit = Circuit(
        "R 0 1\nX_ERROR(0.3) 0\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    )
    kwargs = {"use_detector_reference_sample": reference}
    expected = _cpu(circuit.compile_detector_sampler, seed=1).sample(
        40_000, append_observables=True, **kwargs
    )
    got = _gpu(circuit.compile_detector_sampler(seed=2)).sample(
        40_000, append_observables=True, **kwargs
    )
    assert_counts_match(got, expected)
    measured = _gpu(circuit.compile_sampler(seed=3)).sample(40_000)
    assert measured.shape == (40_000, 2)
    np.testing.assert_array_equal(measured[:, 0], measured[:, 1])


def test_stabilizer_rank_strategies_stay_on_jax():
    circuit = Circuit(T_STATE_X)
    assert circuit.compile_detector_sampler(strategy="cat5")._shots is None


def _widest(circuit):
    """Return the active width of the circuit's widest statevector component."""
    components = circuit.compile_sampler(seed=1)._program.components
    return max(c.width for c in components if isinstance(c, StatevectorComponent))


def _random_unitary_circuit(width, layers, seed=0, noise=True):
    """Layers of random U3 pairs and CX gates, then every qubit measured.

    With ``noise``, X errors come first, so that every rotation reads the shot's
    f-bits and none of them runs before the shots split (``common_prefix``).
    """
    rng = np.random.default_rng(seed)
    lines = ["X_ERROR(0.2) " + " ".join(map(str, range(width)))] if noise else []
    for _ in range(layers):
        perm = rng.permutation(width)
        for a, b in zip(perm[::2], perm[1::2], strict=False):
            for q in (a, b):
                t = rng.uniform(-1, 1, 3)
                lines.append(f"U3({t[0]},{t[1]},{t[2]}) {q}")
            lines.append(f"CX {a} {b}")
    lines.append("M " + " ".join(map(str, range(width))))
    return Circuit("\n".join(lines))


def _assert_matches_cpu(circuit, width, shots=200_000):
    expected = _cpu(circuit.compile_sampler, seed=1).sample(shots)
    got = _gpu(circuit.compile_sampler(seed=2)).sample(shots)
    np.testing.assert_allclose(
        got.mean(axis=0), expected.mean(axis=0), atol=5 * 0.5 / np.sqrt(shots / 2)
    )
    # NOTE: pairs of bits, which a wrong shuffle partner would scramble.
    for a, b in [(0, 1), (0, width - 1), (width // 2, width - 1)]:
        assert_counts_match(got[:, [a, b]], expected[:, [a, b]])


@pytest.mark.parametrize(("width", "layers"), [(6, 8), (8, 8), (9, 10), (11, 12)])
def test_cuda_wide_components_match_the_cpu_executor(width, layers):
    # NOTE: widths 6, 8, 9 and 11 run 2, 8, 16 and 32 threads per shot.
    circuit = _random_unitary_circuit(width, layers)
    assert _widest(circuit) == width
    _assert_matches_cpu(circuit, width)


@pytest.mark.parametrize(("width", "layers"), [(6, 6), (8, 8), (11, 6)])
def test_cuda_looping_tapes_match_the_cpu_executor(monkeypatch, width, layers):
    # NOTE: a tape past the unroll budget loops, applying runs of rotations together.
    monkeypatch.setattr(codegen, "UNROLL_BUDGET", 0)
    _assert_matches_cpu(_random_unitary_circuit(width, layers, seed=width), width)


@pytest.mark.parametrize("width", [6, 9, 11])
def test_cuda_shots_start_after_the_common_prefix(width):
    # NOTE: without noise the rotations before the first measurement run once,
    # on the host; the shots load their amplitudes and run the rest.
    circuit = _random_unitary_circuit(width, width + 2, seed=width, noise=False)
    _assert_matches_cpu(circuit, width)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_cuda_looping_tapes_match_stabilizer_rank(monkeypatch, name):
    # NOTE: measurements, projections and pruning in looping tapes.
    monkeypatch.setattr(codegen, "UNROLL_BUDGET", 0)
    circuit = Circuit(CIRCUITS[name])
    shots = 40_000
    expected = circuit.compile_detector_sampler(strategy="cat5", seed=1).sample(
        shots, batch_size=10_000, append_observables=True
    )
    got = _gpu(circuit.compile_detector_sampler(seed=2)).sample(
        shots, append_observables=True
    )
    assert_counts_match(got, expected)


def test_components_wider_than_the_limit_run_on_the_cpu():
    circuit = _random_unitary_circuit(13, 12)
    assert _widest(circuit) > cuda_sampler.MAX_WIDTH
    with pytest.warns(UserWarning, match="sampling on the CPU"):
        sampler = circuit.compile_sampler(seed=1)
    assert type(sampler._shots) is ShotSampler


def _sampler_in_child(_):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler = Circuit(T_STATE_X).compile_sampler(seed=1)
        sampler.sample(10)
    return type(sampler._shots).__name__, [str(w.message) for w in caught]


def test_processes_forked_after_the_gpu_was_used_sample_on_the_cpu():
    # NOTE: CUDA cannot be used in a child forked after its parent initialized it.
    _gpu(Circuit(T_STATE_X).compile_sampler(seed=1)).sample(10)
    with multiprocessing.get_context("fork").Pool(1) as pool:
        name, messages = pool.map(_sampler_in_child, [0])[0]
    assert name == "ShotSampler"
    assert any("forked" in m for m in messages)


def test_noise_of_tiny_probabilities_fires_at_its_rate():
    # NOTE: 24-bit skips alone fire every segment at least 2^-24 of the time,
    # here ~90 faults instead of ~0.4.
    n, p = 1000, 1e-12
    text = "\n".join(f"X_ERROR({p * (1 + i / n)!r}) {i}" for i in range(n))
    sampler = _gpu(
        Circuit(f"{text}\nM " + " ".join(map(str, range(n)))).compile_sampler(seed=1)
    )
    # NOTE: faults are rare enough that each lands in a byte of its own.
    faults = sum(
        np.count_nonzero(sampler.sample(1 << 22, bit_packed=True)) for _ in range(64)
    )
    assert faults <= 5


@pytest.mark.parametrize("p0", [3e-8, 1e-6])
def test_rare_outcome_zero_is_drawn_at_its_rate(p0):
    # NOTE: w1 / total rounds to a multiple of 2^-24 below 1 in float32; the draw
    # must use w0, as the CPU's ``_cut`` does.
    cp = runtime.cupy()
    source = codegen.PRELUDE + r"""
extern "C" __global__ void zeros(float w0, unsigned long long n, unsigned long long* count) {
    unsigned long long local = 0;
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x)
        local += !draw_one(mix64(i * GOLDEN + 12345), w0, 1.f, w0 + 1.f);
    atomicAdd(count, local);
}
"""
    kernel = cp.RawModule(
        code=source, options=(runtime.arch(), "-std=c++17")
    ).get_function("zeros")
    count = cp.zeros(1, dtype=cp.uint64)
    n = 1 << 30
    kernel((1024,), (256,), (np.float32(p0), np.uint64(n), count))
    expected = p0 * n
    assert abs(int(count.get()[0]) - expected) <= 5 * np.sqrt(expected) + 1


@pytest.mark.parametrize(("probability", "shots"), [(1e-10, 1 << 28), (1e-6, 1 << 26)])
def test_cuda_draws_small_outcome_probabilities_at_their_rate(probability, shots):
    # NOTE: a 24-bit draw takes any positive probability at least 2^-24 = 6e-8 of
    # the time; float32 rounding gives certain outcomes such probabilities.
    theta = float(2 * np.arcsin(np.sqrt(probability)) / np.pi)
    circuit = Circuit(f"R 0\nI[R_X(theta={theta!r}*pi)] 0\nM 0")
    sampler = _gpu(circuit.compile_sampler(seed=3))
    ones = sum(int(sampler.sample(1 << 24).sum()) for _ in range(shots >> 24))
    expected = probability * shots
    assert abs(ones - expected) <= 5 * np.sqrt(expected) + 1


def _out_of_memory_above(monkeypatch, sampler, limit):
    """Make ``sampler``'s GPU calls of batches above ``limit`` run out of memory."""
    shots = _gpu_shots(sampler)
    sample_accepted = shots.sample_accepted
    batches = []

    def fake(n, *, batch_size=None, **kwargs):
        batches.append(batch_size)
        if batch_size is None or batch_size > limit:
            cp = runtime.cupy()
            raise cp.cuda.memory.OutOfMemoryError(1 << 40, 0)
        return sample_accepted(n, batch_size=batch_size, **kwargs)

    monkeypatch.setattr(shots, "sample_accepted", fake)
    return batches


def test_calls_out_of_gpu_memory_retry_in_smaller_batches(monkeypatch):
    sampler = _gpu(Circuit(GHZ_WITH_DIRECT_DETECTOR).compile_detector_sampler(seed=1))
    batches = _out_of_memory_above(monkeypatch, sampler, 1 << 20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sampler.sample(1000).shape == (1000, 3)
    assert batches == [1 << 22, 1 << 21, 1 << 20]
    # NOTE: later calls start from the batch that fit.
    sampler.sample(1000)
    assert batches[-1] == 1 << 20


def test_calls_out_of_gpu_memory_fall_back_to_the_cpu(monkeypatch):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    sampler = _gpu(circuit.compile_detector_sampler(seed=1))
    _out_of_memory_above(monkeypatch, sampler, 0)
    with pytest.warns(UserWarning, match="out of memory"):
        got = sampler.sample(20_000, append_observables=True)
    expected = _cpu(circuit.compile_detector_sampler, seed=2).sample(
        20_000, append_observables=True
    )
    assert_counts_match(got, expected)
