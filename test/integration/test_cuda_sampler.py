"""device="cuda" end to end, against the stabilizer-rank strategy and the CPU executor."""

from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import numpy as np
import pytest

from tsim.circuit import Circuit
from tsim.statevector.cuda import available
from tsim.statevector.program import StatevectorComponent

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


def _accepted(sampler, shots, mask, **kwargs):
    detectors, observables = sampler.sample(
        shots, separate_observables=True, postselection_mask=mask, **kwargs
    )
    keep = ~(detectors & mask).any(axis=1) if mask is not None else slice(None)
    return np.concatenate([detectors, observables], axis=1)[keep]


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_cuda_detector_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 40_000
    expected = circuit.compile_detector_sampler(strategy="cat5", seed=1).sample(
        shots, batch_size=10_000, append_observables=True
    )
    got = circuit.compile_detector_sampler(seed=2, device="cuda").sample(
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
    got = circuit.compile_sampler(seed=2, device="cuda").sample(shots)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_cuda_postselection_keeps_the_accepted_distribution(name):
    circuit = Circuit(CIRCUITS[name])
    if circuit.num_detectors == 0:
        pytest.skip("no detectors to postselect")
    shots = 40_000
    mask = np.ones(circuit.num_detectors, dtype=bool)
    expected = _accepted(circuit.compile_detector_sampler(seed=1), shots, mask)
    got = _accepted(
        circuit.compile_detector_sampler(seed=2, device="cuda"), shots, mask
    )
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
    got = circuit.compile_sampler(seed=seed + 10, device="cuda").sample(shots)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("reference", [False, True])
def test_cuda_reference_samples_and_pruning(reference):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    shots = 40_000
    kwargs = {"use_detector_reference_sample": reference}
    expected = _accepted(
        circuit.compile_detector_sampler(seed=3), shots, mask, **kwargs
    )
    got = _accepted(
        circuit.compile_detector_sampler(seed=3, device="cuda"), shots, mask, **kwargs
    )
    rate, cuda_rate = len(expected) / shots, len(got) / shots
    assert abs(cuda_rate - rate) <= 5 * np.sqrt(rate * (1 - rate) * 2 / shots)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sample_postselected_returns_only_passing_shots(device):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[[0, 2]] = True
    sampler = circuit.compile_detector_sampler(seed=4, device=device)
    packed, discarded = sampler.sample_postselected(20_000, postselection_mask=mask)
    assert discarded == 20_000 - len(packed)
    kept = circuit.num_detectors - mask.sum() + 1
    bits = np.unpackbits(packed, axis=1, bitorder="little")
    assert not bits[:, kept:].any()
    full = _accepted(circuit.compile_detector_sampler(seed=5), 20_000, mask)
    assert abs(len(bits) - len(full)) <= 5 * np.sqrt(len(full))
    outside = np.concatenate([~mask, [True]])
    assert_counts_match(bits[:, :kept].astype(bool), full[:, outside])


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


def test_cuda_returns_the_outputs_outside_the_mask():
    circuit = Circuit(NOISY_DETECTORS)
    rng = np.random.default_rng(0)
    mask = np.zeros(150, dtype=bool)
    # NOTE: a few masked detectors, then unmasked runs across words, then a mix.
    mask[[0, 3, 4]] = True
    mask[100:] = rng.random(50) < 0.1
    full, _ = circuit.compile_detector_sampler(
        seed=11, device="cuda"
    ).sample_postselected(5_000)
    rows, discarded = circuit.compile_detector_sampler(
        seed=11, device="cuda"
    ).sample_postselected(5_000, postselection_mask=mask)
    full_bits = np.unpackbits(full, axis=1, count=152, bitorder="little")
    passing = ~full_bits[:, :150][:, mask].any(axis=1)
    outside = np.concatenate([~mask, [True, True]])
    expected = np.packbits(full_bits[passing][:, outside], axis=1, bitorder="little")
    assert discarded == 5_000 - passing.sum() > 0
    np.testing.assert_array_equal(rows, expected)


def test_observable_expectations_average_to_the_sampled_observable():
    circuit = Circuit(T_STATE_X)
    sampler = circuit.compile_detector_sampler(seed=6, device="cuda")
    packed, values, _ = sampler.sample_postselected(
        200_000, observable_expectations=True
    )
    assert values.shape == (len(packed), 1)
    # NOTE: given the shot's Z error, the expectation is ±cos(π/4).
    np.testing.assert_allclose(np.abs(values), np.sqrt(0.5), atol=1e-5)
    observed = 1 - 2 * np.unpackbits(packed, axis=1, bitorder="little")[:, 0].astype(
        float
    )
    sampled = circuit.compile_detector_sampler(seed=7).sample(
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
    sampler = circuit.compile_detector_sampler(seed=9, device="cuda")
    _, values, _ = sampler.sample_postselected(
        shots, postselection_mask=mask, observable_expectations=True
    )
    detectors, observables = circuit.compile_detector_sampler(seed=10).sample(
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
    sampler = circuit.compile_detector_sampler(seed=8, device="cuda")
    packed, values, _ = sampler.sample_postselected(
        10_000, observable_expectations=True
    )
    bits = np.unpackbits(packed, axis=1, bitorder="little")
    np.testing.assert_array_equal(values[:, 0], 1 - 2 * bits[:, 1].astype(np.float32))


@pytest.mark.parametrize("reference", [False, True])
def test_cuda_samples_programs_without_components(reference):
    circuit = Circuit(
        "R 0 1\nX_ERROR(0.3) 0\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    )
    kwargs = {"use_detector_reference_sample": reference}
    expected = circuit.compile_detector_sampler(seed=1).sample(
        40_000, append_observables=True, **kwargs
    )
    got = circuit.compile_detector_sampler(seed=2, device="cuda").sample(
        40_000, append_observables=True, **kwargs
    )
    assert_counts_match(got, expected)
    measured = circuit.compile_sampler(seed=3, device="cuda").sample(40_000)
    assert measured.shape == (40_000, 2)
    np.testing.assert_array_equal(measured[:, 0], measured[:, 1])


def test_expectations_need_the_cuda_device():
    circuit = Circuit(T_STATE_X)
    with pytest.raises(ValueError, match="needs device='cuda'"):
        circuit.compile_detector_sampler(seed=1).sample_postselected(
            10, observable_expectations=True
        )


def test_cuda_needs_the_statevector_strategy():
    circuit = Circuit(T_STATE_X)
    with pytest.raises(ValueError, match="device='cuda' needs"):
        circuit.compile_detector_sampler(strategy="cat5", device="cuda")
    with pytest.raises(ValueError, match="device must be"):
        circuit.compile_detector_sampler(device="tpu")  # type: ignore[arg-type]


def _widest(circuit):
    """Return the active width of the circuit's widest statevector component."""
    components = circuit.compile_sampler(seed=1)._program.components
    return max(c.width for c in components if isinstance(c, StatevectorComponent))


def _random_unitary_circuit(width, layers, seed=0):
    """Layers of random U3 pairs and CX gates, then every qubit measured."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(layers):
        perm = rng.permutation(width)
        for a, b in zip(perm[::2], perm[1::2], strict=False):
            for q in (a, b):
                t = rng.uniform(-1, 1, 3)
                lines.append(f"U3({t[0]},{t[1]},{t[2]}) {q}")
            lines.append(f"CX {a} {b}")
    lines.append("M " + " ".join(map(str, range(width))))
    return Circuit("\n".join(lines))


@pytest.mark.parametrize(("width", "layers"), [(8, 8), (9, 10), (11, 12)])
def test_cuda_wide_components_match_the_cpu_executor(width, layers):
    # NOTE: widths 8, 9 and 11 run 8, 16 and 32 threads per shot.
    circuit = _random_unitary_circuit(width, layers)
    assert _widest(circuit) == width
    shots = 200_000
    expected = circuit.compile_sampler(seed=1).sample(shots)
    got = circuit.compile_sampler(seed=2, device="cuda").sample(shots)
    np.testing.assert_allclose(
        got.mean(axis=0), expected.mean(axis=0), atol=5 * 0.5 / np.sqrt(shots / 2)
    )
    # NOTE: pairs of bits, which a wrong shuffle partner would scramble.
    for a, b in [(0, 1), (0, width - 1), (width // 2, width - 1)]:
        assert_counts_match(got[:, [a, b]], expected[:, [a, b]])


def test_cuda_rejects_components_wider_than_its_limit():
    circuit = _random_unitary_circuit(13, 12)
    assert _widest(circuit) > 11
    with pytest.raises(ValueError, match="active qubits"):
        circuit.compile_sampler(seed=1, device="cuda")


@pytest.mark.parametrize(("probability", "shots"), [(1e-10, 1 << 28), (1e-6, 1 << 26)])
def test_cuda_draws_small_outcome_probabilities_at_their_rate(probability, shots):
    # NOTE: a 24-bit draw takes any positive probability at least 2^-24 = 6e-8 of
    # the time; float32 rounding gives certain outcomes such probabilities.
    theta = float(2 * np.arcsin(np.sqrt(probability)) / np.pi)
    circuit = Circuit(f"R 0\nI[R_X(theta={theta!r}*pi)] 0\nM 0")
    sampler = circuit.compile_sampler(seed=3, device="cuda")
    ones = sum(int(sampler.sample(1 << 24).sum()) for _ in range(shots >> 24))
    expected = probability * shots
    assert abs(ones - expected) <= 5 * np.sqrt(expected) + 1
