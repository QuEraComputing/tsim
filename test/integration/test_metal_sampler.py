"""The Metal executor end to end, against the stabilizer-rank strategy and the CPU executor."""

import multiprocessing
import pickle
import threading
import warnings
from test.helpers.gen import gen_stim_circuit
from test.helpers.statevector import CIRCUITS, assert_counts_match

import numpy as np
import pytest

from tsim.circuit import Circuit
from tsim.statevector.gpu import tape
from tsim.statevector.metal import (
    codegen,
    noise as metal_noise,
    runtime,
    sampler as metal_sampler,
)
from tsim.statevector.metal.sampler import MetalShotSampler
from tsim.statevector.program import StatevectorComponent
from tsim.statevector.shots import ShotSampler

pytestmark = pytest.mark.skipif(
    not runtime.available(), reason="needs PyObjC's Metal and an Apple GPU"
)

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


def _gpu_shots(sampler) -> MetalShotSampler:
    """Return the GPU shot sampler of ``sampler``."""
    assert isinstance(sampler._shots, MetalShotSampler)
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
def test_metal_detector_sampling_matches_stabilizer_rank(name):
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
def test_metal_measurement_sampling_matches_stabilizer_rank(name):
    circuit = Circuit(CIRCUITS[name])
    shots = 40_000
    expected = circuit.compile_sampler(strategy="cat5", seed=1).sample(
        shots, batch_size=10_000
    )
    got = _gpu(circuit.compile_sampler(seed=2)).sample(shots)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_metal_postselection_keeps_the_accepted_distribution(name):
    circuit = Circuit(CIRCUITS[name])
    if circuit.num_detectors == 0:
        pytest.skip("no detectors to postselect")
    shots = 40_000
    mask = np.ones(circuit.num_detectors, dtype=bool)
    expected = _accepted(_cpu(circuit.compile_detector_sampler, seed=1), shots, mask)
    got = _accepted(_gpu(circuit.compile_detector_sampler(seed=2)), shots, mask)
    rate, gpu_rate = len(expected) / shots, len(got) / shots
    sigma = np.sqrt(rate * (1 - rate) * 2 / shots)
    assert abs(gpu_rate - rate) <= 5 * sigma + 3 / shots
    if len(got) and len(expected):
        assert_counts_match(got, expected)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_metal_sampling_of_random_noisy_circuits(seed):
    stim_circuit = gen_stim_circuit(3, 60, p_t=3, p_x_error=0.5, p_r_z=0.5, seed=seed)
    circuit = Circuit.from_stim_program(stim_circuit)
    shots = 8_192
    expected = circuit.compile_sampler(strategy="cat5", seed=seed).sample(
        shots, batch_size=shots
    )
    got = _gpu(circuit.compile_sampler(seed=seed + 10)).sample(shots)
    assert_counts_match(got, expected)


@pytest.mark.parametrize("reference", [False, True])
def test_metal_reference_samples_and_pruning(reference):
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
    rate, gpu_rate = len(expected) / shots, len(got) / shots
    assert abs(gpu_rate - rate) <= 5 * np.sqrt(rate * (1 - rate) * 2 / shots)
    assert_counts_match(got, expected)


def test_metal_postselection_keeps_every_column():
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
def test_metal_arranges_the_columns_as_the_cpu_does(bit_packed):
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


def test_metal_samples_no_shots():
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


def test_metal_postselection_drops_the_rows_that_fail(monkeypatch):
    # NOTE: the masked run must draw the unmasked run's noise, so none is deferred.
    monkeypatch.setattr(metal_sampler, "DEFER_SHARE", 2.0)
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


def test_metal_defers_the_noise_that_postselection_cannot_see():
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
def test_metal_batches_return_the_rows_of_one_pass(monkeypatch, text):
    # NOTE: 1000-shot batches, queued on one set of buffers; without deferral, a
    # masked and an unmasked run on the same batches draw the same shots.
    monkeypatch.setattr(metal_sampler, "DEFER_SHARE", 2.0)
    circuit = Circuit(text)
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[[0, 2]] = True
    outputs = circuit.num_detectors + circuit.num_observables

    def packed(mask=None):
        sampler = _gpu(circuit.compile_detector_sampler(seed=16))
        return sampler.sample(
            10_500,
            append_observables=True,
            bit_packed=True,
            postselection_mask=mask,
            batch_size=1000,
        )

    full, rows = packed(), packed(mask)
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
def test_metal_samples_programs_without_components(reference):
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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
def test_metal_wide_components_match_the_cpu_executor(width, layers):
    # NOTE: widths 6, 8, 9 and 11 run 2, 8, 16 and 32 threads per shot.
    circuit = _random_unitary_circuit(width, layers)
    assert _widest(circuit) == width
    _assert_matches_cpu(circuit, width)


@pytest.mark.parametrize(("width", "layers"), [(6, 6), (8, 8), (11, 6)])
def test_metal_looping_tapes_match_the_cpu_executor(monkeypatch, width, layers):
    # NOTE: a tape past the unroll budget loops, applying runs of rotations together.
    monkeypatch.setattr(tape, "UNROLL_BUDGET", 0)
    _assert_matches_cpu(_random_unitary_circuit(width, layers, seed=width), width)


@pytest.mark.parametrize("width", [6, 9, 11])
def test_metal_shots_start_after_the_common_prefix(width):
    # NOTE: without noise the rotations before the first measurement run once,
    # on the host; the shots load their amplitudes and run the rest.
    circuit = _random_unitary_circuit(width, width + 2, seed=width, noise=False)
    _assert_matches_cpu(circuit, width)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_metal_looping_tapes_match_stabilizer_rank(monkeypatch, name):
    # NOTE: measurements, projections and pruning in looping tapes.
    monkeypatch.setattr(tape, "UNROLL_BUDGET", 0)
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
    assert _widest(circuit) > metal_sampler.MAX_WIDTH
    with pytest.warns(UserWarning, match="sampling on the CPU"):
        sampler = circuit.compile_sampler(seed=1)
    assert type(sampler._shots) is ShotSampler


@pytest.mark.parametrize(("probability", "shots"), [(1e-10, 1 << 28), (1e-6, 1 << 26)])
def test_metal_draws_small_outcome_probabilities_at_their_rate(probability, shots):
    # NOTE: a 24-bit draw takes any positive probability at least 2^-24 = 6e-8 of
    # the time; float32 rounding gives certain outcomes such probabilities.
    theta = float(2 * np.arcsin(np.sqrt(probability)) / np.pi)
    circuit = Circuit(f"R 0\nI[R_X(theta={theta!r}*pi)] 0\nM 0")
    sampler = _gpu(circuit.compile_sampler(seed=3))
    ones = sum(int(sampler.sample(1 << 24).sum()) for _ in range(shots >> 24))
    expected = probability * shots
    assert abs(ones - expected) <= 5 * np.sqrt(expected) + 1


def test_metal_reruns_a_call_whose_rows_overflow_the_output(monkeypatch):
    # NOTE: room for 10 rows: the call runs again with room for every shot, on
    # the same streams, so it returns the rows of a call with enough room.
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[[0, 2]] = True

    def packed():
        sampler = _gpu(circuit.compile_detector_sampler(seed=21))
        return sampler.sample(
            5_000,
            append_observables=True,
            bit_packed=True,
            postselection_mask=mask,
            batch_size=1000,
        )

    expected = packed()
    monkeypatch.setattr(MetalShotSampler, "_capacity", lambda self, *args: 10)
    rows = packed()
    assert len(rows) > 10
    np.testing.assert_array_equal(rows, expected)


def test_metal_unmasked_expectations_after_a_masked_call():
    # NOTE: a masked call lists its accepted shots in the batch buffers; an
    # unmasked call on the same buffers must still read every row's own value.
    circuit = Circuit(T_STATE_X_AFTER_DETECTOR)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    sampler = _gpu(circuit.compile_detector_sampler(seed=5))
    _expectations(sampler, 20_000, mask)
    shots = _gpu_shots(sampler)
    state = shots._rng.bit_generator.state
    rows, values = _expectations(sampler, 20_000)
    shots._rng.bit_generator.state = state
    shots._slots.clear()
    again, values_again = _expectations(sampler, 20_000)
    np.testing.assert_array_equal(again, rows)
    np.testing.assert_array_equal(values_again, values)


def test_metal_reports_a_nan_normalization():
    sampler = _gpu(Circuit(T_STATE_X).compile_detector_sampler(seed=1))
    counts = np.zeros(4, dtype=np.uint32)
    counts[2] = 0x7FC00000
    assert np.isnan(_gpu_shots(sampler)._deviation(counts))


def test_metal_samples_rows_too_wide_for_threadgroup_memory():
    # NOTE: 3200 detectors are 51 words: the noise goes into device memory.
    circuit = Circuit(
        "\n".join(
            [
                "R " + " ".join(map(str, range(3200))),
                "X_ERROR(0.05) " + " ".join(map(str, range(3200))),
                "M " + " ".join(map(str, range(3200))),
                *(f"DETECTOR rec[{i - 3200}]" for i in range(3200)),
                "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]",
            ]
        )
    )
    sampler = _gpu(circuit.compile_detector_sampler(seed=31))
    assert not _gpu_shots(sampler)._noise.shared_words
    got = sampler.sample(20_000, append_observables=True)
    np.testing.assert_allclose(got[:, :3200].mean(), 0.05, atol=5e-4)
    np.testing.assert_allclose(got[:, -1].mean(), 2 * 0.05 * 0.95, atol=0.01)
    np.testing.assert_array_equal(got[:, -1], got[:, -2] ^ got[:, -3])


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_metal_noise_in_device_memory_matches_the_cpu(monkeypatch, name):
    monkeypatch.setattr(metal_noise, "SHARED_BYTES", 2048 + 8)
    circuit = Circuit(CIRCUITS[name])
    shots = 40_000
    mask = np.ones(circuit.num_detectors, dtype=bool) if circuit.num_detectors else None
    expected = _accepted(_cpu(circuit.compile_detector_sampler, seed=1), shots, mask)
    sampler = _gpu(circuit.compile_detector_sampler(seed=2))
    assert not _gpu_shots(sampler)._noise.shared_words
    got = _accepted(sampler, shots, mask)
    rate, gpu_rate = len(expected) / shots, len(got) / shots
    assert (
        abs(gpu_rate - rate) <= 5 * np.sqrt(rate * (1 - rate) * 2 / shots) + 3 / shots
    )
    if len(got) and len(expected):
        assert_counts_match(got, expected)


def test_metal_unrolled_width_11_tapes_match_the_cpu_executor():
    # NOTE: 64 amplitudes per thread, unrolled, in two register arrays.
    circuit = _random_unitary_circuit(11, 7, seed=0)
    assert _widest(circuit) == 11
    shots = _gpu_shots(_gpu(circuit.compile_sampler(seed=1)))
    assert not tape.loops(shots.plan(None).programs[0], 5, 0)
    _assert_matches_cpu(circuit, 11)


def test_metal_returned_rows_are_not_reused_while_alive():
    circuit = Circuit(NOISY_DETECTORS)
    sampler = _gpu(circuit.compile_detector_sampler(seed=3))
    first = _packed(sampler, 200_000)
    kept = first.copy()
    second = _packed(sampler, 200_000)
    np.testing.assert_array_equal(first, kept)
    assert not np.shares_memory(first, second)
    del first
    third = _packed(sampler, 200_000)
    assert not np.shares_memory(second, third)
    assert not (third == second).all()


def test_metal_leaves_programs_without_shot_words_on_the_cpu():
    circuit = Circuit("R 0\nH 0\nT 0\nH 0\nM 0")
    sampler = circuit.compile_detector_sampler(seed=1)
    assert type(sampler._shots) is ShotSampler
    assert sampler.sample(10).shape == (10, 0)


def _sampler_in_child(_):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler = Circuit(T_STATE_X).compile_sampler(seed=1)
        sampler.sample(10)
    return type(sampler._shots).__name__, [str(w.message) for w in caught]


def test_processes_forked_after_the_gpu_was_used_sample_on_the_cpu():
    # NOTE: Metal's device does not survive a fork of a process that used it.
    _gpu(Circuit(T_STATE_X).compile_sampler(seed=1)).sample(10)
    with multiprocessing.get_context("fork").Pool(1) as pool:
        name, messages = pool.map(_sampler_in_child, [0])[0]
    assert name == "ShotSampler"
    assert any("forked" in m for m in messages)


def test_noise_of_tiny_probabilities_fires_at_its_rate():
    # NOTE: 24-bit skips alone fire every segment at least 2^-24 of the time,
    # here ~350 faults instead of ~0.3.
    n, p = 1000, 1e-12
    text = "\n".join(f"X_ERROR({p * (1 + i / n)!r}) {i}" for i in range(n))
    sampler = _gpu(
        Circuit(f"{text}\nM " + " ".join(map(str, range(n)))).compile_sampler(seed=1)
    )
    # NOTE: faults are rare enough that each lands in a byte of its own.
    faults = sum(
        np.count_nonzero(sampler.sample(1 << 22, bit_packed=True)) for _ in range(44)
    )
    assert faults <= 5


@pytest.mark.parametrize("p0", [3e-8, 1e-6])
def test_rare_outcome_zero_is_drawn_at_its_rate(p0):
    # NOTE: w1 / total rounds to a multiple of 2^-24 below 1 in float32; the draw
    # must use w0, as the CPU's ``_cut`` does.
    source = codegen.PRELUDE + r"""
kernel void zeros(constant float& w0 [[buffer(0)]], constant ulong& n [[buffer(1)]],
    device atomic_uint* count [[buffer(2)]], uint gid [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    uint local = 0;
    for (u64 i = gid; i < n; i += grid)
        local += !draw_one(mix64(i * GOLDEN + 12345), w0, 1.f, w0 + 1.f);
    if (local) atomic_fetch_add_explicit(count, local, memory_order_relaxed);
}
"""
    pipeline = runtime.Library(source).pipeline("zeros")
    count = runtime.buffer(1, np.uint32)
    n = 1 << 30
    encoder = runtime.Encoder()
    encoder.dispatch(pipeline, [np.float32(p0), np.uint64(n), count], 1024, 256)
    runtime.wait(encoder.commit())
    expected = p0 * n
    assert abs(int(count.array[0]) - expected) <= 5 * np.sqrt(expected) + 1


def test_calls_out_of_gpu_memory_fall_back_to_the_cpu(monkeypatch):
    circuit = Circuit(GHZ_WITH_DIRECT_DETECTOR)
    sampler = _gpu(circuit.compile_detector_sampler(seed=1))

    def out_of_memory(*args, **kwargs):
        raise MemoryError("the GPU ran out of memory")

    monkeypatch.setattr(_gpu_shots(sampler), "sample_accepted", out_of_memory)
    with pytest.warns(UserWarning, match="out of memory; sampling on the CPU"):
        got = sampler.sample(20_000, append_observables=True)
    expected = _cpu(circuit.compile_detector_sampler, seed=2).sample(
        20_000, append_observables=True
    )
    assert_counts_match(got, expected)


def test_samplers_out_of_gpu_memory_at_setup_run_on_the_cpu(monkeypatch):
    def out_of_memory(*args, **kwargs):
        raise MemoryError("no Metal buffer")

    monkeypatch.setattr(MetalShotSampler, "__init__", out_of_memory)
    with pytest.warns(UserWarning, match="out of memory; sampling on the CPU"):
        sampler = Circuit(T_STATE_X).compile_sampler(seed=1)
    assert type(sampler._shots) is ShotSampler


def test_tsim_device_cpu_keeps_the_samplers_on_the_cpu(monkeypatch):
    monkeypatch.setenv("TSIM_DEVICE", "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = Circuit(T_STATE_X).compile_sampler(seed=1)
    assert type(sampler._shots) is ShotSampler
    monkeypatch.setenv("TSIM_DEVICE", "")
    _gpu(Circuit(T_STATE_X).compile_sampler(seed=1))


@pytest.mark.parametrize("text", [GHZ_WITH_DIRECT_DETECTOR, NOISY_DETECTORS])
def test_metal_unpacks_booleans_as_it_packs_rows(text):
    # NOTE: samplers of one seed draw the same shots; 4 and 152 outputs, neither a
    # multiple of 8.
    circuit = Circuit(text)
    outputs = circuit.num_detectors + circuit.num_observables
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[0] = True
    for postselection_mask in (None, mask):

        def sample(bit_packed, postselection_mask=postselection_mask):
            sampler = _gpu(circuit.compile_detector_sampler(seed=23))
            return sampler.sample(
                30_000,
                append_observables=True,
                bit_packed=bit_packed,
                postselection_mask=postselection_mask,
            )

        packed, bools = sample(True), sample(False)
        assert bools.dtype == np.bool_ and bools.flags.c_contiguous
        unpacked = np.unpackbits(packed, axis=1, count=outputs, bitorder="little")
        np.testing.assert_array_equal(bools, unpacked.astype(bool))
        detectors = _gpu(circuit.compile_detector_sampler(seed=23)).sample(
            30_000, postselection_mask=postselection_mask
        )
        assert detectors.flags.c_contiguous
        np.testing.assert_array_equal(detectors, bools[:, : circuit.num_detectors])


def test_threads_sharing_a_gpu_sampler_get_their_own_rows():
    # NOTE: a call's batches share the sampler's buffers; four threads at once
    # must not see each other's rows.
    circuit = Circuit(NOISY_DETECTORS)
    sampler = _gpu(circuit.compile_detector_sampler(seed=24))
    mask = np.zeros(150, dtype=bool)
    mask[:3] = True
    failures = []

    def work(masked):
        for _ in range(6):
            rows = sampler.sample(
                20_000,
                batch_size=1000,
                append_observables=True,
                postselection_mask=mask if masked else None,
            )
            if not masked and len(rows) != 20_000:
                failures.append(f"{len(rows)} rows")
            if masked and rows[:, :3].any():
                failures.append("a masked detector is set")
            if (rows[:, 150] != rows[:, 149] ^ rows[:, 143]).any():
                failures.append("an observable disagrees with its detectors")

    threads = [threading.Thread(target=work, args=(i % 2,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures[:5]


def test_an_interrupted_call_leaves_the_next_call_alone(monkeypatch):
    # NOTE: a slow width-11 tape and kernels compiled beforehand, so the
    # interrupted call's batches are still on the GPU when the next call starts.
    width = 11
    lines = [str(_random_unitary_circuit(width, 30, seed=0))]
    lines += [f"DETECTOR rec[{i - width}]" for i in range(width)]
    circuit = Circuit("\n".join(lines))
    sampler = _gpu(circuit.compile_detector_sampler(seed=25))
    shots = _gpu_shots(sampler)
    all_detectors = np.ones(width, dtype=bool)
    mask = np.zeros(width, dtype=bool)
    mask[1] = True
    for warm in (mask, all_detectors):
        sampler.sample(1 << 17, batch_size=1 << 16, postselection_mask=warm)
    encode_pack = MetalShotSampler._encode_pack
    calls = []

    def interrupt_late(self, *args, **kwargs):
        calls.append(None)
        if len(calls) == 20:
            raise KeyboardInterrupt
        return encode_pack(self, *args, **kwargs)

    monkeypatch.setattr(MetalShotSampler, "_encode_pack", interrupt_late)
    with pytest.raises(KeyboardInterrupt):
        sampler.sample(1 << 23, batch_size=1 << 16, postselection_mask=all_detectors)
    monkeypatch.setattr(MetalShotSampler, "_encode_pack", encode_pack)
    state = shots._rng.bit_generator.state
    rows = sampler.sample(1 << 18, batch_size=1 << 16, postselection_mask=mask)
    shots._rng.bit_generator.state = state
    expected = sampler.sample(1 << 18, batch_size=1 << 16, postselection_mask=mask)
    np.testing.assert_array_equal(rows, expected)


SAMPLER_BEFORE_FORK = None


def _sample_in_child(_):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert SAMPLER_BEFORE_FORK is not None
        rows = SAMPLER_BEFORE_FORK.sample(100)
    return rows.shape, [str(w.message) for w in caught]


def test_samplers_compiled_before_a_fork_sample_on_the_cpu_in_the_child():
    global SAMPLER_BEFORE_FORK
    SAMPLER_BEFORE_FORK = _gpu(Circuit(T_STATE_X).compile_detector_sampler(seed=1))
    with multiprocessing.get_context("fork").Pool(1) as pool:
        shape, messages = pool.map_async(_sample_in_child, [0]).get(timeout=60)[0]
    assert shape == (100, 0)
    assert any("forked" in m for m in messages)


def test_gpu_samplers_pickle_and_continue_their_streams():
    circuit = Circuit(NOISY_DETECTORS)
    sampler = _gpu(circuit.compile_detector_sampler(seed=26))
    sampler.sample(1000)
    loaded = pickle.loads(pickle.dumps(sampler))  # noqa: S301
    assert isinstance(loaded._shots, MetalShotSampler)
    np.testing.assert_array_equal(loaded.sample(5000), sampler.sample(5000))


def test_noise_beyond_the_gpu_tables_samples_on_the_cpu():
    # NOTE: a chain of 8 correlated errors has 255 outcomes, more than the 127 of
    # a GPU noise record.
    chain = ["E(0.01) X0", *(f"ELSE_CORRELATED_ERROR(0.01) X{k}" for k in range(1, 8))]
    text = "\n".join(
        ["R 0 1 2 3 4 5 6 7 8", "H 8", "T 8", "H 8", *chain, "M 0 1 2 3 4 5 6 7 8"]
        + [f"DETECTOR rec[{k - 9}]" for k in range(9)]
    )
    with pytest.warns(UserWarning, match="cannot sample this circuit"):
        sampler = Circuit(text).compile_detector_sampler(seed=1)
    assert type(sampler._shots) is ShotSampler
    assert sampler.sample(10).shape == (10, 9)


def test_direct_outputs_out_of_gpu_memory_fall_back_to_the_cpu(monkeypatch):
    circuit = Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    sampler = _gpu(circuit.compile_detector_sampler(seed=1))

    def out_of_memory(*args, **kwargs):
        raise MemoryError("the GPU ran out of memory")

    monkeypatch.setattr(_gpu_shots(sampler), "sample_accepted", out_of_memory)
    with pytest.warns(UserWarning, match="out of memory; sampling on the CPU"):
        rows = sampler.sample(20_000)
    assert rows.shape == (20_000, 1) and abs(rows.mean() - 0.1) < 0.01


def test_calls_larger_than_the_counters_run_in_parts(monkeypatch):
    # NOTE: the 32-bit counters hold MAX_CALL_SHOTS; a larger call is the calls of
    # its parts.
    monkeypatch.setattr(metal_sampler, "MAX_CALL_SHOTS", 1000)
    mask = np.zeros(150, dtype=bool)
    mask[:3] = True
    circuit = Circuit(NOISY_DETECTORS)
    whole = _gpu(circuit.compile_detector_sampler(seed=27)).sample(
        2500, postselection_mask=mask, bit_packed=True
    )
    parts = _gpu(circuit.compile_detector_sampler(seed=27))
    expected = [
        parts.sample(n, postselection_mask=mask, bit_packed=True)
        for n in (1000, 1000, 500)
    ]
    np.testing.assert_array_equal(whole, np.concatenate(expected))


def test_the_repr_says_where_a_sampler_samples():
    circuit = Circuit(T_STATE_X)
    assert "samples on metal" in repr(_gpu(circuit.compile_detector_sampler(seed=1)))
    cpu = _cpu(circuit.compile_detector_sampler, seed=1)
    assert "samples on cpu" in repr(cpu) and cpu.stats.device == "cpu"
