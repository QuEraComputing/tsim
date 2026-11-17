import subprocess
import sys
import warnings
from test.helpers.statevector import shot_sampler
from unittest.mock import patch

import numpy as np
import pytest

import tsim.sampler as sampler_module
import tsim.statevector.shared as shared
from tsim.circuit import Circuit
from tsim.statevector.shots import ShotSampler
from tsim.utils.kernels import PARALLEL_WORK


def test_detector_sampler_args():
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler()
    d = sampler.sample(1)
    assert np.array_equal(d, np.array([[0, 0]]))

    d = sampler.sample(1, append_observables=True)
    assert np.array_equal(d, np.array([[0, 0, 1]]))

    d = sampler.sample(1, prepend_observables=True)
    assert np.array_equal(d, np.array([[1, 0, 0]]))

    d = sampler.sample(1, prepend_observables=True, append_observables=True)
    assert np.array_equal(d, np.array([[1, 0, 0, 1]]))

    d, o = sampler.sample(1, separate_observables=True)
    assert np.array_equal(d, np.array([[0, 0]]))
    assert np.array_equal(o, np.array([[1]]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"separate_observables": True, "append_observables": True},
            "separate_observables",
        ),
        (
            {"separate_observables": True, "prepend_observables": True},
            "separate_observables",
        ),
        (
            {
                "separate_observables": True,
                "append_observables": True,
                "prepend_observables": True,
            },
            "separate_observables",
        ),
    ],
)
def test_detector_sampler_invalid_observable_flag_combos_raise(kwargs, match):
    """Conflicting observable placement flags must raise rather than silently drop columns."""
    c = Circuit("""
        R 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler()
    with pytest.raises(ValueError, match=match):
        sampler.sample(1, **kwargs)


def test_measurement_sampler_zero_shots():
    """Measurement sampler with shots=0 returns an empty (0, num_measurements) array."""
    c = Circuit("H 0\nM 0")
    sampler = c.compile_sampler()
    result = sampler.sample(0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 1)


def test_detector_sampler_zero_shots():
    """Detector sampler with shots=0 returns an empty (0, num_detectors) array."""
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler()
    result = sampler.sample(0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)


def test_detector_sampler_zero_shots_separate_observables():
    """Detector sampler with shots=0 and separate_observables returns two empty arrays."""
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler()
    dets, obs = sampler.sample(0, separate_observables=True)
    assert dets.shape == (0, 2)
    assert obs.shape == (0, 1)


def test_detector_sampler_zero_shots_with_reference():
    """Detector sampler with shots=0 and reference enabled returns an empty array with the expected columns."""
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler()
    result = sampler.sample(
        0,
        append_observables=True,
        use_detector_reference_sample=True,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 3)


@pytest.mark.parametrize("strategy", ["active_statevector", "cat5"])
@pytest.mark.parametrize("shots", [0, 2000])
def test_measurement_sampler_bit_packed(strategy, shots):
    text = "H 0 1 2\nT 0\nX_ERROR(0.1) 3 4 5 6 7 8 9 10 11\nM 0 1 2 3 4 5 6 7 8 9 10 11"

    def sampler():
        return Circuit(text).compile_sampler(seed=4, strategy=strategy)

    expected = sampler().sample(shots, batch_size=1000)
    result = sampler().sample(shots, batch_size=1000, bit_packed=True)
    assert result.dtype == np.uint8
    assert result.shape == (shots, 2)
    assert np.array_equal(result, np.packbits(expected, axis=1, bitorder="little"))


def test_measurement_sampler_no_measurements():
    """Measurement sampler on a circuit with no measurements returns (shots, 0)."""
    c = Circuit("H 0")
    sampler = c.compile_sampler()
    result = sampler.sample(5)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    assert result.shape == (5, 0)


def test_detector_sampler_no_detectors():
    """Detector sampler on a circuit with no detectors returns (shots, 0)."""
    c = Circuit("H 0\nM 0")
    sampler = c.compile_detector_sampler()
    result = sampler.sample(5)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    assert result.shape == (5, 0)


def test_detector_sampler_no_detectors_separate_observables():
    """Detector sampler with separate_observables and no detectors returns two (shots, 0) arrays."""
    c = Circuit("H 0\nM 0")
    sampler = c.compile_detector_sampler()
    dets, obs = sampler.sample(5, separate_observables=True)
    assert dets.dtype == np.bool_
    assert dets.shape == (5, 0)
    assert obs.dtype == np.bool_
    assert obs.shape == (5, 0)


def test_detector_sampler_no_detectors_bit_packed():
    """Detector sampler with bit_packed and no detectors returns (shots, 0) uint8."""
    c = Circuit("H 0\nM 0")
    sampler = c.compile_detector_sampler()
    result = sampler.sample(5, bit_packed=True)
    assert result.dtype == np.uint8
    assert result.shape == (5, 0)


# NOTE: noise bits only, and a T component next to them.
PACKED_CIRCUITS = [
    "X_ERROR(0.1) 0 1 2 3\nM 0 1 2 3\n"
    "DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\n"
    "OBSERVABLE_INCLUDE(0) rec[-1]\n",
    "X_ERROR(0.1) 0 1\nH 2\nT 2\nH 2\nM 0 1 2\n"
    "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
    "OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]\n",
]


@pytest.mark.parametrize("text", PACKED_CIRCUITS)
@pytest.mark.parametrize(
    "layout",
    [{}, {"append_observables": True}, {"prepend_observables": True}],
)
def test_bit_packed_outputs_are_the_packed_bool_outputs(text, layout):
    unpacked = Circuit(text).compile_detector_sampler(seed=5)
    packed = Circuit(text).compile_detector_sampler(seed=5)
    expected = unpacked.sample(9000, **layout)
    result = packed.sample(9000, bit_packed=True, **layout)
    assert result.dtype == np.uint8
    assert np.array_equal(result, np.packbits(expected, axis=1, bitorder="little"))


# NOTE: noise outputs only, one of them constant, so bool samples come straight
# from the noise sampler and packed ones from the shot words.
DIRECT_ONLY = """
X 1
X_ERROR(0.2) 0 2 3
M 0 1 2 3
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
OBSERVABLE_INCLUDE(1) rec[-4] rec[-1]
"""


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize(
    "layout",
    [
        {},
        {"append_observables": True},
        {"prepend_observables": True},
        {"prepend_observables": True, "append_observables": True},
        {"separate_observables": True},
    ],
)
def test_bool_noise_outputs_are_the_shot_word_outputs(layout, reference):
    def sample(bit_packed):
        sampler = Circuit(DIRECT_ONLY).compile_detector_sampler(seed=3)
        result = sampler.sample(
            5000,
            bit_packed=bit_packed,
            use_detector_reference_sample=reference,
            use_observable_reference_sample=reference,
            **layout,
        )
        return result if isinstance(result, tuple) else (result,)

    for bools, packed in zip(sample(False), sample(True), strict=True):
        assert bools.dtype == np.bool_
        expected = np.packbits(bools, axis=1, bitorder="little")
        np.testing.assert_array_equal(packed, expected)


@pytest.mark.parametrize("shots", [0, 5000])
def test_bit_packed_direct_outputs_span_several_words(shots):
    """72 detectors over two words: whole last byte, a later observable, and no shots."""
    n = 72
    targets = " ".join(map(str, range(n + 1)))
    text = f"X_ERROR(0.02) {targets}\nM {targets}\n"
    text += "".join(f"DETECTOR rec[-{n + 1 - i}]\n" for i in range(n))
    text += "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    unpacked = Circuit(text).compile_detector_sampler(seed=2)
    packed = Circuit(text).compile_detector_sampler(seed=2)
    for append_observables in (False, True):
        expected = unpacked.sample(shots, append_observables=append_observables)
        result = packed.sample(
            shots, append_observables=append_observables, bit_packed=True
        )
        expected = np.packbits(expected, axis=1, bitorder="little")
        assert result.shape == expected.shape
        assert np.array_equal(result, expected)


def test_detector_sampler_no_detectors_with_reference_sample():
    """Detector sampler with use_detector_reference_sample=True and no detectors returns (shots, 0)."""
    c = Circuit("H 0\nM 0")
    sampler = c.compile_detector_sampler()
    result = sampler.sample(5, use_detector_reference_sample=True)
    assert result.dtype == np.bool_
    assert result.shape == (5, 0)


def test_sampler_negative_shots_raises():
    sampler = Circuit("H 0\nM 0").compile_sampler(seed=0)
    with pytest.raises(ValueError, match="shots must be non-negative"):
        sampler.sample(-1)


def test_sampler_zero_batch_size_raises():
    sampler = Circuit("H 0\nM 0").compile_sampler(seed=0)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        sampler.sample(1, batch_size=0)


def test_sampler_negative_batch_size_raises():
    sampler = Circuit("H 0\nM 0").compile_sampler(seed=0)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        sampler.sample(1, batch_size=-3)


def test_state_probs_rejects_wrong_state_length():
    from tsim.sampler import CompiledStateProbs

    sampler = CompiledStateProbs(Circuit("M 0"))
    with pytest.raises(ValueError, match=r"state must have shape \(1,\)"):
        sampler.probability_of(np.array([False, True]), batch_size=1)


def test_state_probs_rejects_invalid_batch_size():
    from tsim.sampler import CompiledStateProbs

    sampler = CompiledStateProbs(Circuit("M 0"))
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        sampler.probability_of(np.array([False]), batch_size=0)


def test_sampler_repr_no_measurements():
    """repr() on a sampler with no measurements should not error."""
    c = Circuit("H 0")
    sampler = c.compile_sampler()
    repr_str = repr(sampler)
    assert "CompiledMeasurementSampler" in repr_str
    assert "0 outputs" in repr_str


def test_seed():
    c = Circuit("""
        H 0
        M 0
        """)
    for _ in range(2):
        sampler = c.compile_sampler(strategy="cat5", seed=0)
        assert np.count_nonzero(sampler.sample(100)) == 45
        assert np.count_nonzero(sampler.sample(100)) == 50
        assert np.count_nonzero(sampler.sample(100)) == 46
        assert np.count_nonzero(sampler.sample(100)) == 46


def test_seeds_sharing_30_bits_give_different_noise():
    # NOTE: these seeds used to draw the same 30-bit noise seed (#261).
    c = Circuit("X_ERROR(0.1) 0\nM 0")
    a = c.compile_sampler(seed=37430).sample(1000)
    b = c.compile_sampler(seed=43909).sample(1000)
    assert not np.array_equal(a, b)


def test_no_two_seeds_share_a_stream():
    # NOTE: as a seed list, [s + 2**32, 0] is the same entropy as [s, 1], so
    # streams keyed by such lists could be another seed's stream.
    c = Circuit("X_ERROR(0.1) 0\nH 1\nT 1\nH 1\nM 0 1")
    seeds = [5 + k * 2**32 for k in range(4)]
    first_draws = set()
    for seed in seeds:
        sampler = c.compile_sampler(seed=seed)
        shots = shot_sampler(sampler)
        for rng in (
            sampler._channel_sampler._rng,
            shots._seeds,
            shots._reference_seeds,
        ):
            first_draws.add(int(rng.integers(2**64, dtype=np.uint64)))
    assert len(first_draws) == 3 * len(seeds)


@pytest.mark.parametrize("strategy", ["active_statevector", "cat5"])
@pytest.mark.parametrize("seed", [5, 2**64 + 5])
def test_component_samples_depend_on_every_bit_of_the_seed(strategy, seed):
    # NOTE: jax.random.key(seed) used to keep only the low 32 bits, and failed
    # from 2**63 on.
    c = Circuit("H 0\nT 0\nH 0\nM 0")
    a = c.compile_sampler(strategy=strategy, seed=seed).sample(1000)
    b = c.compile_sampler(strategy=strategy, seed=seed + 2**32).sample(1000)
    assert not np.array_equal(a, b)


def test_sampler_repr():
    c = Circuit("""
        X_ERROR(0.1) 0 1
        M 0 1
        """)
    sampler = c.compile_sampler()
    repr_str = repr(sampler)
    assert "CompiledMeasurementSampler" in repr_str
    assert "2 error channel bits" in repr_str


def test_sampler_repr_shows_only_used_backends():
    c = Circuit("""
        H 0
        T 0
        H 0
        M 0
        """)
    statevector = repr(c.compile_sampler())
    assert "statevector components" in statevector
    assert "graphs" not in statevector
    assert "A terms" not in statevector
    zx = repr(c.compile_sampler(strategy="cat5"))
    assert "statevector" not in zx
    assert "graphs" in zx
    assert "A terms" in zx


T_CIRCUIT = """
    H 0 1
    T 0 1
    CX 0 1
    T 1
    H 0 1
    T 0 1
    CX 1 0
    H 0 1
    M 0 1
    DETECTOR rec[-1]
    DETECTOR rec[-2]
    """


@pytest.mark.parametrize("detectors", [False, True])
def test_stats_of_clifford_circuit(detectors):
    c = Circuit("""
        X_ERROR(0.1) 0 1
        H 2
        CX 2 3
        M 0 1 2 3
        DETECTOR rec[-4]
        DETECTOR rec[-3]
        """)
    sampler = c.compile_detector_sampler() if detectors else c.compile_sampler()
    stats = sampler.stats
    assert sampler.active_width == stats.active_width == 0
    assert stats.component_widths is not None
    assert set(stats.component_widths) <= {0}
    assert stats.error_channel_bits == 2


@pytest.mark.parametrize("detectors", [False, True])
def test_stats_of_t_circuit(detectors):
    c = Circuit(T_CIRCUIT)
    sampler = c.compile_detector_sampler() if detectors else c.compile_sampler()
    stats = sampler.stats
    assert sampler.active_width == 2
    assert stats.num_direct_outputs == 0
    assert stats.component_widths == (2,)
    assert stats.num_actions is not None and stats.num_actions > 0
    assert stats.peak_bytes_per_shot > 4 * 16
    assert stats.compiled_bytes > 0
    assert "width ≤ 2" in repr(sampler)
    with pytest.raises(AttributeError):
        stats.active_width = 0  # type: ignore[misc]


def test_stats_of_stabilizer_rank_strategy():
    sampler = Circuit(T_CIRCUIT).compile_detector_sampler(strategy="cat5")
    stats = sampler.stats
    assert sampler.active_width is None
    assert stats.num_components == 1
    assert stats.component_widths is None
    assert stats.num_actions is None
    assert stats.peak_bytes_per_shot > 0
    # NOTE: without components, nothing is simulated under any strategy.
    noise = Circuit("X_ERROR(0.1) 0\nM 0").compile_sampler(strategy="cat5")
    assert noise.active_width == 0


@pytest.mark.parametrize(
    ("shots", "expected_batch_size"),
    [(100, 25), (101, 26)],
)
def test_auto_batch(shots, expected_batch_size):
    c = Circuit("""
        H 0
        M 0
        """)
    sampler = c.compile_sampler(strategy="cat5", seed=42)

    # Mock _estimate_batch_size to return a small value so auto-batching kicks in.
    with (
        patch.object(type(sampler), "_estimate_batch_size", return_value=30),
        patch.object(
            sampler._channel_sampler,
            "sample",
            wraps=sampler._channel_sampler.sample,
        ) as channel_sample,
    ):
        result = sampler.sample(shots)

    assert result.shape == (shots, 1)
    assert channel_sample.call_count == 4  # 4 batches of equal size
    assert [call.args[0] for call in channel_sample.call_args_list] == [
        expected_batch_size
    ] * 4


@pytest.fixture()
def make_sampler():
    """Create a sampler with mocked internals for fast batch-logic tests."""

    def _make(max_batch_size: int = 30):
        c = Circuit("H 0\nM 0")
        sampler = c.compile_sampler(strategy="cat5", seed=0)
        return sampler, max_batch_size

    return _make


@pytest.mark.parametrize(
    ("shots", "max_batch", "batch_size", "compute_ref", "expected_bs", "expected_n"),
    [
        # Has leeway: 25*4=100 > 99 → stays 25
        (99, 30, None, True, 25, 4),
        # No leeway: 25*4=100 == 100 → bumped to 26
        (100, 30, None, True, 26, 4),
        # Many shots, still uniform
        (200, 30, None, True, 29, 7),
        # No reference: batch_size unaffected
        (100, 30, None, False, 25, 4),
        # Explicit batch_size, no leeway → bump
        (100, None, 50, True, 51, 2),
        # Explicit batch_size, has leeway → stays
        (100, None, 51, True, 51, 2),
        # Small explicit batch_size where bump shrinks ceil(shots/batch_size):
        # num_batches must reflect the pre-bump count or samples are short.
        (2, None, 1, True, 2, 2),
        (12, None, 3, True, 4, 4),
    ],
    ids=[
        "leeway-no-bump",
        "no-leeway-bump",
        "many-shots-uniform",
        "no-reference",
        "explicit-no-leeway",
        "explicit-leeway",
        "explicit-small-no-leeway-bump",
        "explicit-small-no-leeway-bump-multi",
    ],
)
def test_batch_size_with_reference(
    make_sampler, shots, max_batch, batch_size, compute_ref, expected_bs, expected_n
):
    """Verify batch sizes are uniform (no JIT retrace) and sized correctly."""
    sampler, mb = make_sampler(max_batch or 9999)

    with (
        patch.object(type(sampler), "_estimate_batch_size", return_value=mb),
        patch.object(
            sampler._channel_sampler,
            "sample",
            wraps=sampler._channel_sampler.sample,
        ) as spy,
    ):
        referenced = np.ones(1, dtype=np.bool_) if compute_ref else None
        result = sampler._sample_batches(
            shots, batch_size=batch_size, referenced=referenced
        )

    assert result.shape == (shots, 1)

    batch_sizes = [call.args[0] for call in spy.call_args_list]
    assert all(bs == expected_bs for bs in batch_sizes)
    assert len(batch_sizes) == expected_n


def test_reference_sample_basic():
    """Reference sample XORs noiseless outcome with detector results."""
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler()

    # With reference sample: XOR with noiseless outcome
    # For a noiseless circuit, ref == raw, so XOR gives all zeros
    d_ref = sampler.sample(
        1,
        append_observables=True,
        use_detector_reference_sample=True,
        use_observable_reference_sample=True,
    )
    assert np.array_equal(d_ref, np.zeros_like(d_ref))


def test_reference_sample_selective_xor():
    """Test that use flags independently control detector vs observable XOR."""
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)

    # Use detector reference only: detectors XORed, observable raw
    sampler = c.compile_detector_sampler()
    d, o = sampler.sample(
        1,
        separate_observables=True,
        use_detector_reference_sample=True,
        use_observable_reference_sample=False,
    )
    assert np.array_equal(d, np.zeros_like(d))
    assert np.array_equal(o, np.array([[1]]))

    # Use observable reference only: detectors raw, observable XORed
    sampler2 = c.compile_detector_sampler()
    d2, o2 = sampler2.sample(
        1,
        separate_observables=True,
        use_detector_reference_sample=False,
        use_observable_reference_sample=True,
    )
    assert np.array_equal(d2, np.array([[0, 0]]))
    assert np.array_equal(o2, np.zeros_like(o2))


def test_reference_sample_with_bit_packed():
    """Reference sample works correctly with bit-packed output."""
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler()
    d = sampler.sample(
        1,
        append_observables=True,
        bit_packed=True,
        use_detector_reference_sample=True,
        use_observable_reference_sample=True,
    )
    # All zeros after XOR with reference, bit-packed
    expected = np.packbits(np.zeros(3, dtype=np.bool_), bitorder="little").reshape(
        1, -1
    )
    assert np.array_equal(d, expected)


def test_reference_sample_defaults_unchanged():
    """Default use=False preserves existing behavior (no XOR applied)."""
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler()
    d1 = sampler.sample(1, append_observables=True)

    sampler2 = c.compile_detector_sampler()
    d2 = sampler2.sample(
        1,
        append_observables=True,
        use_detector_reference_sample=False,
        use_observable_reference_sample=False,
    )
    assert np.array_equal(d1, d2)


def test_detector_sampler_empty_annotations():
    """Empty DETECTOR / OBSERVABLE_INCLUDE produce deterministic-zero columns
    that sit alongside non-empty ones, matching Stim semantics."""
    c = Circuit("""
        X 0
        M 0 1
        DETECTOR rec[-2]
        DETECTOR
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1)
        """)
    assert c.num_detectors == 2
    assert c.num_observables == 2

    sampler = c.compile_detector_sampler(seed=0)
    det, obs = sampler.sample(4, separate_observables=True)
    assert det.shape == (4, 2)
    assert obs.shape == (4, 2)
    # First detector reads the X-induced 1; second detector and both observables
    # are deterministically zero.
    assert np.all(det[:, 0])
    assert not np.any(det[:, 1])
    assert not np.any(obs)


def test_detector_sampler_sparse_observable_index():
    """OBSERVABLE_INCLUDE(2) only must produce 3 observable columns, with the
    measured value landing in column 2 and columns 0/1 deterministically zero."""
    c = Circuit("X 0\nM 0\nOBSERVABLE_INCLUDE(2) rec[-1]")
    assert c.num_observables == 3

    sampler = c.compile_detector_sampler(seed=0)
    samples = sampler.sample(4, append_observables=True)
    assert samples.shape == (4, 3)
    assert not np.any(samples[:, 0])
    assert not np.any(samples[:, 1])
    assert np.all(samples[:, 2])


def test_detector_sampler_out_of_order_observable_indices():
    """Out-of-order OBSERVABLE_INCLUDE must produce columns in sorted index order."""
    # rec[-2] is qubit-0 (=1 after X), rec[-1] is qubit-1 (=0).
    # Map: obs[2] = rec[-2] = 1, obs[0] = rec[-1] = 0, obs[1] is unmentioned = 0.
    c = Circuit("""
        X 0
        M 0 1
        OBSERVABLE_INCLUDE(2) rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    assert c.num_observables == 3

    sampler = c.compile_detector_sampler(seed=0)
    _, obs = sampler.sample(2, separate_observables=True)
    assert obs.shape == (2, 3)
    # Sorted-by-index order: [obs0, obs1, obs2] = [0, 0, 1]
    assert not np.any(obs[:, 0])
    assert not np.any(obs[:, 1])
    assert np.all(obs[:, 2])


def test_a_nan_normalization_deviation_warns(monkeypatch):
    sampler = Circuit("R 0\nH 0\nT 0\nH 0\nM 0").compile_sampler(
        strategy="active_statevector", seed=0
    )
    original = shared.run_lanes

    def nan_deviation(*args):
        return (float("nan"), *original(*args)[1:])

    monkeypatch.setattr(shared, "run_lanes", nan_deviation)
    with pytest.warns(UserWarning, match="not normalized"):
        sampler.sample(10)


def test_direct_detectors_overwrite_every_column_of_their_rows():
    f = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    out = np.ones((3, 4), dtype=np.bool_)
    discarded = sampler_module._write_direct_detectors(
        0,
        f,
        np.array([1, 0]),
        np.array([False, True]),
        np.array([3, 0]),
        np.array([True, False]),
        out,
    )
    expected = np.zeros((3, 4), dtype=np.bool_)
    expected[:, 3] = f[:, 1] != 0
    expected[:, 0] = (f[:, 0] != 0) ^ True
    assert np.array_equal(out, expected)
    assert np.array_equal(discarded, f[:, 1] != 0)


@pytest.mark.parametrize("work", [0, PARALLEL_WORK])
def test_direct_detectors_match_a_reference(work):
    rng = np.random.default_rng(3)
    rows, num_f, num_outputs, num_direct = 500, 30, 50, 40
    f = (rng.random((rows, num_f)) < 0.05).astype(np.uint8)
    f_indices = rng.integers(-1, num_f, num_direct)
    flips = rng.random(num_direct) < 0.1
    f_indices[:2], flips[:2] = -1, [True, False]
    columns = rng.permutation(num_outputs)[:num_direct]
    # NOTE: postselected detectors are unflipped, so that only some rows go.
    postselect = (rng.random(num_direct) < 0.3) & ~flips
    out = np.ones((rows, num_outputs), dtype=np.bool_)
    discarded = sampler_module._write_direct_detectors(
        work, f, f_indices, flips, columns, postselect, out
    )
    bits = np.where(f_indices >= 0, f[:, f_indices] != 0, False) ^ flips
    expected = np.zeros((rows, num_outputs), dtype=np.bool_)
    expected[:, columns] = bits
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(discarded, (bits & postselect).any(axis=1))
    assert 0 < discarded.sum() < rows


_JAX_BACKEND_CHECK = """
import jax._src.xla_bridge as xb
import numpy as np
import tsim

assert not xb.backends_are_initialized(), "import tsim"
c = tsim.Circuit(
    "R 0 1\\nH 0\\nT 0\\nCNOT 0 1\\nX_ERROR(0.1) 0\\nH 0\\nM 0 1\\n"
    "DETECTOR rec[-1]\\nOBSERVABLE_INCLUDE(0) rec[-2]"
)
c.compile_sampler(seed=1).sample(10, bit_packed=True)
sampler = c.compile_detector_sampler(seed=1)
sampler.sample(10, postselection_mask=np.array([True]))
sampler.sample(10, use_detector_reference_sample=True, separate_observables=True)
assert not xb.backends_are_initialized(), "active_statevector"
c.compile_detector_sampler(seed=1, strategy="cat5").sample(10)
assert xb.backends_are_initialized(), "cat5"
"""


def test_only_the_stabilizer_rank_strategies_start_jax():
    # NOTE: JAX's backend reserves most of a GPU's memory when it starts, so
    # importing Tsim and sampling with "active_statevector" must not start it. A
    # fresh interpreter, since other tests start it.
    subprocess.run([sys.executable, "-c", _JAX_BACKEND_CHECK], check=True)  # noqa: S603


# NOTE: a T state measured after noise: one component with shot words.
NOISY_T_STATE = "R 0\nH 0\nT 0\nZ_ERROR(0.1) 0\nH 0\nM 0\nDETECTOR rec[-1]"


@pytest.mark.parametrize("compile", ["compile_sampler", "compile_detector_sampler"])
def test_device_cpu_samples_on_the_cpu(compile):
    sampler = getattr(Circuit(NOISY_T_STATE), compile)(seed=1, device="cpu")
    assert type(shot_sampler(sampler)) is ShotSampler
    assert sampler.stats.device == "cpu"


def test_unknown_devices_raise():
    with pytest.raises(
        ValueError, match="device must be 'auto', 'cpu', 'metal' or 'cuda'"
    ):
        Circuit(NOISY_T_STATE).compile_sampler(device="gpu")  # type: ignore[arg-type]


@pytest.mark.parametrize("device", ["cpu", "metal", "cuda"])
def test_stabilizer_rank_strategies_take_only_the_auto_device(device):
    with pytest.raises(ValueError, match="needs strategy='active_statevector'"):
        Circuit(NOISY_T_STATE).compile_sampler(strategy="cat5", device=device)


@pytest.mark.parametrize(
    ("device", "name", "reason"),
    [("metal", "Metal", "no Apple GPU"), ("cuda", "CUDA", "no CUDA device")],
)
def test_gpu_devices_raise_where_the_gpu_cannot_run(monkeypatch, device, name, reason):
    from tsim.statevector import cuda, metal

    runtime = {"metal": metal.runtime, "cuda": cuda.runtime}[device]
    monkeypatch.setattr(runtime, "unavailable", lambda: reason)
    with pytest.raises(RuntimeError, match=f"Cannot sample on {name}: {reason}"):
        Circuit(NOISY_T_STATE).compile_detector_sampler(device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = Circuit(NOISY_T_STATE).compile_detector_sampler(seed=1)
    assert sampler.stats.device == "cpu"


def _wide_circuit() -> Circuit:
    """Return a circuit of one component of 13 active qubits: random U3 and CX layers."""
    rng = np.random.default_rng(0)
    lines = []
    for _ in range(12):
        perm = rng.permutation(13)
        for a, b in zip(perm[::2], perm[1::2], strict=False):
            lines += [
                f"U3({t[0]},{t[1]},{t[2]}) {q}"
                for q in (a, b)
                for t in [rng.uniform(-1, 1, 3)]
            ]
            lines.append(f"CX {a} {b}")
    lines.append("M " + " ".join(map(str, range(13))))
    return Circuit("\n".join(lines))


@pytest.mark.parametrize(("device", "name"), [("metal", "Metal"), ("cuda", "CUDA")])
@pytest.mark.parametrize(
    ("text", "message"),
    [
        (None, "a component needs 13 active qubits, more than the 11"),
        (
            "R 0\nH 0\nR_Z(1e-5) 0\nH 0\nM 0\nDETECTOR rec[-1]",
            r"a rotation's angle is 1\.0e-05 π from a Clifford angle.*6\.4e-05 π",
        ),
    ],
)
def test_gpu_devices_raise_for_circuits_the_gpu_cannot_run(
    monkeypatch, device, name, text, message
):
    # NOTE: the check runs before any GPU work, so a GPU that seems usable will do.
    from tsim.statevector import cuda, metal

    runtime = {"metal": metal.runtime, "cuda": cuda.runtime}[device]
    monkeypatch.setattr(runtime, "unavailable", lambda: None)
    circuit = _wide_circuit() if text is None else Circuit(text)
    with pytest.raises(ValueError, match=f"Cannot sample on {name}: {message}"):
        circuit.compile_sampler(seed=1, device=device)
    sampler = circuit.compile_sampler(seed=1)
    assert type(shot_sampler(sampler)) is ShotSampler


@pytest.mark.parametrize("cuda_runs", [True, False])
def test_the_auto_device_tries_cuda_then_metal(monkeypatch, cuda_runs):
    import copy

    from tsim.statevector import cuda, metal

    tried = []

    def backend(name, runs):
        def shot_sampler(host, *args, required, **kwargs):
            tried.append((name, required))
            # NOTE: a copy stands in for a GPU sampler: it is not the CPU's.
            return copy.copy(host) if runs else host

        return shot_sampler

    monkeypatch.setattr(cuda, "shot_sampler", backend("cuda", cuda_runs))
    monkeypatch.setattr(metal, "shot_sampler", backend("metal", True))
    sampler = Circuit(NOISY_T_STATE).compile_detector_sampler(seed=1)
    if cuda_runs:
        assert tried == [("cuda", False)]
    else:
        assert tried == [("cuda", False), ("metal", False)]
    assert sampler.sample(10).shape == (10, 1)
