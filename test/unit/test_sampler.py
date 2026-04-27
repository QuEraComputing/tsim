from unittest.mock import patch

import numpy as np
import pytest

from tsim.circuit import Circuit


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

    d, o = sampler.sample(1, separate_observables=True)
    assert np.array_equal(d, np.array([[0, 0]]))
    assert np.array_equal(o, np.array([[1]]))


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
        sampler = c.compile_sampler(seed=0)
        assert np.count_nonzero(sampler.sample(100)) == 48
        assert np.count_nonzero(sampler.sample(100)) == 53
        assert np.count_nonzero(sampler.sample(100)) == 52
        assert np.count_nonzero(sampler.sample(100)) == 50


def test_sampler_repr():
    c = Circuit("""
        X_ERROR(0.1) 0 1
        M 0 1
        """)
    sampler = c.compile_sampler()
    repr_str = repr(sampler)
    assert "CompiledMeasurementSampler" in repr_str
    assert "2 error channel bits" in repr_str


@pytest.mark.parametrize(
    ("shots", "expected_batch_size"),
    [(100, 25), (101, 26)],
)
def test_auto_batch(shots, expected_batch_size):
    c = Circuit("""
        H 0
        M 0
        """)
    sampler = c.compile_sampler(seed=42)

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
        sampler = c.compile_sampler(seed=0)
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
        result = sampler._sample_batches(
            shots, batch_size=batch_size, compute_reference=compute_ref
        )

    if compute_ref:
        samples, ref = result
        assert samples.shape == (shots, 1)
        assert ref.shape == (1,)
    else:
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
