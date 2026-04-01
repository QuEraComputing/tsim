import warnings
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
        skip_detector_reference_sample=False,
        skip_observable_reference_sample=False,
    )
    assert np.array_equal(d_ref, np.zeros_like(d_ref))


def test_reference_sample_selective_xor():
    """Test that skip flags independently control detector vs observable XOR."""
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)

    # Skip observable reference only: detectors XORed, observable raw
    sampler = c.compile_detector_sampler()
    d, o = sampler.sample(
        1,
        separate_observables=True,
        skip_detector_reference_sample=False,
        skip_observable_reference_sample=True,
    )
    assert np.array_equal(d, np.zeros_like(d))
    assert np.array_equal(o, np.array([[1]]))

    # Skip detector reference only: detectors raw, observable XORed
    sampler2 = c.compile_detector_sampler()
    d2, o2 = sampler2.sample(
        1,
        separate_observables=True,
        skip_detector_reference_sample=True,
        skip_observable_reference_sample=False,
    )
    assert np.array_equal(d2, np.array([[0, 0]]))
    assert np.array_equal(o2, np.zeros_like(o2))


def test_reference_sample_cross_batch_warning():
    """Warn when reference samples differ across batches (non-deterministic circuit)."""
    c = Circuit("""
        H 0
        M 0
        DETECTOR rec[-1]
        """)
    sampler = c.compile_detector_sampler(seed=0)

    with (
        patch.object(type(sampler), "_estimate_batch_size", return_value=2),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        # 10 shots with batch_size=2 means 5 batches.
        # H|0> is non-deterministic, so reference samples will likely differ.
        sampler.sample(
            10,
            skip_detector_reference_sample=False,
        )

    # Check that a UserWarning about non-deterministic reference was raised
    ref_warnings = [x for x in w if "Reference samples differ" in str(x.message)]
    assert len(ref_warnings) >= 1


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
        skip_detector_reference_sample=False,
        skip_observable_reference_sample=False,
    )
    # All zeros after XOR with reference, bit-packed
    expected = np.packbits(np.zeros(3, dtype=np.bool_), bitorder="little").reshape(
        1, -1
    )
    assert np.array_equal(d, expected)


def test_reference_sample_defaults_unchanged():
    """Default skip=True preserves existing behavior (no XOR applied)."""
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
        skip_detector_reference_sample=True,
        skip_observable_reference_sample=True,
    )
    assert np.array_equal(d1, d2)
