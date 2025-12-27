"""Tests for channel_compiler module.

Each test samples from channels before and after simplification,
verifying that the sampling statistics match within tolerance.
"""

import jax.numpy as jnp
import numpy as np

from tsim.channel_compiler import (
    Channel,
    Sampler,
    absorb_subset_channels,
    depolarize_1,
    depolarize_2,
    error,
    expand_channel,
    merge_identical_channels,
    reduce_null_bits,
    simplify_channels,
    xor_convolve,
)


def assert_sampling_matches(
    matrix: jnp.ndarray,
    channels_before: list[Channel],
    channels_after: list[Channel],
    n_samples: int = 500_000,
    seed: int = 42,
    rtol: float = 0.05,
):
    """Assert that sampling statistics match before and after simplification.

    Compares the mean of each output bit (f-variable) between the two channel sets.
    """
    sampler1 = Sampler(matrix=matrix, seed=seed)
    bits1 = sampler1.sample(channels_before, n_samples)
    freq1 = np.mean(np.asarray(bits1), axis=0)

    sampler2 = Sampler(matrix=matrix, seed=seed + 1)  # Different seed for independence
    bits2 = sampler2.sample(channels_after, n_samples)
    freq2 = np.mean(np.asarray(bits2), axis=0)

    # Check that frequencies match within tolerance
    np.testing.assert_allclose(
        freq1,
        freq2,
        rtol=rtol,
        err_msg=f"Sampling frequencies don't match: {freq1} vs {freq2}",
    )


class TestXorConvolve:
    """Tests for xor_convolve function."""

    def test_two_bernoulli(self):
        """Test XOR convolution of two Bernoulli distributions."""
        p, q = 0.1, 0.2
        probs_a = jnp.array([1 - p, p], dtype=jnp.float32)
        probs_b = jnp.array([1 - q, q], dtype=jnp.float32)

        result = xor_convolve(probs_a, probs_b)

        # Expected: P(XOR=1) = p(1-q) + q(1-p) = 0.1*0.8 + 0.2*0.9 = 0.26
        expected_p1 = p * (1 - q) + q * (1 - p)
        assert result.shape == (2,)
        np.testing.assert_allclose(result[1], expected_p1, rtol=1e-5)
        np.testing.assert_allclose(result[0], 1 - expected_p1, rtol=1e-5)

    def test_two_2bit_channels(self):
        """Test XOR convolution of two 2-bit distributions."""
        # Uniform distributions
        probs_a = jnp.array([0.25, 0.25, 0.25, 0.25], dtype=jnp.float32)
        probs_b = jnp.array([0.25, 0.25, 0.25, 0.25], dtype=jnp.float32)

        result = xor_convolve(probs_a, probs_b)

        # XOR of two uniform distributions is still uniform
        np.testing.assert_allclose(result, jnp.ones(4) / 4, rtol=1e-5)

    def test_identity_convolve(self):
        """Convolving with delta at 0 should return the same distribution."""
        probs = jnp.array([0.7, 0.1, 0.1, 0.1], dtype=jnp.float32)
        delta = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

        result = xor_convolve(probs, delta)

        np.testing.assert_allclose(result, probs, rtol=1e-5)


class TestMergeIdenticalChannels:
    """Tests for merge_identical_channels (Phase 1)."""

    def test_merge_two_1bit_same_signature(self):
        """Two 1-bit channels with same signature should merge."""
        c1 = Channel(probs=error(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error(0.2), unique_col_ids=(0,))
        channels = [c1, c2]

        result = merge_identical_channels(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0,)
        # p_combined = 0.1*0.8 + 0.2*0.9 = 0.26
        np.testing.assert_allclose(result[0].probs[1], 0.26, rtol=1e-5)

    def test_merge_two_2bit_same_signature(self):
        """Two 2-bit channels with same signature should merge."""
        c1 = Channel(probs=depolarize_1(0.1), unique_col_ids=(0, 1))
        c2 = Channel(probs=depolarize_1(0.2), unique_col_ids=(0, 1))
        channels = [c1, c2]

        result = merge_identical_channels(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 1)

    def test_no_merge_different_signatures(self):
        """Channels with different signatures should not merge."""
        c1 = Channel(probs=error(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error(0.2), unique_col_ids=(1,))
        channels = [c1, c2]

        result = merge_identical_channels(channels)

        assert len(result) == 2

    def test_sampling_matches_after_merge(self):
        """Sampling statistics should match before and after merging."""
        mat = jnp.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=jnp.uint8,
        )

        c1 = Channel(probs=error(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error(0.15), unique_col_ids=(0,))
        c3 = Channel(probs=error(0.2), unique_col_ids=(1,))
        channels = [c1, c2, c3]

        merged = merge_identical_channels(channels)

        assert len(merged) == 2  # Two unique signature sets
        assert_sampling_matches(mat, channels, merged)


class TestExpandChannel:
    """Tests for expand_channel function."""

    def test_expand_1bit_to_2bit(self):
        """Expand a 1-bit channel to a 2-bit signature set."""
        c = Channel(probs=error(0.3), unique_col_ids=(0,))

        expanded = expand_channel(c, (0, 1))

        assert expanded.unique_col_ids == (0, 1)
        assert expanded.num_bits == 2
        # Bit 1 is always 0, so only outcomes 0b00 and 0b01 have probability
        np.testing.assert_allclose(expanded.probs[0], 0.7, rtol=1e-5)  # 0b00
        np.testing.assert_allclose(expanded.probs[1], 0.3, rtol=1e-5)  # 0b01
        np.testing.assert_allclose(expanded.probs[2], 0.0, rtol=1e-5)  # 0b10
        np.testing.assert_allclose(expanded.probs[3], 0.0, rtol=1e-5)  # 0b11

    def test_expand_1bit_to_2bit_different_position(self):
        """Expand 1-bit channel to 2-bit where source is in second position."""
        c = Channel(probs=error(0.3), unique_col_ids=(5,))

        expanded = expand_channel(c, (3, 5))

        assert expanded.unique_col_ids == (3, 5)
        # Signature 5 is at position 1 in target, so bit 1 has the probability
        # Bit 0 (signature 3) is always 0
        np.testing.assert_allclose(expanded.probs[0], 0.7, rtol=1e-5)  # 0b00
        np.testing.assert_allclose(expanded.probs[1], 0.0, rtol=1e-5)  # 0b01
        np.testing.assert_allclose(expanded.probs[2], 0.3, rtol=1e-5)  # 0b10
        np.testing.assert_allclose(expanded.probs[3], 0.0, rtol=1e-5)  # 0b11

    def test_expand_same_set_no_change(self):
        """Expanding to same signature set should return equivalent channel."""
        c = Channel(probs=error(0.3), unique_col_ids=(0,))

        expanded = expand_channel(c, (0,))

        np.testing.assert_allclose(expanded.probs, c.probs, rtol=1e-5)


class TestAbsorbSubsetChannels:
    """Tests for absorb_subset_channels (Phase 2)."""

    def test_absorb_1bit_into_2bit(self):
        """A 1-bit channel should be absorbed into a 2-bit superset."""
        c1 = Channel(probs=error(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=depolarize_1(0.2), unique_col_ids=(0, 1))
        channels = [c1, c2]

        result = absorb_subset_channels(channels)

        assert len(result) == 1
        assert set(result[0].unique_col_ids) == {0, 1}

    def test_no_absorb_disjoint(self):
        """Disjoint channels should not be absorbed."""
        c1 = Channel(probs=error(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error(0.2), unique_col_ids=(1,))
        channels = [c1, c2]

        result = absorb_subset_channels(channels)

        assert len(result) == 2

    def test_no_absorb_partial_overlap(self):
        """Partially overlapping (not subset) channels should not be absorbed."""
        c1 = Channel(probs=depolarize_1(0.1), unique_col_ids=(0, 1))
        c2 = Channel(probs=depolarize_1(0.2), unique_col_ids=(1, 2))
        channels = [c1, c2]

        result = absorb_subset_channels(channels)

        assert len(result) == 2

    def test_sampling_matches_after_absorb(self):
        """Sampling statistics should match before and after absorption."""
        mat = jnp.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=jnp.uint8,
        )

        # c1 has signature (0,), c2 has signatures (0, 1)
        # c1 should be absorbed into c2
        c1 = Channel(probs=error(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=depolarize_1(0.15), unique_col_ids=(0, 1))
        channels = [c1, c2]

        absorbed = absorb_subset_channels(channels)

        assert len(absorbed) == 1
        assert_sampling_matches(mat, channels, absorbed)


class TestSimplifyChannels:
    """Tests for the full simplify_channels function."""

    def test_simplify_mixed_channels(self):
        """Test simplification with a mix of channel types."""
        mat = jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 1, 1, 0],
            ],
            dtype=jnp.uint8,
        )

        # Create channels:
        # - Two 1-bit with same signature (should merge)
        # - One 1-bit that's subset of a 2-bit (should absorb)
        # - One 2-bit
        c1 = Channel(probs=error(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error(0.15), unique_col_ids=(0,))  # Same as c1
        c3 = Channel(probs=error(0.2), unique_col_ids=(1,))  # Subset of c4
        c4 = Channel(probs=depolarize_1(0.1), unique_col_ids=(1, 2))

        channels = [c1, c2, c3, c4]

        simplified = simplify_channels(channels)

        # c1 and c2 merge into one, c3 absorbed into c4
        assert len(simplified) == 2
        assert_sampling_matches(mat, channels, simplified)

    def test_simplify_many_1bit_channels(self):
        """Test simplification of many 1-bit channels with same signature."""
        mat = jnp.array([[1], [1]], dtype=jnp.uint8)

        # 10 channels all with the same signature
        channels = [Channel(probs=error(0.05), unique_col_ids=(0,)) for _ in range(10)]

        simplified = simplify_channels(channels)

        assert len(simplified) == 1
        assert_sampling_matches(mat, channels, simplified, rtol=0.1)

    def test_simplify_preserves_independent_channels(self):
        """Channels with disjoint signatures should remain separate."""
        mat = jnp.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=jnp.uint8,
        )

        c1 = Channel(probs=error(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error(0.2), unique_col_ids=(1,))
        c3 = Channel(probs=error(0.3), unique_col_ids=(2,))
        channels = [c1, c2, c3]

        simplified = simplify_channels(channels)

        assert len(simplified) == 3
        assert_sampling_matches(mat, channels, simplified)


class TestSamplerCorrectness:
    """Additional tests to verify Sampler correctness."""

    def test_sampler_single_channel(self):
        """Test that sampler produces correct frequencies for a single channel."""
        mat = jnp.array([[1]], dtype=jnp.uint8)
        c = Channel(probs=error(0.3), unique_col_ids=(0,))

        sampler = Sampler(matrix=mat, seed=42)
        bits = sampler.sample([c], 100_000)
        freq = np.mean(np.asarray(bits[:, 0]))

        np.testing.assert_allclose(freq, 0.3, rtol=0.05)

    def test_sampler_xor_two_channels(self):
        """Test that sampler correctly XORs two independent channels."""
        # Matrix shape: (num_signatures, num_f_vars)
        # Both signatures (0 and 1) affect f0
        mat = jnp.array([[1], [1]], dtype=jnp.uint8)

        c1 = Channel(probs=error(0.2), unique_col_ids=(0,))
        c2 = Channel(probs=error(0.3), unique_col_ids=(1,))

        sampler = Sampler(matrix=mat, seed=42)
        bits = sampler.sample([c1, c2], 100_000)
        freq = np.mean(np.asarray(bits[:, 0]))

        # P(f0=1) = P(e0 XOR e1 = 1) = 0.2*0.7 + 0.3*0.8 = 0.14 + 0.24 = 0.38
        expected = 0.2 * 0.7 + 0.3 * 0.8
        np.testing.assert_allclose(freq, expected, rtol=0.05)


class TestReduceNullBits:
    """Tests for reduce_null_bits function."""

    # =========================================================================
    # 1-bit channels
    # =========================================================================

    def test_1bit_all_none_removed(self):
        """A 1-bit channel with only None should be removed entirely."""
        c = Channel(probs=error(0.3), unique_col_ids=(None,))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 0

    def test_1bit_no_none_unchanged(self):
        """A 1-bit channel with no None entries should be unchanged."""
        c = Channel(probs=error(0.3), unique_col_ids=(0,))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0,)
        np.testing.assert_allclose(result[0].probs, c.probs, rtol=1e-5)

    # =========================================================================
    # 2-bit channels
    # =========================================================================

    def test_2bit_one_none_marginalize(self):
        """A 2-bit channel with one None should reduce to 1-bit."""
        # probs: [P(00), P(01), P(10), P(11)] = [0.4, 0.3, 0.2, 0.1]
        probs = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
        c = Channel(probs=probs, unique_col_ids=(0, None))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0,)
        assert result[0].num_bits == 1
        # Marginalize over bit 1 (the None bit):
        # P(bit0=0) = P(00) + P(10) = 0.4 + 0.2 = 0.6
        # P(bit0=1) = P(01) + P(11) = 0.3 + 0.1 = 0.4
        np.testing.assert_allclose(result[0].probs[0], 0.6, rtol=1e-5)
        np.testing.assert_allclose(result[0].probs[1], 0.4, rtol=1e-5)

    def test_2bit_first_none_marginalize(self):
        """A 2-bit channel with None in first position should marginalize correctly."""
        # probs: [P(00), P(01), P(10), P(11)] = [0.4, 0.3, 0.2, 0.1]
        probs = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
        c = Channel(probs=probs, unique_col_ids=(None, 5))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (5,)
        assert result[0].num_bits == 1
        # Marginalize over bit 0 (the None bit):
        # P(bit1=0) = P(00) + P(01) = 0.4 + 0.3 = 0.7
        # P(bit1=1) = P(10) + P(11) = 0.2 + 0.1 = 0.3
        np.testing.assert_allclose(result[0].probs[0], 0.7, rtol=1e-5)
        np.testing.assert_allclose(result[0].probs[1], 0.3, rtol=1e-5)

    def test_2bit_all_none_removed(self):
        """A 2-bit channel with all None entries should be removed."""
        probs = jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32)
        c = Channel(probs=probs, unique_col_ids=(None, None))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 0

    # =========================================================================
    # 3-bit channels
    # =========================================================================

    def test_3bit_one_none_marginalize(self):
        """A 3-bit channel with one None should reduce to 2-bit."""
        # 8 outcomes: 000, 001, 010, 011, 100, 101, 110, 111
        probs = jnp.array([0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.1, 0.1], dtype=jnp.float32)
        c = Channel(probs=probs, unique_col_ids=(0, None, 2))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 2)
        assert result[0].num_bits == 2
        # Marginalize over bit 1 (the None bit, middle position):
        # new 00 (bit0=0, bit2=0): P(000) + P(010) = 0.2 + 0.15 = 0.35
        # new 01 (bit0=1, bit2=0): P(001) + P(011) = 0.1 + 0.05 = 0.15
        # new 10 (bit0=0, bit2=1): P(100) + P(110) = 0.2 + 0.1 = 0.3
        # new 11 (bit0=1, bit2=1): P(101) + P(111) = 0.1 + 0.1 = 0.2
        np.testing.assert_allclose(result[0].probs[0], 0.35, rtol=1e-5)
        np.testing.assert_allclose(result[0].probs[1], 0.15, rtol=1e-5)
        np.testing.assert_allclose(result[0].probs[2], 0.3, rtol=1e-5)
        np.testing.assert_allclose(result[0].probs[3], 0.2, rtol=1e-5)

    def test_3bit_two_none_marginalize(self):
        """A 3-bit channel with two None entries should reduce to 1-bit."""
        probs = jnp.array([0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.1, 0.1], dtype=jnp.float32)
        c = Channel(probs=probs, unique_col_ids=(None, 1, None))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (1,)
        assert result[0].num_bits == 1
        # Only bit 1 survives. Marginalize over bits 0 and 2:
        # P(bit1=0) = P(000) + P(001) + P(100) + P(101) = 0.2 + 0.1 + 0.2 + 0.1 = 0.6
        # P(bit1=1) = P(010) + P(011) + P(110) + P(111) = 0.15 + 0.05 + 0.1 + 0.1 = 0.4
        np.testing.assert_allclose(result[0].probs[0], 0.6, rtol=1e-5)
        np.testing.assert_allclose(result[0].probs[1], 0.4, rtol=1e-5)

    def test_3bit_all_none_removed(self):
        """A 3-bit channel with all None entries should be removed."""
        probs = jnp.array([0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.1, 0.1], dtype=jnp.float32)
        c = Channel(probs=probs, unique_col_ids=(None, None, None))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 0

    # =========================================================================
    # 4-bit channels
    # =========================================================================

    def test_4bit_one_none_marginalize(self):
        """A 4-bit channel with one None should reduce to 3-bit."""
        # 16 outcomes, uniform for simplicity
        probs = jnp.ones(16, dtype=jnp.float32) / 16
        c = Channel(probs=probs, unique_col_ids=(0, 1, None, 3))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 1, 3)
        assert result[0].num_bits == 3
        # Uniform marginalizes to uniform
        np.testing.assert_allclose(result[0].probs, jnp.ones(8) / 8, rtol=1e-5)

    def test_4bit_two_none_marginalize(self):
        """A 4-bit channel with two None entries should reduce to 2-bit."""
        probs = jnp.ones(16, dtype=jnp.float32) / 16
        c = Channel(probs=probs, unique_col_ids=(None, 1, None, 3))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (1, 3)
        assert result[0].num_bits == 2
        np.testing.assert_allclose(result[0].probs, jnp.ones(4) / 4, rtol=1e-5)

    def test_4bit_three_none_marginalize(self):
        """A 4-bit channel with three None entries should reduce to 1-bit."""
        probs = jnp.ones(16, dtype=jnp.float32) / 16
        c = Channel(probs=probs, unique_col_ids=(None, None, 2, None))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (2,)
        assert result[0].num_bits == 1
        np.testing.assert_allclose(result[0].probs, jnp.ones(2) / 2, rtol=1e-5)

    def test_4bit_all_none_removed(self):
        """A 4-bit channel with all None entries should be removed."""
        probs = jnp.ones(16, dtype=jnp.float32) / 16
        c = Channel(probs=probs, unique_col_ids=(None, None, None, None))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 0

    def test_4bit_depolarize_one_none(self):
        """Test marginalization of a depolarizing channel with one None."""
        # depolarize_2 returns 16 outcomes for 2-qubit (4 bits)
        probs = depolarize_2(0.15)
        c = Channel(probs=probs, unique_col_ids=(0, 1, 2, None))
        channels = [c]

        result = reduce_null_bits(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 1, 2)
        assert result[0].num_bits == 3
        # Probabilities should sum to 1
        np.testing.assert_allclose(jnp.sum(result[0].probs), 1.0, rtol=1e-5)

    # =========================================================================
    # Multiple channels
    # =========================================================================

    def test_multiple_channels_mixed(self):
        """Test with multiple channels, some with None, some without."""
        c1 = Channel(probs=error(0.3), unique_col_ids=(0,))  # No None, keep
        c2 = Channel(probs=error(0.2), unique_col_ids=(None,))  # All None, remove
        c3 = Channel(
            probs=jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32),
            unique_col_ids=(1, None),
        )  # One None, reduce
        channels = [c1, c2, c3]

        result = reduce_null_bits(channels)

        assert len(result) == 2  # c2 removed
        assert result[0].unique_col_ids == (0,)
        assert result[1].unique_col_ids == (1,)

    def test_probs_sum_to_one_after_marginalization(self):
        """Verify that probabilities sum to 1 after marginalization."""
        probs = jnp.array(
            [0.1, 0.2, 0.15, 0.25, 0.05, 0.1, 0.1, 0.05], dtype=jnp.float32
        )
        c = Channel(probs=probs, unique_col_ids=(0, None, 2))
        channels = [c]

        result = reduce_null_bits(channels)

        np.testing.assert_allclose(jnp.sum(result[0].probs), 1.0, rtol=1e-5)
