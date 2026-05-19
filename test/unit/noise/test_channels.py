import numpy as np
import pytest
from numpy.testing import assert_allclose

from tsim.noise.channels import (
    Channel,
    ChannelSampler,
    absorb_subset_channels,
    correlated_error_probs,
    error_probs,
    expand_channel,
    fold_duplicate_channel_bits,
    heralded_pauli_channel_1_probs,
    merge_identical_channels,
    normalize_channels,
    pauli_channel_1_probs,
    pauli_channel_2_probs,
    reduce_null_bits,
    simplify_channels,
    xor_convolve,
)


class TestProbabilityConstructors:
    """Tests for probability distribution constructors."""

    def test_error(self):
        """Test single-bit error probability distribution."""
        probs = error_probs(0.1)
        assert probs.shape == (2,)
        assert probs.dtype == np.float64
        assert_allclose(probs[0], 0.9, rtol=1e-10)
        assert_allclose(probs[1], 0.1, rtol=1e-10)
        assert_allclose(np.sum(probs), 1.0, rtol=1e-10)

    def test_pauli_channel_1(self):
        """Test single-qubit Pauli channel probability distribution."""
        # Pure X error
        probs = pauli_channel_1_probs(px=1.0, py=0.0, pz=0.0)
        assert probs.shape == (4,)
        assert probs.dtype == np.float64
        # Order: [I, Z, X, Y] mapped to bits [00, 01, 10, 11]
        assert_allclose(probs[2], 1.0, rtol=1e-10)  # index 2 (0b10): X

        # Pure Y error
        probs = pauli_channel_1_probs(px=0.0, py=1.0, pz=0.0)
        assert_allclose(probs[3], 1.0, rtol=1e-10)  # index 3 (0b11): Y

        # Pure Z error
        probs = pauli_channel_1_probs(px=0.0, py=0.0, pz=1.0)
        assert_allclose(probs[1], 1.0, rtol=1e-10)  # index 1 (0b01): Z

    def test_pauli_channel_2(self):
        """Test two-qubit Pauli channel."""
        # Test IX (X on second qubit) - all other Paulis have 0 probability
        probs = pauli_channel_2_probs(
            pix=1.0,
            piy=0,
            piz=0,
            pxi=0,
            pxx=0,
            pxy=0,
            pxz=0,
            pyi=0,
            pyx=0,
            pyy=0,
            pyz=0,
            pzi=0,
            pzx=0,
            pzy=0,
            pzz=0,
        )
        assert probs.shape == (16,)
        assert probs.dtype == np.float64
        assert_allclose(probs[8], 1.0, rtol=1e-10)  # index 8 (0b1000): IX
        assert_allclose(np.sum(probs), 1.0, rtol=1e-10)

    def test_heralded_pauli_channel_1(self):
        """Test heralded Pauli channel probability distribution."""
        probs = heralded_pauli_channel_1_probs(pi=0.01, px=0.02, py=0.03, pz=0.04)
        assert probs.shape == (8,)
        assert probs.dtype == np.float64
        # Little-endian bits: bit0=herald, bit1=Z error, bit2=X error.
        assert_allclose(probs[0], 0.9)  # index 0 (0b000): no fire
        assert_allclose(probs[1], 0.01)  # index 1 (0b001): herald + I
        assert_allclose(probs[3], 0.04)  # index 3 (0b011): herald + Z
        assert_allclose(probs[5], 0.02)  # index 5 (0b101): herald + X
        assert_allclose(probs[7], 0.03)  # index 7 (0b111): herald + Y
        assert_allclose(probs[2], 0.0)  # index 2 (0b010): impossible
        assert_allclose(probs[4], 0.0)  # index 4 (0b100): impossible
        assert_allclose(probs[6], 0.0)  # index 6 (0b110): impossible
        assert_allclose(np.sum(probs), 1.0, rtol=1e-10)

    def test_heralded_pauli_channel_1_pure_z(self):
        """Heralded channel that always fires with Z."""
        probs = heralded_pauli_channel_1_probs(pi=0.0, px=0.0, py=0.0, pz=1.0)
        assert_allclose(probs[3], 1.0)  # index 3 (0b011): herald + Z
        assert_allclose(np.sum(probs), 1.0, rtol=1e-10)


class TestCorrelatedErrorProbs:
    """Tests for correlated_error_probs probability constructor."""

    def test_single_error(self):
        """Single error with probability p."""
        probs = correlated_error_probs([0.3])
        assert probs.shape == (2,)
        assert_allclose(probs[0], 0.7)  # index 0 (0b0): no error
        assert_allclose(probs[1], 0.3)  # index 1 (0b1): error occurred
        assert_allclose(np.sum(probs), 1.0)

    def test_two_errors(self):
        """Chain of two errors."""
        probs = correlated_error_probs([0.2, 0.25])
        assert probs.shape == (4,)
        # P(00) = (1-0.2)*(1-0.25) = 0.8*0.75 = 0.6
        # P(01) = 0.2 (first error)
        # P(10) = (1-0.2)*0.25 = 0.8*0.25 = 0.2 (second error)
        # P(11) = 0 (mutually exclusive)
        assert_allclose(probs[0], 0.6)
        assert_allclose(probs[1], 0.2)
        assert_allclose(probs[2], 0.2)
        assert_allclose(probs[3], 0.0)
        assert_allclose(np.sum(probs), 1.0)

    def test_three_errors_uniform(self):
        """The example from stim docs: 60% distributed uniformly among 3 errors."""
        # CORRELATED_ERROR(0.2) -> ELSE(0.25) -> ELSE(0.333...)
        # P1 = 0.2, P2 = 0.8*0.25 = 0.2, P3 = 0.8*0.75*0.333... = 0.2
        probs = correlated_error_probs([0.2, 0.25, 1 / 3])
        assert probs.shape == (8,)
        assert_allclose(probs[0], 0.4, rtol=1e-5)  # index 0 (0b000): no error
        assert_allclose(probs[1], 0.2, rtol=1e-5)  # index 1 (0b001): first
        assert_allclose(probs[2], 0.2, rtol=1e-5)  # index 2 (0b010): second
        assert_allclose(probs[4], 0.2, rtol=1e-5)  # index 4 (0b100): third
        # All multi-bit outcomes are 0
        assert_allclose(probs[3], 0.0)
        assert_allclose(probs[5], 0.0)
        assert_allclose(probs[6], 0.0)
        assert_allclose(probs[7], 0.0)
        assert_allclose(np.sum(probs), 1.0)

    def test_zero_probability(self):
        """Test with zero probability (always no error)."""
        probs = correlated_error_probs([0.0])
        assert_allclose(probs[0], 1.0)
        assert_allclose(probs[1], 0.0)

    def test_one_probability(self):
        """Test with probability 1 (always error)."""
        probs = correlated_error_probs([1.0])
        assert_allclose(probs[0], 0.0)
        assert_allclose(probs[1], 1.0)

    def test_chain_with_certain_first_error(self):
        """If first error is certain, subsequent errors have 0 probability."""
        probs = correlated_error_probs([1.0, 0.5, 0.5])
        assert probs.shape == (8,)
        assert_allclose(probs[0], 0.0)  # index 0 (0b000): no error
        assert_allclose(probs[1], 1.0)  # index 1 (0b001): first error
        assert_allclose(probs[2], 0.0)  # index 2 (0b010): second error
        assert_allclose(probs[4], 0.0)  # index 4 (0b100): third error


class TestChannelValidation:
    """Channel.__post_init__ rejects malformed probability distributions so
    they cannot reach the sampler and get silently dropped."""

    def test_rejects_negative_entry(self):
        with pytest.raises(ValueError):
            Channel(probs=np.array([1.2, -0.2]), unique_col_ids=(0,))

    def test_rejects_entry_above_one(self):
        with pytest.raises(ValueError):
            Channel(probs=np.array([-0.5, 1.5]), unique_col_ids=(0,))

    def test_rejects_sum_below_one(self):
        with pytest.raises(ValueError):
            Channel(probs=np.array([0.5, 0.4]), unique_col_ids=(0,))

    def test_rejects_sum_above_one(self):
        with pytest.raises(ValueError):
            Channel(probs=np.array([0.7, 0.7]), unique_col_ids=(0,))

    def test_rejects_helper_with_arguments_summing_above_one(self):
        # pauli_channel_1_probs(0.6, 0.6, 0.6) returns probs[0] = -0.8.
        # Wrapping in a Channel surfaces the negative entry as a clear error.
        with pytest.raises(ValueError):
            Channel(probs=pauli_channel_1_probs(0.6, 0.6, 0.6), unique_col_ids=(0, 1))

    def test_channel_sampler_rejects_invalid_channel(self):
        # ChannelSampler wraps probs in a Channel, so invalid distributions
        # get caught at construction time rather than silently dropped.
        bad_probs = np.array([1.2, -0.1, -0.05, -0.05])
        transform = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        with pytest.raises(ValueError):
            ChannelSampler([bad_probs], transform, seed=42)


def _sample_channels(channels, matrix, n_samples, seed=42):
    """Sample from channels using the ChannelSampler infrastructure."""
    sampler = object.__new__(ChannelSampler)
    sampler.channels = channels
    sampler.signature_matrix = np.asarray(matrix, dtype=np.uint8)
    sampler._rng = np.random.default_rng(seed)
    sampler._sparse_data = ChannelSampler._precompute_sparse(
        channels, sampler.signature_matrix
    )
    return sampler.sample(n_samples)


def assert_sampling_matches(
    matrix: np.ndarray,
    channels_before: list[Channel],
    channels_after: list[Channel],
    n_samples: int = 500_000,
    seed: int = 42,
    rtol: float = 0.05,
):
    """Assert that sampling statistics match before and after simplification.

    Compares the mean of each output bit (f-variable) between the two channel sets.
    """
    bits1 = _sample_channels(channels_before, matrix, n_samples, seed=seed)
    freq1 = np.mean(bits1, axis=0)

    bits2 = _sample_channels(channels_after, matrix, n_samples, seed=seed + 1)
    freq2 = np.mean(bits2, axis=0)

    assert_allclose(
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
        probs_a = np.array([1 - p, p], dtype=np.float64)
        probs_b = np.array([1 - q, q], dtype=np.float64)

        result = xor_convolve(probs_a, probs_b)

        # Expected: P(XOR=1) = p(1-q) + q(1-p) = 0.1*0.8 + 0.2*0.9 = 0.26
        expected_p1 = p * (1 - q) + q * (1 - p)
        assert result.shape == (2,)
        assert_allclose(result[1], expected_p1, rtol=1e-5)
        assert_allclose(result[0], 1 - expected_p1, rtol=1e-5)

    def test_two_2bit_channels(self):
        """Test XOR convolution of two 2-bit distributions."""
        # Uniform distributions
        probs_a = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        probs_b = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

        result = xor_convolve(probs_a, probs_b)

        # XOR of two uniform distributions is still uniform
        assert_allclose(result, np.ones(4) / 4, rtol=1e-5)

    def test_identity_convolve(self):
        """Convolving with delta at 0 should return the same distribution."""
        probs = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
        delta = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        result = xor_convolve(probs, delta)

        assert_allclose(result, probs, rtol=1e-5)


class TestMergeIdenticalChannels:
    """Tests for merge_identical_channels."""

    def test_merge_two_1bit_same_signature(self):
        """Two 1-bit channels with same signature should merge."""
        c1 = Channel(probs=error_probs(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error_probs(0.2), unique_col_ids=(0,))
        channels = [c1, c2]

        result = merge_identical_channels(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0,)
        # p_combined = 0.1*0.8 + 0.2*0.9 = 0.26
        assert_allclose(result[0].probs[1], 0.26, rtol=1e-5)

    def test_merge_two_2bit_same_signature(self):
        """Two 2-bit channels with same signature should merge."""
        c1 = Channel(
            probs=pauli_channel_1_probs(0.1 / 3, 0.1 / 3, 0.1 / 3),
            unique_col_ids=(0, 1),
        )
        c2 = Channel(
            probs=pauli_channel_1_probs(0.2 / 3, 0.2 / 3, 0.2 / 3),
            unique_col_ids=(0, 1),
        )
        channels = [c1, c2]

        result = merge_identical_channels(channels)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 1)

    def test_no_merge_different_signatures(self):
        """Channels with different signatures should not merge."""
        c1 = Channel(probs=error_probs(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error_probs(0.2), unique_col_ids=(1,))
        channels = [c1, c2]

        result = merge_identical_channels(channels)

        assert len(result) == 2

    def test_sampling_matches_after_merge(self):
        """Sampling statistics should match before and after merging."""
        mat = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=np.uint8,
        )

        c1 = Channel(probs=error_probs(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error_probs(0.15), unique_col_ids=(0,))
        c3 = Channel(probs=error_probs(0.2), unique_col_ids=(1,))
        channels = [c1, c2, c3]

        merged = merge_identical_channels(channels)

        assert len(merged) == 2  # Two unique signature sets
        assert_sampling_matches(mat, channels, merged)


class TestExpandChannel:
    """Tests for expand_channel function."""

    def test_expand_1bit_to_2bit(self):
        """Expand a 1-bit channel to a 2-bit signature set."""
        c = Channel(probs=error_probs(0.3), unique_col_ids=(0,))

        expanded = expand_channel(c, (0, 1))

        expected = np.zeros(4, dtype=np.float64)
        expected[0b00] = 0.7
        expected[0b01] = 0.3
        assert expanded.unique_col_ids == (0, 1)
        assert expanded.num_bits == 2
        assert_allclose(expanded.probs, expected)

    def test_expand_1bit_to_2bit_different_position(self):
        """Expand 1-bit channel to 2-bit where source is in second position."""
        c = Channel(probs=error_probs(0.3), unique_col_ids=(5,))

        expanded = expand_channel(c, (3, 5))

        expected = np.zeros(4, dtype=np.float64)
        expected[0b00] = 0.7
        expected[0b10] = 0.3
        assert expanded.unique_col_ids == (3, 5)
        assert expanded.num_bits == 2
        assert_allclose(expanded.probs, expected)

    @pytest.mark.parametrize("unique_col_id", [3, 4, 5])
    def test_expand_1bit_to_3bit(self, unique_col_id):
        """Expand a 1-bit channel to a 3-bit signature set."""
        target_col_ids = (3, 4, 5)
        c = Channel(probs=error_probs(0.3), unique_col_ids=(unique_col_id,))

        expanded = expand_channel(c, target_col_ids)

        expected = np.zeros(8, dtype=np.float64)
        expected[0b000] = 0.7
        expected[1 << target_col_ids.index(unique_col_id)] = 0.3
        assert expanded.unique_col_ids == target_col_ids
        assert expanded.num_bits == 3
        assert_allclose(expanded.probs, expected)

    @pytest.mark.parametrize("unique_col_id", [3, 4, 5, 6, 7])
    def test_expand_1bit_to_5bit(self, unique_col_id):
        """Expand a 1-bit channel to a 5-bit signature set."""
        target_col_ids = (3, 4, 5, 6, 7)
        c = Channel(probs=error_probs(0.3), unique_col_ids=(unique_col_id,))

        expanded = expand_channel(c, target_col_ids)

        expected = np.zeros(32, dtype=np.float64)
        expected[0b00000] = 0.7
        expected[1 << target_col_ids.index(unique_col_id)] = 0.3
        assert expanded.unique_col_ids == target_col_ids
        assert expanded.num_bits == 5
        assert_allclose(expanded.probs, expected)

    @pytest.mark.parametrize("source_col_ids", [(0, 1), (0, 3), (1, 3), (2, 3)])
    def test_expand_2bit_to_4bit_preserves_source_bit_positions(self, source_col_ids):
        """Expand a 2-bit channel into non-adjacent target signature positions."""
        probs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        target_col_ids = (0, 1, 2, 3)
        c = Channel(probs=probs, unique_col_ids=source_col_ids)

        expanded = expand_channel(c, target_col_ids)

        expected = np.zeros(16, dtype=np.float64)
        expected[0b0000] = 0.1
        expected[1 << target_col_ids.index(source_col_ids[0])] = 0.2
        expected[1 << target_col_ids.index(source_col_ids[1])] = 0.3
        expected[
            (1 << target_col_ids.index(source_col_ids[0]))
            | (1 << target_col_ids.index(source_col_ids[1]))
        ] = 0.4
        assert expanded.unique_col_ids == target_col_ids
        assert expanded.num_bits == 4
        assert_allclose(expanded.probs, expected)

    def test_expand_duplicate_source_col_ids_cancel_mod_2(self):
        """Duplicate source column bits should combine by XOR, not OR."""
        # Bits 0 and 1 both map to target column 0. Outcomes 00 and 11 map to
        # target index 0b00, while 01 and 10 map to target index 0b01.
        probs = np.array([0.1, 0.2, 0.4, 0.3], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(0, 0))

        expanded = expand_channel(c, (0, 1))

        assert expanded.unique_col_ids == (0, 1)
        assert_allclose(expanded.probs, [0.4, 0.6, 0.0, 0.0])

    def test_expand_deterministic_duplicate_bits_cancel_to_identity(self):
        """A deterministic 11 outcome on duplicate columns should become 00."""
        c = Channel(
            probs=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            unique_col_ids=(0, 0),
        )

        expanded = expand_channel(c, (0, 1))

        assert_allclose(expanded.probs, [1.0, 0.0, 0.0, 0.0])

    def test_expand_rejects_duplicate_target_col_ids(self):
        """Target signature sets must not contain duplicate column IDs."""
        c = Channel(probs=error_probs(0.3), unique_col_ids=(0,))

        with pytest.raises(ValueError, match="Target must not contain duplicates"):
            expand_channel(c, (0, 1, 1))

    def test_expand_rejects_unsorted_source_col_ids(self):
        """Source signature sets must be sorted."""
        c = Channel(
            probs=np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64),
            unique_col_ids=(1, 0),
        )

        with pytest.raises(ValueError, match="Source must be sorted"):
            expand_channel(c, (0, 1, 2))

    def test_expand_rejects_unsorted_target_col_ids(self):
        """Target signature sets must be sorted."""
        c = Channel(probs=error_probs(0.3), unique_col_ids=(0,))

        with pytest.raises(ValueError, match="Target must be sorted"):
            expand_channel(c, (1, 0))

    @pytest.mark.parametrize("target_col_ids", [(0,), (1, 2)])
    def test_expand_rejects_target_that_is_not_strict_superset(self, target_col_ids):
        """Target signature sets must strictly contain all source column IDs."""
        c = Channel(probs=error_probs(0.3), unique_col_ids=(0,))

        with pytest.raises(ValueError, match="Source must be strict subset"):
            expand_channel(c, target_col_ids)


class TestFoldDuplicateChannelBits:
    """Tests for fold_duplicate_channel_bits."""

    def test_fold_two_duplicate_bits_by_xor(self):
        """Duplicate bits with one column ID become one parity bit."""
        c = Channel(
            probs=np.array([0.1, 0.2, 0.4, 0.3], dtype=np.float64),
            unique_col_ids=(0, 0),
        )

        folded = fold_duplicate_channel_bits([c])

        assert len(folded) == 1
        assert folded[0].unique_col_ids == (0,)
        assert_allclose(folded[0].probs, [0.4, 0.6])

    def test_simplify_folds_duplicate_bits_before_absorption(self):
        """Subset absorption should only expand into duplicate-free targets."""
        channels = [
            Channel(
                probs=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
                unique_col_ids=(0, 0),
            ),
            Channel(probs=error_probs(0.3), unique_col_ids=(0,)),
        ]

        simplified = simplify_channels(channels)

        assert len(simplified) == 1
        assert simplified[0].unique_col_ids == (0,)

    def test_no_duplicates_pass_through(self):
        """A channel with no duplicate col_ids should be returned unchanged."""
        probs = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(0, 1))

        folded = fold_duplicate_channel_bits([c])

        assert len(folded) == 1
        assert folded[0].unique_col_ids == (0, 1)
        assert_allclose(folded[0].probs, probs)

    def test_empty_list_returns_empty(self):
        """An empty input list should return an empty list."""
        assert fold_duplicate_channel_bits([]) == []

    def test_fold_three_duplicate_bits_by_parity(self):
        """Three bits sharing one column ID should fold to one parity bit."""
        # col_ids (0, 0, 0): all three bits map to col 0. The fold collects the
        # mass of even-parity old indices into new[0] and odd-parity into new[1].
        probs = np.arange(1, 9, dtype=np.float64)
        probs /= probs.sum()
        c = Channel(probs=probs, unique_col_ids=(0, 0, 0))

        folded = fold_duplicate_channel_bits([c])

        assert len(folded) == 1
        assert folded[0].unique_col_ids == (0,)
        even_sum = probs[0] + probs[3] + probs[5] + probs[6]  # 000, 011, 101, 110
        odd_sum = probs[1] + probs[2] + probs[4] + probs[7]  # 001, 010, 100, 111
        assert_allclose(folded[0].probs, [even_sum, odd_sum])

    def test_fold_partial_duplicates(self):
        """Only-duplicate bits should fold; unique bits should be preserved."""
        # col_ids (0, 0, 1): bits 0 and 1 fold onto new pos 0,
        # bit 2 maps to new pos 1.
        probs = np.arange(1, 9, dtype=np.float64)
        probs /= probs.sum()
        c = Channel(probs=probs, unique_col_ids=(0, 0, 1))

        folded = fold_duplicate_channel_bits([c])

        assert len(folded) == 1
        assert folded[0].unique_col_ids == (0, 1)
        expected = np.zeros(4, dtype=np.float64)
        expected[0b00] = probs[0b000] + probs[0b011]
        expected[0b01] = probs[0b001] + probs[0b010]
        expected[0b10] = probs[0b100] + probs[0b111]
        expected[0b11] = probs[0b101] + probs[0b110]
        assert_allclose(folded[0].probs, expected)

    def test_fold_preserves_probability_mass(self):
        """Folded channel should still sum to 1."""
        probs = np.array([0.1, 0.2, 0.4, 0.3], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(0, 0))

        folded = fold_duplicate_channel_bits([c])

        assert_allclose(np.sum(folded[0].probs), 1.0)

    def test_fold_multiple_channels_mixed(self):
        """A list mixing folded and unchanged channels should process each in place."""
        c1 = Channel(
            probs=np.array([0.1, 0.2, 0.4, 0.3], dtype=np.float64),
            unique_col_ids=(0, 0),
        )
        c2 = Channel(probs=error_probs(0.25), unique_col_ids=(1,))
        c3 = Channel(
            probs=np.array([0.5, 0.1, 0.3, 0.1], dtype=np.float64),
            unique_col_ids=(2, 3),
        )

        folded = fold_duplicate_channel_bits([c1, c2, c3])

        assert len(folded) == 3
        assert folded[0].unique_col_ids == (0,)
        assert_allclose(folded[0].probs, [0.4, 0.6])
        assert folded[1].unique_col_ids == (1,)
        assert_allclose(folded[1].probs, c2.probs)
        assert folded[2].unique_col_ids == (2, 3)
        assert_allclose(folded[2].probs, c3.probs)

    def test_fold_preserves_sampling_statistics(self):
        """Sampling stats should match before and after folding duplicate bits."""
        mat = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        probs = np.array([0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(0, 0, 1))

        folded = fold_duplicate_channel_bits([c])

        assert_sampling_matches(mat, [c], folded)


class TestNormalizeChannels:
    """Tests for normalize_channels."""

    def test_already_sorted_unchanged(self):
        """A channel with sorted col_ids should be unchanged."""
        c = Channel(
            probs=np.array([0.2, 0.3, 0.4, 0.1], dtype=np.float64),
            unique_col_ids=(0, 1),
        )
        result = normalize_channels([c])

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 1)
        assert_allclose(result[0].probs, c.probs)

    def test_2bit_reorder(self):
        """A 2-bit channel with reversed col_ids should be reordered."""
        # Little-endian indices: bit0 contributes 1, bit1 contributes 2.
        # With col_ids (1, 0): bit 0 -> col 1, bit 1 -> col 0
        probs = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(1, 0))

        result = normalize_channels([c])

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 1)
        # After normalization to (0, 1): bit 0 -> col 0, bit 1 -> col 1
        # new[0] = old[0] (00 -> 00)
        # new[1] (0b01) = old[2] (0b10): col0=1 means old bit1=1
        # new[2] (0b10) = old[1] (0b01): col1=1 means old bit0=1
        # new[3] = old[3] (11 -> 11)
        expected = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float64)
        assert_allclose(result[0].probs, expected)

    def test_3bit_reorder(self):
        """A 3-bit channel with unsorted col_ids should be reordered correctly."""
        # With col_ids (2, 0, 1): bit 0 -> col 2, bit 1 -> col 0, bit 2 -> col 1.
        probs = np.array([0.4, 0.1, 0.15, 0.05, 0.1, 0.05, 0.1, 0.05], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(2, 0, 1))

        result = normalize_channels([c])

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 1, 2)
        # Verify probs sum to 1
        assert_allclose(np.sum(result[0].probs), 1.0)

    def test_preserves_sampling_statistics(self):
        """Normalization should preserve sampling statistics."""
        probs = np.array([0.6, 0.15, 0.15, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(1, 0))

        normalized = normalize_channels([c])

        mat = np.eye(2, dtype=np.uint8)
        assert_sampling_matches(mat, [c], normalized)


class TestAbsorbSubsetChannels:
    """Tests for absorb_subset_channels."""

    def test_absorb_1bit_into_2bit(self):
        """A 1-bit channel should be absorbed into a 2-bit superset."""
        c1 = Channel(probs=error_probs(0.1), unique_col_ids=(0,))
        c2 = Channel(
            probs=pauli_channel_1_probs(0.1 / 3, 0.1 / 3, 0.1 / 3),
            unique_col_ids=(0, 1),
        )
        channels = [c1, c2]

        result = absorb_subset_channels(channels)

        assert len(result) == 1
        assert set(result[0].unique_col_ids) == {0, 1}

    def test_no_absorb_disjoint(self):
        """Disjoint channels should not be absorbed."""
        c1 = Channel(probs=error_probs(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error_probs(0.2), unique_col_ids=(1,))
        channels = [c1, c2]

        result = absorb_subset_channels(channels)

        assert len(result) == 2

    def test_no_absorb_partial_overlap(self):
        """Partially overlapping (not subset) channels should not be absorbed."""
        c1 = Channel(
            probs=pauli_channel_1_probs(0.1 / 3, 0.1 / 3, 0.1 / 3),
            unique_col_ids=(0, 1),
        )
        c2 = Channel(
            probs=pauli_channel_1_probs(0.2 / 3, 0.2 / 3, 0.2 / 3),
            unique_col_ids=(1, 2),
        )
        channels = [c1, c2]

        result = absorb_subset_channels(channels)

        assert len(result) == 2

    def test_sampling_matches_after_absorb(self):
        """Sampling statistics should match before and after absorption."""
        mat = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=np.uint8,
        )

        # c1 has signature (0,), c2 has signatures (0, 1)
        # c1 should be absorbed into c2
        c1 = Channel(probs=error_probs(0.1), unique_col_ids=(0,))
        c2 = Channel(
            probs=pauli_channel_1_probs(0.15 / 3, 0.15 / 3, 0.15 / 3),
            unique_col_ids=(0, 1),
        )
        channels = [c1, c2]

        absorbed = absorb_subset_channels(channels)

        assert len(absorbed) == 1
        assert_sampling_matches(mat, channels, absorbed)

    def test_absorb_duplicate_subset_channel_uses_xor_expansion(self):
        """A duplicate-signature subset should cancel when absorbed into a superset."""
        superset = Channel(
            probs=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            unique_col_ids=(0, 1),
        )
        duplicate_subset = Channel(
            probs=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            unique_col_ids=(0, 0),
        )

        absorbed = absorb_subset_channels([superset, duplicate_subset])

        assert len(absorbed) == 1
        assert absorbed[0].unique_col_ids == (0, 1)
        assert_allclose(absorbed[0].probs, [1.0, 0.0, 0.0, 0.0])


class TestSimplifyChannels:
    """Tests for the full simplify_channels function."""

    def test_simplify_mixed_channels(self):
        """Test simplification with a mix of channel types."""
        mat = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 1, 1, 0],
            ],
            dtype=np.uint8,
        )

        # Create channels:
        # - Two 1-bit with same signature (should merge)
        # - One 1-bit that's subset of a 2-bit (should absorb)
        # - One 2-bit
        c1 = Channel(probs=error_probs(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error_probs(0.15), unique_col_ids=(0,))  # Same as c1
        c3 = Channel(probs=error_probs(0.2), unique_col_ids=(1,))  # Subset of c4
        c4 = Channel(
            probs=pauli_channel_1_probs(0.1 / 3, 0.1 / 3, 0.1 / 3),
            unique_col_ids=(1, 2),
        )

        channels = [c1, c2, c3, c4]

        simplified = simplify_channels(channels)

        # c1 and c2 merge into one, c3 absorbed into c4
        assert len(simplified) == 2
        assert_sampling_matches(mat, channels, simplified)

    def test_simplify_many_1bit_channels(self):
        """Test simplification of many 1-bit channels with same signature."""
        mat = np.array([[1], [1]], dtype=np.uint8)

        # 10 channels all with the same signature
        channels = [
            Channel(probs=error_probs(0.05), unique_col_ids=(0,)) for _ in range(10)
        ]

        simplified = simplify_channels(channels)

        assert len(simplified) == 1
        assert_sampling_matches(mat, channels, simplified, rtol=0.1)

    def test_simplify_preserves_independent_channels(self):
        """Channels with disjoint signatures should remain separate."""
        mat = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )

        c1 = Channel(probs=error_probs(0.1), unique_col_ids=(0,))
        c2 = Channel(probs=error_probs(0.2), unique_col_ids=(1,))
        c3 = Channel(probs=error_probs(0.3), unique_col_ids=(2,))
        channels = [c1, c2, c3]

        simplified = simplify_channels(channels)

        assert len(simplified) == 3
        assert_sampling_matches(mat, channels, simplified)


class TestSampleChannels:
    """Tests for channel sampling."""

    def test_single_channel(self):
        """Test that sampling produces correct frequencies for a single channel."""
        mat = np.array([[1]], dtype=np.uint8)
        c = Channel(probs=error_probs(0.3), unique_col_ids=(0,))

        bits = _sample_channels([c], mat, 100_000)
        freq = np.mean(bits[:, 0])

        assert_allclose(freq, 0.3, rtol=0.05)

    def test_xor_two_channels(self):
        """Test that sampling correctly XORs two independent channels."""
        mat = np.array([[1], [1]], dtype=np.uint8)

        c1 = Channel(probs=error_probs(0.2), unique_col_ids=(0,))
        c2 = Channel(probs=error_probs(0.3), unique_col_ids=(1,))

        bits = _sample_channels([c1, c2], mat, 100_000)
        freq = np.mean(bits[:, 0])

        # P(f0=1) = P(e0 XOR e1 = 1) = 0.2*0.7 + 0.3*0.8 = 0.14 + 0.24 = 0.38
        expected = 0.2 * 0.7 + 0.3 * 0.8
        assert_allclose(freq, expected, rtol=0.05)


class TestReduceNullBits:
    """Tests for reduce_null_bits function."""

    # Use 99 as the null column ID in tests
    NULL = 99

    # =========================================================================
    # 1-bit channels
    # =========================================================================

    def test_1bit_all_null_removed(self):
        """A 1-bit channel with only null col should be removed entirely."""
        c = Channel(probs=error_probs(0.3), unique_col_ids=(self.NULL,))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert len(result) == 0

    def test_1bit_no_null_unchanged(self):
        """A 1-bit channel with no null entries should be unchanged."""
        c = Channel(probs=error_probs(0.3), unique_col_ids=(0,))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0,)
        assert_allclose(result[0].probs, c.probs, rtol=1e-5)

    # =========================================================================
    # 2-bit channels
    # =========================================================================

    def test_2bit_one_null_marginalize(self):
        """A 2-bit channel with one null should reduce to 1-bit."""
        # Little-endian probs: [P(00), P(bit0), P(bit1), P(bit0+bit1)].
        probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(0, self.NULL))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0,)
        assert result[0].num_bits == 1
        # Marginalize over bit 1 (the null bit):
        # P(bit0=0) = P(00) + P(10) = 0.4 + 0.2 = 0.6
        # P(bit0=1) = P(01) + P(11) = 0.3 + 0.1 = 0.4
        assert_allclose(result[0].probs[0], 0.6, rtol=1e-5)
        assert_allclose(result[0].probs[1], 0.4, rtol=1e-5)

    def test_2bit_first_null_marginalize(self):
        """A 2-bit channel with null in first position should marginalize correctly."""
        # Little-endian probs: [P(00), P(bit0), P(bit1), P(bit0+bit1)].
        probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(self.NULL, 5))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert len(result) == 1
        assert result[0].unique_col_ids == (5,)
        assert result[0].num_bits == 1
        # Marginalize over bit 0 (the null bit):
        # P(bit1=0) = P(00) + P(01) = 0.4 + 0.3 = 0.7
        # P(bit1=1) = P(10) + P(11) = 0.2 + 0.1 = 0.3
        assert_allclose(result[0].probs[0], 0.7, rtol=1e-5)
        assert_allclose(result[0].probs[1], 0.3, rtol=1e-5)

    def test_2bit_all_null_removed(self):
        """A 2-bit channel with all null entries should be removed."""
        probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(self.NULL, self.NULL))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert len(result) == 0

    # =========================================================================
    # 3-bit channels
    # =========================================================================

    def test_3bit_one_null_marginalize(self):
        """A 3-bit channel with one null should reduce to 2-bit."""
        # 8 little-endian outcomes, where bit i contributes index 1 << i.
        probs = np.array([0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(0, self.NULL, 2))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert len(result) == 1
        assert result[0].unique_col_ids == (0, 2)
        assert result[0].num_bits == 2
        # Marginalize over bit 1 (the null bit, middle position):
        # new 00 (bit0=0, bit2=0): P(000) + P(010) = 0.2 + 0.15 = 0.35
        # new 01 (bit0=1, bit2=0): P(001) + P(011) = 0.1 + 0.05 = 0.15
        # new 10 (bit0=0, bit2=1): P(100) + P(110) = 0.2 + 0.1 = 0.3
        # new 11 (bit0=1, bit2=1): P(101) + P(111) = 0.1 + 0.1 = 0.2
        assert_allclose(result[0].probs[0], 0.35, rtol=1e-5)
        assert_allclose(result[0].probs[1], 0.15, rtol=1e-5)
        assert_allclose(result[0].probs[2], 0.3, rtol=1e-5)
        assert_allclose(result[0].probs[3], 0.2, rtol=1e-5)

    def test_3bit_two_null_marginalize(self):
        """A 3-bit channel with two null entries should reduce to 1-bit."""
        probs = np.array([0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(self.NULL, 1, self.NULL))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert len(result) == 1
        assert result[0].unique_col_ids == (1,)
        assert result[0].num_bits == 1
        # Only bit 1 survives. Marginalize over bits 0 and 2:
        # P(bit1=0) = P(000) + P(001) + P(100) + P(101) = 0.2 + 0.1 + 0.2 + 0.1 = 0.6
        # P(bit1=1) = P(010) + P(011) + P(110) + P(111) = 0.15 + 0.05 + 0.1 + 0.1 = 0.4
        assert_allclose(result[0].probs[0], 0.6, rtol=1e-5)
        assert_allclose(result[0].probs[1], 0.4, rtol=1e-5)

    def test_3bit_all_null_removed(self):
        """A 3-bit channel with all null entries should be removed."""
        probs = np.array([0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(self.NULL, self.NULL, self.NULL))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert len(result) == 0

    def test_probs_sum_to_one_after_marginalization(self):
        """Verify that probabilities sum to 1 after marginalization."""
        probs = np.array([0.1, 0.2, 0.15, 0.25, 0.05, 0.1, 0.1, 0.05], dtype=np.float64)
        c = Channel(probs=probs, unique_col_ids=(0, self.NULL, 2))
        channels = [c]

        result = reduce_null_bits(channels, null_col_id=self.NULL)

        assert_allclose(np.sum(result[0].probs), 1.0, rtol=1e-5)


class TestChannelSampler:
    """Tests for ChannelSampler with simplification."""

    def test_simple_xor_two_errors(self):
        """Test XOR of two independent error channels."""
        # Two 1-bit error channels, both affecting f0
        probs = [error_probs(0.2), error_probs(0.3)]
        transform = np.array([[1, 1]], dtype=np.uint8)  # f0 = e0 XOR e1

        sampler = ChannelSampler(probs, transform, seed=42)
        samples = sampler.sample(100_000)

        # P(f0=1) = P(e0 XOR e1 = 1) = 0.2*0.7 + 0.3*0.8 = 0.38
        freq = np.mean(np.asarray(samples[:, 0]))
        expected = 0.2 * 0.7 + 0.3 * 0.8
        assert_allclose(freq, expected, rtol=0.05)

    def test_independent_channels(self):
        """Test independent error channels affecting different f-vars."""
        probs = [error_probs(0.1), error_probs(0.2)]
        transform = np.array([[1, 0], [0, 1]], dtype=np.uint8)  # f0=e0, f1=e1

        sampler = ChannelSampler(probs, transform, seed=42)

        assert len(sampler.channels) == 2

        samples = sampler.sample(100_000)

        freq0 = np.mean(np.asarray(samples[:, 0]))
        freq1 = np.mean(np.asarray(samples[:, 1]))

        assert_allclose(freq0, 0.1, rtol=0.1)
        assert_allclose(freq1, 0.2, rtol=0.1)

    def test_channel_simplification(self):
        """Test that channels with same signature are merged."""
        # Three channels all affecting the same f-var
        probs = [error_probs(0.1), error_probs(0.1), error_probs(0.1)]
        transform = np.array([[1, 1, 1]], dtype=np.uint8)  # f0 = e0 XOR e1 XOR e2

        sampler = ChannelSampler(probs, transform, seed=42)

        # Should have simplified to fewer channels
        assert len(sampler.channels) == 1

        samples = sampler.sample(100_000)
        freq = np.mean(np.asarray(samples[:, 0]))

        # XOR of three Bernoulli(0.1)
        # P(odd number of 1s) = 3*0.1*0.9^2 + 0.1^3 = 0.244
        expected = 3 * 0.1 * 0.9**2 + 0.1**3
        assert_allclose(freq, expected, rtol=0.05)

    def test_empty_transform(self):
        """Test with no f-variables (empty transform)."""
        probs = [error_probs(0.1)]
        transform = np.zeros((0, 1), dtype=np.uint8)

        sampler = ChannelSampler(probs, transform, seed=42)
        samples = sampler.sample(100)

        assert samples.shape == (100, 0)
