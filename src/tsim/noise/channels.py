"""Pauli noise channels and error sampling infrastructure."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Channel:
    """A probability distribution over error outcomes.

    Attributes:
        probs: Shape (2^k,) probability array, sums to 1, dtype float64
        unique_col_ids: Tuple of column IDs, where each ID corresponds to a bit of the channel.

    """

    probs: np.ndarray
    unique_col_ids: tuple[int, ...]

    @property
    def num_bits(self) -> int:
        """Number of bits in the channel (k where probs has shape 2^k)."""
        return int(np.log2(len(self.probs)))


def error_probs(p: float) -> np.ndarray:
    """Single-bit error channel. Returns shape (2,)."""
    return np.array([1 - p, p], dtype=np.float64)


def pauli_channel_1_probs(px: float, py: float, pz: float) -> np.ndarray:
    """Single-qubit Pauli channel. Returns shape (4,).

    Order: [I, Z, X, Y] mapped to bits [00, 01, 10, 11].
    """
    return np.array([1 - px - py - pz, pz, px, py], dtype=np.float64)


def pauli_channel_2_probs(
    pix: float,
    piy: float,
    piz: float,
    pxi: float,
    pxx: float,
    pxy: float,
    pxz: float,
    pyi: float,
    pyx: float,
    pyy: float,
    pyz: float,
    pzi: float,
    pzx: float,
    pzy: float,
    pzz: float,
) -> np.ndarray:
    """Two-qubit Pauli channel. Returns shape (16,)."""
    remainder = (
        1
        - pix
        - piy
        - piz
        - pxi
        - pxx
        - pxy
        - pxz
        - pyi
        - pyx
        - pyy
        - pyz
        - pzi
        - pzx
        - pzy
        - pzz
    )
    probs = np.array(
        [
            remainder,  # 00,00
            pzi,  # 10,00
            pxi,  # 01,00
            pyi,  # 11,00
            piz,  # 00,10
            pzz,  # 10,10
            pxz,  # 01,10
            pyz,  # 11,10
            pix,  # 00,01
            pzx,  # 10,01
            pxx,  # 01,01
            pyx,  # 11,01
            piy,  # 00,11
            pzy,  # 10,11
            pxy,  # 01,11
            pyy,  # 11,11
        ],
        dtype=np.float64,
    )
    return probs


def correlated_error_probs(probabilities: list[float]) -> np.ndarray:
    """Build probability distribution for correlated error chain.

    Given conditional probabilities [p1, p2, ..., pk] from a chain of
    CORRELATED_ERROR(p1) ELSE_CORRELATED_ERROR(p2) ... ELSE_CORRELATED_ERROR(pk),
    computes the joint probability distribution over 2^k outcomes.

    Since errors are mutually exclusive, only outcomes with at most one bit set
    have non-zero probability:
    - P(0) = (1-p1)(1-p2)...(1-pk)  (no error)
    - P(2^i) = (1-p1)...(1-p_i) * p_{i+1}  (error i+1 occurred)

    Args:
        probabilities: List of conditional probabilities [p1, p2, ..., pk]

    Returns:
        Array of shape (2^k,) with probabilities for each outcome.

    """
    k = len(probabilities)
    probs = np.zeros(2**k, dtype=np.float64)

    no_error_so_far = 1.0
    for i, p in enumerate(probabilities):
        probs[1 << i] = no_error_so_far * p
        no_error_so_far *= 1 - p

    probs[0] = no_error_so_far
    return probs


def xor_convolve(probs_a: np.ndarray, probs_b: np.ndarray) -> np.ndarray:
    """XOR convolution of two probability distributions.

    Computes P(A XOR B = o) = sum_{a ^ b = o} P(A=a) * P(B=b)

    Args:
        probs_a: Shape (2^k,) probabilities for channel A
        probs_b: Shape (2^k,) probabilities for channel B (same size as A)

    Returns:
        Shape (2^k,) probabilities for the combined channel

    """
    n = len(probs_a)
    if len(probs_b) != n:
        raise ValueError("Both channels must have same number of outcomes")

    # NOTE: The convolution could be done in O(n*log(n)) using Walsh-Hadamard transform.
    # But since probability arrays are usually limited to <=16 entries, this is not
    # worth the complexity.
    result = np.zeros(n, dtype=np.float64)
    for a in range(n):
        for b in range(n):
            o = a ^ b
            result[o] += probs_a[a] * probs_b[b]

    return result


def reduce_null_bits(
    channels: list[Channel], null_col_id: int | None = None
) -> list[Channel]:
    """Remove bits corresponding to the null column (all-zero column).

    If a channel has bits mapped to null_col_id (representing an all-zero
    column in the transform matrix), those bits don't affect any f-variable
    and can be marginalized out by summing over them.

    Args:
        channels: List of channels
        null_col_id: Column ID representing the all-zero column, or None if
            there is no all-zero column.

    Returns:
        List of channels with null bits marginalized out. Channels with all
        null entries are removed entirely (they have no effect on outputs).

    """
    if null_col_id is None:
        # No null column, nothing to reduce
        return channels

    result: list[Channel] = []

    for channel in channels:
        n = channel.num_bits
        non_null_positions = [
            i
            for i, col_id in enumerate(channel.unique_col_ids)
            if col_id != null_col_id
        ]

        if len(non_null_positions) == 0:
            # All entries are null, channel has no effect - remove it
            continue

        # Marginalize out the null bits by summing over them
        new_col_ids = tuple(channel.unique_col_ids[i] for i in non_null_positions)
        new_num_bits = len(non_null_positions)
        sum_axes = tuple(i for i in range(n) if i not in non_null_positions)
        probs_tensor = channel.probs.reshape((2,) * n, order="F")
        new_probs = probs_tensor.sum(axis=sum_axes).reshape(2**new_num_bits, order="F")

        result.append(Channel(probs=new_probs, unique_col_ids=new_col_ids))

    return result


def normalize_channels(channels: list[Channel]) -> list[Channel]:
    """Normalize channels by sorting unique_col_ids, permuting probs accordingly.

    This ensures channels affecting the same set of columns have identical
    unique_col_ids tuples, enabling merge_identical_channels to group them.

    Args:
        channels: List of channels

    Returns:
        List of channels with sorted unique_col_ids

    """
    result: list[Channel] = []

    for channel in channels:
        n = channel.num_bits
        source_col_ids = np.array(channel.unique_col_ids)
        axis_perm = np.argsort(source_col_ids, stable=True)
        probs_tensor = channel.probs.reshape((2,) * n, order="F")
        new_probs = probs_tensor.transpose(axis_perm).reshape(2**n, order="F")

        result.append(
            Channel(probs=new_probs, unique_col_ids=tuple(source_col_ids[axis_perm]))
        )

    return result


def expand_channel(channel: Channel, target_col_ids: tuple[int, ...]) -> Channel:
    """Expand a channel's distribution to a larger signature set.

    The channel's existing col_ids must be a strict subset of target_col_ids.
    Both must be sorted. New bit positions are treated as "don't care" (always 0).

    Args:
        channel: Channel to expand (must have sorted unique_col_ids)
        target_col_ids: Target signature set (must be sorted superset)

    Returns:
        New channel with expanded distribution

    """
    source_col_ids = channel.unique_col_ids
    assert source_col_ids == tuple(sorted(source_col_ids)), "Source must be sorted"
    assert target_col_ids == tuple(sorted(target_col_ids)), "Target must be sorted"
    assert set(source_col_ids) < set(target_col_ids), "Source must be strict subset"

    # Map source columns to their positions in target
    source_to_target = {s: target_col_ids.index(s) for s in source_col_ids}
    n_target = len(target_col_ids)
    new_probs = np.zeros(2**n_target, dtype=np.float64)

    for old_idx in range(len(channel.probs)):
        # Map old bit pattern to new bit pattern (new bits stay 0)
        new_idx = 0
        for src_pos, src_col in enumerate(source_col_ids):
            if (old_idx >> src_pos) & 1:
                new_idx |= 1 << source_to_target[src_col]
        new_probs[new_idx] += channel.probs[old_idx]

    return Channel(probs=new_probs, unique_col_ids=target_col_ids)


def merge_identical_channels(channels: list[Channel]) -> list[Channel]:
    """Merge all channels with identical signature sets.

    Groups channels by their unique_col_ids and convolves all channels
    in each group into a single channel.

    Args:
        channels: List of channels

    Returns:
        List with at most one channel per unique signature set

    """
    groups: dict[tuple[int, ...], list[Channel]] = defaultdict(list)

    for channel in channels:
        key = channel.unique_col_ids
        groups[key].append(channel)

    result: list[Channel] = []

    for col_ids, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Convolve all channels in the group
            combined_probs = group[0].probs.copy()
            for channel in group[1:]:
                combined_probs = xor_convolve(combined_probs, channel.probs)
            result.append(Channel(probs=combined_probs, unique_col_ids=col_ids))

    return result


def absorb_subset_channels(channels: list[Channel], max_bits: int = 4) -> list[Channel]:
    """Absorb channels whose signatures are subsets of others.

    If channel A's signatures are a strict subset of channel B's signatures,
    and |B| <= max_bits, then A is absorbed into B.

    Args:
        channels: List of channels
        max_bits: Maximum number of bits allowed per channel

    Returns:
        List with no channel being a strict subset of another

    """
    # Sort by number of bits (largest first) for efficient processing
    channels = sorted(channels, key=lambda c: -len(c.unique_col_ids))

    result: list[Channel] = []
    absorbed: set[int] = set()

    for i, channel_i in enumerate(channels):
        if i in absorbed:
            continue

        set_i = set(channel_i.unique_col_ids)

        # Try to absorb smaller channels into this one
        current_probs = channel_i.probs.copy()
        current_col_ids = channel_i.unique_col_ids

        for j, channel_j in enumerate(channels):
            if j <= i or j in absorbed:
                continue

            set_j = set(channel_j.unique_col_ids)

            # Check if j is a strict subset of i
            if set_j < set_i and len(set_i) <= max_bits:
                # Expand channel_j to match channel_i's signatures and convolve
                expanded_j = expand_channel(channel_j, current_col_ids)
                current_probs = xor_convolve(current_probs, expanded_j.probs)
                absorbed.add(j)

        result.append(Channel(probs=current_probs, unique_col_ids=current_col_ids))

    return result


def simplify_channels(
    channels: list[Channel], max_bits: int = 4, null_col_id: int | None = None
) -> list[Channel]:
    """Simplify channels by removing null columns, merging identical and absorbing subsets.

    Args:
        channels: List of channels to simplify
        max_bits: Maximum number of bits allowed per channel
        null_col_id: Column ID representing the all-zero column, or None if
            there is no all-zero column.

    Returns:
        Simplified list of channels

    """
    channels = reduce_null_bits(channels, null_col_id)
    channels = normalize_channels(channels)
    channels = merge_identical_channels(channels)
    channels = absorb_subset_channels(channels, max_bits)
    return channels


class ChannelSampler:
    """Samples from multiple error channels and transforms to a reduced basis.

    This class combines multiple error channels (each producing error bits e0, e1, ...)
    and applies a linear transformation over GF(2) to convert samples from the original
    "e" basis to a reduced "f" basis using geometric-skip sampling optimized for
    low-noise regimes.

    f_i = error_transform_ij * e_j mod 2

    Channels are automatically simplified by:
    1. Removing bits e_i that do not affect any f-variable (i.e. all-zero columns in error_transform)
    2. Merging channels with identical column signatures, i.e. channels whose corresponding
        columns in error_transform are identical.
    3. Absorbing channels whose signatures are subsets of others, i.e. channels whose corresponding
        columns in error_transform are a strict subset of another channel's columns.

    Example:
        >>> probs = [error_probs(0.1), error_probs(0.2)]  # two 1-bit channels
        >>> transform = np.array([[1, 1]])  # f0 = e0 XOR e1
        >>> sampler = ChannelSampler(probs, transform)
        >>> samples = sampler.sample(1000)  # shape (1000, 1)

    """

    def __init__(
        self,
        channel_probs: list[np.ndarray],
        error_transform: np.ndarray,
        seed: int | None = None,
    ):
        """Initialize the sampler with channel probabilities and a basis transformation.

        Args:
            channel_probs: List of probability arrays. Channel i has shape (2^k_i,)
                and produces k_i error bits starting at index sum(k_0:k_{i-1}).
                For example, if channels have shapes [(4,), (2,), (4,)], they
                produce variables [e0,e1], [e2], [e3,e4].
            error_transform: Binary matrix of shape (num_f, num_e) where entry [i, j] = 1
                means f_i depends on e_j. For example, if row 0 is [0, 1, 0, 1],
                then f0 = e1 XOR e3.
            seed: Random seed for sampling. If None, a random seed is generated.

        """
        unique_cols, inverse = np.unique(error_transform, axis=1, return_inverse=True)

        # Signature matrix: each row is a unique column signature
        signature_matrix = unique_cols.T  # shape (num_signatures, num_f)

        # Find null_col_id: the index of the all-zero column (or None)
        zero_col_indices = np.flatnonzero(np.all(unique_cols == 0, axis=0))
        null_col_id = int(zero_col_indices[0]) if len(zero_col_indices) else None

        # Create Channel objects with unique_col_ids from inverse mapping
        channels: list[Channel] = []
        e_offset = 0
        for probs in channel_probs:
            num_bits = int(np.log2(len(probs)))
            col_ids = tuple(int(inverse[e_offset + i]) for i in range(num_bits))
            channels.append(Channel(probs=probs, unique_col_ids=col_ids))
            e_offset += num_bits

        self.channels = simplify_channels(channels, null_col_id=null_col_id)
        self.signature_matrix = signature_matrix.astype(np.uint8)

        self._rng = np.random.default_rng(
            seed if seed is not None else np.random.default_rng().integers(0, 2**30)
        )
        self._sparse_data = self._precompute_sparse(
            self.channels, self.signature_matrix
        )

    @staticmethod
    def _precompute_sparse(
        channels: list[Channel], signature_matrix: np.ndarray
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Precompute per-channel data for geometric-skip sampling.

        For each channel with non-trivial fire probability, computes:
        - p_fire: probability of any non-identity outcome
        - cond_cdf: conditional CDF over non-identity outcomes
        - xor_patterns: precomputed XOR output patterns per outcome

        Args:
            channels: List of noise channels to precompute data for.
            signature_matrix: Binary matrix of shape (num_e, num_f) mapping
                error-variable columns to output f-variables.

        Returns:
            List of (p_fire, cond_cdf, xor_patterns) tuples, one per channel
            with non-trivial fire probability. ``p_fire`` is a float,
            ``cond_cdf`` is a float64 array of shape (n_outcomes - 1,), and
            ``xor_patterns`` is a uint8 array of shape (n_outcomes - 1, num_f).

        """
        data: list[tuple[float, np.ndarray, np.ndarray]] = []
        for ch in channels:
            probs = ch.probs.astype(np.float64)
            p_fire = 1.0 - float(probs[0])
            n_outcomes = len(probs)

            if p_fire <= 1e-15 or n_outcomes <= 1:
                continue

            cond_cdf = np.cumsum(probs[1:] / p_fire, dtype=np.float64)
            cond_cdf /= cond_cdf[-1]

            col_ids = np.asarray(ch.unique_col_ids)
            num_bits = len(col_ids)
            outcomes = np.arange(1, n_outcomes)
            bits_mask = ((outcomes[:, None] >> np.arange(num_bits)) & 1).astype(
                np.uint8
            )
            xor_patterns = (bits_mask @ signature_matrix[col_ids] % 2).astype(np.uint8)

            data.append((p_fire, cond_cdf, xor_patterns))
        return data

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Sample from all error channels and transform to new error basis.

        Uses geometric-skip sampling, optimized for low-noise regimes where
        P(non-identity) << 1 per channel.

        Args:
            num_samples: Number of samples to draw.

        Returns:
            NumPy array of shape (num_samples, num_f) with uint8 values indicating
            which f-variables are set for each sample.

        """
        num_outputs = self.signature_matrix.shape[1]
        result = np.zeros((num_samples, num_outputs), dtype=np.uint8)

        for p_fire, cond_cdf, xor_pats in self._sparse_data:
            expected = num_samples * p_fire
            sigma = np.sqrt(expected * (1.0 - p_fire))
            # At 7 sigma, we undersample in about 1 out of 1e12 cases
            n_draws = int(expected + 7.0 * sigma) + 100

            positions = np.cumsum(self._rng.geometric(p_fire, size=n_draws)) - 1
            positions = positions[positions < num_samples]

            if len(positions) == 0:
                continue

            outcome_idx = np.searchsorted(
                cond_cdf, self._rng.uniform(size=len(positions))
            )
            result[positions] ^= xor_pats[outcome_idx]

        return result
