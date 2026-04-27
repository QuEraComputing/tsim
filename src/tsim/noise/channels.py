"""Pauli noise channels and error sampling infrastructure."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Channel:
    """A probability distribution over error outcomes.

    Outcome indices: bit position ``i`` corresponds
    to ``1 << i`` in ``probs``. For example, in a 2-bit channel, index 1
    (0b01) is bit pattern ``bit0=1, bit1=0`` and index 2 (0b10) is
    ``bit0=0, bit1=1``.

    Attributes:
        probs: Shape ``(2**k,)`` probability array, sums to 1, dtype float64.
        unique_col_ids: Tuple of column IDs. Entry ``i`` is the transform-column
            signature affected by channel bit ``i``.

    """

    probs: np.ndarray
    unique_col_ids: tuple[int, ...]

    @property
    def num_bits(self) -> int:
        """Number of bits in the channel (k where probs has shape 2^k)."""
        return int(np.log2(len(self.probs)))


_PROB_SUM_TOL = 1e-9


def _validate_probabilities(*probs: float) -> None:
    """Validate that each probability is in ``[0, 1]`` and they sum to at most 1.

    Raises:
        ValueError: If any probability is outside ``[0, 1]`` or the sum exceeds 1.

    """
    if any(p < 0.0 or p > 1.0 for p in probs):
        raise ValueError("probabilities must lie in [0, 1]")
    if sum(probs) > 1.0 + _PROB_SUM_TOL:
        raise ValueError("probabilities must sum to at most 1")


def error_probs(p: float) -> np.ndarray:
    """Single-bit error channel.

    Returns ``[P(bit0=0), P(bit0=1)]``.
    """
    _validate_probabilities(p)
    return np.array([1 - p, p], dtype=np.float64)


def heralded_pauli_channel_1_probs(
    pi: float, px: float, py: float, pz: float
) -> np.ndarray:
    """Heralded single-qubit Pauli channel. Returns shape (8,).

    Bit layout:
    - bit 0: herald bit, written to the measurement record
    - bit 1: Z error component
    - bit 2: X error component

    The non-zero outcomes are:
    - index 0 (0b000): no herald, no Pauli error
    - index 1 (0b001): herald + I
    - index 3 (0b011): herald + Z
    - index 5 (0b101): herald + X
    - index 7 (0b111): herald + Y, represented as X+Z
    """
    _validate_probabilities(pi, px, py, pz)
    probs = np.zeros(8, dtype=np.float64)
    probs[0] = 1 - pi - px - py - pz
    probs[1] = pi
    probs[3] = pz
    probs[5] = px
    probs[7] = py
    return probs


def pauli_channel_1_probs(px: float, py: float, pz: float) -> np.ndarray:
    """Single-qubit Pauli channel. Returns shape (4,).

    Bit layout:
    - bit 0: Z error component
    - bit 1: X error component

    The outcomes are:
    - index 0 (0b00): I
    - index 1 (0b01): Z
    - index 2 (0b10): X
    - index 3 (0b11): Y
    """
    _validate_probabilities(px, py, pz)
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
    """Two-qubit Pauli channel. Returns shape (16,).

    Bit layout:
    - bit 0: Z error component on ``qubit_i``
    - bit 1: X error component on ``qubit_i``
    - bit 2: Z error component on ``qubit_j``
    - bit 3: X error component on ``qubit_j``

    With that layout, index ``z_i + 2*x_i + 4*z_j + 8*x_j`` stores the
    probability for the corresponding two-qubit Pauli outcome. The arguments
    follow Stim's naming convention: ``pix`` is I on ``qubit_i`` and X on
    ``qubit_j``, ``pzi`` is Z on ``qubit_i`` and I on ``qubit_j``, etc.
    """
    _validate_probabilities(
        pix, piy, piz, pxi, pxx, pxy, pxz, pyi, pyx, pyy, pyz, pzi, pzx, pzy, pzz
    )
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
            remainder,  # index 0 (0b0000): II
            pzi,  # index 1 (0b0001): ZI
            pxi,  # index 2 (0b0010): XI
            pyi,  # index 3 (0b0011): YI
            piz,  # index 4 (0b0100): IZ
            pzz,  # index 5 (0b0101): ZZ
            pxz,  # index 6 (0b0110): XZ
            pyz,  # index 7 (0b0111): YZ
            pix,  # index 8 (0b1000): IX
            pzx,  # index 9 (0b1001): ZX
            pxx,  # index 10 (0b1010): XX
            pyx,  # index 11 (0b1011): YX
            piy,  # index 12 (0b1100): IY
            pzy,  # index 13 (0b1101): ZY
            pxy,  # index 14 (0b1110): XY
            pyy,  # index 15 (0b1111): YY
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
    have non-zero probability.
    - ``probs[0]`` is the probability that no branch fires.
    - ``probs[1 << i]`` is the probability that branch ``i`` fires after all
      previous branches did not fire.

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

        # Marginalize out null bits by summing their tensor axes. The Fortran
        # order reshape makes axis i correspond to little-endian bit i.
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
    ``unique_col_ids`` tuples, enabling ``merge_identical_channels`` to group
    them. The probability tensor is transposed using the same axis permutation
    so little-endian outcome bits continue to refer to the matching column IDs.

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


def fold_duplicate_channel_bits(channels: list[Channel]) -> list[Channel]:
    """Canonicalize channels by XOR-folding duplicate column IDs.

    If two bits in the same channel have identical column signatures, sampling
    both bits only affects the reduced error basis through their parity. This
    replaces those duplicate bits with one bit whose probability is the sum of
    all old outcomes with the same XOR-folded value.

    Args:
        channels: List of channels with sorted unique_col_ids

    Returns:
        List of channels whose unique_col_ids contain no duplicates

    """
    result: list[Channel] = []

    for channel in channels:
        old_col_ids = channel.unique_col_ids
        new_col_ids = tuple(dict.fromkeys(old_col_ids))

        if len(new_col_ids) == len(old_col_ids):
            result.append(channel)
            continue

        col_to_new_pos = {col: pos for pos, col in enumerate(new_col_ids)}
        new_probs = np.zeros(2 ** len(new_col_ids), dtype=np.float64)

        for old_idx in range(len(channel.probs)):
            new_idx = 0
            for old_pos, col in enumerate(old_col_ids):
                if (old_idx >> old_pos) & 1:
                    new_idx ^= 1 << col_to_new_pos[col]
            new_probs[new_idx] += channel.probs[old_idx]

        result.append(Channel(probs=new_probs, unique_col_ids=new_col_ids))

    return result


def expand_channel(channel: Channel, target_col_ids: tuple[int, ...]) -> Channel:
    """Expand a channel's distribution to a larger signature set.

    The channel's existing column IDs must be a strict subset of
    ``target_col_ids`` when considered as sets, and both tuples must be sorted.
    New target bit positions are treated as always zero.

    Duplicate source column IDs are allowed. When multiple source bits map to
    the same target bit, their contribution is XORed, matching GF(2)
    composition. Duplicate target column IDs are not allowed; channels with
    duplicate IDs should be canonicalized before subset absorption.

    Args:
        channel: Channel to expand (must have sorted unique_col_ids)
        target_col_ids: Target signature set (must be sorted superset)

    Returns:
        New channel with expanded distribution

    """
    source_col_ids = channel.unique_col_ids
    if source_col_ids != tuple(sorted(source_col_ids)):
        raise ValueError("Source must be sorted")
    if target_col_ids != tuple(sorted(target_col_ids)):
        raise ValueError("Target must be sorted")
    if len(set(target_col_ids)) != len(target_col_ids):
        raise ValueError("Target must not contain duplicates")
    if not set(source_col_ids) < set(target_col_ids):
        raise ValueError("Source must be strict subset")

    # Map source columns to their positions in target
    source_to_target = {s: target_col_ids.index(s) for s in source_col_ids}
    n_target = len(target_col_ids)
    new_probs = np.zeros(2**n_target, dtype=np.float64)

    for old_idx in range(len(channel.probs)):
        # Map old bit pattern to the expanded pattern.
        # Use XOR so duplicate source columns cancel mod 2.
        new_idx = 0
        for src_pos, src_col in enumerate(source_col_ids):
            if (old_idx >> src_pos) & 1:
                new_idx ^= 1 << source_to_target[src_col]
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
    """Simplify channels by removing null columns, folding, merging identical and absorbing subsets.

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
    channels = fold_duplicate_channel_bits(channels)
    channels = merge_identical_channels(channels)
    channels = absorb_subset_channels(channels, max_bits)
    return channels


class ChannelSampler:
    """Samples from multiple error channels and transforms to a reduced basis.

    This class combines multiple error channels (each producing error bits e0, e1, ...)
    and applies a linear transformation over GF(2) to convert samples from the original
    "e" basis to a reduced "f" basis using geometric-skip sampling optimized for
    low-noise regimes.

    ``f_i = error_transform_ij * e_j mod 2``. Within each channel,
    probability-array bit 0 corresponds to the first produced error bit, bit 1
    to the second, and so on.

    Channels are automatically simplified by:
    1. Removing bits e_i that do not affect any f-variable (i.e. all-zero columns in error_transform)
    2. Folding duplicate column IDs, i.e. channels whose column signatures contain duplicate IDs.
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

            if p_fire < -_PROB_SUM_TOL:
                raise ValueError("channel has invalid probabilities")

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
