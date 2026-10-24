"""Pauli noise channels and error sampling infrastructure."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple

import numba
import numpy as np

from tsim.utils.kernels import jit, kernel
from tsim.utils.rng import next_skip, next_uniform

# NOTE: shots are sampled in blocks of this many rows, in parallel, each block
# from its own random stream; a block's rows stay in cache while every channel
# visits them.
_BLOCK_SHOTS = 4096
# NOTE: a block of f rows up to this size stays in a core's cache, so its
# faults are XOR-ed in channel by channel instead of being sorted by row first.
_CACHED_BYTES = 2**18


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

    def __post_init__(self) -> None:
        """Validate channel probabilities."""
        # NOTE: NumPy's overhead dominates for a channel's few outcomes, and
        # simplification makes many channels, so the range is checked in plain
        # Python and the sum without np.isclose, with its tolerance.
        tol = 1e-6
        values = self.probs.tolist()
        if min(values, default=0.0) < -tol or max(values, default=0.0) > 1.0 + tol:
            raise ValueError(f"Probabilities must lie in [0, 1], but got: {self.probs}")
        if not abs(float(self.probs.sum()) - 1.0) <= 1e-8 + 1e-5:
            raise ValueError(
                f"Probabilities must sum to 1, but got: {self.probs} "
                f"(sum {np.sum(self.probs)})"
            )

    @property
    def num_bits(self) -> int:
        """Number of bits in the channel (k where probs has shape 2^k)."""
        return channel_num_bits(self.probs)


def channel_num_bits(probs: np.ndarray) -> int:
    """Return the number of bits ``k`` of a channel with ``2^k`` outcomes.

    Raises:
        ValueError: If the number of outcomes is not a power of two.

    """
    num_bits = len(probs).bit_length() - 1
    if num_bits < 0 or len(probs) != 1 << num_bits:
        raise ValueError(f"a channel has 2^k outcomes, but got {len(probs)}")
    return num_bits


def error_probs(p: float) -> np.ndarray:
    """Single-bit error channel.

    Returns ``[P(bit0=0), P(bit0=1)]``.
    """
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
        List in which no channel is a strict subset of another of at most
        ``max_bits`` signatures

    """
    # NOTE: largest first, so that every channel is absorbed into the first
    # channel that contains it.
    channels = sorted(channels, key=lambda c: -len(c.unique_col_ids))

    # NOTE: a channel of at most max_bits signatures has few strict subsets, so
    # they are looked up instead of compared with every other channel.
    by_signatures: dict[frozenset[int], list[int]] = defaultdict(list)
    for j, channel in enumerate(channels):
        by_signatures[frozenset(channel.unique_col_ids)].append(j)

    result: list[Channel] = []
    absorbed: set[int] = set()

    for i, channel_i in enumerate(channels):
        if i in absorbed:
            continue

        signatures = sorted(set(channel_i.unique_col_ids))
        current_probs = channel_i.probs.copy()
        current_col_ids = channel_i.unique_col_ids

        if len(signatures) <= max_bits:
            subsets = (
                frozenset(subset)
                for size in range(len(signatures))
                for subset in combinations(signatures, size)
            )
            subset_channels = sorted(
                j
                for subset in subsets
                for j in by_signatures.get(subset, ())
                if j > i and j not in absorbed
            )
            for j in subset_channels:
                # Expand channel_j to match channel_i's signatures and convolve
                expanded_j = expand_channel(channels[j], current_col_ids)
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


def unique_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``np.unique(matrix, axis=1, return_inverse=True)``, with a 1-D inverse.

    ``np.unique`` sorts every column, comparing whole columns, which is slow for
    a large sparse matrix. Here a dict over the bit-packed columns finds the
    distinct ones, and only those are sorted, as bytes. Big-endian packing keeps
    the lexicographic order of the columns, so the result is the same.

    Args:
        matrix: Binary matrix with entries 0 or 1 (nonzero entries count as 1).

    Returns:
        The distinct columns in lexicographic order, and the index of each
        column of ``matrix`` among them.

    """
    nonzero = (np.asarray(matrix) != 0).view(np.uint8)
    # NOTE: np.packbits along axis 0 reads the matrix column by column; OR-ing
    # every eighth row into the bytes reads it row by row, which is faster.
    packed = np.zeros((-(-nonzero.shape[0] // 8), nonzero.shape[1]), dtype=np.uint8)
    for bit in range(8):
        rows = nonzero[bit::8]
        packed[: len(rows)] |= rows << np.uint8(7 - bit)
    packed = np.ascontiguousarray(packed.T)
    ids: dict[bytes, int] = {}
    first_list: list[int] = []
    column_ids = np.empty(len(packed), dtype=np.int64)
    for i, column in enumerate(packed):
        key = column.tobytes()
        if key not in ids:
            ids[key] = len(first_list)
            first_list.append(i)
        column_ids[i] = ids[key]
    first = np.array(first_list, dtype=np.int64)
    keys = list(ids)
    order = np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return matrix[:, first[order]], rank[column_ids]


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
    3. Merging channels with identical column signatures, i.e. channels whose corresponding
        columns in error_transform are identical.
    4. Absorbing channels whose signatures are subsets of others, i.e. channels whose corresponding
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
        unique_cols, inverse = unique_columns(error_transform)

        # Signature matrix: each row is a unique column signature
        signature_matrix = unique_cols.T  # shape (num_signatures, num_f)

        # Find null_col_id: the index of the all-zero column (or None)
        zero_col_indices = np.flatnonzero(np.all(unique_cols == 0, axis=0))
        null_col_id = int(zero_col_indices[0]) if len(zero_col_indices) else None

        # Create Channel objects with unique_col_ids from inverse mapping
        channels: list[Channel] = []
        e_offset = 0
        for probs in channel_probs:
            num_bits = channel_num_bits(probs)
            col_ids = tuple(int(inverse[e_offset + i]) for i in range(num_bits))
            channels.append(Channel(probs=probs, unique_col_ids=col_ids))
            e_offset += num_bits

        self.channels = simplify_channels(channels, null_col_id=null_col_id)
        self.signature_matrix = signature_matrix.astype(np.uint8)

        self._rng = np.random.default_rng(
            seed if seed is not None else np.random.default_rng().integers(0, 2**30)
        )
        self._tables = _fire_tables(self.channels, self.signature_matrix)

    @property
    def fire_tables(self) -> FireTables:
        """Return the sampling tables of the channels that can fire."""
        return self._tables

    def block_seeds(self, num_samples: int) -> np.ndarray:
        """Draw the seeds of the blocks of the next ``num_samples`` samples."""
        num_blocks = -(-num_samples // _BLOCK_SHOTS)
        return self._rng.integers(0, 2**64, size=num_blocks, dtype=np.uint64)

    def sample(
        self,
        num_samples: int = 1,
        seeds: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample from all error channels and transform to new error basis.

        Uses geometric-skip sampling, optimized for low-noise regimes where
        P(non-identity) << 1 per channel. Blocks of shots are sampled in
        parallel, each from its own random stream, so the samples depend on the
        seed and ``num_samples`` but not on the number of threads.

        Args:
            num_samples: Number of samples to draw.
            seeds: The blocks' seeds from ``block_seeds(num_samples)``; drawn
                here if ``None``.
            out: A uint8 array of shape (num_samples, num_f) to write into, or
                ``None`` for a new one.

        Returns:
            NumPy array of shape (num_samples, num_f) with uint8 values indicating
            which f-variables are set for each sample.

        """
        num_f = self.signature_matrix.shape[1]
        result = np.empty((num_samples, num_f), dtype=np.uint8) if out is None else out
        if result.shape != (num_samples, num_f) or result.dtype != np.uint8:
            raise ValueError(
                f"out must be a uint8 array of shape {(num_samples, num_f)}"
            )
        if seeds is None:
            seeds = self.block_seeds(num_samples)
        num_blocks = -(-num_samples // _BLOCK_SHOTS)
        if len(seeds) != num_blocks:
            raise ValueError(
                f"{num_samples} samples need {num_blocks} block seeds, "
                f"got {len(seeds)}"
            )
        work = num_samples * len(self._tables.p_fire)
        _sample_blocks(work, result, seeds, *self._tables)
        return result


class FireTables(NamedTuple):
    """Geometric-skip sampling tables of the channels that can fire.

    Channel ``c`` fires with probability ``p_fire[c]``. Its non-identity
    outcomes are ``outcome_start[c] … outcome_start[c+1] - 1``, picked by the
    conditional CDF ``cdf`` over those indices, and outcome ``k`` flips the
    f-variables ``flip_cols[flip_start[k] : flip_start[k+1]]``.
    """

    p_fire: np.ndarray
    outcome_start: np.ndarray
    cdf: np.ndarray
    flip_start: np.ndarray
    flip_cols: np.ndarray


def _fire_tables(channels: list[Channel], signature_matrix: np.ndarray) -> FireTables:
    """Return the sampling tables of ``channels``.

    A channel fires with the total probability of its non-identity outcomes,
    clipped to [0, 1]: ``1 - probs[0]`` can exceed 1 by a rounding error, and it
    loses precision for small probabilities.

    Args:
        channels: The noise channels.
        signature_matrix: Binary matrix of shape (num_signatures, num_f) mapping
            each column ID to the f-variables it flips.

    """
    p_fire: list[float] = []
    cdf: list[float] = []
    flip_cols: list[int] = []
    outcome_start, flip_start = [0], [0]
    # NOTE: signatures are sparse, so an outcome's flips are built as sets: those
    # of its lowest bit XOR those of the outcome without that bit.
    supports: dict[int, frozenset[int]] = {}
    for channel in channels:
        probs = channel.probs.astype(np.float64)
        conditional = np.cumsum(np.clip(probs[1:], 0, None))
        fire = min(float(conditional[-1]), 1.0) if len(conditional) else 0.0
        if fire <= 1e-15:
            continue
        p_fire.append(fire)
        cdf.extend(conditional / conditional[-1])
        outcome_start.append(len(cdf))
        for col_id in channel.unique_col_ids:
            if col_id not in supports:
                supports[col_id] = frozenset(
                    np.flatnonzero(signature_matrix[col_id]).tolist()
                )
        bit_supports = [supports[col_id] for col_id in channel.unique_col_ids]
        flips: list[frozenset[int]] = [frozenset()]
        for outcome in range(1, len(probs)):
            lowest = outcome & -outcome
            bit = lowest.bit_length() - 1
            flips.append(flips[outcome ^ lowest] ^ bit_supports[bit])
            flip_cols.extend(sorted(flips[outcome]))
            flip_start.append(len(flip_cols))
    return FireTables(
        p_fire=np.array(p_fire, dtype=np.float64),
        outcome_start=np.array(outcome_start, dtype=np.int64),
        cdf=np.array(cdf, dtype=np.float64),
        flip_start=np.array(flip_start, dtype=np.int64),
        flip_cols=np.array(flip_cols, dtype=np.int64),
    )


@jit
def _block_faults(rows, state, p_fire, outcome_start, cdf, fault_row, fault_outcome):
    """Draw the faults of a block of ``rows`` shots, channel by channel.

    Writes each fault's row and outcome; returns their count, or -1 if they
    don't fit.
    """
    count = 0
    for c in range(len(p_fire)):
        log_keep = np.log1p(-p_fire[c])
        first, last = outcome_start[c], outcome_start[c + 1]
        row = -1.0
        while True:
            state, skip = next_skip(state, log_keep)
            row += 1.0 + skip
            if row >= rows:
                break
            state, u = next_uniform(state)
            if count == len(fault_row):
                return -1
            fault_row[count] = int(row)
            fault_outcome[count] = first + np.searchsorted(
                cdf[first:last], u, side="right"
            )
            count += 1
    return count


@jit
def _xor_faults(out, state, p_fire, outcome_start, cdf, flip_start, flip_cols):
    """XOR every channel's faults into the rows of ``out``, channel by channel."""
    for c in range(len(p_fire)):
        log_keep = np.log1p(-p_fire[c])
        first, last = outcome_start[c], outcome_start[c + 1]
        row = -1.0
        while True:
            state, skip = next_skip(state, log_keep)
            row += 1.0 + skip
            if row >= out.shape[0]:
                break
            state, u = next_uniform(state)
            k = first + np.searchsorted(cdf[first:last], u, side="right")
            for j in range(flip_start[k], flip_start[k + 1]):
                out[int(row), flip_cols[j]] ^= 1


@jit
def _sample_block(out, state, p_fire, outcome_start, cdf, flip_start, flip_cols):
    """Write the rows of ``out`` from the stream ``state``.

    A block that fits in cache takes each channel's faults in turn. A larger
    one draws them first, then sorts them by row, so that each row is built
    from its own faults while it is in cache: rows far apart in a block miss
    the cache when channels XOR into them in turn. Both give the same rows.
    """
    rows = out.shape[0]
    if out.size <= _CACHED_BYTES:
        out[:] = 0
        _xor_faults(out, state, p_fire, outcome_start, cdf, flip_start, flip_cols)
        return
    expected = rows * np.sum(p_fire)
    capacity = int(expected + 8 * np.sqrt(expected)) + 64
    while True:
        fault_row = np.empty(capacity, dtype=np.int64)
        fault_outcome = np.empty(capacity, dtype=np.int64)
        count = _block_faults(
            rows, state, p_fire, outcome_start, cdf, fault_row, fault_outcome
        )
        if count >= 0:
            break
        capacity *= 2
    start = np.zeros(rows + 1, dtype=np.int64)
    for f in range(count):
        start[fault_row[f] + 1] += 1
    for r in range(rows):
        start[r + 1] += start[r]
    filled = start[:rows].copy()
    by_row = np.empty(count, dtype=np.int64)
    for f in range(count):
        r = fault_row[f]
        by_row[filled[r]] = fault_outcome[f]
        filled[r] += 1
    for r in range(rows):
        row = out[r]
        row[:] = 0
        for i in range(start[r], start[r + 1]):
            k = by_row[i]
            for j in range(flip_start[k], flip_start[k + 1]):
                row[flip_cols[j]] ^= 1


@kernel
def _sample_blocks(out, seeds, p_fire, outcome_start, cdf, flip_start, flip_cols):
    """Sample block ``b`` of ``_BLOCK_SHOTS`` rows from the stream ``seeds[b]``."""
    for b in numba.prange(len(seeds)):
        rows = out[b * _BLOCK_SHOTS : (b + 1) * _BLOCK_SHOTS]
        _sample_block(rows, seeds[b], p_fire, outcome_start, cdf, flip_start, flip_cols)
