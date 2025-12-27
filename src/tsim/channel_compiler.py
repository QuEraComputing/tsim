from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Channel:
    probs: jax.Array  # shape (2^k,), sums to 1, dtype float32
    unique_col_ids: tuple[int | None, ...]  # None indicates all-zero column

    @property
    def num_bits(self) -> int:
        return int(jnp.log2(len(self.probs)))

    @property
    def col_ids(self) -> tuple[int, ...]:
        """Get unique_col_ids, asserting no None values present."""
        assert None not in self.unique_col_ids, "Call reduce_null_bits first"
        return self.unique_col_ids  # type: ignore[return-value]

    @property
    def logits(self) -> jax.Array:
        """Convert to logits for sampling."""
        return jnp.log(self.probs)


def reduce_null_bits(channels: list[Channel]) -> list[Channel]:
    """Remove bits corresponding to None entries in unique_col_ids.

    If a channel has None entries in its unique_col_ids (representing all-zero
    columns in the transform matrix), those bits don't affect any f-variable
    and can be marginalized out by summing over them.

    Args:
        channels: List of channels, where unique_col_ids may contain None

    Returns:
        List of channels with None bits marginalized out. Channels with all
        None entries are removed entirely (they have no effect on outputs).
    """
    result: list[Channel] = []

    for channel in channels:
        # Find which positions are None vs non-None
        none_positions = [
            i for i, col_id in enumerate(channel.unique_col_ids) if col_id is None
        ]
        non_none_positions = [
            i for i, col_id in enumerate(channel.unique_col_ids) if col_id is not None
        ]

        if len(none_positions) == 0:
            # No None entries, keep as-is
            result.append(channel)
            continue

        if len(non_none_positions) == 0:
            # All entries are None, channel has no effect - remove it
            continue

        # Marginalize out the None bits by summing over them
        new_col_ids = tuple(
            channel.unique_col_ids[i] for i in non_none_positions
        )  # type: ignore[misc]
        new_num_bits = len(non_none_positions)
        new_probs = jnp.zeros(2**new_num_bits, dtype=jnp.float32)

        # For each old outcome, compute the new outcome and add probability
        for old_idx in range(len(channel.probs)):
            # Extract only the bits at non_none_positions
            new_idx = 0
            for new_bit_pos, old_bit_pos in enumerate(non_none_positions):
                if (old_idx >> old_bit_pos) & 1:
                    new_idx |= 1 << new_bit_pos
            new_probs = new_probs.at[new_idx].add(channel.probs[old_idx])

        result.append(Channel(probs=new_probs, unique_col_ids=new_col_ids))

    return result


class Sampler:
    def __init__(self, matrix: jax.Array, seed: int | None = None):
        self.matrix = matrix
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self._key = jax.random.key(seed)

    def sample_one(self, channel: Channel, num_samples: int = 1) -> jax.Array:
        num_bits = channel.num_bits
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.categorical(subkey, channel.logits, shape=(num_samples,))
        bits = ((samples[:, None] >> jnp.arange(num_bits)) & 1).astype(jnp.uint8)
        res = jnp.zeros((num_samples, self.matrix.shape[1]), dtype=jnp.uint8)

        assert channel.num_bits == len(channel.unique_col_ids)
        for i, col_id in enumerate(channel.unique_col_ids):
            assert col_id is not None
            res = res ^ (bits[:, i : i + 1] * self.matrix[col_id : col_id + 1, :])
        return res

    def sample(self, channels: list[Channel], num_samples: int = 1) -> jax.Array:
        res = jnp.zeros((num_samples, self.matrix.shape[1]), dtype=jnp.uint8)
        for channel in channels:
            res = res ^ self.sample_one(channel, num_samples)
        return res


# =============================================================================
# Channel constructors (return probabilities, dtype float32)
# =============================================================================


def pauli_channel_1(px: float, py: float, pz: float) -> jax.Array:
    return jnp.array([1 - px - py - pz, pz, px, py], dtype=jnp.float32)


def depolarize_1(p: float) -> jax.Array:
    return pauli_channel_1(p / 3, p / 3, p / 3)


def pauli_channel_2(
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
):
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
    probs = jnp.array(
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
        ]
    )
    return probs


def depolarize_2(p: float) -> jax.Array:
    return pauli_channel_2(
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
    )


def error(p: float) -> jax.Array:
    """Single-bit error channel. Returns shape (2,)."""
    return jnp.array([1 - p, p], dtype=jnp.float32)


def y_error(p: float) -> jax.Array:
    """Y error channel (correlated X and Z). Returns shape (4,).

    A Y error is equivalent to both X and Z errors occurring together.
    Outcomes: 00 (no error), 01 (Z only), 10 (X only), 11 (Y = XZ).
    Only 00 and 11 have non-zero probability.
    """
    return jnp.array([1 - p, 0, 0, p], dtype=jnp.float32)


# =============================================================================
# Channel combination
# =============================================================================


def xor_convolve(probs_a: jax.Array, probs_b: jax.Array) -> jax.Array:
    """XOR convolution of two probability distributions.

    Computes P(A XOR B = o) = sum_{a ^ b = o} P(A=a) * P(B=b)

    Args:
        probs_a: Shape (2^k,) probabilities for channel A
        probs_b: Shape (2^k,) probabilities for channel B (same size as A)

    Returns:
        Shape (2^k,) probabilities for the combined channel
    """
    n = len(probs_a)
    assert len(probs_b) == n, "Both channels must have same number of outcomes"

    result = jnp.zeros(n, dtype=jnp.float32)
    for a in range(n):
        for b in range(n):
            o = a ^ b
            result = result.at[o].add(probs_a[a] * probs_b[b])

    return result


def combine_bernoulli_probs(probs_a: jax.Array, probs_b: jax.Array) -> jax.Array:
    """Combine two Bernoulli channels via XOR (specialized 1-bit case).

    If A ~ Bernoulli(p) and B ~ Bernoulli(q) independently,
    then A XOR B ~ Bernoulli(p(1-q) + q(1-p)).

    Args:
        probs_a: Shape (2,) probabilities [1-p, p]
        probs_b: Shape (2,) probabilities [1-q, q]

    Returns:
        Shape (2,) probabilities for the combined channel
    """
    p = probs_a[1]
    q = probs_b[1]

    # XOR probability: p(1-q) + q(1-p) = p + q - 2pq
    p_combined = p + q - 2 * p * q

    return jnp.array([1 - p_combined, p_combined], dtype=jnp.float32)


def expand_channel(channel: Channel, target_col_ids: tuple[int, ...]) -> Channel:
    """Expand a channel's distribution to a larger signature set.

    The channel's existing signatures must be a subset of target_col_ids.
    New bit positions are treated as "don't care" (uniformly 0).

    Args:
        channel: Channel to expand (must have no None in col_ids)
        target_col_ids: Target signature set (must be superset of channel's)

    Returns:
        New channel with expanded distribution
    """
    source_col_ids = channel.col_ids
    source_set = set(source_col_ids)
    target_set = set(target_col_ids)
    assert source_set <= target_set, "Source must be subset of target"

    if source_set == target_set:
        # Same signatures, just reorder if needed
        if source_col_ids == target_col_ids:
            return channel
        # Need to reorder bits
        source_to_target = {s: target_col_ids.index(s) for s in source_col_ids}
        n_target = len(target_col_ids)
        new_probs = jnp.zeros(2**n_target, dtype=jnp.float32)

        for old_idx in range(len(channel.probs)):
            # Map old bit pattern to new bit pattern
            new_idx = 0
            for src_pos, src_col in enumerate(source_col_ids):
                if (old_idx >> src_pos) & 1:
                    new_idx |= 1 << source_to_target[src_col]
            new_probs = new_probs.at[new_idx].add(channel.probs[old_idx])

        return Channel(probs=new_probs, unique_col_ids=target_col_ids)

    # Expand to larger set: new bits are always 0
    source_to_target = {s: target_col_ids.index(s) for s in source_col_ids}
    n_target = len(target_col_ids)
    new_probs = jnp.zeros(2**n_target, dtype=jnp.float32)

    for old_idx in range(len(channel.probs)):
        # Map old bit pattern to new bit pattern (new bits stay 0)
        new_idx = 0
        for src_pos, src_col in enumerate(source_col_ids):
            if (old_idx >> src_pos) & 1:
                new_idx |= 1 << source_to_target[src_col]
        new_probs = new_probs.at[new_idx].add(channel.probs[old_idx])

    return Channel(probs=new_probs, unique_col_ids=target_col_ids)


def merge_identical_channels(channels: list[Channel]) -> list[Channel]:
    """Phase 1: Merge all channels with identical signature sets.

    Groups channels by their unique_col_ids and convolves all channels
    in each group into a single channel.

    Args:
        channels: List of channels

    Returns:
        List with at most one channel per unique signature set
    """
    from collections import defaultdict

    groups: dict[tuple[int, ...], list[Channel]] = defaultdict(list)

    for channel in channels:
        key = channel.col_ids
        groups[key].append(channel)

    result: list[Channel] = []

    for col_ids, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Convolve all channels in the group
            combined_probs = group[0].probs
            for channel in group[1:]:
                combined_probs = xor_convolve(combined_probs, channel.probs)
            result.append(Channel(probs=combined_probs, unique_col_ids=col_ids))

    return result


def absorb_subset_channels(channels: list[Channel], max_bits: int = 4) -> list[Channel]:
    """Phase 2: Absorb channels whose signatures are subsets of others.

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

        set_i = set(channel_i.col_ids)

        # Try to absorb smaller channels into this one
        current_probs = channel_i.probs
        current_col_ids = channel_i.col_ids

        for j, channel_j in enumerate(channels):
            if j <= i or j in absorbed:
                continue

            set_j = set(channel_j.col_ids)

            # Check if j is a strict subset of i
            if set_j < set_i and len(set_i) <= max_bits:
                # Expand channel_j to match channel_i's signatures and convolve
                expanded_j = expand_channel(channel_j, current_col_ids)
                current_probs = xor_convolve(current_probs, expanded_j.probs)
                absorbed.add(j)

        result.append(Channel(probs=current_probs, unique_col_ids=current_col_ids))

    return result


def simplify_channels(channels: list[Channel], max_bits: int = 4) -> list[Channel]:
    """Simplify channels by merging identical and absorbing subsets.

    Applies Phase 1 (merge identical) and Phase 2 (absorb subsets).

    Args:
        channels: List of channels to simplify
        max_bits: Maximum number of bits allowed per channel

    Returns:
        Simplified list of channels
    """
    channels = reduce_null_bits(channels)

    channels = merge_identical_channels(channels)

    channels = absorb_subset_channels(channels, max_bits)

    return channels
