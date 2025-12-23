import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class Channel(abc.ABC):
    """Abstract base class for quantum error channels."""

    logits: Array
    num_bits: int

    def __init__(self, key: Array):
        self._key = key

    @abc.abstractmethod
    def sample(self, num_samples: int = 1) -> jax.Array:
        """Sample errors from the channel.

        Args:
            num_samples: Number of samples to draw from the channel.

        Returns:
            A jax.numpy array of shape (num_samples, num_qubits) containing the
            sampled errors.

        """

    def __repr__(self):
        return f"{self.__class__.__name__}(probs={jnp.exp(self.logits)})"


class PauliChannel1(Channel):
    """Single-qubit Pauli error channel."""

    def __init__(self, px: float, py: float, pz: float, key: Array):
        """Initialize channel with X, Y, Z error probabilities."""
        self.num_bits = 2
        self._key = key
        probs = jnp.array([1 - px - py - pz, pz, px, py])
        self.logits = jnp.log(probs)

    def sample(self, num_samples: int = 1) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.categorical(subkey, self.logits, shape=(num_samples,))
        bits = ((samples[:, None] >> jnp.arange(2)) & 1).astype(jnp.uint8)
        return bits


class Depolarize1(PauliChannel1):
    """Single-qubit depolarizing channel."""

    def __init__(self, p: float, key: Array):
        """Initialize with total depolarizing probability p."""
        super().__init__(p / 3, p / 3, p / 3, key=key)


class PauliChannel2(Channel):
    """Two-qubit Pauli error channel."""

    def __init__(
        self,
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
        key: Array,
    ):
        """Initialize with probabilities for all 15 two-qubit Pauli errors."""
        self._key = key
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
        self.num_bits = 4
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
        self.logits = jnp.log(probs)

    def sample(self, num_samples: int = 1) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.categorical(subkey, self.logits, shape=(num_samples,))
        bits = ((samples[:, None] >> jnp.arange(4)) & 1).astype(jnp.uint8)
        return bits


class Depolarize2(PauliChannel2):
    """Two-qubit depolarizing channel."""

    def __init__(self, p: float, key: Array):
        """Initialize with total depolarizing probability p."""
        super().__init__(
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
            key=key,
        )


class Error(Channel):
    """Single bit error channel used to sample X/Y/Z flips."""

    def __init__(self, p: float, key: Array):
        """Initialize with error probability p."""
        self.num_bits = 1
        self._key = key
        self.p = p

    def sample(self, num_samples: int = 1) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.bernoulli(subkey, self.p, shape=(num_samples,)).astype(
            jnp.uint8
        )
        return samples[:, None]


@dataclass
class ErrorSpec:
    """Specification of an error channel.

    Used during graph building to record what errors exist. Actual Channel
    objects are created later when a sampler is compiled with a key.
    """

    error_type: type[Channel]
    params: tuple[float, ...]

    def create_channel(self, key: Array) -> Channel:
        """Create the actual Channel with the given random key."""
        return self.error_type(*self.params, key=key)


def create_channels_from_specs(specs: list[ErrorSpec], key: Array) -> list[Channel]:
    """Create Channel objects from ErrorSpecs with the given random key."""
    channels = []
    for spec in specs:
        key, subkey = jax.random.split(key)
        channels.append(spec.create_channel(subkey))
    return channels


class ChannelSampler:
    """Samples from multiple error channels and transforms to a reduced basis.

    This class combines multiple error channels (each producing error bits e0, e1, ...)
    and applies a linear transformation over GF(2) to convert samples from the original
    "e" basis to a reduced "f" basis.

    f_i = error_transform_ij * e_j mod 2

    Channels whose variables don't appear in the transform are automatically filtered
    out to avoid unnecessary sampling.

    Attributes:
        error_channels: Filtered list of channels that contribute to the transform.
        error_transform: Matrix of shape (num_e, num_f) for basis conversion.

    Example:
        >>> channels = [Error(0.1, key1), Error(0.2, key2)]  # produces e0, e1
        >>> transform = {"f0": {"e0", "e1"}}  # f0 = e0 XOR e1
        >>> sampler = ChannelSampler(channels, transform)
        >>> samples = sampler.sample(1000)  # shape (1000, 1)
    """

    def __init__(
        self,
        error_channels: list[Channel],
        error_transform: dict[str, set[str]],
    ):
        """Initialize the sampler with error channels and a basis transformation.

        Args:
            error_channels: List of channels. Channel i produces error bits starting
                at index sum(channels[0:i].num_bits). For example, if channels have
                num_bits [2, 1, 2], they produce variables [e0,e1], [e2], [e3,e4].
            error_transform: Mapping from new basis variables to sets of original
                variables. Each new variable f_i is the XOR of its associated e
                variables. E.g., {"f0": {"e1", "e3"}, "f1": {"e2"}} means
                f0 = e1 XOR e3 and f1 = e2.
        """
        from itertools import count

        counter = count()
        channel_evars: list[list[str]] = [
            [f"e{next(counter)}" for _ in range(ch.num_bits)] for ch in error_channels
        ]

        # Filter to channels whose variables are used
        used_evars = set().union(*error_transform.values())
        filtered = [
            (ch, evars)
            for ch, evars in zip(error_channels, channel_evars)
            if set(evars) & used_evars
        ]

        self.error_channels = [ch for ch, _ in filtered]
        kept_evars = [evar for _, evars in filtered for evar in evars]
        e2idx = {evar: i for i, evar in enumerate(kept_evars)}

        # Build transformation matrix: shape (num_e_vars, num_f_vars)
        transform = np.zeros((len(e2idx), len(error_transform)), dtype=np.uint8)
        for col, e_vars in enumerate(error_transform.values()):
            transform[[e2idx[evar] for evar in e_vars], col] = 1

        self.error_transform = jnp.array(transform).astype(jnp.float32)

    def sample(self, num_samples: int = 1) -> jax.Array:
        """Sample from all error channels and transform to new error basis."""
        if len(self.error_channels) == 0:
            return jnp.zeros((num_samples, 0), dtype=jnp.bool)
        samples = []
        for channel in self.error_channels:
            samples.append(channel.sample(num_samples))
        total_samples = jnp.concatenate(samples, axis=1).astype(jnp.float32)
        return (total_samples @ self.error_transform % 2).astype(jnp.bool)
