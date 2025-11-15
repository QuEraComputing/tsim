import abc

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class Channel(abc.ABC):
    """Abstract base class for quantum error channels."""

    logits: jnp.ndarray

    @abc.abstractmethod
    def sample(self, num_samples: int = 1):
        """Sample errors from the channel.

        Args:
            num_samples: Number of samples to draw from the channel.

        Returns:
            A jax.numpy array of shape (num_samples, num_qubits) containing the
            sampled errors.

        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(probs={jnp.exp(self.logits)})"


class PauliChannel1:
    """Single-qubit Pauli error channel."""

    def __init__(self, px: float, py: float, pz: float, key: Array):
        """Initialize channel with X, Y, Z error probabilities."""
        self._key = key
        probs = jnp.array([1 - px - py - pz, pz, px, py])
        self.logits = jnp.log(probs)

    def sample(self, num_samples: int = 1):
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.categorical(subkey, self.logits, shape=(num_samples,))
        bits = ((samples[:, None] >> jnp.arange(2)) & 1).astype(jnp.uint8)
        return bits


class Depolarize1(PauliChannel1):
    """Single-qubit depolarizing channel."""

    def __init__(self, p: float, key: Array):
        """Initialize with total depolarizing probability p."""
        super().__init__(p / 3, p / 3, p / 3, key=key)


class PauliChannel2:
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

    def sample(self, num_samples: int = 1):
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
        self._key = key
        self.p = p

    def sample(self, num_samples: int = 1):  # type: ignore[override]
        self._key, subkey = jax.random.split(self._key)
        samples = jax.random.bernoulli(subkey, self.p, shape=(num_samples,)).astype(
            jnp.uint8
        )
        return samples[:, None]


class ErrorSampler:
    """Samples from multiple error channels simultaneously."""

    def __init__(
        self,
        error_channels: list[Channel],
        error_transform: np.ndarray,
    ):
        """Initialize with a list of error channels."""
        self.error_channels = error_channels
        self.error_transform = jnp.array(error_transform.T)

    def sample(self, num_samples: int = 1) -> jax.Array:
        """Sample from all error channels and transform to new error basis."""
        if len(self.error_channels) == 0:
            return jnp.zeros((num_samples, 0), dtype=jnp.uint8)
        samples = []
        for channel in self.error_channels:
            samples.append(channel.sample(num_samples))
        total_samples = jnp.concatenate(samples, axis=1)
        return total_samples @ self.error_transform % 2
