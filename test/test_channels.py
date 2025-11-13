import jax
import jax.numpy as jnp

from tsim.channels import Error, PauliChannel1, PauliChannel2


def test_error():
    error = Error(0.1, jax.random.key(0))
    samples = error.sample(1000)
    assert jnp.count_nonzero(samples) == 132


def test_pauli_channel_1():
    p = PauliChannel1(1, 0, 0, jax.random.key(0))
    samples = p.sample(10)
    x = jnp.array([[0, 1]] * 10)
    assert jnp.all(x == samples)

    p = PauliChannel1(0, 1, 0, jax.random.key(0))
    samples = p.sample(10)
    y = jnp.array([[1, 1]] * 10)
    assert jnp.all(y == samples)

    p = PauliChannel1(0, 0, 1, jax.random.key(0))
    samples = p.sample(10)
    z = jnp.array([[1, 0]] * 10)
    assert jnp.all(z == samples)

    p = PauliChannel1(0.5, 0, 0.5, jax.random.key(0))
    samples = p.sample(10)
    assert jnp.all(jnp.count_nonzero(samples, axis=1))


def test_pauli_channel_2():
    pauli2bits = {"i": [0, 0], "x": [0, 1], "y": [1, 1], "z": [1, 0]}
    paulis2bits = {
        f"p{p1}{p2}": jnp.array(pauli2bits[p1] + pauli2bits[p2])
        for p1 in pauli2bits
        for p2 in pauli2bits
    }
    for kw_name, bits in paulis2bits.items():
        print(kw_name, bits)
        kwargs = {}
        for kw in paulis2bits:
            kwargs[kw] = 0
        kwargs[kw_name] = 1
        kwargs.pop("pii")  # pii is not an arg since it is always the leftover prob
        p = PauliChannel2(**kwargs, key=jax.random.key(0))
        samples = p.sample(10)
        assert jnp.all(samples == bits)
