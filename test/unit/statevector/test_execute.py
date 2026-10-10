from test.helpers.statevector import (
    CIRCUITS,
    assert_matches_distribution,
    f_patterns,
    pauli_matrix,
    plan_distribution,
    quantum_components,
    reference_measure,
)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tsim.statevector.execute as execute
from tsim.statevector.execute import (
    initial_state,
    pack_bits,
    run,
    sample_statevector_component,
)
from tsim.statevector.heisenberg import ComponentLayout
from tsim.statevector.plan import (
    AffineRow,
    MeasureActive,
    MeasureDormant,
    Plan,
    Record,
    Rotate,
)
from tsim.statevector.program import lower, plan_component, resolve_dtype


def _component(
    actions, *, width, num_f=1, num_branches=0, num_outputs=0, dtype="complex64"
):
    planned = Plan(
        width=width,
        num_f=num_f,
        num_branches=num_branches,
        num_outputs=num_outputs,
        actions=tuple(actions),
    )
    layout = ComponentLayout(tuple(range(num_f)), {}, tuple(range(num_outputs)))
    return lower(planned, layout, dtype)


def _random_states(lanes, width, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(lanes, 2**width)) + 1j * rng.normal(size=(lanes, 2**width))
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def test_pack_bits():
    bits = np.zeros((2, 40), dtype=bool)
    bits[0, [0, 31, 32, 39]] = True
    bits[1, 5] = True
    words = np.asarray(pack_bits(jnp.asarray(bits), 2))
    assert words.dtype == np.uint32
    assert words.tolist() == [[1 | (1 << 31), 1 | (1 << 7)], [1 << 5, 0]]


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("x, z", [(0, 0b101), (0b011, 0), (0b110, 0b011), (1, 1)])
def test_rotations_match_dense_matrices(x, z, dtype):
    with jax.enable_x64(dtype == "complex128"):
        sign = AffineRow(False, frozenset({0}))
        component = _component([Rotate(x, z, 0.37, sign)], width=3, dtype=dtype)
        psi = _random_states(2, 3)
        f_bits = jnp.array([[False], [True]])
        amps = jnp.asarray(psi, jnp.dtype(dtype))
        state = initial_state(component, f_bits)._replace(amps=amps)
        out = run(component, state, jax.random.key(0), 0, 1)
        assert out.amps.dtype == jnp.dtype(dtype)
        pauli = pauli_matrix(x, z, 3)
        atol = 1e-5 if dtype == "complex64" else 1e-12
        for lane, negate in ((0, 1), (1, -1)):
            rotation = np.cos(0.37) * np.eye(8) - 1j * np.sin(0.37) * negate * pauli
            np.testing.assert_allclose(
                np.asarray(out.amps[lane]), rotation @ psi[lane], atol=atol
            )


@pytest.mark.parametrize(
    "x, z, pivot", [(0b011, 0b110, 0), (0b010, 0b011, 1), (0, 0b101, 2), (0, 0b011, 0)]
)
def test_measurements_match_the_reference_kernel(monkeypatch, x, z, pivot):
    action = MeasureActive(
        x, z, pivot, branch=1, sign=AffineRow(False, frozenset({0})), slot=0
    )
    component = _component([action], width=3, num_branches=1, num_outputs=1)
    outcomes = jnp.array([False, True, True])
    monkeypatch.setattr(execute, "choose_outcome", lambda p1, count, key: outcomes)
    psi = _random_states(3, 3)
    f_bits = jnp.array([[False], [False], [True]])
    state = initial_state(component, f_bits)._replace(
        amps=jnp.asarray(psi, jnp.complex64)
    )
    out = run(component, state, jax.random.key(0), 0, 1)
    for lane, outcome in enumerate([0, 1, 1]):
        expected, _ = reference_measure(psi[lane], x, z, pivot, outcome)
        np.testing.assert_allclose(np.asarray(out.amps[lane]), expected, atol=1e-5)
    assert np.asarray(out.records[:, 0]).tolist() == [False, True, False]
    assert ((np.asarray(out.symbols[:, 0]) >> 1) & 1).tolist() == [0, 1, 1]


def test_dormant_measurements_and_records_write_outcome_xor_sign(monkeypatch):
    component = _component(
        [
            MeasureDormant(branch=1, sign=AffineRow(True, frozenset()), slot=0),
            Record(sign=AffineRow(False, frozenset({0, 1})), slot=1),
        ],
        width=0,
        num_branches=1,
        num_outputs=2,
    )
    monkeypatch.setattr(
        execute, "choose_outcome", lambda p1, count, key: jnp.array([False, True])
    )
    state = initial_state(component, jnp.array([[True], [True]]))
    out = run(component, state, jax.random.key(0), 0, 2)
    assert np.asarray(out.records[:, :2]).tolist() == [[True, True], [False, False]]


def test_a_component_without_actions_samples_no_outputs():
    component = _component([], width=0, num_f=0)
    assert component.num_actions == 0
    state = initial_state(component, jnp.zeros((3, 0), dtype=bool))
    out = run(component, state, jax.random.key(0), 0, component.num_actions)
    for got, initial in zip(out, state, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(initial))
    samples, _, deviation = sample_statevector_component(
        component, jnp.zeros((3, 0), dtype=jnp.uint8), jax.random.key(0)
    )
    assert samples.shape == (3, 0)
    assert float(deviation) == 0.0


@pytest.mark.parametrize(
    "name",
    [
        "t_ghz",
        "midcircuit_feedforward",
        "two_components",
        "clifford_component",
        "rotations",
    ],
)
def test_sampling_matches_the_plan_distribution(name):
    shots = 20_000
    components, layout = quantum_components(CIRCUITS[name], sample_detectors=True)
    for component in components:
        planned, component_layout = plan_component(component, layout)
        compiled = lower(planned, component_layout, "complex64")
        num_columns = max(component_layout.f_selection, default=-1) + 1
        for f in f_patterns(planned.num_f, limit=2):
            f_params = np.zeros((shots, num_columns), dtype=np.uint8)
            if component_layout.f_selection:
                f_params[:, list(component_layout.f_selection)] = f
            samples, _, deviation = sample_statevector_component(
                compiled, jnp.asarray(f_params), jax.random.key(7)
            )
            assert float(deviation) < 1e-4
            assert_matches_distribution(
                np.asarray(samples), plan_distribution(planned, f)
            )


def test_complex128_amplitudes():
    with jax.enable_x64(True):
        (component,), layout = quantum_components(
            CIRCUITS["t_ghz"], sample_detectors=True
        )
        planned, component_layout = plan_component(component, layout)
        compiled = lower(planned, component_layout, resolve_dtype(None))
        assert compiled.dtype == "complex128"
        state = initial_state(compiled, jnp.zeros((4, planned.num_f), dtype=bool))
        assert state.amps.dtype == jnp.complex128
        out = run(compiled, state, jax.random.key(0), 0, compiled.num_actions)
        assert out.amps.dtype == jnp.complex128


def _count_outcomes(p1, dtype, draws, chunk=2**22):
    @jax.jit
    def ones_in_chunk(key):
        probabilities = jnp.full(chunk, p1, dtype=dtype)
        count = jnp.ones(chunk, dtype=jnp.int32)
        return jnp.sum(execute.choose_outcome(probabilities, count, key))

    return sum(
        int(ones_in_chunk(jax.random.key(seed))) for seed in range(draws // chunk)
    )


@pytest.mark.parametrize("p1", [0.0, 1e-15])
def test_choose_outcome_never_returns_true_below_its_resolution(p1):
    assert _count_outcomes(p1, jnp.float32, 2**26) == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_choose_outcome_always_returns_true_for_p1_one(dtype):
    with jax.enable_x64(dtype == "float64"):
        assert _count_outcomes(1.0, jnp.dtype(dtype), 2**22) == 2**22


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p1", [0.5, 0.3, 1e-3, 1e-6])
def test_choose_outcome_frequency_matches_p1(p1, dtype):
    draws = 2**24
    with jax.enable_x64(dtype == "float64"):
        ones = _count_outcomes(p1, jnp.dtype(dtype), draws)
    expected = draws * p1
    assert abs(ones - expected) <= 5 * np.sqrt(draws * p1 * (1 - p1)) + 1
