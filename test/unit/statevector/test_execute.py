from test.helpers.statevector import (
    CIRCUITS,
    apply_pauli,
    assert_matches_distribution,
    f_patterns,
    output_indices,
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
    outputs,
    pack_bits,
    run,
    sample_statevector_component,
)
from tsim.statevector.heisenberg import ComponentLayout
from tsim.statevector.plan import AffineRow, MeasureActive, MeasureDormant, Plan, Rotate
from tsim.statevector.program import (
    MEASURE,
    MEASURE_DIAG,
    MEASURE_DORMANT,
    compile_statevector_component,
    lower,
    plan_component,
)


def _component(actions, *, width, num_f=1, outputs=(), dtype="complex64"):
    planned = Plan(
        width=width, num_f=num_f, actions=tuple(actions), outputs=tuple(outputs)
    )
    layout = ComponentLayout(tuple(range(num_f)), {}, tuple(range(len(outputs))))
    return lower(planned, layout, dtype)


def _random_states(lanes, width, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(lanes, 2**width)) + 1j * rng.normal(size=(lanes, 2**width))
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def _measuring(x, z, pivot, psi, f_bits=None):
    """Return a component that measures ``x``, ``z``, and lanes in the states ``psi``.

    Its one output is the outcome XOR the f-bit.
    """
    component = _component(
        [MeasureActive(x, z, pivot, branch=1)],
        width=3,
        outputs=[AffineRow(False, frozenset({0, 1}))],
    )
    if f_bits is None:
        f_bits = jnp.zeros((len(psi), 1), dtype=bool)
    amps = jnp.asarray(psi, jnp.complex64)
    return component, initial_state(component, f_bits)._replace(amps=amps)


def _only_component(name):
    (component,), layout = quantum_components(CIRCUITS[name], sample_detectors=True)
    return component, output_indices(component), layout


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


_MEASUREMENTS = [
    (0b011, 0b110, 0),
    (0b010, 0b011, 1),
    (0b110, 0b010, 2),
    (0, 0b101, 2),
    (0, 0b011, 0),
]


@pytest.mark.parametrize("x, z, pivot", _MEASUREMENTS)
def test_measurements_match_the_reference_kernel(monkeypatch, x, z, pivot):
    psi = _random_states(3, 3)
    f_bits = jnp.array([[False], [False], [True]])
    component, state = _measuring(x, z, pivot, psi, f_bits)
    outcomes = jnp.array([False, True, True])
    monkeypatch.setattr(execute, "choose_outcome", lambda p1, count, key: outcomes)
    out = run(component, state, jax.random.key(0), 0, 1)
    for lane, outcome in enumerate([0, 1, 1]):
        expected, _ = reference_measure(psi[lane], x, z, pivot, outcome)
        np.testing.assert_allclose(np.asarray(out.amps[lane]), expected, atol=1e-5)
    assert np.asarray(outputs(component, out)[:, 0]).tolist() == [False, True, False]
    assert ((np.asarray(out.symbols[:, 0]) >> 1) & 1).tolist() == [0, 1, 1]


def test_outputs_are_parities_of_the_symbols(monkeypatch):
    component = _component(
        [MeasureDormant(branch=1)],
        width=0,
        outputs=[AffineRow(True, frozenset({1})), AffineRow(False, frozenset({0, 1}))],
    )
    monkeypatch.setattr(
        execute, "choose_outcome", lambda p1, count, key: jnp.array([False, True])
    )
    state = initial_state(component, jnp.array([[True], [True]]))
    out = run(component, state, jax.random.key(0), 0, 1)
    assert np.asarray(outputs(component, out)).tolist() == [
        [True, True],
        [False, False],
    ]


def test_outputs_read_symbols_beyond_the_first_word(monkeypatch):
    component = _component(
        [MeasureDormant(branch=40)],
        width=0,
        num_f=40,
        outputs=[
            AffineRow(False, frozenset({40})),
            AffineRow(True, frozenset({3, 40})),
        ],
    )
    assert component.num_words == 2
    monkeypatch.setattr(
        execute, "choose_outcome", lambda p1, count, key: jnp.array([False, True])
    )
    f_bits = np.zeros((2, 40), dtype=bool)
    f_bits[1, 3] = True
    state = initial_state(component, jnp.asarray(f_bits))
    out = run(component, state, jax.random.key(0), 0, 1)
    assert np.asarray(outputs(component, out)).tolist() == [[False, True], [True, True]]


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
        indices = output_indices(component)
        planned, component_layout = plan_component(component, indices, layout)
        f_selection = component_layout.f_selection
        compiled = lower(planned, component_layout, "complex64")
        num_columns = max(f_selection, default=-1) + 1
        for f in f_patterns(planned.num_f, limit=2):
            f_params = np.zeros((shots, num_columns), dtype=np.uint8)
            if f_selection:
                f_params[:, list(f_selection)] = f
            samples, _, deviation = sample_statevector_component(
                compiled, jnp.asarray(f_params), jax.random.key(7)
            )
            assert float(deviation) < 1e-4
            assert_matches_distribution(
                np.asarray(samples), plan_distribution(planned, f)
            )


@pytest.mark.parametrize("x64", [False, True])
def test_amplitudes_are_complex128_with_64_bit_types(x64):
    dtype = "complex128" if x64 else "complex64"
    with jax.enable_x64(x64):
        compiled = compile_statevector_component(*_only_component("t_ghz"))
        assert compiled.dtype == dtype
        num_f = compiled.f_selection.shape[0]
        state = initial_state(compiled, jnp.zeros((4, num_f), dtype=bool))
        out = run(compiled, state, jax.random.key(0), 0, compiled.num_actions)
        assert out.amps.dtype == jnp.dtype(dtype)


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


def _projected(vectors, x, z, eigenvalue):
    projected = np.array([v + eigenvalue * apply_pauli(v, x, z) for v in vectors])
    return projected / np.linalg.norm(projected, axis=1, keepdims=True)


def _run_recording_p1(monkeypatch, component, state):
    recorded = []
    choose = execute.choose_outcome

    def recording(p1, count, key):
        recorded.append(np.asarray(p1, dtype=np.float64))
        return choose(p1, count, key)

    monkeypatch.setattr(execute, "choose_outcome", recording)
    with jax.disable_jit():
        out = run(component, state, jax.random.key(3), 0, 1)
    (p1,) = recorded
    return p1, out


@pytest.mark.parametrize("eigenvalue", [1, -1])
@pytest.mark.parametrize("x, z, pivot", _MEASUREMENTS)
def test_measuring_an_eigenstate_never_samples_the_impossible_outcome(
    monkeypatch, x, z, pivot, eigenvalue
):
    lanes = 4096
    # NOTE: the noise is of the size rotations leave in complex64 amplitudes.
    rounding = 1e-7 * _random_states(lanes, 3, seed=1)
    psi = _projected(_random_states(lanes, 3), x, z, eigenvalue) + rounding
    component, state = _measuring(x, z, pivot, psi)
    p1, out = _run_recording_p1(monkeypatch, component, state)
    impossible = p1 if eigenvalue == 1 else 1 - p1
    assert impossible.max() < 1e-12
    assert np.all(np.asarray(outputs(component, out)[:, 0]) == (eigenvalue == -1))


@pytest.mark.parametrize("unlikely", [0, 1])
@pytest.mark.parametrize("x, z, pivot", _MEASUREMENTS)
def test_an_unlikely_outcome_leaves_a_normalized_state(
    monkeypatch, x, z, pivot, unlikely
):
    lanes = 64
    vectors = _random_states(lanes, 3)
    likely = _projected(vectors, x, z, (-1) ** (1 - unlikely))
    rare = _projected(vectors, x, z, (-1) ** unlikely)
    p = np.logspace(-5, -3, lanes)[:, None]
    component, state = _measuring(
        x, z, pivot, np.sqrt(1 - p) * likely + np.sqrt(p) * rare
    )
    outcomes = jnp.full(lanes, bool(unlikely))
    monkeypatch.setattr(execute, "choose_outcome", lambda p1, count, key: outcomes)
    out = run(component, state, jax.random.key(0), 0, 1)
    norms = np.linalg.norm(np.asarray(out.amps, dtype=np.complex128), axis=1) ** 2
    assert np.abs(norms - 1).max() < 1e-5


def test_running_a_tape_in_two_parts_equals_running_it_whole():
    compiled = compile_statevector_component(*_only_component("midcircuit_feedforward"))
    lanes, split, stop = 512, 4, compiled.num_actions
    kinds = np.asarray(compiled.kind)
    measurements = np.isin(kinds, [MEASURE_DIAG, MEASURE, MEASURE_DORMANT])
    assert measurements[:split].any() and measurements[split:].any()
    rng = np.random.default_rng(0)
    num_f = compiled.f_selection.shape[0]
    f_bits = jnp.asarray(rng.integers(0, 2, (lanes, num_f)).astype(bool))
    state = initial_state(compiled, f_bits)
    key = jax.random.key(5)
    whole = run(compiled, state, key, 0, stop)
    parts = run(compiled, run(compiled, state, key, 0, split), key, split, stop)
    for got, expected in zip(parts, whole, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
