"""Tests for the float scalar backends at the term, evaluate and sampler level."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import stim

import tsim
import tsim.sampler
from tsim.compile.compile import compile_scalar_graphs
from tsim.compile.evaluate import evaluate
from tsim.compile.terms import HalfPiPhases, NodePhases, PhasePairs, PiProducts
from tsim.core.exact_scalar import ExactScalarArray
from tsim.core.float_scalar import Float32ScalarArray, Float64ScalarArray
from tsim.core.graph import prepare_graph
from tsim.core.scalar import scalar_type

X64 = bool(jax.config.read("jax_enable_x64"))
FLOAT_PRECISIONS = ["float32"] + (["float64"] if X64 else [])
BACKENDS = [Float32ScalarArray] + ([Float64ScalarArray] if X64 else [])


def _tol(cls):
    return 1e-5 if cls is Float32ScalarArray else 1e-12


@pytest.fixture
def modules():
    rng = np.random.default_rng(1)
    G, T, P = 4, 6, 7

    def bits(*shape):
        return jnp.asarray(rng.integers(0, 2, size=shape).astype(np.uint8))

    node = NodePhases(
        jnp.asarray(rng.integers(0, 8, (G, T)).astype(np.uint8)),
        bits(G, T, P),
        jnp.asarray(np.array([T, T - 1, 2, 0], dtype=np.int32)),
    )
    half = HalfPiPhases(
        jnp.asarray(rng.choice([0, 2, 4, 6], (G, T)).astype(np.uint8)), bits(G, T, P)
    )
    pi = PiProducts(bits(G, T), bits(G, T, P), bits(G, T), bits(G, T, P))
    pairs = PhasePairs(
        jnp.asarray(rng.integers(0, 8, (G, T)).astype(np.uint8)),
        bits(G, T, P),
        jnp.asarray(rng.integers(0, 8, (G, T)).astype(np.uint8)),
        bits(G, T, P),
        jnp.asarray(np.array([T, 3, 1, 0], dtype=np.int32)),
    )
    pv = bits(9, P)
    return [node, half, pi, pairs], pv


@pytest.mark.parametrize("cls", BACKENDS)
def test_terms_match_exact_backend(cls, modules):
    mods, pv = modules
    for mod in mods:
        expected = np.asarray(mod.evaluate(pv, ExactScalarArray).to_complex())
        got = np.asarray(mod.evaluate(pv, cls).to_complex())
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=_tol(cls), atol=_tol(cls))


def test_pi_products_sign_is_minus_one(modules):
    """(-1)^1 must be -1 for every backend (regression: unsigned wrap gave 255 / 2^32-1)."""
    P = 3
    pi = PiProducts(
        jnp.ones((1, 1), dtype=jnp.uint8),
        jnp.zeros((1, 1, P), dtype=jnp.uint8),
        jnp.ones((1, 1), dtype=jnp.uint8),
        jnp.zeros((1, 1, P), dtype=jnp.uint8),
    )
    pv = jnp.zeros((2, P), dtype=jnp.uint8)
    for cls in [ExactScalarArray, *BACKENDS]:
        np.testing.assert_allclose(np.asarray(pi.evaluate(pv, cls).to_complex()), -1.0)


def _small_t_circuit():
    return tsim.Circuit("""
        H 0 1 2
        T 0
        CX 0 1
        T 1
        CX 1 2
        T 2
        T_DAG 0
        H 0 1 2
        M 0 1 2
        """)


@pytest.mark.parametrize("precision", FLOAT_PRECISIONS)
def test_evaluate_matches_exact(precision):
    """The float backends agree with the exact evaluator on compiled graphs."""
    prepared = prepare_graph(_small_t_circuit(), sample_detectors=False)
    from tsim.compile.pipeline import compile_program

    program = compile_program(prepared, mode="sequential")
    comp = max(program.components, key=lambda c: len(c.output_indices))
    rng = np.random.default_rng(0)
    tol = 1e-5 if precision == "float32" else 1e-12
    for circ in comp.compiled_scalar_graphs:
        pv = jnp.asarray(rng.integers(0, 2, size=(16, circ.n_params)).astype(np.uint8))
        exact = np.asarray(evaluate(circ, pv))
        got = np.asarray(evaluate(dataclasses.replace(circ, precision=precision), pv))
        assert got.dtype == scalar_type(precision).complex_dtype()
        np.testing.assert_allclose(got, exact, rtol=tol, atol=tol)


@pytest.mark.parametrize("precision", FLOAT_PRECISIONS)
def test_sampler_precision_statistics(precision):
    """Float-backend samplers reproduce the exact sampler's marginals."""
    circuit = _small_t_circuit()
    shots = 20000
    exact = circuit.compile_sampler(seed=0).sample(shots)
    approx = circuit.compile_sampler(seed=0, precision=precision).sample(shots)
    p_exact = exact.mean(axis=0)
    p_approx = approx.mean(axis=0)
    se = np.sqrt(p_exact * (1 - p_exact) / shots) * np.sqrt(2) + 1e-9
    assert np.all(np.abs(p_exact - p_approx) < 5 * se)


@pytest.mark.parametrize("precision", FLOAT_PRECISIONS)
def test_detector_sampler_precision(precision):
    circuit = tsim.Circuit.from_stim_program(
        stim.Circuit.generated(
            "repetition_code:memory",
            distance=3,
            rounds=2,
            after_clifford_depolarization=0.05,
        )
    )
    circuit = tsim.Circuit(
        str(circuit.stim_circuit).replace("TICK", "TICK") + "\nT 0\nDETECTOR rec[-1]"
    )
    sampler = circuit.compile_detector_sampler(seed=3, precision=precision)
    assert f"precision={precision}" in repr(sampler)
    dets = sampler.sample(1000)
    assert dets.shape[0] == 1000
    assert dets.dtype == np.bool_


def test_probability_estimator_precision():
    circuit = _small_t_circuit()
    state = np.array([0, 1, 1], dtype=np.uint8)
    exact = tsim.sampler.CompiledStateProbs(circuit, seed=0).probability_of(
        state, batch_size=4
    )
    approx = tsim.sampler.CompiledStateProbs(
        circuit, seed=0, precision="float32"
    ).probability_of(state, batch_size=4)
    np.testing.assert_allclose(approx, exact, rtol=1e-4, atol=1e-6)


def test_invalid_precision():
    with pytest.raises(ValueError, match="precision"):
        _small_t_circuit().compile_sampler(precision="float16")  # type: ignore[arg-type]
    if not X64:
        with pytest.raises(RuntimeError, match="x64"):
            _small_t_circuit().compile_sampler(precision="float64")


def test_compile_scalar_graphs_default_precision():
    prepared = prepare_graph(_small_t_circuit(), sample_detectors=False)
    from tsim.compile.pipeline import compile_program

    program = compile_program(prepared, mode="joint", precision="float32")
    for comp in program.components:
        for circ in comp.compiled_scalar_graphs:
            assert circ.precision == "float32"
    assert compile_scalar_graphs([], []).precision == "exact"
