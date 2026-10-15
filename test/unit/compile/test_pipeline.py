"""compile_program on zxparam: plugging normalization, the direct and constant paths."""

import itertools
from fractions import Fraction
from test.helpers.gate_matrices import CNOT, H, T

import jax.numpy as jnp
import numpy as np
import pytest
from zxparam import Domain, EdgeType, Graph, Phase, VertexType, decompose

import tsim.sampler as sampler_module
from tsim.circuit import Circuit
from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import (
    _plugged,
    classify_constant,
    classify_direct,
    compile_program,
)
from tsim.core.graph import prepare_graph
from tsim.core.params import ParamLayout
from tsim.sampler import CompiledStateProbs

I2 = np.eye(2)


def _abs_value(compiled, bits: list[int]) -> float:
    params = jnp.asarray(np.array([bits], dtype=np.uint8).reshape(1, len(bits)))
    return float(np.abs(np.asarray(evaluate(compiled, params)))[0])


def test_sequential_plugged_graphs_are_consistent_marginals():
    circuit = Circuit("R 0 1\nH 0\nT 0\nH 0\nCNOT 0 1\nM 0 1")
    program = compile_program(
        prepare_graph(circuit, sample_detectors=False), mode="sequential"
    )
    assert any(len(c.output_indices) > 1 for c in program.components)
    for component in program.components:
        graphs = component.compiled_scalar_graphs
        for k in range(len(graphs) - 1):
            for prefix in itertools.product((0, 1), repeat=k):
                parent = _abs_value(graphs[k], list(prefix))
                children = sum(_abs_value(graphs[k + 1], [*prefix, b]) for b in (0, 1))
                np.testing.assert_allclose(children, parent, rtol=1e-4, atol=1e-6)


def test_joint_probabilities_match_the_state_vector():
    circuit = Circuit("R 0 1\nH 0\nT 0\nH 0\nCNOT 0 1\nM 0 1")
    program = compile_program(
        prepare_graph(circuit, sample_detectors=False), mode="joint"
    )
    assert any(len(c.output_indices) > 1 for c in program.components)
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1
    for u in (np.kron(H, I2), np.kron(T, I2), np.kron(H, I2), CNOT):
        psi = u @ psi
    probs = CompiledStateProbs(circuit, seed=0)
    for index, bits in enumerate(itertools.product((False, True), repeat=2)):
        got = probs.probability_of(np.array(bits), batch_size=1)[0]
        np.testing.assert_allclose(got, abs(psi[index]) ** 2, atol=1e-5)


def _outcome_probabilities(text: str) -> list[float]:
    probs = CompiledStateProbs(Circuit(text), seed=0)
    return [
        float(probs.probability_of(np.array([m]), batch_size=1)[0])
        for m in (False, True)
    ]


# NOTE: error probabilities of 0 and 1 make every sampled error bit
# deterministic, so the probabilities below are conditioned on the error bits.


@pytest.mark.parametrize("e0, e1", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_z_errors_around_a_z_rotation_condition_the_x_outcome(e0, e1):
    # Z^e1 R_Z(θ) Z^e0 |+> measured in the X basis: P(+) = (1 ± cos πθ) / 2,
    # with + for even e0 + e1 (both errors map to one basis bit).
    text = f"RX 0\nZ_ERROR({e0}) 0\nR_Z(0.17) 0\nZ_ERROR({e1}) 0\nH 0\nM 0"
    p_plus = (1 + (-1) ** (e0 + e1) * np.cos(0.17 * np.pi)) / 2
    np.testing.assert_allclose(
        _outcome_probabilities(text), [p_plus, 1 - p_plus], atol=1e-5
    )


@pytest.mark.parametrize("e", [0, 1])
def test_x_error_before_a_u3_conditions_the_x_outcome(e):
    # U3(θ, φ, λ) X^e |0> measured in the X basis: P(+) = (1 ± sin πθ cos πφ) / 2,
    # with + for e = 0; λ is a global phase of each branch.
    text = f"R 0\nX_ERROR({e}) 0\nU3(0.37, 0.2, 0.6) 0\nH 0\nM 0"
    p_plus = (1 + (-1) ** e * np.sin(0.37 * np.pi) * np.cos(0.2 * np.pi)) / 2
    np.testing.assert_allclose(
        _outcome_probabilities(text), [p_plus, 1 - p_plus], atol=1e-5
    )


def test_single_error_detectors_take_the_direct_path():
    circuit = Circuit(
        "R 0 1\nX_ERROR(0.1) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]"
    )
    program = compile_program(
        prepare_graph(circuit, sample_detectors=True), mode="sequential"
    )
    assert program.components == ()
    assert sorted(np.asarray(program.direct_f_indices).tolist()) == [0, 1]
    assert not np.asarray(program.direct_flips).any()


# Detector 0 is random (H T H |0>); detectors 1-3 are deterministic with values
# 0, 1 and 1, the last one through a T · T_DAG pair that cancels.
NOISELESS_DETERMINISTIC_DETECTORS = """
R 0 1 2 3
H 0
T 0
H 0
X 2
H 3
T 3
T_DAG 3
H 3
X 3
M 0 1 2 3
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
"""


def test_deterministic_detectors_of_a_noiseless_circuit_take_the_constant_path():
    circuit = Circuit(NOISELESS_DETERMINISTIC_DETECTORS)
    program = compile_program(
        prepare_graph(circuit, sample_detectors=True), mode="sequential"
    )
    assert len(program.components) == 1
    assert program.components[0].output_indices == (0,)
    assert len(program.direct_f_indices) == 0
    assert np.asarray(program.constant_bits).tolist() == [False, True, True]
    assert np.asarray(program.output_order).tolist() == [1, 2, 3, 0]
    assert program.output_reindex is not None

    samples = circuit.compile_detector_sampler(seed=0).sample(200, batch_size=50)
    assert samples.shape == (200, 4)
    assert (samples[:, 1:] == [False, True, True]).all()
    assert samples[:, 0].any() and not samples[:, 0].all()


def test_all_constant_detectors_skip_jax(monkeypatch):
    circuit = Circuit(
        "R 0 1\nX 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]"
    )
    sampler = circuit.compile_detector_sampler(seed=0)
    assert sampler._program.components == ()

    def fail(*args, **kwargs):
        raise AssertionError("sample_program must not be called")

    monkeypatch.setattr(sampler_module, "sample_program", fail)
    dets, obs = sampler.sample(10, separate_observables=True)
    assert (dets == [False, True]).all()
    assert obs.all()
    dets, obs = sampler.sample(
        10,
        separate_observables=True,
        use_detector_reference_sample=True,
        use_observable_reference_sample=True,
    )
    assert not dets.any() and not obs.any()


@pytest.mark.parametrize("with_component", [False, True], ids=["numpy", "jax"])
def test_direct_and_constant_detectors_are_reassembled_in_order(with_component):
    # Detector 0 is constant 1, detector 1 a direct error bit and detector 2
    # constant 0, so the direct output comes first in the concatenated layout;
    # detector 3 (if present) is a compiled component, sampled with JAX.
    text = (
        "R 0 1 2 3\nX 0\nX_ERROR(0.5) 1\nM 0 1 2\n"
        "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]"
    )
    if with_component:
        text += "\nH 3\nM 3\nDETECTOR rec[-1]"
    sampler = Circuit(text).compile_detector_sampler(seed=0)
    program = sampler._program
    assert len(program.components) == int(with_component)
    assert len(program.direct_f_indices) == 1
    assert np.asarray(program.constant_bits).tolist() == [True, False]
    assert program.output_reindex is not None

    samples = sampler.sample(200, batch_size=50)
    assert samples[:, 0].all() and not samples[:, 2].any()
    for column in range(1, samples.shape[1], 2):
        assert samples[:, column].any() and not samples[:, column].all()


def test_constant_outputs_condition_the_joint_probability():
    # Measurement 0 is random (H T H |0>); measurements 1 and 2 are constant.
    circuit = Circuit("R 0 1 2\nH 0\nT 0\nH 0\nX 2\nM 0 1 2")
    probs = CompiledStateProbs(circuit, seed=0)
    assert len(probs._program.constant_bits) == 2
    p_one = np.sin(np.pi / 8) ** 2
    for m0, p in ((False, 1 - p_one), (True, p_one)):
        got = probs.probability_of(np.array([m0, False, True]), batch_size=1)[0]
        np.testing.assert_allclose(got, p, atol=1e-5)
        for wrong in ([m0, True, True], [m0, False, False]):
            assert probs.probability_of(np.array(wrong), batch_size=1)[0] == 0


LAYOUT = ParamLayout(num_errors=3)


def _direct_candidate(
    value: int | Fraction = 0,
    params: tuple[int, ...] = (LAYOUT.basis(1),),
    edge_type: EdgeType = EdgeType.HADAMARD,
    output_domain: Domain = Domain.CLASSICAL,
    spider_type: VertexType = VertexType.Z,
) -> Graph:
    g = Graph()
    out = g.add_vertex(VertexType.BOUNDARY, domain=output_domain)
    g.set_port(out, 0)
    phase = Phase(Fraction(value), frozenset(params))
    spider = g.add_vertex(spider_type, phase, domain=output_domain)
    g.add_edge(out, spider, edge_type)
    return g


@pytest.mark.parametrize("value, flip", [(0, False), (1, True)])
@pytest.mark.parametrize("basis_index", [1, 5])
def test_classify_direct_reads_the_basis_index_and_flip(value, flip, basis_index):
    component = _direct_candidate(value=value, params=(LAYOUT.basis(basis_index),))
    assert classify_direct(component, LAYOUT) == (basis_index, flip)


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(_direct_candidate(params=(LAYOUT.error(0),)), id="error-param"),
        pytest.param(_direct_candidate(params=(LAYOUT.output(0),)), id="output-param"),
        pytest.param(
            _direct_candidate(params=(LAYOUT.basis(0), LAYOUT.basis(1))),
            id="two-params",
        ),
        pytest.param(_direct_candidate(params=()), id="no-param"),
        pytest.param(_direct_candidate(value=1, params=()), id="no-param-flip"),
        pytest.param(_direct_candidate(edge_type=EdgeType.SIMPLE), id="simple-edge"),
        pytest.param(_direct_candidate(spider_type=VertexType.X), id="x-spider"),
        pytest.param(
            _direct_candidate(output_domain=Domain.QUANTUM),
            id="quantum-output",
        ),
    ],
)
def test_classify_direct_rejects_other_components(candidate):
    assert classify_direct(candidate, LAYOUT) is None


@pytest.mark.parametrize("value, bit", [(0, False), (1, True)])
def test_classify_constant_reads_the_bit(value, bit):
    assert classify_constant(_direct_candidate(value=value, params=())) is bit


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(_direct_candidate(), id="basis-param"),
        pytest.param(_direct_candidate(params=(LAYOUT.error(0),)), id="error-param"),
        pytest.param(
            _direct_candidate(params=(), edge_type=EdgeType.SIMPLE), id="simple-edge"
        ),
        pytest.param(
            _direct_candidate(params=(), spider_type=VertexType.X), id="x-spider"
        ),
        pytest.param(
            _direct_candidate(params=(), output_domain=Domain.QUANTUM),
            id="quantum-output",
        ),
        pytest.param(
            _direct_candidate(value=Fraction(1, 2), params=()), id="non-pauli-phase"
        ),
    ],
)
def test_classify_constant_rejects_other_components(candidate):
    assert classify_constant(candidate) is None


@pytest.mark.parametrize("output_domain", [Domain.CLASSICAL, Domain.QUANTUM])
def test_plugged_outputs_give_outcome_probabilities(output_domain):
    g = Graph()
    zero = g.add_vertex(VertexType.X, domain=output_domain)
    out = g.add_vertex(VertexType.BOUNDARY, domain=output_domain)
    g.set_port(out, 0)
    g.add_edge(zero, out)
    g.mul_sqrt2(-1, output_domain)
    m = LAYOUT.output(0)
    measured = decompose(_plugged(g, LAYOUT, 1))
    assert measured.evaluate({m: 0}) == pytest.approx(1)
    assert measured.evaluate({m: 1}) == pytest.approx(0)
    assert decompose(_plugged(g, LAYOUT, 0)).evaluate({}) == pytest.approx(1)
