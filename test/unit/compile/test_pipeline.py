"""compile_program on zx_param: plugging normalization and the direct path."""

import itertools
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
from zx_param import Domain, EdgeType, Graph, Phase, VertexType

from tsim.circuit import Circuit
from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import _plugged, classify_direct, compile_program
from tsim.core.graph import prepare_graph
from tsim.core.params import ParamLayout
from tsim.sampler import CompiledStateProbs

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)])
I2 = np.eye(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


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


LAYOUT = ParamLayout(num_errors=3, num_basis=2, num_outputs=1)


def _direct_candidate(
    value: int = 0,
    params: tuple[int, ...] = (LAYOUT.basis(1),),
    edge_type: EdgeType = EdgeType.HADAMARD,
    output_domain: Domain = Domain.CLASSICAL,
    spider_domain: Domain = Domain.CLASSICAL,
) -> Graph:
    g = Graph()
    out = g.add_vertex(VertexType.BOUNDARY, domain=output_domain)
    g.set_port(out, 0)
    phase = Phase(Fraction(value), frozenset(params))
    spider = g.add_vertex(VertexType.Z, phase, domain=spider_domain)
    g.add_edge(out, spider, edge_type)
    return g


@pytest.mark.parametrize("value, flip", [(0, False), (1, True)])
def test_classify_direct_reads_the_basis_index_and_flip(value, flip):
    component = _direct_candidate(value=value)
    assert classify_direct(component, LAYOUT) == (1, flip)


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
        pytest.param(
            _direct_candidate(output_domain=Domain.QUANTUM),
            id="quantum-output",
        ),
        pytest.param(
            _direct_candidate(
                output_domain=Domain.QUANTUM, spider_domain=Domain.QUANTUM
            ),
            id="quantum-output-and-spider",
        ),
    ],
)
def test_classify_direct_rejects_other_components(candidate):
    assert classify_direct(candidate, LAYOUT) is None


def test_plugging_a_quantum_output_is_an_error():
    component = _direct_candidate(output_domain=Domain.QUANTUM)
    with pytest.raises(ValueError, match="not classical"):
        _plugged(component, LAYOUT, 0)
