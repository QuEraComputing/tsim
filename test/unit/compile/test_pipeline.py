"""compile_program on zx_param: plugging normalization and the direct path."""

import itertools

import jax.numpy as jnp
import numpy as np

from tsim.circuit import Circuit
from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.sampler import CompiledStateProbs

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)])
I2 = np.eye(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def _abs_value(compiled, bits: list[int]) -> float:
    params = jnp.asarray(np.array([bits], dtype=np.uint8).reshape(1, len(bits)))
    return float(np.abs(np.asarray(evaluate(compiled, params)))[0])


def test_sequential_plugged_graphs_are_consistent_marginals():
    circuit = Circuit(
        "R 0 1 2\nH 0\nT 0\nCNOT 0 1\nT 1\nCNOT 1 2\nH 2\nT 2\nH 1\nM 0 1 2"
    )
    program = compile_program(
        prepare_graph(circuit, sample_detectors=False), mode="sequential"
    )
    assert program.components
    for component in program.components:
        graphs = component.compiled_scalar_graphs
        for k in range(len(graphs) - 1):
            for prefix in itertools.product((0, 1), repeat=k):
                parent = _abs_value(graphs[k], list(prefix))
                children = sum(_abs_value(graphs[k + 1], [*prefix, b]) for b in (0, 1))
                np.testing.assert_allclose(children, parent, rtol=1e-4, atol=1e-6)


def test_joint_probabilities_match_the_state_vector():
    circuit = Circuit("R 0 1\nH 0\nT 0\nCNOT 0 1\nH 0\nM 0 1")
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1
    for u in (np.kron(H, I2), np.kron(T, I2), CNOT, np.kron(H, I2)):
        psi = u @ psi
    probs = CompiledStateProbs(circuit, seed=0)
    for index, bits in enumerate(itertools.product((False, True), repeat=2)):
        got = probs.probability_of(np.array(bits), batch_size=1)[0]
        np.testing.assert_allclose(got, abs(psi[index]) ** 2, atol=1e-5)


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
