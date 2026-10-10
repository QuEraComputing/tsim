"""Closing for sampling, the circuit graph and preparation on zx_param."""

import numpy as np
import stim
import zx_param
from zx_param import Domain, VertexType

from tsim.circuit import Circuit
from tsim.core.graph import build_sampling_graph, circuit_graph, prepare_graph
from tsim.core.parse import parse_stim_circuit


def _distribution(text: str, *, sample_detectors: bool = False) -> np.ndarray:
    built = parse_stim_circuit(stim.Circuit(text))
    g = build_sampling_graph(built, sample_detectors=sample_detectors)
    t = np.real(g.to_tensor().reshape(-1))
    return t / t.sum()


def test_bell_measurements_are_correlated():
    np.testing.assert_allclose(
        _distribution("H 0\nCNOT 0 1\nM 0 1"), [0.5, 0, 0, 0.5], atol=1e-9
    )


def test_second_measurement_after_h_is_independent():
    np.testing.assert_allclose(
        _distribution("M 0\nH 0\nM 0"), [0.5, 0.5, 0, 0], atol=1e-9
    )
    np.testing.assert_allclose(
        _distribution("H 0\nM 0\nH 0\nM 0"), [0.25, 0.25, 0.25, 0.25], atol=1e-9
    )


def test_feedforward_copies_the_record():
    np.testing.assert_allclose(
        _distribution("H 0\nM 0\nCX rec[-1] 1\nM 1"), [0.5, 0, 0, 0.5], atol=1e-9
    )


def test_detector_and_observable_outputs():
    text = "H 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    np.testing.assert_allclose(
        _distribution(text, sample_detectors=True), [0.5, 0, 0, 0.5], atol=1e-9
    )


def test_detector_with_a_repeated_record_is_zero():
    text = "H 0\nM 0\nDETECTOR rec[-1] rec[-1]"
    np.testing.assert_allclose(
        _distribution(text, sample_detectors=True), [1, 0], atol=1e-9
    )


def test_sampling_outputs_are_classical_and_ordered():
    built = parse_stim_circuit(stim.Circuit("H 0 1\nM 0 1\nDETECTOR rec[-1]"))
    g = build_sampling_graph(built)
    assert [g.port(v) for v in g.outputs()] == [0, 1]
    assert all(g.vertex_domain(v) == Domain.CLASSICAL for v in g.outputs())
    assert not g.inputs()
    assert all(not g.has_vertex(v) for v in built.detectors)


def test_circuit_graph_ports_and_helper_lanes():
    built = parse_stim_circuit(stim.Circuit("H 0\nMPP X0*Z1\nM 0"))
    g = circuit_graph(built)
    boundaries = [v for v in g.vertices() if g.vertex_type(v) == VertexType.BOUNDARY]
    ports = [g.port(v) for v in boundaries]
    assert len(ports) == 4
    assert set(ports) == {-2, -1, 0, 1}


def test_circuit_graph_without_annotations_drops_detectors():
    built = parse_stim_circuit(stim.Circuit("M 0\nDETECTOR rec[-1]"))
    g = circuit_graph(built, annotations=False)
    assert all(not g.has_vertex(v) for v in built.detectors)


def test_prepare_graph_maps_errors_to_basis_ids():
    circuit = Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    prepared = prepare_graph(circuit, sample_detectors=True)
    layout = prepared.layout
    assert (layout.num_errors, layout.num_basis, layout.num_outputs) == (1, 1, 1)
    params = {
        p for v in prepared.graph.vertices() for p in prepared.graph.phase(v).parameters
    }
    assert params == {layout.basis(0)}
    np.testing.assert_array_equal(prepared.error_transform, [[1]])
    assert isinstance(prepared.graph, zx_param.Graph)
