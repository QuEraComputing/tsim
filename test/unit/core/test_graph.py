"""Building open and closed graphs, and preparing them for sampling."""

from fractions import Fraction
from test.helpers.gate_matrices import CNOT, H
from test.helpers.tensors import doubled

import numpy as np
import pytest
import stim
import zxparam
from zxparam import Domain, VertexType
from zxparam.phase import Bit, Phase

from tsim.circuit import Circuit
from tsim.core.graph import prepare_graph, transform_error_basis
from tsim.core.parse import parse_stim_circuit


def _distribution(text: str, *, sample_detectors: bool = False) -> np.ndarray:
    builder = parse_stim_circuit(stim.Circuit(text))
    g = builder.build(
        closed=True, outputs="detectors" if sample_detectors else "records"
    )
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
    # NOTE: after the second H, qubit 0's wire no longer equals rec0, so a
    # control taken from the wire would correlate rec2 with rec1, not rec0.
    np.testing.assert_allclose(
        _distribution("H 0\nM 0\nH 0\nCX rec[-1] 1\nM 0 1"),
        [0.25, 0, 0.25, 0, 0, 0.25, 0, 0.25],
        atol=1e-9,
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
    builder = parse_stim_circuit(stim.Circuit("H 0 1\nM 0 1\nDETECTOR rec[-1]"))
    g = builder.build(closed=True, outputs="records")
    assert [g.port(v) for v in g.outputs()] == [0, 1]
    assert all(g.vertex_domain(v) == Domain.CLASSICAL for v in g.outputs())
    assert not g.inputs()
    assert all(not g.has_vertex(v) for v in builder.detectors)


LABELLED = (
    "R 0\nH 0 1\nM 0 1\nMR 0\nDETECTOR rec[-1] rec[-3]\n"
    "OBSERVABLE_INCLUDE(0) rec[-2]"
)


def _labels(g: zxparam.Graph) -> dict[str, VertexType]:
    labels = {}
    for v in g.vertices():
        label = g.vertex_label(v)
        if label is not None:
            labels[label] = g.vertex_type(v)
    return labels


def _output_labels(g: zxparam.Graph) -> list[str | None]:
    return [g.vertex_label(v) for v in g.outputs()]


def test_open_graph_labels_records_and_detector_outputs():
    builder = parse_stim_circuit(stim.Circuit(LABELLED))
    g = builder.build(closed=False, outputs="detectors")
    assert _labels(g) == {
        "rec[0]": VertexType.Z,
        "rec[1]": VertexType.Z,
        "rec[2]": VertexType.Z,
        "m[0]": VertexType.Z,
        "det[0]": VertexType.BOUNDARY,
        "obs[0]": VertexType.BOUNDARY,
    }
    assert _output_labels(g)[-2:] == ["det[0]", "obs[0]"]


def test_detector_sampling_graph_labels_only_its_outputs():
    builder = parse_stim_circuit(stim.Circuit(LABELLED))
    g = builder.build(closed=True, outputs="detectors")
    assert _labels(g) == {"det[0]": VertexType.BOUNDARY, "obs[0]": VertexType.BOUNDARY}
    assert _output_labels(g) == ["det[0]", "obs[0]"]


def test_measurement_sampling_graph_labels_only_its_outputs():
    builder = parse_stim_circuit(stim.Circuit(LABELLED))
    g = builder.build(closed=True, outputs="records")
    assert set(_labels(g).values()) == {VertexType.BOUNDARY}
    assert _output_labels(g) == ["rec[0]", "rec[1]", "rec[2]"]


def test_open_graph_with_record_outputs():
    builder = parse_stim_circuit(stim.Circuit(LABELLED))
    g = builder.build(closed=False, outputs="records")
    assert [g.port(v) for v in g.outputs()] == [0, 1, 2, 3, 4]
    assert _output_labels(g)[2:] == ["rec[0]", "rec[1]", "rec[2]"]
    assert _labels(g)["m[0]"] == VertexType.Z
    assert all(not g.has_vertex(v) for v in [*builder.detectors, *builder.observables])


def test_open_graph_ports_and_helper_lanes():
    builder = parse_stim_circuit(stim.Circuit("H 0\nMPP X0*Z1\nM 0"))
    g = builder.build(closed=False, outputs="detectors")
    boundaries = [v for v in g.vertices() if g.vertex_type(v) == VertexType.BOUNDARY]
    ports = [g.port(v) for v in boundaries]
    assert len(ports) == 4
    assert set(ports) == {-2, -1, 0, 1}


def test_open_graph_without_outputs_drops_detectors():
    builder = parse_stim_circuit(
        stim.Circuit("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    )
    g = builder.build(closed=False, outputs="none")
    assert all(not g.has_vertex(v) for v in [*builder.detectors, *builder.observables])
    assert [g.port(v) for v in g.outputs()] == [0]


def test_open_graph_gives_detectors_classical_outputs():
    builder = parse_stim_circuit(
        stim.Circuit(
            "H 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]"
        )
    )
    g = builder.build(closed=False, outputs="detectors")
    outputs = g.outputs()
    assert [g.port(v) for v in outputs] == [0, 1, 2, 3, 4]
    annotations = [*builder.detectors, *builder.observables]
    for b, v in zip(outputs[2:], annotations, strict=True):
        assert g.vertex_domain(b) == Domain.CLASSICAL
        assert list(g.neighbors(b)) == [v]


def _with_annotation_bits(terms: list[tuple[np.ndarray, list[int]]]) -> np.ndarray:
    """Return Σ doubled(K) ⊗ |bits⟩, with the bit axes after the qubit outputs."""
    total = 0
    for kraus, bits in terms:
        t = doubled(kraus)
        for bit in bits:
            t = np.multiply.outer(t, np.eye(2)[bit])
        half = (t.ndim - len(bits)) // 2
        axes = [*range(half), *range(2 * half, t.ndim), *range(half, 2 * half)]
        total = total + t.transpose(axes)
    return np.asarray(total)


def test_open_graph_tensor_is_the_channel_with_detector_bits():
    builder = parse_stim_circuit(
        stim.Circuit("H 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    )
    expected = _with_annotation_bits(
        [(np.outer(np.eye(2)[k], np.eye(2)[k]) @ H, [k, k]) for k in (0, 1)]
    )
    g = builder.build(closed=False, outputs="detectors")
    np.testing.assert_allclose(g.to_tensor(), expected, atol=1e-9)


def test_open_graph_tensor_xors_records_into_detectors():
    builder = parse_stim_circuit(
        stim.Circuit(
            "H 0\nCNOT 0 1\nH 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
        )
    )
    u = np.kron(np.eye(2), H) @ CNOT @ np.kron(H, np.eye(2))
    terms = []
    for k0 in (0, 1):
        for k1 in (0, 1):
            e = np.eye(4)[2 * k0 + k1]
            terms.append((np.outer(e, e) @ u, [k0 ^ k1, k0]))
    expected = _with_annotation_bits(terms)
    g = builder.build(closed=False, outputs="detectors")
    np.testing.assert_allclose(g.to_tensor(), expected, atol=1e-9)


def test_open_graph_tensor_is_the_channel_with_record_bits():
    builder = parse_stim_circuit(stim.Circuit("H 0\nM 0\nH 0\nM 0"))
    terms = []
    for k0 in (0, 1):
        for k1 in (0, 1):
            p0 = np.outer(np.eye(2)[k0], np.eye(2)[k0])
            p1 = np.outer(np.eye(2)[k1], np.eye(2)[k1])
            terms.append((p1 @ H @ p0 @ H, [k0, k1]))
    expected = _with_annotation_bits(terms)
    g = builder.build(closed=False, outputs="records")
    np.testing.assert_allclose(g.to_tensor(), expected, atol=1e-9)


def test_prepare_graph_maps_errors_to_basis_ids():
    circuit = Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    prepared = prepare_graph(circuit, sample_detectors=True)
    layout = prepared.layout
    assert (layout.num_errors, prepared.num_outputs) == (1, 1)
    assert prepared.graph.parameters() == {layout.basis(0)}
    np.testing.assert_array_equal(prepared.error_transform, [[1]])
    assert isinstance(prepared.graph, zxparam.Graph)


NORMALIZATION_CIRCUITS = {
    "mpp": "H 0\nMPP X0*Z1 Y0*Y1\nM 0 1\nDETECTOR rec[-1] rec[-3]\n"
    "OBSERVABLE_INCLUDE(0) rec[-2]",
    "mxx-myy-mzz": "H 0\nMXX 0 1\nMYY(0.1) 0 1\nMZZ 0 1\nDETECTOR rec[-1] rec[-3]\n"
    "OBSERVABLE_INCLUDE(0) rec[-2]",
    "mpad": "H 0\nMPAD 0 1\nM 0\nDETECTOR rec[-2] rec[-1]\n"
    "OBSERVABLE_INCLUDE(0) rec[-3]",
    "heralded": "H 0\nHERALDED_ERASE(0.2) 0\n"
    "HERALDED_PAULI_CHANNEL_1(0.1, 0.1, 0.1, 0.1) 1\nM 0 1\n"
    "DETECTOR rec[-4]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]",
    "correlated": "H 0\nCORRELATED_ERROR(0.1) X0 Z1\n"
    "ELSE_CORRELATED_ERROR(0.2) Y0 X1\nM 0 1\nDETECTOR rec[-1]\n"
    "OBSERVABLE_INCLUDE(0) rec[-2]",
    "noisy-measurement": "H 0\nCX 0 1\nM(0.1) 0 1\nDETECTOR rec[-1] rec[-2]\n"
    "OBSERVABLE_INCLUDE(0) rec[-1]",
    "feedforward": "H 0 2\nM 0\nCX rec[-1] 1\nM 2\nH 3\nCZ rec[-1] 3\nH 3\n"
    "X_ERROR(0.1) 1 3\nM 1 3\nDETECTOR rec[-4] rec[-2]\nDETECTOR rec[-3] rec[-1]\n"
    "OBSERVABLE_INCLUDE(0) rec[-1]",
    "resets": "H 0\nMR 0\nRX 1\nRY 2\nT 1 2\nDEPOLARIZE1(0.1) 0 1 2\nMRX(0.1) 1\n"
    "MRY 2\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-3]",
    "annotations": "H 0\nT 0\nM 0 1\nDETECTOR\nDETECTOR rec[-2] rec[-2]\n"
    "DETECTOR rec[-1] rec[-2] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    "OBSERVABLE_INCLUDE(0) rec[-2]",
}


@pytest.mark.parametrize("sample_detectors", [False, True])
@pytest.mark.parametrize(
    "text", NORMALIZATION_CIRCUITS.values(), ids=NORMALIZATION_CIRCUITS.keys()
)
def test_sampling_graph_sums_to_one(text, sample_detectors):
    circuit = Circuit(text)
    num_errors = parse_stim_circuit(circuit.stim_circuit).num_error_bits
    rng = np.random.default_rng(0)
    zeros: dict[int, Bit] = {j: 0 for j in range(num_errors)}
    random: dict[int, Bit] = {j: 1 if rng.integers(2) else 0 for j in zeros}
    for bindings in (zeros, random):
        g = circuit.get_sampling_graph(sample_detectors=sample_detectors)
        np.testing.assert_allclose(np.sum(g.to_tensor(bindings)), 1, atol=1e-9)


def _positions_are_distinct(g: zxparam.Graph) -> bool:
    positions = [(g.col(v), g.row(v)) for v in g.vertices()]
    return len(set(positions)) == len(positions)


@pytest.mark.parametrize(
    "text",
    [
        "H 0\nT 0\nSWAP 0 1\nM 0 1",
        "H 0\nCX 0 1\nSWAP 1 2\nM 0 1 2",
        "MPP X0\nDETECTOR",
        "MPP X0*Z1\nDETECTOR\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "HERALDED_ERASE(0.1) 0\nMPAD 1\nM 0\nDETECTOR rec[-1] rec[-2]",
    ],
)
def test_circuit_graph_positions_are_distinct(text: str):
    assert _positions_are_distinct(Circuit(text).get_graph())


@pytest.mark.parametrize("sample_detectors", [False, True])
def test_sampling_graph_positions_are_distinct(sample_detectors: bool):
    c = Circuit("H 0\nM 0\nM 0\nMPP X0\nM 0\nDETECTOR rec[-1] rec[-3]")
    assert _positions_are_distinct(c.get_sampling_graph(sample_detectors))


def test_transform_error_basis_rejects_a_parameter_that_is_not_an_error_bit():
    g = zxparam.Graph()
    g.add_vertex(VertexType.Z, Phase(Fraction(0), frozenset({3})))
    with pytest.raises(ValueError, match="not an error bit"):
        transform_error_basis(g, num_errors=2)
