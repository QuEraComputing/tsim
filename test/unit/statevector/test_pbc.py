from fractions import Fraction
from math import pi
from test.helpers.statevector import (
    CIRCUITS,
    assert_distributions_close,
    component_ops,
    f_patterns,
    ops_distribution,
    output_indices,
    quantum_components,
    stabilizer_rank_distribution,
)

import numpy as np
import pytest
from zxparam import Domain, Phase, VertexType

from tsim.circuit import Circuit
from tsim.compile.pipeline import classify_direct, compile_program
from tsim.core.graph import prepare_graph
from tsim.core.types import CompiledComponent
from tsim.statevector.pbc import (
    MEASUREMENT,
    ROTATION,
    ComponentLayout,
    PauliOps,
    _with_commutant_measurements,
    anticommute,
    read_pauli_ops,
)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
@pytest.mark.parametrize("sample_detectors", [True, False])
def test_ops_match_stabilizer_rank(name, sample_detectors):
    prepared = prepare_graph(Circuit(CIRCUITS[name]), sample_detectors=sample_detectors)
    joint = compile_program(prepared, mode="joint", strategy="cat5").components
    by_outputs = {c.output_indices: c for c in joint}
    for component in prepared.graph.connected_components():
        if classify_direct(component, prepared.layout) is not None:
            continue
        indices = output_indices(component)
        layout = ComponentLayout.of(component, indices, prepared.layout)
        ops = read_pauli_ops(component, layout)
        reference = by_outputs[tuple(indices)]
        assert isinstance(reference, CompiledComponent)
        for f in f_patterns(ops.num_f):
            assert_distributions_close(
                ops_distribution(ops, f),
                stabilizer_rank_distribution(reference, f),
                atol=2e-5,
            )


def test_a_t_gate_becomes_one_rotation_by_pi_over_8():
    (component,), layout = quantum_components(
        "R 0\nH 0\nT 0\nH 0\nM 0", sample_detectors=False
    )
    ops = read_pauli_ops(component, ComponentLayout.of(component, [0], layout))
    assert ops.kind.tolist().count(ROTATION) == 1
    assert abs(ops.angle[ops.kind == ROTATION][0]) == pytest.approx(pi / 8)
    assert ops.slot.tolist().count(0) == 1


def test_ops_are_forgotten_measurements_then_rotations_then_outputs():
    for ops in component_ops():
        head = ops.num_ops - ops.num_outputs
        assert ops.slot[head:].tolist() == list(range(ops.num_outputs))
        assert (ops.kind[head:] == MEASUREMENT).all()
        assert (ops.slot[:head] == -1).all()
        rotations = np.flatnonzero(ops.kind[:head] == ROTATION)
        assert (ops.kind[rotations.min(initial=head) : head] == ROTATION).all()


def test_commutant_measurements_commute_with_every_op():
    for ops in component_ops():
        forgotten = (ops.kind == MEASUREMENT) & (ops.slot < 0)
        anti = anticommute(ops.x[forgotten], ops.z[forgotten], ops.x, ops.z)
        assert not anti.any()


def test_commutant_measurements_have_independent_x_parts():
    # NOTE: Y⊗X and X⊗Y commute with rotations about X⊗Y and Y⊗X and share their
    # X part, so X⊗Y measures deterministically once Y⊗X has been measured.
    ops = PauliOps(
        kind=np.array([ROTATION, ROTATION], dtype=np.uint8),
        x=np.array([[1, 1], [1, 1]], dtype=bool),
        z=np.array([[0, 1], [1, 0]], dtype=bool),
        angle=np.array([0.3, 0.3]),
        sign=np.zeros((2, 1), dtype=bool),
        slot=np.array([-1, -1], dtype=np.int32),
    )
    measured = _with_commutant_measurements(ops)
    assert measured.num_ops == ops.num_ops + 1
    assert measured.kind[0] == MEASUREMENT and measured.slot[0] == -1


def test_every_line_is_flipped_by_some_op():
    for ops in component_ops():
        assert ops.x.any(axis=0).all()


def test_component_layout_reads_f_bits_and_ports():
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    component_layout = ComponentLayout.of(component, output_indices(component), layout)
    assert list(component_layout.f_selection) == sorted(component_layout.f_selection)
    assert len(component_layout.f_selection) == 3
    assert component_layout.output_indices == (0, 1, 2)
    assert sorted(component_layout.f_position.values()) == [0, 1, 2]


def _t_ghz():
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    return component, ComponentLayout.of(component, output_indices(component), layout)


def test_an_x_spider_is_rejected():
    component, layout = _t_ghz()
    spider = next(
        v for v in component.vertices() if component.vertex_type(v) == VertexType.Z
    )
    component.set_vertex_type(spider, VertexType.X)
    with pytest.raises(ValueError, match="Z spider"):
        read_pauli_ops(component, layout)


def test_a_classical_spider_with_a_non_pauli_phase_is_rejected():
    component, layout = _t_ghz()
    spider = next(
        v
        for v in component.vertices()
        if component.vertex_type(v) == VertexType.Z
        and component.vertex_domain(v) == Domain.CLASSICAL
    )
    component.set_phase(spider, Phase(Fraction(1, 2)))
    with pytest.raises(ValueError, match="phase"):
        read_pauli_ops(component, layout)


def test_a_plain_edge_at_a_quantum_spider_is_rejected():
    component, layout = _t_ghz()
    u, w = next(
        (u, w)
        for u, w in component.edges()
        if component.vertex_domain(u) == Domain.QUANTUM
        and component.vertex_domain(w) == Domain.QUANTUM
    )
    component.remove_edge(u, w)
    component.add_edge(u, w)
    with pytest.raises(ValueError, match="plain"):
        read_pauli_ops(component, layout)
