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
import stim
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
    _frame,
    _mask,
    _Reading,
    _Relations,
    _rotation_axes,
    _unmask,
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


def test_commutant_measurements_pairwise_commute():
    # NOTE: X1 and Z1 both commute with a rotation about Z0, but not with each other.
    ops = PauliOps(
        kind=np.array([ROTATION], dtype=np.uint8),
        x=np.array([[0, 0]], dtype=bool),
        z=np.array([[1, 0]], dtype=bool),
        angle=np.array([0.3]),
        sign=np.zeros((1, 1), dtype=bool),
        slot=np.array([-1], dtype=np.int32),
    )
    measured = _with_commutant_measurements(ops)
    assert measured.num_ops == ops.num_ops + 1
    assert measured.x[0].tolist() == [False, True]
    assert not measured.z[0].any()


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


@pytest.mark.parametrize("seed", range(5))
def test_graph_state_stabilizers_are_products_of_the_generators(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 100))
    edges = np.triu(rng.random((n, n)) < rng.choice([0.05, 0.3, 0.8]), 1)
    # NOTE: a reading's stabilizers only need its graph state.
    reading = object.__new__(_Reading)
    reading.num_spiders = n
    reading.neighbours = [_mask(row) for row in edges | edges.T]
    for _ in range(20):
        t = _mask(rng.random(n) < rng.random())
        product = stim.PauliString(n)
        for u in (u for u in range(n) if t >> u & 1):
            k_u = stim.PauliString(n)
            k_u[u] = "X"
            for w in (w for w in range(n) if reading.neighbours[u] >> w & 1):
                k_u[w] = "Z"
            product *= k_u
        assert reading.stabilizer(t) == product


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 70])
def test_unmask_reads_the_low_bits_of_a_mask(length):
    for mask in [0, 1, 0b1011, 2**69 + 2**8 + 1, (1 << 80) - 1]:
        expected = [bool(mask >> i & 1) for i in range(length)]
        assert _unmask(mask, length).tolist() == expected


def test_an_input_is_rejected():
    component, layout = _t_ghz()
    boundary = component.add_vertex(VertexType.BOUNDARY)
    component.set_vertex_domain(boundary, Domain.CLASSICAL)
    with pytest.raises(ValueError, match="input"):
        read_pauli_ops(component, layout)


def test_a_sign_of_one_variable_is_linear():
    relations = _Relations(1)
    relations.add_sign(0, 0)
    assert relations.lin == [1] and relations.pairs == [0]


def test_contradicting_equalities_are_rejected():
    relations = _Relations(1)
    relations.equalities = [(1, 0), (1, 1)]
    with pytest.raises(ValueError, match="inconsistent"):
        relations.reduce([0])


def test_an_equality_that_reduces_to_one_is_rejected():
    relations = _Relations(2)
    relations.equalities = [(0b11, 0), (0b11, 1)]
    with pytest.raises(ValueError, match="inconsistent"):
        relations.eliminate(0)


def test_a_sign_left_after_reducing_is_rejected():
    relations = _Relations(2)
    relations.add_sign(0, 1)
    with pytest.raises(ValueError, match="stochastic"):
        relations.reduce([0, 1])


def _reading(n, *, angle=None, posts=()):
    """Return a reading of ``n`` unconnected X spiders."""
    reading = object.__new__(_Reading)
    reading.num_spiders = n
    reading.neighbours = [0] * n
    reading.letter = ["X"] * n
    reading.negative = 0
    reading.angle = angle or {}
    reading.posts = list(posts)
    return reading


def test_a_rotation_that_no_order_commutes_with_its_postselection_is_rejected():
    # NOTE: Z0 anticommutes with the postselection of X0, and no unrotated
    # spider can fix that.
    with pytest.raises(ValueError, match="no flow"):
        _rotation_axes(_reading(1, angle={0: 0.3}), ([0], [1]))


@pytest.mark.parametrize("rhs", [0, 1])
def test_a_dependent_postselection_must_hold_with_certainty(rhs):
    # NOTE: |+⟩ postselected twice on X: again on +1 holds, on -1 never does.
    reading = _reading(1, posts=[(1, 0), (1, rhs)])
    if rhs:
        with pytest.raises(ValueError, match="probability 0"):
            _frame(reading, [0], 0)
    else:
        _, targets = _frame(reading, [0], 0)
        assert targets.shape == (0, 0)
