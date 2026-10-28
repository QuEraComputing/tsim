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
    _mask,
    _Reading,
    _Relations,
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


def _affine(form: int, f: int) -> int:
    """Return the affine form ``form`` of ``(1, f_0, f_1, …)`` at the f-bits ``f``."""
    return (form & (1 | f << 1)).bit_count() & 1


@pytest.mark.parametrize("seed", range(300))
def test_relations_eliminate_and_reduce_like_brute_force(seed):
    """Summing the sign over eliminated variables leaves the reduced rows' indicator.

    The weight of an assignment ``v`` is ``(-1)^Φ`` times whether every equality
    holds. Summed over the eliminated variables, it must be, for each f-bit
    pattern, a constant times the indicator of the reduced rows; the reduction
    raises exactly when some pattern breaks that form.
    """
    rng = np.random.default_rng(seed)
    num_vars, num_f = int(rng.integers(1, 7)), int(rng.integers(0, 3))
    forms = 1 << (1 + num_f)
    relations = _Relations(num_vars)
    for _ in range(int(rng.integers(0, 2 * num_vars))):
        i, j = (int(v) for v in rng.integers(0, num_vars, 2))
        relations.add_sign(i, j)
    for i in range(num_vars):
        if rng.random() < 0.3:
            relations.lin[i] ^= int(rng.integers(0, forms))
    for _ in range(int(rng.integers(0, num_vars + 1))):
        mask = int(rng.integers(1, 1 << num_vars))
        relations.equalities.append((mask, int(rng.integers(0, forms))))
    pairs, lin = list(relations.pairs), list(relations.lin)
    equalities = list(relations.equalities)
    eliminated = [v for v in range(num_vars) if rng.random() < 0.5]
    kept = [int(v) for v in rng.permutation(num_vars) if v not in eliminated]

    def summed(f: int) -> dict[int, int]:
        """Return the weight of each kept assignment, summed over the others."""
        total: dict[int, int] = {}
        for v in range(1 << num_vars):
            if not all(
                (mask & v).bit_count() & 1 == _affine(rhs, f)
                for mask, rhs in equalities
            ):
                continue
            # NOTE: pairs are symmetric, so the pairs below i count each once; a
            # diagonal pair is in lin, where add_sign keeps it.
            phase = sum(
                (v >> i & 1)
                * (((pairs[i] & v) & ((1 << i) - 1)).bit_count() + _affine(lin[i], f))
                for i in range(num_vars)
            )
            key = sum((v >> k & 1) << k for k in kept)
            total[key] = total.get(key, 0) + (-1) ** (phase % 2)
        return total

    def kept_assignments():
        for bits in range(1 << len(kept)):
            yield sum((bits >> n & 1) << k for n, k in enumerate(kept))

    weights = [summed(f) for f in range(1 << num_f)]

    def representable(total: dict[int, int]) -> bool:
        values = {w for w in total.values() if w}
        return len(values) == 1

    try:
        for v in eliminated:
            relations.eliminate(v)
        rows = relations.reduce(kept)
    except ValueError:
        assert not all(representable(total) for total in weights)
        return
    for f, total in enumerate(weights):
        assert representable(total)
        for v in kept_assignments():
            holds = all(
                (mask & v).bit_count() & 1 == _affine(rhs, f) for _, mask, rhs in rows
            )
            assert bool(total.get(v, 0)) == holds
