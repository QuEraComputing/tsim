from fractions import Fraction
from test.helpers.tensors import channel, doubled, doubled_state

import numpy as np
from zxparam import Domain, EdgeType, Graph, VertexType

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
Q = Domain.QUANTUM


def _wire(g: Graph, qubit: int, edge_type: EdgeType) -> None:
    i = g.add_vertex(VertexType.BOUNDARY, domain=Q, row=qubit)
    o = g.add_vertex(VertexType.BOUNDARY, domain=Q, row=qubit, col=1)
    g.set_port(i, -(qubit + 1))
    g.set_port(o, qubit)
    g.add_edge(i, o, edge_type)


def test_doubled_matches_zxparam_for_a_hadamard_wire():
    g = Graph()
    _wire(g, 0, EdgeType.HADAMARD)
    np.testing.assert_allclose(g.to_tensor(), doubled(H), atol=1e-12)


def test_doubled_orders_qubits_big_endian():
    g = Graph()
    _wire(g, 0, EdgeType.HADAMARD)
    _wire(g, 1, EdgeType.SIMPLE)
    np.testing.assert_allclose(
        g.to_tensor(), doubled(np.kron(H, np.eye(2))), atol=1e-12
    )


def test_channel_matches_a_classical_spider_on_a_wire():
    g = Graph()
    i = g.add_vertex(VertexType.BOUNDARY, domain=Q)
    o = g.add_vertex(VertexType.BOUNDARY, domain=Q, col=2)
    c = g.add_vertex(VertexType.Z, col=1)  # classical: dephasing
    g.set_port(i, -1)
    g.set_port(o, 0)
    g.add_edge(i, c)
    g.add_edge(c, o)
    dephasing = channel(np.outer(ZERO, ZERO), np.outer(ONE, ONE))
    np.testing.assert_allclose(g.to_tensor(), dephasing, atol=1e-12)


def test_doubled_state_matches_a_zero_preparation():
    g = Graph()
    x = g.add_vertex(VertexType.X, Fraction(0), domain=Q)
    o = g.add_vertex(VertexType.BOUNDARY, domain=Q, col=1)
    g.set_port(o, 0)
    g.add_edge(x, o)
    g.mul_sqrt2(-1, Q)
    np.testing.assert_allclose(g.to_tensor(), doubled_state(ZERO), atol=1e-12)
