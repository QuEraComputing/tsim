"""Each instruction's doubled tensor is its exact channel."""

from collections.abc import Callable
from fractions import Fraction
from test.helpers.gate_matrices import (
    ROT_GATE_MATRICES,
    SINGLE_QUBIT_GATE_MATRICES,
    TWO_QUBIT_GATE_MATRICES,
)
from test.helpers.tensors import ONE, ZERO, channel, doubled, doubled_state

import numpy as np
import pytest
from zxparam import Domain, VertexType

from tsim.circuit import Circuit
from tsim.core import instructions
from tsim.core.builder import GraphBuilder

PLUS = (ZERO + ONE) / np.sqrt(2)
MINUS = (ZERO - ONE) / np.sqrt(2)
PLUS_I = (ZERO + 1j * ONE) / np.sqrt(2)
MINUS_I = (ZERO - 1j * ONE) / np.sqrt(2)


def _tensor(*steps: Callable[[GraphBuilder], object]) -> np.ndarray:
    builder = GraphBuilder()
    for step in steps:
        step(builder)
    return builder.build(closed=False, outputs="detectors").to_tensor()


def _dephasing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return channel(np.outer(a, a.conj()), np.outer(b, b.conj()))


def _reset_to(psi: np.ndarray) -> np.ndarray:
    return channel(np.outer(psi, ZERO), np.outer(psi, ONE))


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        (instructions.i, SINGLE_QUBIT_GATE_MATRICES["I"]),
        (instructions.x, SINGLE_QUBIT_GATE_MATRICES["X"]),
        (instructions.y, SINGLE_QUBIT_GATE_MATRICES["Y"]),
        (instructions.z, SINGLE_QUBIT_GATE_MATRICES["Z"]),
        (instructions.t, SINGLE_QUBIT_GATE_MATRICES["T"]),
        (instructions.t_dag, SINGLE_QUBIT_GATE_MATRICES["T_DAG"]),
        (instructions.c_nxyz, SINGLE_QUBIT_GATE_MATRICES["C_NXYZ"]),
        (instructions.c_nzyx, SINGLE_QUBIT_GATE_MATRICES["C_NZYX"]),
        (instructions.c_xnyz, SINGLE_QUBIT_GATE_MATRICES["C_XNYZ"]),
        (instructions.c_xynz, SINGLE_QUBIT_GATE_MATRICES["C_XYNZ"]),
        (instructions.c_xyz, SINGLE_QUBIT_GATE_MATRICES["C_XYZ"]),
        (instructions.c_znyx, SINGLE_QUBIT_GATE_MATRICES["C_ZNYX"]),
        (instructions.c_zynx, SINGLE_QUBIT_GATE_MATRICES["C_ZYNX"]),
        (instructions.c_zyx, SINGLE_QUBIT_GATE_MATRICES["C_ZYX"]),
        (instructions.h, SINGLE_QUBIT_GATE_MATRICES["H"]),
        (instructions.h_nxy, SINGLE_QUBIT_GATE_MATRICES["H_NXY"]),
        (instructions.h_nxz, SINGLE_QUBIT_GATE_MATRICES["H_NXZ"]),
        (instructions.h_nyz, SINGLE_QUBIT_GATE_MATRICES["H_NYZ"]),
        (instructions.h_xy, SINGLE_QUBIT_GATE_MATRICES["H_XY"]),
        (instructions.h_yz, SINGLE_QUBIT_GATE_MATRICES["H_YZ"]),
        (instructions.s, SINGLE_QUBIT_GATE_MATRICES["S"]),
        (instructions.sqrt_x, SINGLE_QUBIT_GATE_MATRICES["SQRT_X"]),
        (instructions.sqrt_x_dag, SINGLE_QUBIT_GATE_MATRICES["SQRT_X_DAG"]),
        (instructions.sqrt_y, SINGLE_QUBIT_GATE_MATRICES["SQRT_Y"]),
        (instructions.sqrt_y_dag, SINGLE_QUBIT_GATE_MATRICES["SQRT_Y_DAG"]),
        (instructions.s_dag, SINGLE_QUBIT_GATE_MATRICES["S_DAG"]),
    ],
)
def test_single_qubit_instruction(gate_func, matrix: np.ndarray):
    np.testing.assert_allclose(
        _tensor(lambda builder: gate_func(builder, 0)), doubled(matrix), atol=1e-9
    )


@pytest.mark.parametrize("frac", [Fraction(1, 5), Fraction(-1, 3), Fraction(1, 7)])
@pytest.mark.parametrize("name", ["R_X", "R_Y", "R_Z"])
def test_rotation(name: str, frac: Fraction):
    gate = getattr(instructions, name.lower())
    np.testing.assert_allclose(
        _tensor(lambda builder: gate(builder, 0, frac)),
        doubled(ROT_GATE_MATRICES[name](frac)),
        atol=1e-9,
    )


@pytest.mark.parametrize("frac_theta", [Fraction(1, 5), Fraction(-1, 3)])
@pytest.mark.parametrize("frac_phi", [Fraction(1, 5), Fraction(1, 7)])
@pytest.mark.parametrize("frac_lambda", [Fraction(-1, 13), Fraction(1, 7)])
def test_u3(frac_theta: Fraction, frac_phi: Fraction, frac_lambda: Fraction):
    np.testing.assert_allclose(
        _tensor(
            lambda builder: instructions.u3(
                builder, 0, frac_theta, frac_phi, frac_lambda
            )
        ),
        doubled(ROT_GATE_MATRICES["U3"](frac_theta, frac_phi, frac_lambda)),
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "gate_func, matrix",
    [
        (instructions.cnot, TWO_QUBIT_GATE_MATRICES["CNOT"]),
        (instructions.cxswap, TWO_QUBIT_GATE_MATRICES["CXSWAP"]),
        (instructions.cy, TWO_QUBIT_GATE_MATRICES["CY"]),
        (instructions.cz, TWO_QUBIT_GATE_MATRICES["CZ"]),
        (instructions.czswap, TWO_QUBIT_GATE_MATRICES["CZSWAP"]),
        (instructions.iswap, TWO_QUBIT_GATE_MATRICES["ISWAP"]),
        (instructions.iswap_dag, TWO_QUBIT_GATE_MATRICES["ISWAP_DAG"]),
        (instructions.sqrt_xx, TWO_QUBIT_GATE_MATRICES["SQRT_XX"]),
        (instructions.sqrt_xx_dag, TWO_QUBIT_GATE_MATRICES["SQRT_XX_DAG"]),
        (instructions.sqrt_yy, TWO_QUBIT_GATE_MATRICES["SQRT_YY"]),
        (instructions.sqrt_yy_dag, TWO_QUBIT_GATE_MATRICES["SQRT_YY_DAG"]),
        (instructions.sqrt_zz, TWO_QUBIT_GATE_MATRICES["SQRT_ZZ"]),
        (instructions.sqrt_zz_dag, TWO_QUBIT_GATE_MATRICES["SQRT_ZZ_DAG"]),
        (instructions.swap, TWO_QUBIT_GATE_MATRICES["SWAP"]),
        (instructions.swapcx, TWO_QUBIT_GATE_MATRICES["SWAPCX"]),
        (instructions.swapcz, TWO_QUBIT_GATE_MATRICES["SWAPCZ"]),
        (instructions.xcx, TWO_QUBIT_GATE_MATRICES["XCX"]),
        (instructions.xcy, TWO_QUBIT_GATE_MATRICES["XCY"]),
        (instructions.xcz, TWO_QUBIT_GATE_MATRICES["XCZ"]),
        (instructions.ycx, TWO_QUBIT_GATE_MATRICES["YCX"]),
        (instructions.ycy, TWO_QUBIT_GATE_MATRICES["YCY"]),
        (instructions.ycz, TWO_QUBIT_GATE_MATRICES["YCZ"]),
    ],
)
def test_two_qubit_instruction(gate_func, matrix: np.ndarray):
    np.testing.assert_allclose(
        _tensor(lambda builder: gate_func(builder, 0, 1)), doubled(matrix), atol=1e-9
    )


def _on_qubit_0(*gates: Callable[..., None]) -> np.ndarray:
    return _tensor(*[lambda builder, gate=gate: gate(builder, 0) for gate in gates])


def test_measurements_dephase():
    m, mx, my = instructions.m, instructions.mx, instructions.my
    np.testing.assert_allclose(_on_qubit_0(m), _dephasing(ZERO, ONE), atol=1e-9)
    np.testing.assert_allclose(_on_qubit_0(mx), _dephasing(PLUS, MINUS), atol=1e-9)
    np.testing.assert_allclose(_on_qubit_0(my), _dephasing(PLUS_I, MINUS_I), atol=1e-9)


@pytest.mark.parametrize(
    "gates, expected",
    [
        ((instructions.r,), doubled_state(ZERO)),
        ((instructions.i, instructions.r), _reset_to(ZERO)),
        ((instructions.mr,), _reset_to(ZERO)),
        ((instructions.rx,), doubled_state(PLUS)),
        ((instructions.i, instructions.rx), _reset_to(PLUS)),
        ((instructions.mrx,), _reset_to(PLUS)),
        ((instructions.ry,), doubled_state(PLUS_I)),
        ((instructions.i, instructions.ry), _reset_to(PLUS_I)),
        ((instructions.mry,), _reset_to(PLUS_I)),
    ],
)
def test_resets(gates, expected: np.ndarray):
    np.testing.assert_allclose(_on_qubit_0(*gates), expected, atol=1e-9)


@pytest.mark.parametrize("basis", ["", "x", "y"])
def test_mr_matches_m_then_r(basis: str):
    """MR/MRX/MRY should produce the same tensor as M;R / MX;RX / MY;RY."""
    mr, m, r = (getattr(instructions, f"{g}{basis}") for g in ("mr", "m", "r"))
    np.testing.assert_allclose(_on_qubit_0(mr), _on_qubit_0(m, r), atol=1e-9)


def test_x_error_is_an_x_spider_with_its_error_bit():
    builder = GraphBuilder()
    instructions.x_error(builder, 0, 0.1)
    noisy = [v for v in builder.graph.vertices() if builder.graph.phase(v).parameters]
    assert len(noisy) == 1
    assert builder.graph.vertex_type(noisy[0]) == VertexType.X
    assert builder.graph.phase(noisy[0]).parameters == {0}
    g = builder.build(closed=False, outputs="detectors")
    np.testing.assert_allclose(g.to_tensor({0: 0}), doubled(np.eye(2)), atol=1e-9)
    x = SINGLE_QUBIT_GATE_MATRICES["X"]
    np.testing.assert_allclose(g.to_tensor({0: 1}), doubled(x), atol=1e-9)


def test_records_and_detectors_are_classical_spiders():
    builder = GraphBuilder()
    instructions.m(builder, 0)
    builder.add_detector([0])
    (rec,) = builder.records
    (det,) = builder.detectors
    g = builder.graph
    assert (g.vertex_type(rec), g.vertex_domain(rec)) == (
        VertexType.Z,
        Domain.CLASSICAL,
    )
    assert (g.vertex_type(det), g.vertex_domain(det)) == (
        VertexType.X,
        Domain.CLASSICAL,
    )
    assert (g.vertex_label(rec), g.vertex_label(det)) == ("rec[0]", "det[0]")
    assert g.has_edge(rec, det)


def test_cz_between_two_measurement_records_does_nothing():
    with_cz = Circuit("H 0 1\nM 0 1\nCZ rec[-1] rec[-2]\nH 0")
    without = Circuit("H 0 1\nM 0 1\nH 0")
    np.testing.assert_allclose(with_cz.to_tensor(), without.to_tensor(), atol=1e-12)


def test_pending_correlated_error_must_be_finalized():
    builder = GraphBuilder()
    builder.correlated_error([0], ["X"], 0.1)
    reads = [
        lambda: builder.channel_probs,
        lambda: builder.num_error_bits,
        lambda: builder.build(closed=True, outputs="records"),
    ]
    for read in reads:
        with pytest.raises(RuntimeError, match="finalize_correlated_error"):
            read()
    builder.finalize_correlated_error()
    assert builder.num_error_bits == 1
    assert [p.shape for p in builder.channel_probs] == [(2,)]


def test_correlated_error_with_mismatched_targets_changes_nothing():
    builder = GraphBuilder()
    with pytest.raises(ValueError):
        builder.correlated_error([0, 1], ["X"], 0.1)
    assert builder.graph.num_vertices() == 0
    builder.correlated_error([0], ["Z"], 0.2)
    builder.finalize_correlated_error()
    g = builder.graph
    noisy = [v for v in g.vertices() if g.phase(v).parameters]
    assert [g.vertex_type(v) for v in noisy] == [VertexType.Z]


def test_add_channel_rejects_non_power_of_two():
    builder = GraphBuilder()
    with pytest.raises(ValueError, match="2\\^k outcomes"):
        builder.add_channel(np.full(3, 1 / 3))
    assert builder.num_error_bits == 0
    assert builder.channel_probs == []


@pytest.mark.parametrize("gate", [instructions.cnot, instructions.cz, instructions.cy])
def test_unset_classical_control_flags_give_the_quantum_gate(gate):
    np.testing.assert_allclose(
        _tensor(
            lambda builder: gate(builder, 0, 1, classically_controlled=[False, False])
        ),
        _tensor(lambda builder: gate(builder, 0, 1)),
        atol=1e-12,
    )
