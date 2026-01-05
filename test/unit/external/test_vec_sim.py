"""
Tests for the VecSim class.

Based on code from:
Gidney, C., Jones, C., & Shutty, N. (2024). "Magic state cultivation: growing
T states as cheap as CNOT gates." arXiv:2409.17595


Code dataset: https://doi.org/10.5281/zenodo.13777072
Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

No modifications were made to the original code in this file.
"""

import numpy as np
import stim

from tsim.external.vec_sim import VecSim


def test_alloc():
    s = VecSim()
    s.do_qalloc_z("a")
    assert not s.do_mz("a", key=None)
    s.do_x("a")
    assert s.do_mz("a", key=None)
    s.do_x("a")
    assert not s.do_mz("a", key=None)


def test_cx():
    s = VecSim()
    s.do_qalloc_z("a")
    s.do_qalloc_z("b")
    assert not s.do_mz("a", key=None)
    assert not s.do_mz("b", key=None)

    s.do_x("a")
    assert s.do_mz("a", key=None)
    assert not s.do_mz("b", key=None)

    s.do_x("b")
    assert s.do_mz("a", key=None)
    assert s.do_mz("b", key=None)

    s.do_cx("a", "b")
    assert s.do_mz("a", key=None)
    assert not s.do_mz("b", key=None)


def test_x_basis():
    s = VecSim()
    s.do_qalloc_z("a")
    s.do_h("a")
    assert not s.do_mx("a", key=None)
    s.do_z("a")
    assert s.do_mx("a", key=None)
    s.do_z("a")

    s.do_qalloc_x("b")
    assert not s.do_mx("a", key=None)
    assert not s.do_mx("b", key=None)
    assert not s.do_mxx("a", "b", key=None)
    s.do_z("b")
    assert s.do_mxx("a", "b", key=None)
    s.do_z("a")
    assert not s.do_mxx("a", "b", key=None)
    s.do_z("b")
    assert s.do_mxx("a", "b", key=None)


def test_y_basis():
    s = VecSim()
    s.do_qalloc_y("a")
    s.do_s_dag("a")
    s.do_h("a")
    assert not s.do_mz("a", key=None)

    s.do_ry("a")
    s.do_s_dag("a")
    s.do_h("a")
    assert not s.do_mz("a", key=None)

    s.do_ry("a")
    assert not s.do_my("a", key=None)
    s.do_x("a")
    assert s.do_my("a", key=None)


def test_h_cx_h():
    s = VecSim()
    s.do_qalloc_z("a")
    s.do_qalloc_z("b")

    s.do_x("b")
    s.do_h("a")
    s.do_h("b")
    s.do_cx("a", "b")
    s.do_h("a")
    s.do_h("b")
    assert s.do_mz("a", key=None)
    assert s.do_mz("b", key=None)


def test_mzz():
    s = VecSim()
    s.do_qalloc_z("a")
    s.do_qalloc_z("b")
    assert not s.do_mzz("a", "b", key=None)
    s.do_x("a")
    assert s.do_mzz("a", "b", key=None)
    s.do_x("b")
    assert not s.do_mzz("a", "b", key=None)
    s.do_x("a")
    assert s.do_mzz("a", "b", key=None)
    assert not s.do_mz("a", key=None)
    assert s.do_mz("b", key=None)


def test_stabilizers():
    sim = VecSim()
    sim.do_qalloc_z(0)
    sim.do_qalloc_z(1)
    sim.do_qalloc_z(2)
    sim.do_h(0)
    sim.do_cx(0, 1)
    sim.do_cx(0, 2)
    s = sim.state[sim.state_slicer({})]
    v = list((s / np.linalg.norm(s)).flat)
    t: stim.Tableau = stim.Tableau.from_state_vector(v, endian="little")
    assert len(t) == 3
    stabilizers = [t.z_output(k) for k in range(3)]
    assert stabilizers == [
        stim.PauliString("XXX"),
        stim.PauliString("ZZ_"),
        stim.PauliString("Z_Z"),
    ]


def test_state_order():
    sim = VecSim()
    sim.do_qalloc_z("a")
    sim.do_qalloc_z("b")
    sim.do_x("b")
    np.testing.assert_array_equal(
        sim.normalized_state(order="ab".index),
        np.array([[0, 1], [0, 0]], dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        sim.normalized_state(order="ba".index),
        np.array([[0, 0], [1, 0]], dtype=np.complex64),
    )

    sim.do_qalloc_z("c")
    sim.do_h("c")
    sim.do_z("c")
    s = 0.5**0.5
    np.testing.assert_allclose(
        sim.normalized_state(order="abc".index).flat,
        np.array([0, 0, s, -s, 0, 0, 0, 0], dtype=np.complex64),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sim.normalized_state(order="acb".index).flat,
        np.array([0, s, 0, -s, 0, 0, 0, 0], dtype=np.complex64),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sim.normalized_state(order="cba".index).flat,
        np.array([0, 0, s, 0, 0, 0, -s, 0], dtype=np.complex64),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sim.normalized_state(order="cab".index).flat,
        np.array([0, s, 0, 0, 0, -s, 0, 0], dtype=np.complex64),
        atol=1e-5,
    )


def test_state_order_more():
    sim = VecSim()

    def check(order_string: str, hot_index: int):
        actual = sim.normalized_state(order=order_string.index).flatten()
        expected = np.zeros(1 << len(sim.q2i), dtype=np.complex64)
        expected[hot_index] = 1
        np.testing.assert_array_equal(actual, expected)

    sim.do_qalloc_z("e")
    sim.do_qalloc_z("b")
    sim.do_qalloc_z("c")
    sim.do_qalloc_z("d")
    sim.do_qalloc_z("a")
    sim.do_mz_discard("e")
    assert sim.q2i == {"b": 1, "c": 2, "d": 3, "a": 0}
    assert list(sim.q2i) == ["b", "c", "d", "a"]
    sim.do_x("c")
    check("abdc", 0b0001)
    check("cabd", 0b1000)
    check("abcd", 0b0010)
    sim.do_x("a")
    check("abcd", 0b1010)
    check("dcba", 0b0101)


def test_do_t_obs_3z():
    sim = VecSim()
    sim.do_qalloc_x("a")
    sim.do_qalloc_x("b")
    sim.do_qalloc_x("c")
    a = (2**-0.5) ** 3
    b = a * 1j**0.5
    sim.do_t_obs({"a": "Z", "b": "Z", "c": "Z"})
    np.testing.assert_allclose(
        sim.normalized_state(order="abc".index).flat,
        np.array([a, b, b, a, b, a, a, b], dtype=np.complex64),
        atol=1e-5,
    )


def test_do_t_obs_xz():
    sim = VecSim()
    sim.do_qalloc_z("a")
    sim.do_qalloc_z("b")
    sim.do_t_obs({"a": "Z", "b": "X"})
    np.testing.assert_allclose(
        sim.normalized_state(order="ab".index).flat,
        np.array([0.853553 + 0.35353j, 0.146447 - 0.353553j, 0, 0], dtype=np.complex64),
        atol=1e-3,
    )


def test_do_t_obs_xz_inverted():
    sim = VecSim()
    sim.do_qalloc_z("a")
    sim.do_qalloc_z("b")
    sim.do_x("a")
    sim.do_t_obs({"a": "Z", "b": "X"})
    np.testing.assert_allclose(
        sim.normalized_state(order="ab".index).flat,
        np.array(
            [0, 0, 0.853553 + 0.35353j, -0.146447 + 0.353553j], dtype=np.complex64
        ),
        atol=1e-3,
    )
