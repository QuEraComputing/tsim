from test.helpers import get_matrix

import numpy as np
import pytest

from tsim.circuit import Circuit
from tsim.sampler import CompiledStateProbs


def test_sample_bell_state():
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        """)
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(100)

    assert np.array_equal(m[:, 0], m[:, 1])
    assert np.count_nonzero(m[:, 0]) == 48


def test_detector_sampler_bell_state_with_measurement_error():
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        X_ERROR(0.3) 0
        M 0 1
        DETECTOR rec[-1] rec[-2]
        """)
    sampler = c.compile_detector_sampler(seed=1)

    d = sampler.sample(10)
    assert np.count_nonzero(d) == 4


def test_t_gate():
    c = Circuit("""
        RX 0
        S[T] 0
        H 0
        M 0
        """)
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(100)
    assert np.count_nonzero(m) == 9


def test_s_gate():
    c = Circuit("""
        RX 0
        S 0
        H 0
        M 0
        """)
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(100)
    assert np.count_nonzero(m) == 48


def test_t_dag_gate():
    c = Circuit("""
        RX 0
        S[T] 0
        S_DAG[T] 0
        H 0
        M 0
        """)
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(10)
    assert np.count_nonzero(m) == 0


def test_s_dag_gate():
    c = Circuit("""
        RX 0
        S 0
        S_DAG 0
        H 0
        M 0
        """)
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(10)
    assert np.count_nonzero(m) == 0


def test_r_gate():
    c = Circuit("""
        RX 0
        RX 0
        M 0
        RX 0
        M 0
        DETECTOR rec[-1] rec[-2]
        R 0
        M 0
        """)
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(10)
    assert np.count_nonzero(m[:, 0]) == 7
    assert np.count_nonzero(m[:, 1]) == 4
    assert np.count_nonzero(m[:, 2]) == 0

    det_sampler = c.compile_detector_sampler(seed=0)
    d = det_sampler.sample(10)
    assert np.count_nonzero(d) == 7


@pytest.mark.parametrize(
    "reset_basis,measure_basis",
    [("X", "Y"), ("X", "Z"), ("Y", "X"), ("Y", "Z"), ("Z", "X"), ("Z", "Y")],
)
def test_measurements_stay_same(reset_basis: str, measure_basis: str):
    if reset_basis == measure_basis:
        return

    c = Circuit(f"""
        R{reset_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        """)
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(20)
    meas_stay_same = (res == res[:, [0]]).all(axis=1)
    assert np.all(meas_stay_same)

    # measurement outcomes should be different for different shots
    col = res[:, 0]
    shots_differ = not np.all(col == col[0])
    assert shots_differ


@pytest.mark.parametrize(
    "reset_basis,measure_basis",
    [("X", "Y"), ("X", "Z"), ("Y", "X"), ("Y", "Z"), ("Z", "X"), ("Z", "Y")],
)
def test_mr(measure_basis: str, reset_basis: str):
    c = Circuit(f"""
        R{reset_basis} 0
        MR{measure_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        """)
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(20)

    assert not np.any(res[:, 1:])

    col = res[:, 0]
    shots_differ = not np.all(col == col[0])
    assert shots_differ


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_reset_same_basis_measurement_always_zero(basis: str):

    c = Circuit(f"""
        H 0
        S[T] 0
        H 0
        R{basis} 0
        M{basis} 0
        M{basis} 0
        M{basis} 0
        """)
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)
    assert not np.any(res)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_mr_same_basis_subsequent_measurements_zero(basis: str):

    c = Circuit(f"""
        H 0
        S[T] 0
        H 0
        MR{basis} 0
        M{basis} 0
        M{basis} 0
        """)
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)

    assert not np.any(res[:, 1:])


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_reset_after_state_change(basis: str):
    """Apply gates to change state, then reset -> measurement should be 0."""
    reset_gate = "R" if basis == "Z" else f"R{basis}"
    measure_gate = "M" if basis == "Z" else f"M{basis}"

    c = Circuit(f"""
        H 0
        S 0
        {reset_gate} 0
        {measure_gate} 0
        """)
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)
    assert not np.any(res)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_multiple_resets_same_basis(basis: str):

    c = Circuit(f"""
        H 0
        R{basis} 0
        R{basis} 0
        R{basis} 0
        M{basis} 0
        M{basis} 0
        """)
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)
    assert not np.any(res)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_mr_on_eigenstate_returns_zero(basis: str):
    """MR on an eigenstate of that basis with +1 eigenvalue -> measurement is 0."""
    reset_gate = "R" if basis == "Z" else f"R{basis}"
    mr_gate = "MR" if basis == "Z" else f"MR{basis}"

    c = Circuit(f"""
        {reset_gate} 0
        {mr_gate} 0
        """)
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)
    # Reset puts qubit in +1 eigenstate, so MR should always measure 0
    assert not np.any(res)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_singlet_state(basis: str):

    c = Circuit(f"""
        R 0 1
        X 0
        H 1
        CNOT 1 0
        Z 0
        M{basis} 0 1
        """)
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(20)
    assert (res[:, 0] != res[:, 1]).all()


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_m_inverted_record(basis: str):
    c = Circuit(f"""
        R 0 1 2
        X 0
        H 1
        CNOT 1 0
        Z 0
        M{basis} 0 !0 !0 0
        """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert (samples[:, 0] == samples[:, 3]).all()
    assert (samples[:, 1] == samples[:, 2]).all()
    assert (samples[:, 0] == ~samples[:, 1]).all()


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_mr_inverted_record(basis: str):
    c = Circuit(f"""
        R 0 1 2
        X 0
        H 1
        CNOT 1 0
        Z 0
        MR{basis} 0 !0 !0 0
        """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    print(samples)
    assert (samples[:, 1] == 1).all()
    assert (samples[:, 2] == 1).all()
    assert (samples[:, 3] == 0).all()


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_mpp_inverted_record(basis: str):
    singlet = """
        R 0 1 2
        X 0
        H 1
        CNOT 1 0
        Z 0
        """

    c = Circuit(f"""
        {singlet}
        MPP {basis}0*{basis}1
        """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert samples.all()

    c = Circuit(f"""
        {singlet}
        MPP !{basis}0*{basis}1
        MPP !{basis}0*{basis}1
        """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert (~samples).all()


def test_cx_rec_control():
    c = Circuit("""
        X 0
        M 0
        X 0
        H 0
        CNOT rec[-1] 1
        M 1
        """)
    s = c.compile_sampler()
    samples = s.sample(100)
    assert np.all(samples)


def test_cz_rec_control():
    c = Circuit("""
        X 0
        M 0
        RX 1
        CZ rec[-1] 1
        H 1
        M 1
        """)
    s = c.compile_sampler()
    samples = s.sample(100)
    assert np.all(samples)


def test_rec_control_with_singlet():
    singlet = """
        R 0 1
        X 0
        H 1
        CNOT 1 0
        Z 0
        """

    c = Circuit(f"""
        {singlet}
        M !0 0
        CNOT rec[-2] 0
        CNOT rec[-1] 1
        M 0 1
        """)
    s = c.compile_sampler()
    samples = s.sample(100)
    assert np.all(samples[:, 2:])

    c = Circuit(f"""
        {singlet}
        MX !0 0
        CZ rec[-2] 0
        CZ rec[-1] 1
        MX 0 1
        """)
    s = c.compile_sampler()
    samples = s.sample(100)
    assert np.all(samples[:, 2:])


def test_rec_controlled_effective_reset():
    c = Circuit("""
        RX 0
        M 0
        CNOT rec[-1] 0
        M 0
        """)
    s = c.compile_sampler()
    samples = s.sample(100)
    assert not np.any(samples[:, 1])

    c = Circuit("""
        R 0
        MX 0
        CZ rec[-1] 0
        MX 0
        """)
    s = c.compile_sampler()
    samples = s.sample(100)
    assert not np.any(samples[:, 1])

    c = Circuit("""
        R 0
        MX 0
        CY rec[-1] 0
        MX 0
        """)
    s = c.compile_sampler()
    samples = s.sample(100)
    assert not np.any(samples[:, 1])


def test_rec_controlled_xcz_ycz_zcz():
    c = Circuit("""
        RX 0 1
        M 1
        ZCZ 0 rec[-1]
        MX 0
        """)
    sampler = c.compile_sampler()
    samples = sampler.sample(shots=10)
    assert np.all(samples[:, 0] == samples[:, 1])

    c = Circuit("""
        R 0
        RX 1
        M 1
        XCZ 0 rec[-1]
        M 0
        """)
    sampler = c.compile_sampler()
    samples = sampler.sample(shots=10)
    assert np.all(samples[:, 0] == samples[:, 1])

    c = Circuit("""
        R 0
        RX 1
        M 1
        YCZ 0 rec[-1]
        M 0
        """)
    sampler = c.compile_sampler()
    samples = sampler.sample(shots=10)
    assert np.all(samples[:, 0] == samples[:, 1])


def test_rec_controlled_raises_error():
    c = Circuit("""
        R 0
        RX 1
        M 1
        YCZ rec[-1] 0
        M 0
        """)
    with pytest.raises(
        ValueError, match="Measurement record editing is not supported."
    ):
        c.compile_sampler()


@pytest.mark.parametrize("alpha", [0.34, 0.24, 0.49])
@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_rot_gates(alpha: float, basis: str):
    alpha_pi = alpha * np.pi

    c = Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        H_X{basis} 1
        R_{basis}({alpha}) 1
        H_X{basis} 1
        M 0 1
        """.replace("H_XX 1", ""))
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)

    expected = (
        np.abs(
            [
                [np.cos(alpha_pi / 2), -1j * np.sin(alpha_pi / 2)],
                [-1j * np.sin(alpha_pi / 2), np.cos(alpha_pi / 2)],
            ]
        )
        ** 2
    )
    assert np.allclose(mat, expected)


@pytest.mark.parametrize(
    "theta, phi, lambda_",
    [(0.3, 0.24, 0.49), (0.1, -0.3, 0.2)],
)
def test_u3_gate(theta: float, phi: float, lambda_: float):
    theta_pi = theta * np.pi
    phi_pi = phi * np.pi
    lambda_pi = lambda_ * np.pi

    c = Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        U3({theta}, {phi}, {lambda_}) 1
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)

    expected = (
        np.abs(
            [
                [np.cos(theta_pi / 2), -np.exp(1j * lambda_pi) * np.sin(theta_pi / 2)],
                [
                    np.exp(1j * phi_pi) * np.sin(theta_pi / 2),
                    np.exp(1j * (phi_pi + lambda_pi)) * np.cos(theta_pi / 2),
                ],
            ]
        )
        ** 2
    )
    assert np.allclose(mat, expected)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_rot_gate_identity(basis: str):
    c = Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        R_{basis}(0.34) 1
        DEPOLARIZE1(0.0) 1  # prevent simplification
        R_{basis}(-0.34) 1
        DEPOLARIZE1(0.0) 1  # prevent simplification
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.eye(2))


def test_u3_gate_identity():
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        U3(0.34, 0.24, 0.49) 1
        DEPOLARIZE1(0.0) 1  # prevent simplification
        U3(-0.34, -0.49, -0.24) 1
        DEPOLARIZE1(0.0) 1  # prevent simplification
        M 0 1
        """)
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.eye(2), atol=1e-6)


@pytest.mark.parametrize("n", [2, 5])
def test_many_rx_gates(n: int):
    a = 0.01
    c = Circuit(
        """
        R 0 1
        H 1
        CNOT 1 0
        """
        + f"""
        R_X({a}) 1
        Z_ERROR(0.0) 1  # prevent simplification
        """ * n
        + f"""
        R_X({-a * n}) 1
        M 0 1
        """
    )
    sampler = CompiledStateProbs(c)
    mat = get_matrix(sampler)
    assert np.allclose(mat, np.eye(2), atol=1e-6)


def test_mpad_zero():
    c = Circuit("""
        MPAD 0
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(10)
    assert (samples == 0).all()


def test_mpad_one():
    c = Circuit("""
        MPAD 1
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(10)
    assert (samples == 1).all()


def test_mpad_multiple():
    c = Circuit("""
        MPAD 0 1 1 0
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(10)
    assert (samples[:, 0] == 0).all()
    assert (samples[:, 1] == 1).all()
    assert (samples[:, 2] == 1).all()
    assert (samples[:, 3] == 0).all()


def test_mpad_with_measurements():
    c = Circuit("""
        R 0
        X 0
        M 0
        MPAD 0
        M 0
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(10)
    # M 0 after X -> 1, MPAD 0 -> 0, M 0 after implicit reset -> 1
    assert (samples[:, 0] == 1).all()
    assert (samples[:, 1] == 0).all()
    assert (samples[:, 2] == 1).all()


def test_mpad_detector():
    c = Circuit("""
        MPAD 1
        DETECTOR rec[-1]
    """)
    sampler = c.compile_detector_sampler()
    samples = sampler.sample(10)
    assert (samples == 1).all()


def test_mxx_bell_plus():
    """MXX on |00>+|11> (XX eigenstate with eigenvalue +1) should give 0."""
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        MXX 0 1
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert (samples == 0).all()


def test_mxx_bell_minus():
    """MXX on |00>-|11> (XX eigenstate with eigenvalue -1) should give 1."""
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        Z 0
        MXX 0 1
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert (samples == 1).all()


def test_myy_bell_plus():
    """MYY on |00>+|11> (YY eigenstate with eigenvalue -1) should give 1."""
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        MYY 0 1
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert (samples == 1).all()


def test_mzz_same_state():
    """MZZ on |00> (ZZ eigenstate with eigenvalue +1) should give 0."""
    c = Circuit("""
        R 0 1
        MZZ 0 1
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert (samples == 0).all()


def test_mzz_different_state():
    """MZZ on |01> (ZZ eigenstate with eigenvalue -1) should give 1."""
    c = Circuit("""
        R 0 1
        X 1
        MZZ 0 1
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert (samples == 1).all()


def test_mxx_inverted():
    """MXX with inverted target should flip the result."""
    c = Circuit("""
        R 0 1
        H 0
        CNOT 0 1
        MXX !0 1
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    assert (samples == 1).all()


def test_mzz_multiple_pairs():
    """MZZ with multiple pairs should measure each independently."""
    c = Circuit("""
        R 0 1 2 3
        X 2
        MZZ 0 1 2 3
    """)
    sampler = c.compile_sampler()
    samples = sampler.sample(20)
    # |00> -> ZZ=+1 -> 0, |01> (qubit 2 flipped, qubit 3 not) -> ZZ=-1 -> 1
    assert (samples[:, 0] == 0).all()
    assert (samples[:, 1] == 1).all()


def _get_probs(circuit_str: str) -> np.ndarray:
    """Get probability vector for a circuit."""
    c = Circuit(circuit_str)
    sampler = CompiledStateProbs(c)
    num_qubits = c.num_qubits
    probs = []
    for i in range(2**num_qubits):
        state = np.array(
            [(i >> j) & 1 for j in range(num_qubits)][::-1], dtype=np.bool_
        )
        probs.append(sampler.probability_of(state, batch_size=1)[0])
    return np.array(probs)


def test_spp_z_equals_s():
    """SPP Z0 should be equivalent to S 0."""
    probs_spp = _get_probs("RX 0\nSPP Z0\nM 0")
    probs_s = _get_probs("RX 0\nS 0\nM 0")
    assert np.allclose(probs_spp, probs_s, atol=1e-6)


def test_spp_dag_z_equals_s_dag():
    """SPP_DAG Z0 should be equivalent to S_DAG 0."""
    probs_spp = _get_probs("RX 0\nSPP_DAG Z0\nM 0")
    probs_s = _get_probs("RX 0\nS_DAG 0\nM 0")
    assert np.allclose(probs_spp, probs_s, atol=1e-6)


def test_spp_x_equals_sqrt_x():
    """SPP X0 should be equivalent to SQRT_X 0."""
    probs_spp = _get_probs("R 0\nSPP X0\nM 0")
    probs_sx = _get_probs("R 0\nSQRT_X 0\nM 0")
    assert np.allclose(probs_spp, probs_sx, atol=1e-6)


def test_spp_inverted_equals_dag():
    """SPP !X0 should be equivalent to SQRT_X_DAG 0."""
    probs_spp = _get_probs("R 0\nSPP !X0\nM 0")
    probs_sxd = _get_probs("R 0\nSQRT_X_DAG 0\nM 0")
    assert np.allclose(probs_spp, probs_sxd, atol=1e-6)


def test_spp_dag_inverted_equals_non_dag():
    """SPP_DAG !X0 should be equivalent to SQRT_X 0."""
    probs_spp = _get_probs("R 0\nSPP_DAG !X0\nM 0")
    probs_sx = _get_probs("R 0\nSQRT_X 0\nM 0")
    assert np.allclose(probs_spp, probs_sx, atol=1e-6)


def test_spp_xx_product():
    """SPP X0*X1 should be equivalent to SQRT_XX 0 1."""
    probs_spp = _get_probs("R 0 1\nSPP X0*X1\nM 0 1")
    probs_sxx = _get_probs("R 0 1\nSQRT_XX 0 1\nM 0 1")
    assert np.allclose(probs_spp, probs_sxx, atol=1e-6)


def test_spp_zz_product():
    """SPP Z0*Z1 should be equivalent to SQRT_ZZ 0 1."""
    probs_spp = _get_probs("R 0 1\nSPP Z0*Z1\nM 0 1")
    probs_szz = _get_probs("R 0 1\nSQRT_ZZ 0 1\nM 0 1")
    assert np.allclose(probs_spp, probs_szz, atol=1e-6)


def test_spp_yy_product():
    """SPP Y0*Y1 should be equivalent to SQRT_YY 0 1."""
    probs_spp = _get_probs("R 0 1\nSPP Y0*Y1\nM 0 1")
    probs_syy = _get_probs("R 0 1\nSQRT_YY 0 1\nM 0 1")
    assert np.allclose(probs_spp, probs_syy, atol=1e-6)
