import numpy as np
import pymatching
import pytest
import stim

from tsim.circuit import Circuit


def test_sample_bell_state():
    c = Circuit(
        """
        R 0 1
        H 0
        CNOT 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        """
    )
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(100)

    assert np.array_equal(m[:, 0], m[:, 1])
    assert np.count_nonzero(m[:, 0]) == 51


def test_detector_sampler_bell_state_with_measurement_error():
    c = Circuit(
        """
        R 0 1
        H 0
        CNOT 0 1
        X_ERROR(0.3) 0
        M 0 1
        DETECTOR rec[-1] rec[-2]
        """
    )
    sampler = c.compile_detector_sampler(seed=1)

    d = sampler.sample(10)
    assert np.count_nonzero(d) == 5


def test_t_gate():
    c = Circuit(
        """
        RX 0
        S[T] 0
        H 0
        M 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(100)
    assert np.count_nonzero(m) == 11


def test_s_gate():
    c = Circuit(
        """
        RX 0
        S 0
        H 0
        M 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(100)
    assert np.count_nonzero(m) == 51


def test_t_dag_gate():
    c = Circuit(
        """
        RX 0
        S[T] 0
        S_DAG[T] 0
        H 0
        M 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(10)
    assert np.count_nonzero(m) == 0


def test_s_dag_gate():
    c = Circuit(
        """
        RX 0
        S 0
        S_DAG 0
        H 0
        M 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(10)
    assert np.count_nonzero(m) == 0


def test_r_gate():
    c = Circuit(
        """
        RX 0
        RX 0
        M 0
        RX 0
        M 0
        DETECTOR rec[-1] rec[-2]
        R 0
        M 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    m = sampler.sample(10)
    assert np.count_nonzero(m[:, 0]) == 5
    assert np.count_nonzero(m[:, 1]) == 6
    assert np.count_nonzero(m[:, 2]) == 0

    det_sampler = c.compile_detector_sampler(seed=0)
    d = det_sampler.sample(10)
    assert np.count_nonzero(d) == 5


@pytest.mark.parametrize(
    "reset_basis,measure_basis",
    [("X", "Y"), ("X", "Z"), ("Y", "X"), ("Y", "Z"), ("Z", "X"), ("Z", "Y")],
)
def test_measurements_stay_same(reset_basis: str, measure_basis: str):
    if reset_basis == measure_basis:
        return

    c = Circuit(
        f"""
        R{reset_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        """
    )
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
    c = Circuit(
        f"""
        R{reset_basis} 0
        MR{measure_basis} 0
        M{measure_basis} 0
        M{measure_basis} 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(20)

    assert not np.any(res[:, 1:])

    col = res[:, 0]
    shots_differ = not np.all(col == col[0])
    assert shots_differ


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_reset_same_basis_measurement_always_zero(basis: str):

    c = Circuit(
        f"""
        H 0
        S[T] 0
        H 0
        R{basis} 0
        M{basis} 0
        M{basis} 0
        M{basis} 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)
    assert not np.any(res)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_mr_same_basis_subsequent_measurements_zero(basis: str):

    c = Circuit(
        f"""
        H 0
        S[T] 0
        H 0
        MR{basis} 0
        M{basis} 0
        M{basis} 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)

    assert not np.any(res[:, 1:])


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_reset_after_state_change(basis: str):
    """Apply gates to change state, then reset -> measurement should be 0."""
    reset_gate = "R" if basis == "Z" else f"R{basis}"
    measure_gate = "M" if basis == "Z" else f"M{basis}"

    c = Circuit(
        f"""
        H 0
        S 0
        {reset_gate} 0
        {measure_gate} 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)
    assert not np.any(res)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_multiple_resets_same_basis(basis: str):

    c = Circuit(
        f"""
        H 0
        R{basis} 0
        R{basis} 0
        R{basis} 0
        M{basis} 0
        M{basis} 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)
    assert not np.any(res)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_mr_on_eigenstate_returns_zero(basis: str):
    """MR on an eigenstate of that basis with +1 eigenvalue -> measurement is 0."""
    reset_gate = "R" if basis == "Z" else f"R{basis}"
    mr_gate = "MR" if basis == "Z" else f"MR{basis}"

    c = Circuit(
        f"""
        {reset_gate} 0
        {mr_gate} 0
        """
    )
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(100)
    # Reset puts qubit in +1 eigenstate, so MR should always measure 0
    assert not np.any(res)


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_singlet_state(basis: str):

    c = Circuit(
        f"""
        R 0 1
        X 0
        H 1
        CNOT 1 0
        Z 0
        M{basis} 0 1
        """
    )
    sampler = c.compile_sampler(seed=0)
    res = sampler.sample(20)
    assert (res[:, 0] != res[:, 1]).all()


@pytest.mark.parametrize(
    "code_task",
    [
        "repetition_code:memory",
        "surface_code:rotated_memory_x",
        "surface_code:rotated_memory_z",
        "surface_code:unrotated_memory_x",
        "surface_code:unrotated_memory_z",
        "color_code:memory_xyz",
    ],
)
def test_quantum_memory_codes_without_noise(code_task: str):

    circ = stim.Circuit.generated(
        code_task,
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.0,
        before_measure_flip_probability=0.0,
        before_round_data_depolarization=0.0,
        after_reset_flip_probability=0.0,
    )
    c = Circuit.from_stim_program(circ)
    sampler = c.compile_detector_sampler(seed=0)
    samples = sampler.sample(10)
    assert not np.any(samples)


@pytest.mark.parametrize(
    "code_task",
    [
        "repetition_code:memory",
        "surface_code:rotated_memory_x",
        "surface_code:rotated_memory_z",
        "surface_code:unrotated_memory_x",
        "surface_code:unrotated_memory_z",
    ],
)
def test_memory_error_correction_and_compare_to_stim(code_task: str):
    p = 0.01
    circ = stim.Circuit.generated(
        code_task,
        distance=3,
        rounds=2,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p * 1.2,
        before_round_data_depolarization=p * 0.8,
        after_reset_flip_probability=p * 0.9,
    )
    error_count = []
    error_count_after_correction = []

    for c in [circ, Circuit.from_stim_program(circ)]:
        sampler = c.compile_detector_sampler(seed=0)
        if isinstance(c, Circuit):
            detection_events, observable_flips = sampler.sample(
                10_000, batch_size=10_000, separate_observables=True
            )
        else:
            detection_events, observable_flips = sampler.sample(
                10_000, separate_observables=True
            )

        detector_error_model = circ.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

        predictions = matcher.decode_batch(detection_events)

        num_errors = np.count_nonzero(observable_flips)
        num_errors_after_correction = np.count_nonzero(
            np.logical_xor(observable_flips, predictions)
        )
        error_count.append(num_errors)
        error_count_after_correction.append(num_errors_after_correction)
        assert num_errors_after_correction <= num_errors

    stim_errors = error_count[0]
    tsim_errors = error_count[1]
    stim_errors_after_correction = error_count_after_correction[0]
    tsim_errors_after_correction = error_count_after_correction[1]

    assert np.abs(stim_errors - tsim_errors) / stim_errors <= 0.1
    assert (
        np.abs(stim_errors_after_correction - tsim_errors_after_correction)
        / stim_errors_after_correction
        <= 0.3
    )
