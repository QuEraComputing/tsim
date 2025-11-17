import numpy as np

from tsim.circuit import Circuit


def test_sample_bell_state():
    c = Circuit.from_stim_program_text(
        """
        R 0 1
        H 0
        CNOT 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        """
    )
    sampler = c.compile_sampler()
    m = sampler.sample(100)

    assert np.array_equal(m[:, 0], m[:, 1])
    assert np.count_nonzero(m[:, 0]) == 54


def test_detector_sampler_bell_state_with_measurement_error():
    c = Circuit.from_stim_program_text(
        """
        R 0 1
        H 0
        CNOT 0 1
        X_ERROR(0.3) 0
        M 0 1
        DETECTOR rec[-1] rec[-2]
        """
    )
    sampler = c.compile_detector_sampler()

    d = sampler.sample(10)
    assert np.count_nonzero(d) == 4


def test_t_gate():
    c = Circuit.from_stim_program_text(
        """
        RX 0
        S[T] 0
        H 0
        M 0
        """
    )
    sampler = c.compile_sampler()
    m = sampler.sample(100)
    assert np.count_nonzero(m) == 15


def test_s_gate():
    c = Circuit.from_stim_program_text(
        """
        RX 0
        S 0
        H 0
        M 0
        """
    )
    sampler = c.compile_sampler()
    m = sampler.sample(100)
    assert np.count_nonzero(m) == 54


def test_t_dag_gate():
    c = Circuit.from_stim_program_text(
        """
        RX 0
        S[T] 0
        S_DAG[T] 0
        H 0
        M 0
        """
    )
    sampler = c.compile_sampler()
    m = sampler.sample(10)
    assert np.count_nonzero(m) == 0


def test_s_dag_gate():
    c = Circuit.from_stim_program_text(
        """
        RX 0
        S 0
        S_DAG 0
        H 0
        M 0
        """
    )
    sampler = c.compile_sampler()
    m = sampler.sample(10)
    assert np.count_nonzero(m) == 0


def test_r_gate():
    c = Circuit.from_stim_program_text(
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
    sampler = c.compile_sampler()
    m = sampler.sample(10)
    assert np.count_nonzero(m[:, 0]) == 6
    assert np.count_nonzero(m[:, 1]) == 4
    assert np.count_nonzero(m[:, 2]) == 0

    det_sampler = c.compile_detector_sampler()
    d = det_sampler.sample(10)
    assert np.count_nonzero(d) == 6
