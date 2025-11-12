import random

import numpy as np
import stim

from tsim.external.vec_sim import VecSampler


def test_vec_sampler_bell_state():
    c = stim.Circuit(
        """
        R 0 1
        H 0
        CNOT 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        """
    )
    random.seed(0)
    sampler = VecSampler(c, False)
    m, d, _ = sampler.sample(10)

    assert np.array_equal(m[:, 0], m[:, 1])
    assert np.count_nonzero(m[:, 0]) == 4
    assert (d == 0).all()


def test_vec_sampler_bell_state_with_measurement_error():
    c = stim.Circuit(
        """
        R 0 1
        H 0
        CNOT 0 1
        X_ERROR(0.3) 0
        M 0 1
        DETECTOR rec[-1] rec[-2]
        """
    )
    sampler = VecSampler(c, False)

    random.seed(0)
    _, d, _ = sampler.sample(10)
    assert np.count_nonzero(d) == 2


def test_t_gate():
    c = stim.Circuit(
        """
        RX 0
        S[T] 0
        H 0
        M 0
        """
    )
    random.seed(0)
    sampler = VecSampler(c, False)
    m, _, _ = sampler.sample(100)
    assert np.count_nonzero(m) == 12


def test_s_gate():
    c = stim.Circuit(
        """
        RX 0
        S 0
        H 0
        M 0
        """
    )
    random.seed(0)
    sampler = VecSampler(c, False)
    m, _, _ = sampler.sample(100)
    assert np.count_nonzero(m) == 42


def test_t_dag_gate():
    c = stim.Circuit(
        """
        RX 0
        S[T] 0
        S_DAG[T] 0
        H 0
        M 0
        """
    )
    random.seed(0)
    sampler = VecSampler(c, False)
    m, _, _ = sampler.sample(10)
    assert np.count_nonzero(m) == 0


def test_s_dag_gate():
    c = stim.Circuit(
        """
        RX 0
        S 0
        S_DAG 0
        H 0
        M 0
        """
    )
    random.seed(0)
    sampler = VecSampler(c, False)
    m, _, _ = sampler.sample(10)
    assert np.count_nonzero(m) == 0
