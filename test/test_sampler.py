import numpy as np

from tsim.circuit import Circuit


def test_detector_sampler_args():
    c = Circuit(
        """
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    sampler = c.compile_detector_sampler()
    d = sampler.sample(1)
    assert np.array_equal(d, np.array([[0, 0]]))

    d = sampler.sample(1, append_observables=True)
    assert np.array_equal(d, np.array([[0, 0, 1]]))

    d = sampler.sample(1, prepend_observables=True)
    assert np.array_equal(d, np.array([[1, 0, 0]]))

    d, o = sampler.sample(1, separate_observables=True)
    assert np.array_equal(d, np.array([[0, 0]]))
    assert np.array_equal(o, np.array([[1]]))


def test_seed():
    c = Circuit(
        """
        H 0
        M 0
        """
    )
    for i in range(2):
        sampler = c.compile_sampler(seed=0)
        assert np.count_nonzero(sampler.sample(100)) == 54
        assert np.count_nonzero(sampler.sample(100)) == 50
        assert np.count_nonzero(sampler.sample(100)) == 54
        assert np.count_nonzero(sampler.sample(100)) == 49
