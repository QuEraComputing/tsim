import pytest
import stim

from tsim.noise.dem import get_detector_error_model


def test_get_detector_error_model():
    c_with_nondet_obs = stim.Circuit(
        """
        RX 6
        S 6
        H 6
        R 0 1 2 3 4 5
        SQRT_Y_DAG 0 1 2 3 4 5
        CZ 1 2 3 4 5 6
        SQRT_Y 6
        CZ 0 3 2 5 4 6
        DEPOLARIZE2(0.01) 0 3 2 5 4 6
        SQRT_Y 2 3 4 5 6
        DEPOLARIZE1(0.01) 0 1 2 3 4 5 6
        CZ 0 1 2 3 4 5
        DEPOLARIZE2(0.01) 0 1 2 3 4 5
        SQRT_Y 1 2 4
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-3]
        M 3 4 5 6
        DETECTOR rec[-7] rec[-6] rec[-5] rec[-4]
        DETECTOR rec[-6] rec[-5] rec[-3] rec[-2]
        DETECTOR rec[-5] rec[-4] rec[-3] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-2]
        """
    )

    c = stim.Circuit(
        """
        R 0 1 2 3 4 5
        SQRT_Y_DAG 0 1 2 3 4 5
        CZ 1 2 3 4 5 6
        SQRT_Y 6
        CZ 0 3 2 5 4 6
        DEPOLARIZE2(0.01) 0 3 2 5 4 6
        SQRT_Y 2 3 4 5 6
        DEPOLARIZE1(0.01) 0 1 2 3 4 5 6
        CZ 0 1 2 3 4 5
        DEPOLARIZE2(0.01) 0 1 2 3 4 5
        SQRT_Y 1 2 4
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-3]
        M 3 4 5 6
        DETECTOR rec[-7] rec[-6] rec[-5] rec[-4]
        DETECTOR rec[-6] rec[-5] rec[-3] rec[-2]
        DETECTOR rec[-5] rec[-4] rec[-3] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-2]
        """
    )

    dem = get_detector_error_model(c_with_nondet_obs)
    dem2 = c.detector_error_model()
    assert dem.approx_equals(dem2, atol=1e-12)


def test_get_detector_error_model_with_gauge_detectors():
    c = stim.Circuit(
        """
        R 0 1
        H 0
        CNOT 0 1
        H 1
        X_ERROR(0.01) 0
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        DETECTOR rec[-2]
        """
    )
    assert get_detector_error_model(c).approx_equals(
        stim.DetectorErrorModel("error(0.01) D0 L0"), atol=1e-12
    )


def test_get_detector_error_model_no_errors():
    c = stim.Circuit(
        """
        R 0 1
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    assert str(get_detector_error_model(c)) == "logical_observable L0"

    c = stim.Circuit(
        """
        R 0 1
        M 0 1
        DETECTOR rec[-1]
        """
    )
    assert str(get_detector_error_model(c)) == "detector D0"

    c = stim.Circuit(
        """
        R 0 1
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )
    assert (
        str(get_detector_error_model(c))
        == "logical_observable L0\nlogical_observable L1"
    )


def test_get_detector_error_model_with_logical_observables():

    with pytest.raises(
        ValueError, match="The number of observables changed after conversion."
    ):
        c = stim.Circuit(
            """
            R 0
            H 0
            X_ERROR(0.01) 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
            """
        )
        get_detector_error_model(c)
