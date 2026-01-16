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
    # Gauge errors that only trigger observables (error(0.5) L0) are removed,
    # but gauge errors that trigger detectors (error(0.5) D0) are kept.
    assert get_detector_error_model(c).approx_equals(
        stim.DetectorErrorModel("error(0.5) D0\n error(0.01) D0 L0"),
        atol=1e-12,
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


def test_get_detector_error_model_with_mpp_single_product():
    """Test that a single MPP product is correctly counted as 1 measurement."""
    # MPP X0*X1*X2 measures a single Pauli product and produces exactly 1 measurement
    c = stim.Circuit(
        """
        R 0 1 2
        H 0 1 2
        X_ERROR(0.01) 0
        MPP X0*X1*X2
        OBSERVABLE_INCLUDE(0) rec[-1]
        M 0
        DETECTOR rec[-1] rec[-2]
        """
    )
    dem = get_detector_error_model(c)
    assert dem.num_detectors == 1
    assert dem.num_observables == 1
    assert "D0" in str(dem)
    assert "L0" in str(dem)


def test_get_detector_error_model_with_mpp_multiple_products():
    """Test that MPP with multiple products produces one measurement per product.

    MPP X0*X1 Z2*Z3 Y4 produces 3 measurements (one per space-separated product).
    """
    c = stim.Circuit(
        """
        R 0 1 2 3 4
        H 0 1 2 3 4
        X_ERROR(0.01) 0
        MPP X0*X1 Z2*Z3 Y4
        OBSERVABLE_INCLUDE(0) rec[-3]
        M 0
        DETECTOR rec[-1] rec[-4]
        """
    )
    dem = get_detector_error_model(c)
    # rec[-3] refers to the first MPP product (X0*X1) since MPP produces 3 measurements
    # and then M produces 1, so rec[-4] is X0*X1, rec[-3] is Z2*Z3, rec[-2] is Y4, rec[-1] is M
    assert dem.num_detectors == 1
    assert dem.num_observables == 1


def test_get_detector_error_model_with_multiple_mpp_instructions():
    """Test multiple separate MPP instructions with OBSERVABLE_INCLUDE between them."""
    c = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1 2 3
        X_ERROR(0.01) 0
        MPP X0*X1
        OBSERVABLE_INCLUDE(0) rec[-1]
        MPP X2*X3
        DETECTOR rec[-1] rec[-2]
        """
    )
    dem = get_detector_error_model(c)
    assert dem.num_detectors == 1
    assert dem.num_observables == 1


def test_get_detector_error_model_mpp_measurement_counting():
    """Test correct measurement counting for MPP vs regular M measurements.

    This test verifies that rec indices are correctly adjusted when MPP instructions
    produce multiple measurements. The circuit has a non-deterministic observable
    (Z2*Z3 measured on |++⟩ state), which should raise a ValueError because stim
    interprets it as a gauge and eliminates it.
    """
    # Circuit with MPP producing 2 measurements + M producing 2 measurements
    c = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1 2 3
        Z_ERROR(0.01) 0
        MPP Z0*Z1 Z2*Z3
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-3]
        DETECTOR rec[-4] rec[-2] rec[-1]
        """
    )
    # MPP Z0*Z1 Z2*Z3 produces 2 measurements: rec[-4] and rec[-3] (before M)
    # M 0 1 produces 2 measurements: rec[-2] and rec[-1]
    # OBSERVABLE_INCLUDE(0) rec[-3] refers to the Z2*Z3 measurement
    # Since the observable is non-deterministic (gauge), it gets eliminated
    with pytest.raises(
        ValueError, match="The number of observables changed after conversion."
    ):
        get_detector_error_model(c)
