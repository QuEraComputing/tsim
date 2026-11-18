"""Tests for ``tsim.noise.faults``: the exact search for shortest logical errors."""

import itertools

import numpy as np
import pytest
import stim

from tsim import Circuit
from tsim.experimental import shortest_errors
from tsim.noise.faults import _combinations, _distinct, _Search

# A Bell pair is stabilized by XX and ZZ, also under T on one qubit and T_DAG on
# the other. An X error between the T layers turns into (X ± Y)/√2, which the XX
# detector catches half of the time; between S layers it turns into Y.
BELL = """
    R 0 1
    H 0
    CX 0 1
    {gate} 0
    {gate}_DAG 1
    X_ERROR(0.01) 0 1
    {gate}_DAG 0
    {gate} 1
    MPP X0*X1
    DETECTOR rec[-1]
    MPP Z0*Z1
    OBSERVABLE_INCLUDE(0) rec[-1]
"""

GHZ = """
    R 0 1 2
    H 0
    CX 0 1 0 2
    T 0 1 2
    DEPOLARIZE1(0.01) 0 1 2
    T_DAG 0 1 2
    TICK
    DEPOLARIZE2(0.01) 0 1
    MPP X0*X1*X2 Z0*Z1
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    MPP Z1*Z2
    OBSERVABLE_INCLUDE(0) rec[-1]
"""

# One round of a three-qubit repetition code, with every kind of noise.
NOISES = """
    QUBIT_COORDS(0, 1) 0
    R 0 1 2
    E(0.01) X0 X1
    ELSE_CORRELATED_ERROR(0.02) X2 Z1
    DEPOLARIZE2(0.01) 0 1
    TICK
    HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 1
    DETECTOR rec[-1]
    MPP(0.01) Z0*Z1 Z1*Z2
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    TICK
    PAULI_CHANNEL_1(0.01, 0.02, 0) 2
    PAULI_CHANNEL_2(0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0) 0 2
    Y_ERROR(0.01) 0
    MZZ(0.01) 0 1 1 2
    DETECTOR rec[-2] rec[-4]
    DETECTOR rec[-1] rec[-3]
    MR(0.01) 0
    M(0.01) 1 2
    DETECTOR rec[-3] rec[-2] rec[-5]
    DETECTOR rec[-2] rec[-1] rec[-4]
    OBSERVABLE_INCLUDE(0) rec[-1]
"""


def _generated(task: str, distance: int) -> stim.Circuit:
    return stim.Circuit.generated(
        task,
        distance=distance,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )


def _with_errors(circuit: Circuit, error) -> Circuit:
    """Return the noiseless circuit with one circuit error of each fault."""
    flat = circuit.stim_circuit.flattened()
    inserted: dict[int, stim.Circuit] = {}
    for fault in error:
        location = fault.circuit_error_locations[0]
        (frame,) = location.stack_frames
        paulis = inserted.setdefault(frame.instruction_offset, stim.Circuit())
        for target in location.flipped_pauli_product:
            pauli = target.gate_target.pauli_type
            paulis.append(f"{pauli}_ERROR", [target.gate_target.value], 1)
    out = stim.Circuit()
    for index in range(len(flat)):
        out += flat[index : index + 1].without_noise()
        out += inserted.get(index, stim.Circuit())
    return Circuit(out)


@pytest.mark.parametrize(
    ("task", "distance"),
    [
        ("repetition_code:memory", 3),
        ("repetition_code:memory", 5),
        ("surface_code:rotated_memory_z", 3),
        # NOTE: hook errors halve the distance of Stim's color code circuit.
        ("color_code:memory_xyz", 3),
    ],
)
def test_the_distance_of_a_clifford_circuit_is_stims(task, distance):
    circuit = _generated(task, distance)
    # NOTE: Stim's search gives an upper bound, which is tight for these circuits.
    expected = len(
        circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=6,
            dont_explore_edges_with_degree_above=6,
            dont_explore_edges_increasing_symptom_degree=False,
        )
    )
    errors = shortest_errors(Circuit(circuit), max_weight=distance)
    assert errors
    assert {len(error) for error in errors} == {expected}
    for error in errors:
        assert error.undetected_probability == pytest.approx(1.0)
        assert error.logical_flip_probability == pytest.approx(1.0)
    assert shortest_errors(Circuit(circuit), max_weight=expected - 1) == []


@pytest.mark.parametrize(
    "task", ["repetition_code:memory", "surface_code:rotated_memory_z"]
)
def test_the_errors_are_undetectable_logical_errors_for_stim(task):
    circuit = _generated(task, 3)
    reference = circuit.reference_sample()
    converter = circuit.compile_m2d_converter()
    quiet = converter.convert(
        measurements=reference[None, :], separate_observables=True
    )
    for error in shortest_errors(Circuit(circuit), max_weight=3)[:20]:
        noisy = _with_errors(Circuit(circuit), error).stim_circuit
        sample = noisy.compile_sampler().sample(1)
        detectors, observables = converter.convert(
            measurements=sample, separate_observables=True
        )
        assert np.array_equal(detectors, quiet[0])
        assert not np.array_equal(observables, quiet[1])


@pytest.mark.parametrize(
    "text", [NOISES, str(_generated("surface_code:rotated_memory_z", 3))]
)
def test_the_circuit_error_locations_are_stims(text):
    circuit = stim.Circuit(text)
    explained = circuit.flattened().explain_detector_error_model_errors()
    expected = {str(loc) for e in explained for loc in e.circuit_error_locations}
    search = _Search(Circuit(circuit))
    got = {
        str(location)
        for index in range(len(search._events))
        for location in search._fault(index).circuit_error_locations
    }
    assert got == expected


def test_t_gates_make_an_error_undetected_half_of_the_time():
    (error,) = shortest_errors(Circuit(BELL.format(gate="T")), max_weight=2)
    assert len(error) == 1
    assert error.undetected_probability == pytest.approx(0.5)
    assert error.logical_flip_probability == pytest.approx(0.5)
    (fault,) = error
    products = [
        [str(t.gate_target) for t in location.flipped_pauli_product]
        for location in fault.circuit_error_locations
    ]
    # NOTE: on a Bell pair, an X on either qubit is the same error.
    assert products == [["stim.target_x(0)"], ["stim.target_x(1)"]]
    assert fault.probability == pytest.approx(0.02)
    location = fault.circuit_error_locations[0]
    assert location.instruction_targets.gate == "X_ERROR"
    assert location.stack_frames[0].instruction_offset == 5
    assert location.flipped_measurement is None

    assert shortest_errors(Circuit(BELL.format(gate="S")), max_weight=2) == []


def test_the_probabilities_are_those_of_sampling_the_errors():
    circuit = Circuit(GHZ)
    errors = shortest_errors(circuit, max_weight=2)
    assert {len(error) for error in errors} == {1}
    assert {round(error.undetected_probability, 6) for error in errors} == {0.5, 1.0}
    shots = 20_000
    for error in errors:
        sampler = _with_errors(circuit, error).compile_detector_sampler(seed=0)
        detectors, observables = sampler.sample(
            shots,
            separate_observables=True,
            use_detector_reference_sample=True,
            use_observable_reference_sample=True,
        )
        quiet = ~detectors.any(axis=1)
        flipped = quiet & observables.any(axis=1)
        for got, expected in (
            (quiet.mean(), error.undetected_probability),
            (flipped.mean(), error.logical_flip_probability),
        ):
            assert abs(got - expected) < 5 * np.sqrt(0.25 / shots)


def test_the_most_likely_error_comes_first():
    errors = shortest_errors(Circuit(GHZ), max_weight=1)

    def likelihood(error):
        return error.logical_flip_probability * np.prod(
            [fault.probability for fault in error]
        )

    assert [likelihood(e) for e in errors] == sorted(map(likelihood, errors))[::-1]


def test_errors_of_one_channel_do_not_combine():
    # NOTE: X and Y each flip the ZZ detector and together are a Z, which flips
    # only the observable. One PAULI_CHANNEL_1 applies one of them.
    one_channel = """
        R 0 1
        H 0
        CX 0 1
        PAULI_CHANNEL_1(0.01, 0.01, 0) 0
        MPP Z0*Z1
        DETECTOR rec[-1]
        MPP X0*X1
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    assert shortest_errors(Circuit(one_channel), max_weight=2) == []
    two_channels = one_channel.replace(
        "PAULI_CHANNEL_1(0.01, 0.01, 0) 0", "X_ERROR(0.01) 0\nY_ERROR(0.01) 0"
    )
    (error,) = shortest_errors(Circuit(two_channels), max_weight=2)
    assert len(error) == 2


def test_no_observable_or_no_noise_has_no_errors():
    assert (
        shortest_errors(Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]"), max_weight=3)
        == []
    )
    assert (
        shortest_errors(Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]"), max_weight=3)
        == []
    )


def test_an_undetected_single_error_has_weight_one():
    (error,) = shortest_errors(
        Circuit("X_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"), max_weight=3
    )
    assert len(error) == 1


def test_random_detectors_and_observables_are_rejected():
    with pytest.raises(ValueError, match="detector 0 is random without noise"):
        shortest_errors(
            Circuit(
                "RX 0\nT 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"
            ),
            max_weight=1,
        )
    with pytest.raises(ValueError, match="observable 0 is random without noise"):
        shortest_errors(
            Circuit("RX 0\nT 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"),
            max_weight=1,
        )


def test_max_weight_is_checked():
    with pytest.raises(ValueError, match="max_weight must be at least 1"):
        shortest_errors(Circuit(GHZ), max_weight=0)


def test_a_search_that_is_too_large_is_refused(monkeypatch):
    monkeypatch.setattr("tsim.noise.faults.MAX_COMBINATIONS", 10)
    with pytest.raises(ValueError, match="lower max_weight"):
        shortest_errors(Circuit(_generated("repetition_code:memory", 3)), max_weight=3)


@pytest.mark.parametrize(("n", "size"), [(5, 0), (5, 1), (6, 2), (7, 3), (3, 4)])
def test_combinations(n, size):
    expected = list(itertools.combinations(range(n), size))
    assert [tuple(row) for row in _combinations(n, size).tolist()] == expected


def test_distinct():
    assert _distinct([{1}, {2}])
    assert _distinct([{1, 2}, {1}])
    assert not _distinct([{1}, {1}])
    assert not _distinct([{1, 2}, {1, 2}, {1, 2}])


def test_full_buffers_grow(monkeypatch):
    circuit = Circuit(_generated("surface_code:rotated_memory_z", 3))
    expected = shortest_errors(circuit, max_weight=3)
    monkeypatch.setattr("tsim.noise.faults.FIRST_CAPACITY", 1)
    got = shortest_errors(circuit, max_weight=3)
    assert [str(e) for e in got] == [str(e) for e in expected]


def test_two_components_combine():
    # NOTE: two Bell pairs with T gates and an observable each are two
    # statevector components.
    pair = BELL.format(gate="T")
    shifted = pair.replace(" 0", " 2").replace(" 1", " 3").replace("X0*X1", "X2*X3")
    shifted = shifted.replace("Z0*Z1", "Z2*Z3").replace(
        "OBSERVABLE_INCLUDE(0)", "OBSERVABLE_INCLUDE(1)"
    )
    circuit = Circuit(pair + shifted)
    assert circuit.num_observables == 2
    errors = shortest_errors(circuit, max_weight=2)
    assert len(errors) == 2
    for error in errors:
        assert len(error) == 1
        assert error.undetected_probability == pytest.approx(0.5)
        assert error.logical_flip_probability == pytest.approx(0.5)
    qubits = {
        location.flipped_pauli_product[0].gate_target.value
        for error in errors
        for location in error.faults[0].circuit_error_locations
    }
    assert qubits == {0, 1, 2, 3}


def test_max_errors_keeps_the_most_likely():
    circuit = Circuit(_generated("surface_code:rotated_memory_z", 3))
    every = shortest_errors(circuit, max_weight=3)
    best = shortest_errors(circuit, max_weight=3, max_errors=5)
    assert len(every) > 5
    assert [str(e) for e in best] == [str(e) for e in every[:5]]
