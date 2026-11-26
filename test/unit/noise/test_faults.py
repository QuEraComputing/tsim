"""Tests for ``tsim.noise.faults``: the exact search for shortest logical errors."""

import collections
import itertools

import numpy as np
import pytest
import stim

from tsim import Circuit
from tsim.experimental import shortest_errors
from tsim.noise import fault_pieces, fault_values
from tsim.noise.faults import MIN_PROBABILITY, _distinct, _Search
from tsim.statevector.branches import gf2_product

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

# Inverted measurements, and two links of a chain with the same effect.
INVERTED = """
    R 0 1 2
    E(0.01) X0
    ELSE_CORRELATED_ERROR(0.02) X0
    M(0.01) !0
    DETECTOR rec[-1]
    H 1 2
    MXX(0.01) !1 2
    DETECTOR rec[-1]
    MPP(0.01) !X1*X2
    DETECTOR rec[-1]
    M 0
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


# NOTE: MPP X0*X1 reads qubit 0 after its T gates, so the detector is a
# statevector output. Z0 and Z1 flip no direct detector: each is a set of one
# fault whose direct detectors cancel, caught by the statevector detector, and
# together they flip the observable undetected.
TWO_PIECES = """
    RX 0 1
    T 0
    X_ERROR(0.01) 0
    T_DAG 0
    Z_ERROR(0.01) 0 1
    MPP X0*X1
    DETECTOR rec[-1]
    MX 1
    OBSERVABLE_INCLUDE(0) rec[-1]
"""

# A Stim surface code circuit with T gates around one qubit's noise, cut down.
T_SURFACE = """
    RX 1 3 5 8 10 12 15 17 19
    H 2 11 16 25
    CX 2 3 16 17 11 12 15 14 10 9 19 18
    S_DAG[T] 15
    DEPOLARIZE2(0.01) 2 3 16 17 11 12 15 14 10 9 19 18
    S[T] 15
    CX 2 1 16 15 11 10 8 14 3 9 12 18
    CX 16 10 11 5 25 19 8 9 17 18 12 13
    CX 16 8 11 3 25 17 1 9 10 18 5 13
    H 2 11 16 25
    MR 2 9 11 13 14 16 18 25
    DETECTOR(2, 4, 0) rec[-3]
    MX 1 3 5 8 10 12 15 17 19
    OBSERVABLE_INCLUDE(0) rec[-3] rec[-6] rec[-9]
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
        # NOTE: the faults' detectors cancel and their observables do not.
        terms = collections.Counter(
            str(term.dem_target) for fault in error for term in fault.dem_error_terms
        )
        odd = {target for target, count in terms.items() if count % 2}
        assert odd and all(target.startswith("L") for target in odd)
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
    "text", [NOISES, INVERTED, str(_generated("surface_code:rotated_memory_z", 3))]
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


@pytest.mark.parametrize(
    "text", [NOISES, INVERTED, str(_generated("surface_code:rotated_memory_z", 3))]
)
def test_the_dem_error_terms_are_stims(text):
    circuit = stim.Circuit(text)
    terms = {
        str(location): explained.dem_error_terms
        for explained in circuit.flattened().explain_detector_error_model_errors()
        for location in explained.circuit_error_locations
    }
    search = _Search(Circuit(circuit))
    for index in range(len(search._events)):
        fault = search._fault(index)
        for location in fault.circuit_error_locations:
            assert list(fault.dem_error_terms) == terms[str(location)]
        explained = fault.to_explained_error()
        assert explained.dem_error_terms == list(fault.dem_error_terms)
        assert explained.circuit_error_locations == list(fault.circuit_error_locations)


def test_the_dem_error_terms_are_the_certain_flips():
    # NOTE: between S layers the X error is a Y, which flips both checks;
    # between T layers it flips the XX detector in only half of the shots.
    for gate, expected in (("S", ["D0", "L0"]), ("T", ["L0"])):
        fault = _Search(Circuit(BELL.format(gate=gate)))._fault(0)
        assert [str(t.dem_target) for t in fault.dem_error_terms] == expected


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
    # NOTE: two independent channels: one of them, not both.
    assert fault.probability == pytest.approx(0.01 + 0.01 - 2 * 0.01 * 0.01)
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


def test_a_fault_combines_its_channels_as_a_detector_error_model():
    # NOTE: X and Y of the PAULI_CHANNEL_1 exclude each other; the X_ERROR is
    # independent of them and undoes them.
    text = """
        PAULI_CHANNEL_1(0.01, 0.02, 0) 0
        X_ERROR(0.1) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    (error,) = shortest_errors(Circuit(text), max_weight=1)
    (fault,) = error
    assert len(fault.circuit_error_locations) == 3
    assert fault.probability == pytest.approx(0.03 + 0.1 - 2 * 0.03 * 0.1)


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
    # NOTE: Stim's detector error model takes X and Y as independent.
    stim_errors = stim.Circuit(one_channel).search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=2,
        dont_explore_edges_with_degree_above=2,
        dont_explore_edges_increasing_symptom_degree=False,
    )
    assert len(stim_errors) == 2
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


@pytest.mark.parametrize("max_weight", [0, -1])
def test_max_weight_is_checked(max_weight):
    with pytest.raises(ValueError, match="max_weight must be at least 1"):
        shortest_errors(Circuit(GHZ), max_weight=max_weight)


@pytest.mark.parametrize("max_errors", [0, -1])
def test_max_errors_is_checked(max_errors):
    # NOTE: a slice with these would drop errors and skip to the next weight.
    with pytest.raises(ValueError, match="max_errors must be at least 1"):
        shortest_errors(Circuit(GHZ), max_weight=2, max_errors=max_errors)


def test_a_search_that_is_too_large_is_refused(monkeypatch):
    monkeypatch.setattr("tsim.noise.faults.MAX_VALUE_BYTES", 1024)
    with pytest.raises(ValueError, match="lower max_weight"):
        shortest_errors(Circuit(_generated("repetition_code:memory", 3)), max_weight=3)


def test_distinct():
    assert _distinct([{1}, {2}])
    assert _distinct([{1, 2}, {1}])
    assert not _distinct([{1}, {1}])
    assert not _distinct([{1, 2}, {1, 2}, {1, 2}])


def test_a_search_with_too_many_pieces_is_refused(monkeypatch):
    monkeypatch.setattr("tsim.noise.faults.MAX_PIECES", 0)
    with pytest.raises(ValueError, match="lower max_weight"):
        shortest_errors(Circuit(_generated("repetition_code:memory", 3)), max_weight=3)


@pytest.mark.parametrize(("batch", "max_pairs"), [(1, 0), (7, 1 << 24)])
def test_small_batches_and_no_pairs_give_the_same_errors(batch, max_pairs, monkeypatch):
    circuit = Circuit(_generated("surface_code:rotated_memory_z", 3))
    expected = shortest_errors(circuit, max_weight=3)
    monkeypatch.setattr("tsim.noise.fault_pieces.BATCH_BYTES", batch)
    monkeypatch.setattr("tsim.noise.fault_pieces.MAX_PAIRS", max_pairs)
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


@pytest.mark.parametrize("text", [GHZ, TWO_PIECES, T_SURFACE])
def test_the_parallel_kernels_find_the_same_errors(text, monkeypatch):
    expected = shortest_errors(Circuit(text), max_weight=4)
    # NOTE: every kernel then runs in parallel, if numba has threads.
    monkeypatch.setattr("tsim.utils.kernels.PARALLEL_WORK", 0)
    got = shortest_errors(Circuit(text), max_weight=4)
    assert [str(e) for e in got] == [str(e) for e in expected]


def _python_kernels(monkeypatch):
    """Run the numba kernels of the search as Python, so coverage sees them."""
    for module in (fault_pieces, fault_values):
        for name, value in vars(module).copy().items():
            if hasattr(value, "py_func"):
                monkeypatch.setattr(module, name, value.py_func)
    # NOTE: a kernel ``name(work, *args)`` runs ``_name(*args)``.
    kernels = {fault_pieces: ["search"], fault_values: ["find_all", "xor_sets"]}
    kernels[fault_values] += ["accepted", "probabilities", "accepted_sums"]
    kernels[fault_values] += ["triples", "join"]
    for module, names in kernels.items():
        for name in names:
            python = getattr(module, f"_{name}")
            monkeypatch.setattr(module, name, lambda _, *a, run=python: run(*a))


@pytest.mark.parametrize("task", ["ghz", "noises", "repetition", "t_surface"])
def test_the_search_kernels_run_as_python(task, monkeypatch):
    text = {
        "ghz": GHZ,
        "noises": NOISES,
        "repetition": str(_generated("repetition_code:memory", 3)),
        "t_surface": T_SURFACE,
    }[task]
    weights = range(1, 5)
    compiled = [_Search(Circuit(text)).logical_errors(w) for w in weights]
    _python_kernels(monkeypatch)
    # NOTE: the hashes multiply uint64 scalars, which wrap as they do in numba.
    with np.errstate(over="ignore"):
        python = [_Search(Circuit(text)).logical_errors(w) for w in weights]
    assert [[str(e) for e in errors] for errors in python] == [
        [str(e) for e in errors] for errors in compiled
    ]


def _every_set(search: _Search, weight: int) -> dict[tuple, tuple[float, float]]:
    """Return the logical errors of ``weight`` faults by trying every set of them.

    The walk of each set takes its f-bits, and the probabilities are products
    over the components, without the search's tables.
    """
    is_detector = search._direct_outputs < search._num_detectors
    known = search._direct_bits >= 0
    detectors = np.unique(search._direct_bits[is_detector & known])
    observables = np.unique(search._direct_bits[~is_detector & known])
    sets = np.array(
        list(itertools.combinations(range(len(search._events)), weight)), dtype=np.int64
    ).reshape(-1, weight)
    f = np.bitwise_xor.reduce(search._bits[sets], axis=1)
    quiet = ~f[:, detectors].any(axis=1)
    sets, f = sets[quiet], f[quiet]
    undetected = np.ones(len(sets))
    unflipped = np.where(f[:, observables].any(axis=1), 0.0, 1.0)
    for branches in search._branches:
        local = f[:, branches.f_selection]
        shifts = gf2_product(local, branches.output_rows)
        for i, (deviations, p) in enumerate(branches.outcomes(local)):
            same = deviations == shifts[i]
            undetected[i] *= p[same[:, ~branches.observable].all(axis=1)].sum()
            unflipped[i] *= p[same.all(axis=1)].sum()
    flip = undetected - unflipped
    keep = (flip > MIN_PROBABILITY) & search._distinct_channels(sets)
    return {
        tuple(search._fault(j) for j in row): (u, p)
        for row, u, p in zip(
            sets[keep].tolist(), undetected[keep], flip[keep], strict=True
        )
    }


@pytest.mark.parametrize(
    "text",
    [
        GHZ,
        BELL.format(gate="T"),
        NOISES,
        INVERTED,
        TWO_PIECES,
        T_SURFACE,
        str(_generated("repetition_code:memory", 3)),
    ],
)
def test_the_shortest_errors_are_those_of_trying_every_set(text):
    search = _Search(Circuit(text))
    for weight in range(1, 5):
        expected = _every_set(search, weight)
        got = {
            error.faults: (error.undetected_probability, error.logical_flip_probability)
            for error in search.logical_errors(weight)
        }
        assert got.keys() == expected.keys()
        for key, (undetected, flip) in got.items():
            assert undetected == pytest.approx(expected[key][0], abs=1e-12)
            assert flip == pytest.approx(expected[key][1], abs=1e-12)
        if expected:
            break


def test_a_shortest_error_can_be_two_pieces():
    errors = shortest_errors(Circuit(TWO_PIECES), max_weight=2)
    paulis = [
        [str(f.circuit_error_locations[0].flipped_pauli_product[0]) for f in error]
        for error in errors
    ]
    assert paulis == [["Z0", "Z1"], ["X0", "Z1"]]
    assert [error.logical_flip_probability for error in errors] == pytest.approx(
        [1.0, 0.5]
    )


def test_a_noisy_padding_flips_its_record():
    # NOTE: Stim explains this error as a flip of the earlier M, so the test
    # cannot compare with Stim.
    text = """
        R 0
        M 0
        DETECTOR rec[-1]
        MPAD(0.01) 0
        DETECTOR rec[-1]
    """
    search = _Search(Circuit(text))
    (location,) = search._fault(0).circuit_error_locations
    # NOTE: Stim's getter returns None for a flip without a measured observable.
    assert "flipped_measurement.measurement_record_index: 1" in str(location)
    assert location.stack_frames[0].instruction_offset == 3
