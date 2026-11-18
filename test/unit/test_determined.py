"""Tests for ``tsim.determined``: determined measurement parities."""

import numpy as np
import pytest
import stim

from tsim.circuit import Circuit
from tsim.determined import (
    _directions_of,
    count_determined_measurements,
    missing_detectors,
    nondeterministic_detectors,
)


@pytest.fixture(autouse=True)
def _uncached():
    # NOTE: some tests change the limits of the exact walk, which the cached
    # directions do not see.
    _directions_of.cache_clear()


STIM_EXAMPLES = [
    "R 0\nM 0",
    "R 0\nH 0\nM 0",
    "M 0",
    "M 0\nM 0 1\nM 0 1 2\nM 0 1 2 3",
    "R 0 1\nMZZ 0 1\nMYY 0 1\nMXX 0 1",
    """
    MZZ 0 1
    MYY 0 1
    MXX 0 1
    DEPOLARIZE1(0.1) 0 1
    MZZ 0 1
    MYY 0 1
    MXX 0 1
    DETECTOR rec[-1] rec[-4]
    DETECTOR rec[-2] rec[-5]
    DETECTOR rec[-3] rec[-6]
    """,
    "RX 0\nMPP X0*Z1 Y0\nMRY 0\nMY 0",
]

GENERATED = [
    ("surface_code:rotated_memory_x", 3),
    ("surface_code:unrotated_memory_z", 3),
    ("repetition_code:memory", 5),
    ("color_code:memory_xyz", 5),
]


def _generated(task: str, distance: int) -> str:
    circuit = stim.Circuit.generated(
        task, distance=distance, rounds=3, after_clifford_depolarization=0.01
    )
    return str(circuit)


def _clifford_cases() -> list[str]:
    return STIM_EXAMPLES + [_generated(task, d) for task, d in GENERATED]


@pytest.mark.parametrize("text", _clifford_cases())
@pytest.mark.parametrize("unknown_input", [False, True])
@pytest.mark.parametrize("shots", [None, 2000])
def test_a_clifford_circuit_has_stims_counts(text, unknown_input, shots):
    expected = stim.Circuit(text)
    circuit = Circuit(text)
    options = {"unknown_input": unknown_input, "shots": shots, "seed": 0}
    assert count_determined_measurements(
        circuit, **options
    ) == expected.count_determined_measurements(unknown_input=unknown_input)
    missing = missing_detectors(circuit, **options)
    # NOTE: Stim's stub says that missing_detectors returns an int.
    reference = stim.Circuit(
        str(expected.missing_detectors(unknown_input=unknown_input))
    )
    assert len(missing) == len(reference)
    assert nondeterministic_detectors(circuit, **options) == []
    # NOTE: with the missing detectors, Stim finds none missing.
    rest = (expected + missing).missing_detectors(unknown_input=unknown_input)
    assert str(rest) == ""


@pytest.mark.parametrize("text", _clifford_cases())
def test_the_missing_detectors_are_determined_in_every_shot(text):
    bare = Circuit(text).without_noise().without_annotations()
    missing = missing_detectors(bare)
    assert len(missing) == count_determined_measurements(bare)
    sampler = (bare + missing).compile_detector_sampler(seed=1)
    assert not sampler.sample(500, use_detector_reference_sample=True).any()


def test_t_gates_randomize_a_measurement_and_t4_is_z():
    assert count_determined_measurements(Circuit("R 0\nH 0\nT 0\nH 0\nM 0")) == 0
    assert count_determined_measurements(Circuit("R 0\nH 0\nT 0 0 0 0\nH 0\nM 0")) == 1


def test_a_nonlinear_function_of_earlier_measurements_is_not_counted():
    # NOTE: m2 = m0 AND m1 is predictable, but no parity of the three is.
    circuit = Circuit("R 0 1 2\nH 0 1\nCCX 0 1 2\nM 0 1 2")
    assert count_determined_measurements(circuit) == 0
    assert count_determined_measurements(circuit, shots=1000, seed=0) == 0


@pytest.mark.parametrize(
    "gate, value", [("Z", 0), ("S", 0), ("T", 1), ("T_DAG", 1), ("SQRT_X", 0)]
)
def test_ghz_parity_after_four_gates(gate, value):
    # NOTE: Gidney, Shutty, Jones (arXiv:2409.17595) Fig. 22, "danger of using S
    # as a proxy for T": XXXX of a GHZ state stays determined after T^4, but -1.
    circuit = Circuit(
        f"R 0 1 2 3\nH 0\nCX 0 1 0 2 0 3\n{gate} 0 1 2 3\nMPP X0*X1*X2*X3"
    )
    assert str(missing_detectors(circuit)).split() == ["DETECTOR", "rec[-1]"]
    samples = circuit.compile_sampler(seed=0).sample(100)
    assert (samples == value).all()


def _fifteen_to_one(tail: str, *, flipped: int | None = None) -> Circuit:
    # NOTE: Litinski, arXiv:1905.06903, Fig. 3: 15 pi/8 rotations Z_4 Z_b of the
    # output qubit 4 and the checks b, which return to |+>.
    lines = ["RX 0 1 2 3 4"]
    for b in range(1, 16):
        pairs = " ".join(f"{q} 4" for q in range(4) if b >> q & 1)
        gate = "T_DAG" if b == flipped else "T"
        lines += [f"CX {pairs}", f"{gate} 4", f"CX {pairs}"]
    return Circuit("\n".join(lines) + "\n" + tail)


CHECKS = "MX 0 1 2 3\n" + "".join(f"DETECTOR rec[-{k}]\n" for k in range(1, 5))


@pytest.mark.parametrize("shots", [None, 5000])
def test_fifteen_to_one_has_four_determined_checks(shots):
    circuit = _fifteen_to_one(CHECKS)
    assert count_determined_measurements(circuit, shots=shots, seed=0) == 4
    assert nondeterministic_detectors(circuit, shots=shots, seed=0) == []
    assert len(missing_detectors(_fifteen_to_one("MX 0 1 2 3"), shots=shots)) == 4
    # NOTE: the output is T_DAG |+>: undoing it makes it determined.
    output = CHECKS + "T 4\nMX 4\nOBSERVABLE_INCLUDE(0) rec[-1]"
    assert count_determined_measurements(_fifteen_to_one(output)) == 5
    output = CHECKS + "MX 4\nOBSERVABLE_INCLUDE(0) rec[-1]"
    random = nondeterministic_detectors(_fifteen_to_one(output))
    assert random == [stim.target_logical_observable_id(0)]


def test_a_wrong_t_gate_randomizes_checks():
    circuit = _fifteen_to_one(CHECKS, flipped=7)
    assert count_determined_measurements(circuit) == 3
    assert nondeterministic_detectors(circuit) == [
        stim.target_relative_detector_id(k) for k in (1, 2, 3)
    ]


def test_five_to_one_distillation_checks_are_random():
    # NOTE: Rodriguez et al., arXiv:2412.15165: noiseless acceptance is 1/6, so
    # no syndrome is determined (``docs/tutorials/magic_state_distillation``).
    theta = -np.arccos(np.sqrt(1 / 3)) / np.pi
    circuit = Circuit(f"""
        R 0 1 2 3 4
        R_X({theta}) 0 1 2 3 4
        T_DAG 0 1 2 3 4
        SQRT_X 0 1 4
        CZ 0 1 2 3
        SQRT_Y 0 3
        CZ 0 2 3 4
        SQRT_X_DAG 0
        CZ 0 4 1 3
        SQRT_X_DAG 0 1 2 3 4
        T 0
        R_X({-theta}) 0
        M 0 1 2 3 4
    """)
    assert count_determined_measurements(circuit) == 0
    assert count_determined_measurements(circuit, shots=20_000, seed=0) == 0


def test_missing_detectors_heal_random_detectors():
    # NOTE: after T, Z0 is random, and so is its copy Z1, but not Z0 Z1.
    circuit = Circuit("""
        R 0 1
        H 0
        T 0
        H 0
        CX 0 1
        M 0 1
        DETECTOR rec[-1]
        DETECTOR rec[-2]
    """)
    assert nondeterministic_detectors(circuit) == [
        stim.target_relative_detector_id(k) for k in (0, 1)
    ]
    assert count_determined_measurements(circuit) == 1
    assert str(missing_detectors(circuit)).split() == ["DETECTOR", "rec[-2]", "rec[-1]"]


def _random_circuit(rng: np.random.Generator, *, measured: bool = True) -> str:
    kinds = ["H", "S", "T", "T_DAG", "CX"] + ["M", "MX", "R", "MPP"] * measured
    lines = []
    for _ in range(int(rng.integers(6, 16))):
        kind = rng.choice(kinds)
        a, b = (int(q) for q in rng.choice(3, size=2, replace=False))
        if kind == "CX":
            lines.append(f"CX {a} {b}")
        elif kind == "MPP":
            lines.append(f"MPP X{a}*Z{b}")
        else:
            lines.append(f"{kind} {a}")
    return "\n".join([*lines, "M 0 1 2"])


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("measured", [True, False])
def test_exact_counts_are_those_of_many_shots(seed, measured):
    # NOTE: without measurements and resets before the end, unknown inputs
    # reach the rotation signs.
    rng = np.random.default_rng(seed)
    circuit = Circuit(_random_circuit(rng, measured=measured))
    for unknown_input in (False, True):
        exact = count_determined_measurements(circuit, unknown_input=unknown_input)
        sampled = count_determined_measurements(
            circuit, unknown_input=unknown_input, shots=4000, seed=seed
        )
        assert exact == sampled


def _two_qubit_circuit(rng: np.random.Generator) -> str:
    lines = []
    for _ in range(int(rng.integers(6, 14))):
        kind = rng.choice(["H", "T", "T_DAG", "CX", "H"])
        a, b = (int(q) for q in rng.choice(2, size=2, replace=False))
        lines.append(f"CX {a} {b}" if kind == "CX" else f"{kind} {a}")
    return "\n".join([*lines, "M 0 1", "CX 0 1", "M 1 0"])


# NOTE: unknown inputs set the sign of a rotation in each of these.
SIGN_CIRCUITS = [
    "H 0\nT 0\nH 0\nT 0\nH 0\nM 0",
    "H 0\nT 0\nH 0\nH 1\nT 0\nH 0\nT_DAG 1\nM 0 1\nCX 0 1\nM 1 0",
    "H 0\nT 0\nH 1\nCX 1 0\nT_DAG 0\nH 0\nT_DAG 0\nH 0\nT 0\nM 0 1\nCX 0 1\nM 1 0",
]


@pytest.mark.parametrize("text", SIGN_CIRCUITS)
def test_unknown_inputs_that_set_rotation_signs_match_many_shots(text, monkeypatch):
    circuit = Circuit(text)
    exact = count_determined_measurements(circuit, unknown_input=True)
    sampled = count_determined_measurements(
        circuit, unknown_input=True, shots=4000, seed=0
    )
    assert exact == sampled
    monkeypatch.setattr("tsim.determined.MAX_SIGN_BITS", 0)
    _directions_of.cache_clear()
    with pytest.raises(ValueError, match="independent rotation signs"):
        count_determined_measurements(circuit, unknown_input=True)


def test_unknown_inputs_reach_rotation_signs():
    # NOTE: an unknown input b of qubit 0 flips the sign of the T rotation that
    # H T H applies; the second measurement repeats the first either way.
    circuit = Circuit("CX 0 1\nH 0\nT 0\nH 0\nM 0 0 1")
    assert count_determined_measurements(circuit) == 2
    assert count_determined_measurements(circuit, unknown_input=True) == 1


def test_an_unlikely_outcome_after_another_is_possible():
    # NOTE: m2 = m0 AND m1 flips only when both of chance 1e-7 land 1; that is
    # unlikely overall, but each given the one before is not.
    angle = 2 * np.arcsin(np.sqrt(1e-7)) / np.pi
    circuit = Circuit(f"R 0 1 2\nR_X({angle}) 0 1\nCCX 0 1 2\nM 0 1 2")
    assert count_determined_measurements(circuit) == 0


def test_too_many_branches_are_refused(monkeypatch):
    monkeypatch.setattr("tsim.statevector.branches.MAX_BRANCHES", 1)
    circuit = Circuit("R 0 1\nH 0 1\nT 0 1\nH 0 1\nM 0 1")
    with pytest.raises(ValueError, match="measurement branches"):
        count_determined_measurements(circuit)


def test_too_few_shots_warn():
    circuit = Circuit(
        "R 0 1 2 3 4 5 6 7 8 9\nH 0 1 2 3 4 5 6 7 8 9\nM 0 1 2 3 4 5 6 7 8 9"
    )
    with pytest.warns(UserWarning, match="use more shots"):
        count_determined_measurements(circuit, shots=12, seed=0)
    with pytest.raises(ValueError, match="at least 1"):
        count_determined_measurements(circuit, shots=0)


def test_padded_measurements_are_determined():
    # NOTE: Stim 1.16's count_determined_measurements rejects MPAD.
    circuit = Circuit("MPAD 1 0\nH 0\nM 0\nDETECTOR rec[-2]")
    assert count_determined_measurements(circuit) == 2
    assert str(missing_detectors(circuit)).split() == ["DETECTOR", "rec[-3]"]


def test_a_circuit_without_measurements():
    assert count_determined_measurements(Circuit("H 0\nT 0")) == 0
    assert len(missing_detectors(Circuit("H 0\nT 0"))) == 0


def test_the_directions_are_cached_unless_shots_have_no_seed():
    circuit = Circuit("R 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]")
    count_determined_measurements(circuit)
    nondeterministic_detectors(circuit)
    assert _directions_of.cache_info().hits == 1
    count_determined_measurements(circuit, shots=10)
    assert _directions_of.cache_info().currsize == 1
