"""Tests for ``tsim.flows``: stabilizer flows."""

import itertools

import numpy as np
import pytest
import stim

from tsim.circuit import Circuit
from tsim.flows import flow_generators, has_all_flows, has_flow

CLIFFORD = [
    "",
    "H 1",
    "S 0",
    "M 0",
    "R 0",
    "MR 0",
    "MXX 0 1",
    "H 0\nCX 0 1\nM 1",
    "X 0\nM 0\nR 1\nM 1",
    "R 0 1\nX 0\nM 0 1",
    "RX 0\nMPP X0*Z1 Y0\nMRY 0\nMY 0",
    "R 0 1\nMZZ 0 1\nMYY 0 1\nMXX 0 1",
    "M 0\nCX rec[-1] 1\nMPAD 1\nCZ rec[-1] 0",
    "C_XYZ 0\nSQRT_Y 1\nCY 0 1\nISWAP 1 2\nMY 2",
    "REPEAT 2 {\n    H 0\n    CX 0 1\n    M !1\n}",
]

GENERATED = [
    ("surface_code:rotated_memory_x", 3),
    ("repetition_code:memory", 5),
    ("color_code:memory_xyz", 3),
]

T_GATE = np.diag([1, np.exp(1j * np.pi / 4)])


def _random_circuit(seed: int, t_gates: bool) -> str:
    """Return a circuit of at most 2 qubits, 2 measurements and 1 reset."""
    rng = np.random.default_rng(seed)
    one = ["H", "S", "S_DAG", "X", "Z"] + ["T", "T_DAG"] * (3 * t_gates)
    lines, limits = [], {"M": 2, "MX": 2, "R": 1, "RX": 1}
    for _ in range(rng.integers(3, 12)):
        name = rng.choice([*one, "CX", "CZ", *limits])
        q = int(rng.integers(2))
        if name in ("CX", "CZ"):
            lines.append(f"{name} {q} {1 - q}")
            continue
        if name in limits:
            if not limits[name]:
                continue
            for other in limits:
                limits[other] -= other[0] == name[0]
        lines.append(f"{name} {q}")
        if name[0] == "M" and rng.random() < 0.5:
            lines.append(f"{rng.choice(['CX', 'CZ'])} rec[-1] {rng.integers(2)}")
    return "\n".join(lines)


def _bare(circuit: Circuit) -> stim.Circuit:
    return circuit.without_noise().without_annotations().stim_circuit


def _negated(flow: stim.Flow) -> stim.Flow:
    return stim.Flow(
        input=flow.input_copy(),
        output=-flow.output_copy(),
        measurements=flow.measurements_copy(),
    )


def _random_product(flows: list[stim.Flow], rng: np.random.Generator) -> stim.Flow:
    product = stim.Flow()
    for flow in itertools.compress(flows, rng.random(len(flows)) < 0.5):
        product *= flow
    return product


def _kraus(circuit: stim.Circuit) -> list[tuple[list[int], np.ndarray]]:
    """Return every Kraus operator of a small circuit, with its record.

    A reset has an operator for each outcome that it discards.
    """
    n = circuit.num_qubits
    start = np.eye(2**n, dtype=complex).reshape((2,) * n + (-1,))
    branches: list[tuple[list[int], np.ndarray]] = [([], start)]

    def apply(gate: np.ndarray, qubits: list[int], k: np.ndarray) -> np.ndarray:
        arity = len(qubits)
        gate = gate.reshape((2,) * (2 * arity))
        k = np.tensordot(gate, k, (list(range(arity, 2 * arity)), qubits))
        return np.moveaxis(k, list(range(arity)), qubits)

    def unitary(name: str) -> np.ndarray:
        gate = stim.Tableau.from_named_gate(name)
        return np.asarray(gate.to_unitary_matrix(endian="big"))

    for inst in circuit:
        assert isinstance(inst, stim.CircuitInstruction)
        for group in inst.target_groups():
            qubits = [t.value for t in group if t.is_qubit_target]
            if group[0].is_measurement_record_target:
                pauli = unitary(inst.name[1])
                branches = [
                    (record, apply(pauli, qubits, k) if record[group[0].value] else k)
                    for record, k in branches
                ]
            elif inst.name in ("M", "MX", "R", "RX"):
                basis = unitary("H" if inst.name[-1] == "X" else "I")
                branches, old = [], branches
                for (record, k), b in itertools.product(old, (0, 1)):
                    outcome = np.zeros((2, 2))
                    outcome[0 if inst.name[0] == "R" else b, b] = 1
                    k = apply(basis @ outcome @ basis, qubits, k)
                    if np.abs(k).max() > 1e-9:
                        branches.append((record + [b] * (inst.name[0] == "M"), k))
            else:
                gate = unitary(inst.name)
                if inst.tag == "T":
                    gate = T_GATE if inst.name == "S" else T_GATE.conj()
                branches = [(record, apply(gate, qubits, k)) for record, k in branches]
    return [(record, k.reshape(2**n, 2**n)) for record, k in branches]


def _matrix(pauli: stim.PauliString) -> np.ndarray:
    return np.asarray(pauli.to_unitary_matrix(endian="big"))


def _holds(flow: stim.Flow, kraus: list[tuple[list[int], np.ndarray]], n: int) -> bool:
    before, after = (
        _matrix(p + stim.PauliString(n - len(p)))
        for p in (flow.input_copy(), flow.output_copy())
    )
    return all(
        np.allclose(
            k @ before,
            (-1) ** sum(record[m] for m in flow.measurements_copy()) * after @ k,
        )
        for record, k in kraus
    )


def _count_flows(kraus: list[tuple[list[int], np.ndarray]], n: int) -> int:
    """Count the flows of the Kraus operators, up to sign."""
    records = np.array([record for record, _ in kraus], dtype=np.int64)
    paulis = [_matrix(p) for p in stim.PauliString.iter_all(n)]
    count = 0
    for before, after in itertools.product(paulis, repeat=2):
        signs = []
        for _, k in kraus:
            if np.allclose(k @ before, after @ k):
                signs.append(0)
            elif np.allclose(k @ before, -after @ k):
                signs.append(1)
        if len(signs) == len(kraus):
            for subset in itertools.product((0, 1), repeat=records.shape[1]):
                count += (
                    np.ptp((records @ np.array(subset, dtype=np.int64) + signs) % 2)
                    == 0
                )
    return count


def _clifford_cases() -> list[str]:
    generated = [
        str(
            stim.Circuit.generated(
                task, distance=d, rounds=3, after_clifford_depolarization=0.01
            )
        )
        for task, d in GENERATED
    ]
    return CLIFFORD + generated + [_random_circuit(s, t_gates=False) for s in range(30)]


@pytest.mark.parametrize("text", _clifford_cases())
def test_generators_of_a_clifford_circuit_are_stims(text):
    circuit = Circuit(text)
    assert flow_generators(circuit) == _bare(circuit).flow_generators()


@pytest.mark.parametrize(
    "text", CLIFFORD + [_random_circuit(s, False) for s in range(10)]
)
def test_has_all_flows_agrees_with_stim_on_clifford_circuits(text):
    circuit = Circuit(text)
    bare = _bare(circuit)
    generators = bare.flow_generators()
    assert has_all_flows(circuit, generators)
    for flow in generators:
        assert not has_flow(circuit, _negated(flow))
        assert has_flow(circuit, _negated(flow), unsigned=True)
    rng = np.random.default_rng(0)
    for _ in range(10):
        product = _random_product(generators, rng)
        # An X that enters on a qubit past the circuit's and does not leave.
        wrong = product * stim.Flow(input=stim.PauliString(f"X{bare.num_qubits}"))
        assert has_flow(circuit, product)
        assert not has_flow(circuit, wrong, unsigned=True)
        assert not bare.has_flow(wrong, unsigned=True)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("T 0", ["Z -> Z"]),
        ("T 0\nT_DAG 0", ["X -> X", "Z -> Z"]),
        ("H 0\nT 0\nH 0", ["X -> X"]),
        ("CX 0 1\nT 1\nCX 0 1", ["_Z -> _Z", "XX -> XX", "Z_ -> Z_"]),
        # A controlled (X + Y) / √2, which is no Clifford gate.
        ("T_DAG 1\nCX 0 1\nT 1", ["_Z -> ZZ", "Z_ -> Z_"]),
        ("T 0\nT 0", [str(f) for f in stim.Circuit("S 0").flow_generators()]),
        ("T 0 0 0 0", [str(f) for f in stim.Circuit("Z 0").flow_generators()]),
        ("X 0\nT 0\nM 0", [str(f) for f in stim.Circuit("X 0\nM 0").flow_generators()]),
    ],
)
def test_generators_with_t_gates(text, expected):
    assert [str(f) for f in flow_generators(Circuit(text))] == expected


def test_repeated_non_pauli_measurement_agrees_with_itself():
    # NOTE: each round measures (X + Y) / √2 of qubit 0, which no Pauli flow
    # passes through, so the two results are random but equal.
    measure = "RX 1\nT_DAG 0\nCX 1 0\nT 0\nMX 1\n"
    circuit = Circuit(measure * 2)
    assert has_flow(circuit, stim.Flow("1 -> rec[0] xor rec[1]"))
    assert not has_flow(circuit, stim.Flow("1 -> rec[0]"), unsigned=True)
    assert stim.Flow("1 -> rec[0] xor rec[1]") in flow_generators(circuit)


@pytest.mark.parametrize("seed", range(40))
def test_flows_of_random_circuits_with_t_gates_match_their_kraus_operators(seed):
    circuit = Circuit(_random_circuit(seed, t_gates=True))
    bare = _bare(circuit)
    n = bare.num_qubits
    kraus = _kraus(bare)
    generators = flow_generators(circuit)
    assert all(_holds(flow, kraus, n) for flow in generators)
    assert _count_flows(kraus, n) == 2 ** len(generators)

    assert has_all_flows(circuit, generators)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        random = stim.Flow(
            input=stim.PauliString(rng.integers(4, size=n).tolist()),
            output=stim.PauliString(rng.integers(4, size=n).tolist()),
            measurements=np.flatnonzero(rng.random(bare.num_measurements) < 0.5),
        )
        product = _random_product(generators, rng)
        for flow in (random, product, _negated(product)):
            assert has_flow(circuit, flow) == _holds(flow, kraus, n)


def test_noise_and_annotations_are_ignored():
    noisy = Circuit("""
        T 0
        DEPOLARIZE1(0.1) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    assert flow_generators(noisy) == flow_generators(Circuit("T 0\nM 0"))
    assert has_flow(noisy, stim.Flow("Z -> rec[0]"))


def test_flows_may_name_qubits_past_the_circuit():
    circuit = Circuit("T 0")
    assert has_flow(circuit, stim.Flow("_X -> _X"))
    assert not has_flow(circuit, stim.Flow("XX -> XX"), unsigned=True)


def test_flows_from_shots():
    circuit = Circuit("H 0\nT 0\nCX 0 1\nM 1\nT_DAG 0\nMX 0")
    exact = flow_generators(circuit)
    assert flow_generators(circuit, shots=200, seed=0) == exact
    assert has_all_flows(circuit, exact, shots=200, seed=0)
    assert not has_flow(circuit, stim.Flow("1 -> rec[1]"), shots=200, seed=0)


def test_no_flows_is_true():
    assert has_all_flows(Circuit("T 0"), [])


@pytest.mark.parametrize(
    "flow, message",
    [
        ("1 -> rec[1]", "the circuit has 1 measurements, so none is rec\\[1\\]"),
        ("1 -> rec[-2]", "the circuit has 1 measurements, so none is rec\\[-2\\]"),
        ("Z -> obs[0]", "flows of observables are not supported"),
    ],
)
def test_unsupported_flows_raise(flow, message):
    with pytest.raises(ValueError, match=message):
        has_flow(Circuit("M 0"), stim.Flow(flow))
