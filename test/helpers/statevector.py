"""Dense reference simulations for testing ``strategy="statevector"``.

Bit ``q`` of a basis-state index is qubit ``q`` (a line, or an array axis).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from math import cos, pi, sin

import jax.numpy as jnp
import numpy as np
from zxparam import Graph, Phase, Scalar
from zxparam.extract import ExtractedCircuit, Gate, extract_circuit

from tsim.circuit import Circuit
from tsim.compile.evaluate import evaluate
from tsim.compile.pipeline import classify_direct
from tsim.core.graph import prepare_graph
from tsim.core.params import ParamLayout
from tsim.core.types import CompiledComponent
from tsim.statevector.heisenberg import (
    ROTATION,
    ComponentLayout,
    HeisenbergOps,
    heisenberg_ops,
)
from tsim.statevector.plan import MeasureActive, Plan, Record, Rotate

Distribution = dict[tuple[int, ...], float]

CIRCUITS: dict[str, str] = {
    "t_interference": """
        R 0
        H 0
        T 0
        H 0
        T 0
        H 0
        X_ERROR(0.1) 0
        T 0
        H 0
        M 0
        DETECTOR rec[-1]
    """,
    "t_ghz": """
        R 0 1 2
        H 0
        CNOT 0 1
        CNOT 1 2
        T 0 1 2
        H 0 1 2
        DEPOLARIZE1(0.05) 0 1 2
        T 0 1
        CZ 0 1
        H 0 1
        M 0 1 2
        DETECTOR rec[-3] rec[-2]
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """,
    "rotations": """
        R 0 1
        U3(0.3, 0.24, 0.49) 0
        CNOT 0 1
        R_Z(0.13) 1
        R_X(0.21) 0
        Z_ERROR(0.2) 0
        R_Y(0.4) 1
        CZ 0 1
        H 1
        M 0 1
        DETECTOR rec[-1]
        DETECTOR rec[-2]
    """,
    "injection_check": """
        R 0 1 2 3
        H 0
        T 0
        CNOT 0 1
        CNOT 0 2
        DEPOLARIZE1(0.02) 0 1 2
        CNOT 0 3
        CNOT 1 3
        M 3
        DETECTOR rec[-1]
        R 3
        CNOT 1 3
        CNOT 2 3
        X_ERROR(0.05) 3
        M 3
        DETECTOR rec[-1]
        CNOT 0 2
        CNOT 0 1
        T_DAG 0
        H 0
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """,
    "midcircuit_feedforward": """
        R 0 1 2
        H 0 1
        T 0 1
        CNOT 0 2
        CNOT 1 2
        X_ERROR(0.1) 2
        M 2
        CX rec[-1] 0
        T 0
        CNOT 0 1
        H 0 1
        Z_ERROR(0.1) 1
        T 1
        H 1
        MR 0
        H 0
        T 0
        H 0
        M 0 1
        DETECTOR rec[-1] rec[-4]
        DETECTOR rec[-2] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """,
    "two_components": """
        R 0 1 2 3 4
        X_ERROR(0.2) 4
        H 0 2
        T 0 2
        H 0 2
        T 0 2
        CNOT 0 1
        CNOT 2 3
        X_ERROR(0.1) 1 3
        T 1 3
        H 0 1 2 3
        M 4 0 1 2 3
        DETECTOR rec[-5]
        DETECTOR rec[-4] rec[-3]
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-4]
    """,
    "clifford_component": """
        X_ERROR(0.5) 0
        R 1
        H 1
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1] rec[-2]
    """,
}


def quantum_components(
    text: str, *, sample_detectors: bool
) -> tuple[list[Graph], ParamLayout]:
    """Return the non-direct components of a circuit's prepared sampling graph."""
    prepared = prepare_graph(Circuit(text), sample_detectors=sample_detectors)
    components = [
        component
        for component in prepared.graph.connected_components()
        if classify_direct(component, prepared.layout) is None
    ]
    return components, prepared.layout


def hand_built_circuits() -> list[tuple[str, ExtractedCircuit, ComponentLayout]]:
    """Return small extracted circuits with a DISCARD and a reused MEASURE line."""
    t = Phase(Fraction(1, 4))
    noise = Phase(Fraction(0), frozenset({7}))
    discard = ExtractedCircuit(
        gates=[
            Gate("Z", (0,), t),
            Gate("CZ", (0, 1)),
            Gate("Z", (1,), Phase(Fraction(1, 4), frozenset({7}))),
            Gate("H", (1,)),
            Gate("DISCARD", (1,)),
            Gate("H", (0,)),
            Gate("Z", (0,), Phase(Fraction(3, 4))),
        ],
        n_lines=2,
        line_start={0: None, 1: None},
        line_end={0: 0, 1: None},
        scalar=Scalar.one(),
        classical_ports=frozenset({0}),
    )
    measure = ExtractedCircuit(
        gates=[
            Gate("Z", (0,), t),
            Gate("CNOT", (0, 1)),
            Gate("Z", (1,), noise),
            Gate("H", (1,)),
            Gate("Z", (1,), Phase(Fraction(1, 8))),
            Gate("MEASURE", (1,)),
            Gate("CNOT", (1, 0)),
            Gate("H", (0,)),
            Gate("Z", (0,), Phase(Fraction(7, 4))),
            Gate("H", (0,)),
        ],
        n_lines=2,
        line_start={0: None, 1: None},
        line_end={0: 0, 1: 1},
        scalar=Scalar.one(),
        classical_ports=frozenset({0, 1}),
    )
    return [
        ("discard", discard, ComponentLayout((0,), {7: 0}, (0,))),
        ("measure", measure, ComponentLayout((0,), {7: 0}, (0, 1))),
    ]


def component_ops() -> Iterator[HeisenbergOps]:
    """Yield the ops of every test component (both modes) and hand-built circuit."""
    for name in sorted(CIRCUITS):
        for sample_detectors in (True, False):
            components, layout = quantum_components(
                CIRCUITS[name], sample_detectors=sample_detectors
            )
            for component in components:
                component_layout = ComponentLayout.of(component, layout)
                yield heisenberg_ops(extract_circuit(component), component_layout)
    for _, circuit, component_layout in hand_built_circuits():
        yield heisenberg_ops(circuit, component_layout)


def f_patterns(num_f: int, limit: int = 32) -> list[tuple[int, ...]]:
    """Return every f-bit pattern, or all-zero plus ``limit`` random ones."""
    if 2**num_f <= limit:
        return [tuple(p) for p in itertools.product((0, 1), repeat=num_f)]
    rng = np.random.default_rng(0)
    random = [tuple(int(b) for b in rng.integers(0, 2, num_f)) for _ in range(limit)]
    return [(0,) * num_f, *random]


def assert_distributions_close(
    got: Distribution, expected: Distribution, atol: float = 1e-6
) -> None:
    """Assert that two outcome distributions agree entry by entry."""
    for key in set(got) | set(expected):
        a, b = got.get(key, 0.0), expected.get(key, 0.0)
        assert abs(a - b) <= atol, (key, a, b)


def parity(values: np.ndarray | int) -> np.ndarray:
    """Return the parity of the set bits of each non-negative integer."""
    remaining = np.asarray(values, dtype=np.uint64).copy()
    result = np.zeros(remaining.shape, dtype=np.uint64)
    while remaining.any():
        result ^= remaining & np.uint64(1)
        remaining >>= np.uint64(1)
    return result.astype(bool)


def mask_of(bits: Sequence[bool] | np.ndarray) -> int:
    """Return the integer whose bit ``q`` is ``bits[q]``."""
    return sum(1 << q for q, bit in enumerate(bits) if bit)


def apply_pauli(psi: np.ndarray, x: int, z: int) -> np.ndarray:
    """Return P·ψ for P = i^|x∧z| X^x Z^z."""
    index = np.arange(len(psi))
    flipped = index ^ x
    i_power = (1, 1j, -1, -1j)[bin(x & z).count("1") % 4]
    signs = np.where(parity(flipped & z), -1, 1)
    return i_power * signs * psi[flipped]


_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def _single_qubit(psi: np.ndarray, num_qubits: int, qubit: int, matrix: np.ndarray):
    tensor = psi.reshape([2] * num_qubits)
    axis = num_qubits - 1 - qubit
    tensor = np.moveaxis(np.tensordot(matrix, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def _extracted_gate(
    psi: np.ndarray, num_lines: int, gate: Gate, bindings: Mapping[int, int]
) -> list[np.ndarray]:
    index = np.arange(len(psi))
    if gate.kind == "H":
        return [_single_qubit(psi, num_lines, gate.lines[0], _H)]
    if gate.kind == "CNOT":
        control, target = gate.lines
        return [psi[index ^ (((index >> control) & 1) << target)]]
    if gate.kind == "CZ":
        a, b = gate.lines
        both = ((index >> a) & (index >> b) & 1) == 1
        return [psi * np.where(both, -1, 1)]
    (line,) = gate.lines
    bit = (index >> line) & 1
    if gate.kind == "Z":
        assert gate.phase is not None
        flips = sum(bindings[p] for p in gate.phase.parameters)
        value = float(gate.phase.value) + flips
        return [psi * np.where(bit == 1, np.exp(1j * pi * value), 1)]
    assert gate.kind in ("MEASURE", "DISCARD")
    return [psi * (bit == 0), psi * (bit == 1)]


def extracted_distribution(
    circuit: ExtractedCircuit, layout: ComponentLayout, bindings: Mapping[int, int]
) -> Distribution:
    """Return the exact output distribution of an extracted circuit.

    Lines start in |+⟩. MEASURE and DISCARD dephase their line in the Z basis
    (each outcome becomes a branch). Output lines are read in the Z basis at
    the end, in slot order. ``bindings`` gives each phase parameter ID its bit.
    """
    num_lines = circuit.n_lines
    branches = [np.full(2**num_lines, 2 ** (-num_lines / 2), dtype=complex)]
    for gate in circuit.gates:
        branches = [
            new
            for psi in branches
            for new in _extracted_gate(psi, num_lines, gate, bindings)
        ]
    slot_of_port = {port: slot for slot, port in enumerate(layout.output_indices)}
    lines = [0] * len(layout.output_indices)
    for line, port in circuit.line_end.items():
        if port is not None:
            lines[slot_of_port[port]] = line
    result: Distribution = {}
    for psi in branches:
        probabilities = np.abs(psi) ** 2
        for i in np.flatnonzero(probabilities > 1e-15):
            key = tuple(int((i >> line) & 1) for line in lines)
            result[key] = result.get(key, 0.0) + float(probabilities[i])
    return result


def ops_distribution(ops: HeisenbergOps, f: Sequence[int]) -> Distribution:
    """Return the exact distribution of the recorded outcomes of ``ops``.

    The state starts in |0…0⟩. A measurement records the eigenvalue bit of its
    signed Pauli; a forgotten one only branches.
    """
    f_vector = np.asarray(f, dtype=np.int64)
    psi = np.zeros(2**ops.num_lines, dtype=complex)
    psi[0] = 1
    branches: list[tuple[np.ndarray, dict[int, int]]] = [(psi, {})]
    for j in range(ops.num_ops):
        sign = (int(ops.sign[j, 0]) + int(ops.sign[j, 1:] @ f_vector)) & 1
        x, z = mask_of(ops.x[j]), mask_of(ops.z[j])
        slot = int(ops.slot[j])
        updated: list[tuple[np.ndarray, dict[int, int]]] = []
        for state, record in branches:
            paulied = (-1) ** sign * apply_pauli(state, x, z)
            if ops.kind[j] == ROTATION:
                rotated = cos(ops.angle[j]) * state - 1j * sin(ops.angle[j]) * paulied
                updated.append((rotated, record))
                continue
            for bit in (0, 1):
                projected = (state + (-1) ** bit * paulied) / 2
                if np.vdot(projected, projected).real < 1e-15:
                    continue
                updated.append(
                    (projected, {**record, slot: bit} if slot >= 0 else record)
                )
        branches = updated
    result: Distribution = {}
    for state, record in branches:
        key = tuple(record[s] for s in range(ops.num_outputs))
        result[key] = result.get(key, 0.0) + float(np.vdot(state, state).real)
    return result


def reference_rotate(
    psi: np.ndarray, x: int, z: int, angle: float, negate: bool
) -> np.ndarray:
    """Return exp(-iθ (±P)) ψ, the ``ROTATE`` and ``ROTATE_DIAG`` kernels."""
    sign = -1 if negate else 1
    return cos(angle) * psi - 1j * sin(angle) * sign * apply_pauli(psi, x, z)


def reference_measure(
    psi: np.ndarray, x: int, z: int, pivot: int, outcome: int
) -> tuple[np.ndarray, float]:
    """Return the ``MEASURE`` / ``MEASURE_DIAG`` kernel's state and outcome probability.

    The kernel projects onto the (-1)^outcome eigenspace of P and resets the
    pivot axis to |0⟩: it keeps the pivot-0 half and renormalizes.
    """
    index = np.arange(len(psi))
    keep = ((index >> pivot) & 1) == 0
    norm = float(np.vdot(psi, psi).real)
    if x:
        paulied = apply_pauli(psi, x, z)
        expectation = float(np.vdot(psi, paulied).real)
        p1 = min(max((norm - expectation) / (2 * norm), 0.0), 1.0)
        p = p1 if outcome else 1 - p1
        projected = (psi + (-1) ** outcome * paulied) / np.sqrt(
            max(2 * norm * p, 1e-300)
        )
        return np.where(keep, projected, 0), p
    odd = parity(index & z)
    p1 = float(np.sum(np.abs(psi[odd]) ** 2)) / norm
    p = p1 if outcome else 1 - p1
    source = index ^ ((odd ^ bool(outcome)).astype(np.int64) << pivot)
    return np.where(keep, psi[source] / np.sqrt(max(norm * p, 1e-300)), 0), p


def plan_distribution(plan: Plan, f: Sequence[int]) -> Distribution:
    """Return the exact output distribution of a plan by enumerating its branches."""
    result: Distribution = {}

    def written(record: list[int], slot: int, value: int) -> list[int]:
        if slot < 0:
            return record
        updated = list(record)
        updated[slot] = value
        return updated

    def walk(
        a: int, psi: np.ndarray, symbols: list[int], record: list[int], weight: float
    ) -> None:
        if weight < 1e-14:
            return
        if a == len(plan.actions):
            key = tuple(record)
            result[key] = result.get(key, 0.0) + weight
            return
        action = plan.actions[a]
        sign = action.sign.evaluate(symbols)
        if isinstance(action, Rotate):
            rotated = reference_rotate(
                psi, action.x, action.z, action.angle, bool(sign)
            )
            walk(a + 1, rotated, symbols, record, weight)
            return
        if isinstance(action, Record):
            walk(a + 1, psi, symbols, written(record, action.slot, sign), weight)
            return
        for outcome in (0, 1):
            if isinstance(action, MeasureActive):
                new, p = reference_measure(
                    psi, action.x, action.z, action.pivot, outcome
                )
            else:
                new, p = psi, 0.5
            branched = list(symbols)
            branched[action.branch] = outcome
            value = outcome ^ sign
            walk(a + 1, new, branched, written(record, action.slot, value), weight * p)

    psi = np.zeros(2**plan.width, dtype=complex)
    psi[0] = 1
    walk(0, psi, [*f, *[0] * plan.num_branches], [0] * plan.num_outputs, 1.0)
    return result


def stabilizer_rank_distribution(
    compiled: CompiledComponent, f: Sequence[int]
) -> Distribution:
    """Return P(outputs | f) of a joint-mode stabilizer-rank component."""
    norm_graphs, joint_graphs = compiled.compiled_scalar_graphs
    outcomes = list(itertools.product((0, 1), repeat=len(compiled.output_indices)))
    params = np.array([[*f, *o] for o in outcomes], dtype=np.uint8)
    params = params.reshape(len(outcomes), len(f) + len(compiled.output_indices))
    joint = np.abs(np.asarray(evaluate(joint_graphs, jnp.asarray(params))))
    f_row = np.array([list(f)], dtype=np.uint8).reshape(1, len(f))
    norm = float(np.abs(np.asarray(evaluate(norm_graphs, jnp.asarray(f_row))))[0])
    return {o: float(p) / norm for o, p in zip(outcomes, joint, strict=True)}


_PAULI_MATRICES = {
    (0, 0): np.eye(2),
    (1, 0): np.array([[0, 1], [1, 0]]),
    (0, 1): np.diag([1, -1]),
    (1, 1): np.array([[0, -1j], [1j, 0]]),
}


def pauli_matrix(x: int, z: int, num_qubits: int) -> np.ndarray:
    """Return the dense matrix of P = i^|x∧z| X^x Z^z (bit q is qubit q)."""
    matrix = np.ones((1, 1), dtype=complex)
    for q in reversed(range(num_qubits)):
        matrix = np.kron(matrix, _PAULI_MATRICES[((x >> q) & 1, (z >> q) & 1)])
    return matrix


def assert_matches_distribution(
    samples: np.ndarray, distribution: Distribution
) -> None:
    """Assert that sampled rows follow ``distribution`` (5σ + 3 per outcome)."""
    shots = samples.shape[0]
    counts: dict[tuple[int, ...], int] = {}
    for row in samples.astype(int):
        key = tuple(row.tolist())
        counts[key] = counts.get(key, 0) + 1
    for key in set(counts) | set(distribution):
        p = distribution.get(key, 0.0)
        expected = shots * p
        sigma = np.sqrt(shots * p * (1 - p))
        assert abs(counts.get(key, 0) - expected) <= 5 * sigma + 3, (
            key,
            counts.get(key, 0),
            expected,
        )


def assert_counts_match(samples: np.ndarray, reference: np.ndarray) -> None:
    """Assert that two sets of sampled rows have the same row distribution."""
    width = samples.shape[1]
    powers = 2 ** np.arange(width)
    bins = 2**width
    a = np.bincount(samples.astype(np.int64) @ powers, minlength=bins)
    b = np.bincount(reference.astype(np.int64) @ powers, minlength=bins)
    scale_a, scale_b = 1 / samples.shape[0], 1 / reference.shape[0]
    pooled = (a + b) / (samples.shape[0] + reference.shape[0])
    sigma = np.sqrt(pooled * (1 - pooled) * (scale_a + scale_b))
    assert np.all(
        np.abs(a * scale_a - b * scale_b) <= 5 * sigma + 3 * (scale_a + scale_b)
    )
