"""Fig. 4a global-rotation demo: ideal state-vector sweep for Reed–Muller codes.

Tsim prepares $|+_L\\rangle$; global $R_Z(\\varphi)^{\\otimes n}$ is applied as a
diagonal phase on the state vector (exact, fast for $n \\le 15$). This matches the
paper's noiseless logical response; it is not the experimental decoder pipeline.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache

import numpy as np
import stim
import tsim

# Hypercube encoder for $|+_L\\rangle$ on $[[7,1,3]]$ (Extended Data Fig. 10a; 4th CZ off).
CODE_7_PLUS_L = """
RX 6
R 0 1 2 3 4 5
SQRT_Y_DAG 0 1 2 3 4 5
CZ 1 2 3 4 5 6
SQRT_Y 6
CZ 0 3 2 5 4 6
SQRT_Y 2 3 4 5 6
CZ 0 1 2 3 4 5
SQRT_Y 1 2 4
"""

STEANE_PREP = f"RX 6\n{CODE_7_PLUS_L.strip()}\n"

# Same 4-weight supports as ``encoding_demo.ipynb`` (X and Z checks coincide).
STEANE_STABILIZER_SUPPORTS = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 4, 6]]
STEANE_LOGICAL_X_SUPPORT = [0, 1, 5]

GATE_LABELS_DEG = {
    -180: r"$Z$",
    -135: r"$(TS)^\dagger$",
    -90: r"$S^\dagger$",
    -45: r"$T^\dagger$",
    0: r"$I$",
    45: r"$T$",
    90: r"$S$",
    135: r"$TS$",
    180: r"$Z$",
}

PAULI_MUL: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("X", "X"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1, "I"),
}


@dataclass(frozen=True)
class Observable:
    coefficient: complex
    pauli: str


@dataclass(frozen=True)
class RotationSystem:
    label: str
    n: int
    base_state: np.ndarray
    x_stabilizers: list[str]
    logical_x: Observable
    logical_y: Observable


@dataclass(frozen=True)
class RotationCurve:
    angles_deg: np.ndarray
    logical_x: np.ndarray
    stabilizer_abs: np.ndarray


def pauli_string(n: int, axis: str, support: list[int] | np.ndarray) -> str:
    text = ["I"] * n
    for qubit in support:
        text[int(qubit)] = axis
    return "".join(text)


def stim_pauli_text(pauli: str) -> str:
    return pauli.replace("I", "_")


def multiply_paulis(left: str, right: str) -> tuple[complex, str]:
    phase = 1
    out: list[str] = []
    for a, b in zip(left, right, strict=True):
        local_phase, local_pauli = PAULI_MUL[(a, b)]
        phase *= local_phase
        out.append(local_pauli)
    return phase, "".join(out)


def generated_group(generators: list[str]) -> list[tuple[complex, str]]:
    identity = "I" * len(generators[0])
    group: list[tuple[complex, str]] = []
    for bits in itertools.product([0, 1], repeat=len(generators)):
        phase = 1
        pauli = identity
        for bit, generator in zip(bits, generators, strict=True):
            if bit:
                local_phase, pauli = multiply_paulis(pauli, generator)
                phase *= local_phase
        group.append((phase, pauli))
    return group


@cache
def pauli_action(pauli: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(pauli)
    indices = np.arange(1 << n, dtype=np.uint64)
    flip_mask = 0
    z_mask = 0
    y_count = 0
    for qubit, axis in enumerate(pauli):
        bit_position = n - 1 - qubit
        if axis in "XY":
            flip_mask |= 1 << bit_position
        if axis in "YZ":
            z_mask |= 1 << bit_position
        if axis == "Y":
            y_count += 1
    parity = np.array(
        [((int(index) & z_mask).bit_count() & 1) for index in indices],
        dtype=np.int8,
    )
    phase = ((1j) ** y_count) * (1 - 2 * parity)
    return indices ^ np.uint64(flip_mask), phase


@cache
def hamming_weights(n: int) -> np.ndarray:
    return np.array([index.bit_count() for index in range(1 << n)], dtype=np.int16)


def pauli_expectation_complex(state: np.ndarray, pauli: str) -> complex:
    partner, phase = pauli_action(pauli)
    return np.sum(np.conjugate(state[partner]) * phase * state)


def pauli_expectation(state: np.ndarray, pauli: str) -> float:
    return float(np.real_if_close(pauli_expectation_complex(state, pauli)).real)


def observable_expectation(state: np.ndarray, observable: Observable) -> float:
    value = observable.coefficient * pauli_expectation_complex(state, observable.pauli)
    return float(np.real_if_close(value).real)


def apply_pauli_to_state(state: np.ndarray, pauli: str) -> np.ndarray:
    partner, phase = pauli_action(pauli)
    out = np.empty_like(state)
    out[partner] = phase * state
    return out


def logical_y_from_xz(logical_x: str, logical_z: str) -> Observable:
    phase, pauli = multiply_paulis(logical_x, logical_z)
    coefficient = 1j * phase
    return Observable(float(np.real(coefficient)), pauli)


_STATE_INIT_OPS = frozenset(
    {"R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ", "MPP", "MXX", "MYY", "MZZ"}
)


def _has_state_initialization(stim_circuit: stim.Circuit) -> bool:
    return any(instruction.name in _STATE_INIT_OPS for instruction in stim_circuit)


def normalized_state(circuit: tsim.Circuit, *, n_qubits: int | None = None) -> np.ndarray:
    """Return a normalized state vector from a Tsim preparation circuit.

    ``Circuit.to_matrix()`` returns a column state vector only when the Stim
    program includes initialization (``R`` / ``RX`` / …). A Clifford-only circuit
    would materialize the full unitary and ``reshape(-1)`` would yield length
    ``2**(2n)``, which is both expensive and incompatible with our phase sweep.
    """
    stim_circuit = circuit.stim_circuit
    if not _has_state_initialization(stim_circuit):
        raise ValueError(
            "Preparation circuit has no R/RX-style initialization; "
            "cannot extract a state vector via to_matrix()."
        )

    matrix = np.asarray(circuit.to_matrix())
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.shape[1] != 1:
        raise ValueError(
            f"Expected a state vector from to_matrix(), got unitary shape "
            f"{matrix.shape}."
        )
    state = matrix.reshape(-1)
    if n_qubits is not None and state.size != 1 << n_qubits:
        raise ValueError(
            f"Expected state vector of length {1 << n_qubits}, got {state.size}."
        )
    return state / np.sqrt(np.vdot(state, state).real)


def rotate_state_from_base(
    base_state: np.ndarray, n: int, angle_degrees: float
) -> np.ndarray:
    theta = np.deg2rad(angle_degrees)
    phases = np.exp(-0.5j * theta * (n - 2 * hamming_weights(n)))
    return base_state * phases


def mean_abs_expectation(state: np.ndarray, observables: list[str]) -> float:
    return float(
        np.mean([abs(pauli_expectation(state, obs)) for obs in observables])
    )


def normalize_signed(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values))
    if scale == 0:
        return values
    return values / scale


def verify_encoding_flows(encoding: str) -> list[stim.Flow]:
    """Pauli flows for the Clifford hypercube encoding subcircuit."""
    return list(stim.Circuit(encoding).flow_generators())


def verify_steane_plus_l_flows() -> list[stim.Flow]:
    """Pauli flows for $|+_L\\rangle$ preparation on $[[7,1,3]]$."""
    return list(stim.Circuit(STEANE_PREP).flow_generators())


def gf2_rank(rows: list[np.ndarray] | np.ndarray) -> int:
    rows_array = np.asarray(rows, dtype=np.uint8)
    if rows_array.ndim == 1:
        rows_array = rows_array.reshape(1, -1)
    n = rows_array.shape[1]
    rows_int = [
        int(sum(int(bit) << i for i, bit in enumerate(row)))
        for row in rows_array
        if np.any(row)
    ]
    rank = 0
    for column in range(n):
        pivot = next(
            (i for i in range(rank, len(rows_int)) if (rows_int[i] >> column) & 1),
            None,
        )
        if pivot is None:
            continue
        rows_int[rank], rows_int[pivot] = rows_int[pivot], rows_int[rank]
        for i in range(len(rows_int)):
            if i != rank and ((rows_int[i] >> column) & 1):
                rows_int[i] ^= rows_int[rank]
        rank += 1
    return rank


def gf2_nullspace_basis(matrix: np.ndarray) -> list[np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.uint8)
    n = matrix.shape[1]
    rows = [int(sum(int(bit) << i for i, bit in enumerate(row))) for row in matrix]
    pivots: list[int] = []
    row_index = 0
    for column in range(n):
        pivot = next(
            (i for i in range(row_index, len(rows)) if (rows[i] >> column) & 1),
            None,
        )
        if pivot is None:
            continue
        rows[row_index], rows[pivot] = rows[pivot], rows[row_index]
        for i in range(len(rows)):
            if i != row_index and ((rows[i] >> column) & 1):
                rows[i] ^= rows[row_index]
        pivots.append(column)
        row_index += 1

    basis: list[np.ndarray] = []
    for free_column in (c for c in range(n) if c not in pivots):
        vector = 1 << free_column
        for row_number, pivot_column in enumerate(pivots):
            if (rows[row_number] >> free_column) & 1:
                vector |= 1 << pivot_column
        basis.append(np.array([(vector >> i) & 1 for i in range(n)], dtype=np.uint8))
    return basis


def build_reed_muller_15() -> tuple[
    list[str],
    list[str],
    Observable,
    Observable,
    np.ndarray,
]:
    """Return X/Z stabilizers, logical X/Y, and $|+_L\\rangle$ state."""
    rm_n = 15
    rm_columns = np.array(
        [[int(bit) for bit in f"{value:04b}"] for value in range(1, 16)],
        dtype=np.uint8,
    ).T
    rm_logical_x_row = np.ones(rm_n, dtype=np.uint8)
    rm_x_rows = [row.copy() for row in rm_columns]
    rm_z_rows = gf2_nullspace_basis(np.vstack([rm_logical_x_row, *rm_x_rows]))

    rm_logical_z_row: np.ndarray | None = None
    for support in itertools.combinations(range(rm_n), 3):
        candidate = np.zeros(rm_n, dtype=np.uint8)
        candidate[list(support)] = 1
        if all(np.dot(candidate, row) % 2 == 0 for row in rm_x_rows) and (
            np.dot(candidate, rm_logical_x_row) % 2 == 1
        ):
            rm_logical_z_row = candidate
            break
    assert rm_logical_z_row is not None

    x_stabilizers = [
        pauli_string(rm_n, "X", np.flatnonzero(row)) for row in rm_x_rows
    ]
    z_stabilizers = [
        pauli_string(rm_n, "Z", np.flatnonzero(row)) for row in rm_z_rows
    ]
    logical_x = Observable(
        1, pauli_string(rm_n, "X", np.flatnonzero(rm_logical_x_row))
    )
    logical_z = pauli_string(rm_n, "Z", np.flatnonzero(rm_logical_z_row))

    stabilizer_strings = x_stabilizers + z_stabilizers + [logical_x.pauli]
    stim_stabilizers = [
        stim.PauliString("+" + stim_pauli_text(p)) for p in stabilizer_strings
    ]
    prep_stim = stim.Tableau.from_stabilizers(
        stim_stabilizers,
        allow_underconstrained=False,
        allow_redundant=False,
    ).to_circuit(method="graph_state")
    # graph_state synthesis prepends RX on all qubits before the Clifford layer.
    prep_circuit = tsim.Circuit.from_stim_program(prep_stim)
    base_state = normalized_state(prep_circuit, n_qubits=rm_n)

    logical_y = logical_y_from_xz(logical_x.pauli, logical_z)
    test_state = rotate_state_from_base(base_state, rm_n, 1.0)
    if observable_expectation(test_state, logical_y) < 0:
        logical_y = Observable(-logical_y.coefficient, logical_y.pauli)

    return x_stabilizers, z_stabilizers, logical_x, logical_y, base_state


def reed_muller_prep_circuit() -> stim.Circuit:
    """Stim preparation circuit for $|+_L\\rangle$ on $[[15,1,3]]$."""
    rm_x, rm_z, logical_x, _, _ = build_reed_muller_15()
    stabilizers = rm_x + rm_z + [logical_x.pauli]
    return stim.Tableau.from_stabilizers(
        [stim.PauliString("+" + stim_pauli_text(p)) for p in stabilizers],
        allow_underconstrained=False,
        allow_redundant=False,
    ).to_circuit(method="graph_state")


def build_steane_7() -> tuple[list[str], Observable, Observable, np.ndarray]:
    """Hypercube $|+_L\\rangle$ on $[[7,1,3]]$ with Steane stabilizer labelling."""
    n = 7
    x_stabilizers = [
        pauli_string(n, "X", support) for support in STEANE_STABILIZER_SUPPORTS
    ]
    logical_x = Observable(1, pauli_string(n, "X", STEANE_LOGICAL_X_SUPPORT))
    logical_z = pauli_string(n, "Z", STEANE_LOGICAL_X_SUPPORT)
    logical_y = logical_y_from_xz(logical_x.pauli, logical_z)

    base_state = normalized_state(tsim.Circuit(STEANE_PREP), n_qubits=n)
    test_state = rotate_state_from_base(base_state, n, 1.0)
    if observable_expectation(test_state, logical_y) < 0:
        logical_y = Observable(-logical_y.coefficient, logical_y.pauli)

    return x_stabilizers, logical_x, logical_y, base_state


def build_systems() -> list[RotationSystem]:
    steane_x, steane_lx, steane_ly, steane_state = build_steane_7()
    rm_x, _, rm_lx, rm_ly, rm_state = build_reed_muller_15()

    product_state = normalized_state(
        tsim.Circuit("RX " + " ".join(map(str, range(15)))),
        n_qubits=15,
    )

    rm_corner_flip = pauli_string(15, "X", [0, 1, 3, 7])
    rm_negative_state = apply_pauli_to_state(rm_state, rm_corner_flip)

    return [
        RotationSystem(
            "2D colour ($[[7,1,3]]$)",
            7,
            steane_state,
            steane_x,
            steane_lx,
            steane_ly,
        ),
        RotationSystem(
            "3D colour ($[[15,1,3]]$)",
            15,
            rm_state,
            rm_x,
            rm_lx,
            rm_ly,
        ),
        RotationSystem(
            "Unentangled",
            15,
            product_state,
            rm_x,
            rm_lx,
            rm_ly,
        ),
        RotationSystem(
            "3D negative stabilizer",
            15,
            rm_negative_state,
            rm_x,
            rm_lx,
            rm_ly,
        ),
    ]


def compute_curve(system: RotationSystem, angles_deg: np.ndarray) -> RotationCurve:
    logical_x_vals: list[float] = []
    stab_vals: list[float] = []
    for angle in angles_deg:
        state = rotate_state_from_base(system.base_state, system.n, float(angle))
        logical_x_vals.append(observable_expectation(state, system.logical_x))
        stab_vals.append(mean_abs_expectation(state, system.x_stabilizers))
    return RotationCurve(
        angles_deg=np.asarray(angles_deg, dtype=float),
        logical_x=normalize_signed(np.asarray(logical_x_vals)),
        stabilizer_abs=normalize_signed(np.asarray(stab_vals)),
    )
