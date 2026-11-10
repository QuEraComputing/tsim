"""Unit tests of the conditioned backend's kernels against stim and dense references."""

from test.helpers import conditioned as exact
from test.helpers.statevector import (
    assert_counts_match,
    assert_distributions_close,
    ops_distribution,
)

import numpy as np
import pytest
import stim

from tsim.circuit import Circuit
from tsim.compile.pipeline import classify_direct
from tsim.core.graph import prepare_graph
from tsim.core.types import SamplingStrategy
from tsim.statevector.conditioned import lane, paulis
from tsim.statevector.conditioned.component import (
    ConditionedComponent,
    conditioned_component,
)
from tsim.statevector.conditioned.gauss import layer_amplitude
from tsim.statevector.pbc import ROTATION, ComponentLayout, read_pauli_ops

_SIGN_POWER = {1: 0, 1j: 1, -1: 2, -1j: 3}
_GATES = {
    "H": lane.H,
    "S": lane.S,
    "S_DAG": lane.S_DAG,
    "X": lane.X,
    "Z": lane.Z,
    "CX": lane.CX,
    "CZ": lane.CZ,
    "CY": lane.CY,
}


def to_row(p: stim.PauliString, words: int) -> np.ndarray:
    """Return ``p`` as a row ``i^e X^x Z^z``."""
    xs, zs = p.to_numpy()
    row = np.zeros(2 * words + 1, dtype=np.uint64)
    for q in np.flatnonzero(xs):
        row[q >> 6] |= np.uint64(1) << np.uint64(q & 63)
    for q in np.flatnonzero(zs):
        row[words + (q >> 6)] |= np.uint64(1) << np.uint64(q & 63)
    row[2 * words] = (int(np.count_nonzero(xs & zs)) + _SIGN_POWER[p.sign]) % 4
    return row


def random_pauli(rng, n: int) -> stim.PauliString:
    p = stim.PauliString.random(n)
    return p if rng.random() < 0.5 else -p


@pytest.mark.parametrize("n", [5, 70])
def test_pauli_rows_multiply_and_commute_as_stim(n):
    rng = np.random.default_rng(n)
    words = -(-n // 64)
    for _ in range(200):
        a, b = random_pauli(rng, n), random_pauli(rng, n)
        row = to_row(a, words)
        paulis.multiply_right(row, to_row(b, words), words)
        assert np.array_equal(row, to_row(a * b, words))
        assert paulis.anticommutes(to_row(a, words), to_row(b, words), words) == (
            not a.commutes(b)
        )


@pytest.mark.parametrize("n", [4, 70])
def test_gates_and_frame_images_match_stim(n):
    rng = np.random.default_rng(n + 1)
    words = -(-n // 64)
    frame, _, _, _, _, _ = lane.new_state(n, words, 1)
    tableau = stim.Tableau(n)
    for _ in range(60):
        name = str(rng.choice(list(_GATES)))
        if _GATES[name] >= lane.CX:
            a, b = (int(q) for q in rng.choice(n, 2, replace=False))
        else:
            a, b = int(rng.integers(n)), 0
        targets = [a, b] if _GATES[name] >= lane.CX else [a]
        p = random_pauli(rng, n)
        row = to_row(p, words)
        lane.row_gate(row, _GATES[name], a, b, words)
        gate = stim.Tableau.from_named_gate(name)
        assert np.array_equal(
            row,
            to_row(
                p.after(stim.Circuit(f"{name} {' '.join(map(str, targets))}")), words
            ),
        )
        lane.frame_gate(frame, _GATES[name], a, b, words)
        tableau.append(gate, targets)
    for _ in range(50):
        p = random_pauli(rng, n)
        xs, zs = p.to_numpy()
        plain = to_row(stim.PauliString.from_numpy(xs=xs, zs=zs), words)
        out = np.empty(2 * words + 1, dtype=np.uint64)
        paulis.image(
            frame,
            plain[:words],
            plain[words : 2 * words],
            int(plain[2 * words]),
            out,
            words,
        )
        assert np.array_equal(
            out, to_row(tableau(stim.PauliString.from_numpy(xs=xs, zs=zs)), words)
        )


def _dense(p: stim.PauliString) -> np.ndarray:
    """Return the matrix of ``p`` in complex128 (bit q of an index is qubit q)."""
    letters = {
        0: np.eye(2),
        1: np.array([[0, 1], [1, 0]]),
        2: np.array([[0, -1j], [1j, 0]]),
        3: np.diag([1, -1]),
    }
    out = np.array([[1.0 + 0j]])
    for q in range(len(p)):
        out = np.kron(letters[p[q]], out)
    return complex(p.sign) * out


def test_layer_amplitude_matches_dense():
    rng = np.random.default_rng(7)
    n, words = 5, 1
    for _ in range(40):
        # Commuting V_k = U Z^{u_k} U^dag, all anticommuting with M = U X^m U^dag.
        u = stim.Tableau(n)
        for _ in range(30):
            a, b = (int(q) for q in rng.choice(n, 2, replace=False))
            name = str(rng.choice(["H", "S", "CX", "CZ"]))
            u.append(
                stim.Tableau.from_named_gate(name), [a, b] if name[0] == "C" else [a]
            )
        m = rng.integers(0, 2, n)
        m[0] = 1
        vs = []
        for _ in range(int(rng.integers(1, 6))):
            zs = rng.integers(0, 2, n).astype(bool)
            if (zs.astype(int) @ m) % 2 == 0:
                zs[np.flatnonzero(m)[0]] ^= True
            v = u(stim.PauliString.from_numpy(xs=np.zeros(n, bool), zs=zs))
            vs.append(v if rng.random() < 0.5 else -v)
        mp = u(stim.PauliString.from_numpy(xs=m.astype(bool), zs=np.zeros(n, bool)))
        thetas = (2 * rng.integers(0, 4, len(vs)) + 1) * np.pi / 8
        rows = np.array([to_row(v, words) for v in vs])
        eps = (np.cos(2 * thetas) < 0).astype(np.int64)
        eta = (np.sin(2 * thetas) < 0).astype(np.int64)
        dense = _dense(mp)
        for v, th in zip(vs, thetas, strict=True):
            dense = (
                np.cos(2 * th) * np.eye(2**n) + 1j * np.sin(2 * th) * _dense(v)
            ) @ dense
        column = dense[:, 0]
        for target in range(2**n):
            t = np.array([target], dtype=np.uint64)
            got = layer_amplitude(rows, len(vs), eps, eta, to_row(mp, words), t, words)
            assert abs(got - column[target]) < 1e-9


def random_t_circuit(rng, n: int, depth: int) -> str:
    """Return a T-heavy random circuit: Clifford and T layers, noise, Pauli-product checks."""
    lines = []
    for _ in range(depth):
        r = rng.random()
        q = [int(x) for x in rng.choice(n, 2, replace=False)]
        if r < 0.3:
            lines.append(f"{rng.choice(['T', 'T_DAG'])} {q[0]}")
        elif r < 0.5:
            lines.append(f"{rng.choice(['H', 'S', 'SQRT_X'])} {q[0]}")
        elif r < 0.75:
            lines.append(f"{rng.choice(['CX', 'CZ'])} {q[0]} {q[1]}")
        elif r < 0.85:
            lines.append(f"DEPOLARIZE1(0.05) {q[0]}")
        elif r < 0.95:
            a, b = (str(rng.choice(["X", "Y", "Z"])) for _ in range(2))
            lines.append(f"MPP {a}{q[0]}*{b}{q[1]}")
        else:
            lines.append(f"X_ERROR(0.1) {q[1]}")
    lines += [f"{rng.choice(['M', 'MX', 'MY'])} {q}" for q in range(n)]
    return "\n".join(lines)


def _random_components(seed: int):
    rng = np.random.default_rng(seed)
    text = random_t_circuit(rng, int(rng.integers(4, 7)), int(rng.integers(25, 45)))
    prepared = prepare_graph(Circuit(text), sample_detectors=False)
    for component in prepared.graph.connected_components():
        if classify_direct(component, prepared.layout) is not None:
            continue
        outputs = [
            p for v in component.outputs() if (p := component.port(v)) is not None
        ]
        if not outputs:
            continue
        layout = ComponentLayout.of(component, outputs, prepared.layout)
        ops = read_pauli_ops(component, layout)
        if ops.num_lines <= 10:  # NOTE: the dense reference doubles per line
            yield ops


@pytest.mark.parametrize("measure_through", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_random_components_are_exact_under_both_plans(seed, measure_through):
    rng = np.random.default_rng(seed)
    for ops in _random_components(seed):
        layout = ComponentLayout(
            tuple(range(ops.num_f)), {}, tuple(range(ops.num_outputs))
        )
        component = conditioned_component(ops, layout, measure_through=measure_through)
        cops = component.ops
        for _ in range(4):
            f = (rng.random(ops.num_f) < 0.3).astype(np.int64)
            signs = (
                cops.sign[:, 0].astype(np.int64) + cops.sign[:, 1:].astype(np.int64) @ f
            ) & 1
            flipped = (((cops.sign[:, 1:].astype(np.int64) @ f) & 1).astype(bool)) & (
                cops.kind == ROTATION
            )
            leaves, probs = exact.key_leaves(component, flipped)
            got: dict[tuple[int, ...], float] = {}
            for raw, p in zip(leaves, probs, strict=True):
                key = tuple(
                    int(raw[s]) ^ int(signs[np.flatnonzero(cops.slot == s)[0]])
                    for s in range(cops.num_outputs)
                )
                got[key] = got.get(key, 0.0) + p
            assert_distributions_close(
                got, ops_distribution(ops, f.tolist()), atol=1e-9
            )


def test_the_reference_tree_walk_draws_what_the_lanes_draw():
    text = """
        RX 0 1 2
        R 3
        T 0 1 2
        CNOT 0 3 1 3
        T_DAG 3
        CNOT 2 3
        T 3
        CNOT 1 3
        T_DAG 3
        CNOT 2 3
        T 3
        CNOT 0 3
        DEPOLARIZE1(0.01) 0 1 2
        M 3
        MX 0 1 2
        DETECTOR rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-3]
        OBSERVABLE_INCLUDE(1) rec[-2]
    """
    circuit = Circuit(text)
    mask = np.ones(circuit.num_detectors, bool)
    out = []
    for cached in (True, False):
        sampler = circuit.compile_detector_sampler(
            seed=4, strategy="conditioned_statevector"
        )
        if not cached:
            for component in sampler._program.components:
                assert isinstance(component, ConditionedComponent)
                component.cache["tree"] = None
        out.append(
            sampler.sample(20_000, postselection_mask=mask, append_observables=True)
        )
    assert np.array_equal(out[0], out[1])


def test_more_than_64_qubits_match_active_statevector():
    n = 70
    lines = [f"RX {' '.join(map(str, range(n)))}"]
    lines += [f"CZ {i} {i + 1}" for i in range(0, n - 1, 2)]
    lines += [
        "T 0 31 64 69",
        "CNOT 0 69 64 31",
        "T_DAG 69 31",
        "DEPOLARIZE1(0.02) 0 31 64 69",
    ]
    lines += [f"MX {' '.join(map(str, range(n)))}"]
    circuit = Circuit("\n".join(lines))
    runs: tuple[tuple[int, SamplingStrategy], ...] = (
        (1, "active_statevector"),
        (2, "conditioned_statevector"),
    )
    samples = [
        circuit.compile_sampler(seed=s, strategy=st).sample(20_000)[
            :, [0, 31, 64, 69, 1, 30, 63, 68]
        ]
        for s, st in runs
    ]
    assert_counts_match(samples[1], samples[0])
