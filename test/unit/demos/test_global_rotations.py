"""Tests for the ``docs/demos/global_rotations.ipynb`` tutorial (issue #119).

These mirror the constructions used in the notebook and guard the physics it
reproduces from Fig. 4a of Bluvstein et al. (Nature 649, 39, 2026): the
Reed-Muller code encoders, the literal hypercube encoding circuit, and the
transversal-T plateau that distinguishes the 3D code from the 2D code.
"""

import numpy as np
import stim

import tsim


def _pauli(kind: str, support: list[int], n: int) -> str:
    chars = ["_"] * n
    for q in support:
        chars[q] = kind
    return "".join(chars)


def _bit(j: int, t: int) -> int:
    return (j >> t) & 1


# [[7,1,3]] Steane and [[15,1,3]] Reed-Muller code definitions.
STEANE = {
    "n": 7,
    "x_stabs": [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]],
    "z_stabs": [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]],
    "x_logical": list(range(7)),
}
_RM_X = [[j for j in range(15) if _bit(j + 1, k)] for k in range(4)]
_RM_Z = [list(s) for s in _RM_X] + [
    [j for j in range(15) if _bit(j + 1, a) and _bit(j + 1, b)]
    for a in range(4)
    for b in range(a + 1, 4)
]
REED_MULLER = {
    "n": 15,
    "x_stabs": _RM_X,
    "z_stabs": _RM_Z,
    "x_logical": list(range(15)),
    "corners": [0, 1, 3, 7],
}


def _encoder(code: dict, corner_flips: list[int] | None = None) -> stim.Circuit:
    n = code["n"]
    gens = [stim.PauliString(_pauli("X", s, n)) for s in code["x_stabs"]]
    gens += [stim.PauliString(_pauli("Z", s, n)) for s in code["z_stabs"]]
    gens.append(stim.PauliString(_pauli("X", code["x_logical"], n)))
    circuit = stim.Tableau.from_stabilizers(
        gens, allow_redundant=False, allow_underconstrained=False
    ).to_circuit("elimination")
    if corner_flips:
        circuit += stim.Circuit("X " + " ".join(str(q) for q in corner_flips))
    return circuit


def _plus_logical_statevector(
    code: dict, corner_flips: list[int] | None = None
) -> np.ndarray:
    sim = stim.TableauSimulator()
    sim.do_circuit(_encoder(code, corner_flips))
    return np.asarray(sim.state_vector(endian="big"), dtype=np.complex128)


def _support_mask(support: list[int], n: int) -> int:
    mask = 0
    for q in support:
        mask |= 1 << (n - 1 - q)
    return mask


def _expectations(code: dict, phi: float, corner_flips: list[int] | None = None):
    n = code["n"]
    psi0 = _plus_logical_statevector(code, corner_flips)
    idx = np.arange(psi0.size)
    popcount = np.array([int(b).bit_count() for b in idx])
    psi = psi0 * np.exp(1j * phi * popcount)
    logical = float(
        np.real(np.vdot(psi, psi[idx ^ _support_mask(code["x_logical"], n)]))
    )
    stab = float(
        np.mean(
            [
                abs(np.real(np.vdot(psi, psi[idx ^ _support_mask(s, n)])))
                for s in code["x_stabs"]
            ]
        )
    )
    return logical, stab


def _hypercube_plus_logical(m: int, r_x: int, r_z: int) -> np.ndarray:
    """Literal hypercube encoder (Gong-Renes / Ext. Data Fig. 10a)."""
    del r_z  # the |0> rows are the complement of the |+> rows for these codes
    n = 1 << m
    lines = [f"H {q}" for q in range(n) if q == 0 or bin(q).count("1") <= r_x]
    lines += [
        f"CX {q} {q | (1 << t)}" for t in range(m) for q in range(n) if _bit(q, t) == 0
    ]
    circuit = stim.Circuit("\n".join(lines))
    sim = stim.TableauSimulator()
    sim.do_circuit(circuit)
    psi = np.asarray(sim.state_vector(endian="big"), dtype=np.complex128).reshape(
        [2] * n
    )
    # Puncture qubit 0 in the X basis (project onto |+>) and drop it.
    branch0 = psi[(0, *([slice(None)] * (n - 1)))].reshape(-1)
    branch1 = psi[(1, *([slice(None)] * (n - 1)))].reshape(-1)
    sub = branch0 + branch1
    return sub / np.linalg.norm(sub)


def test_flow_generators_prepare_logical_x():
    # The encoder maps an input Z onto the all-X logical X_L (PauliString index 1 == X).
    enc = _encoder(REED_MULLER)
    all_x_flows = [
        f
        for f in enc.flow_generators()
        if all(f.output_copy()[q] == 1 for q in range(15))
    ]
    assert all_x_flows


def test_reed_muller_stabilizer_weights():
    assert sorted(len(s) for s in REED_MULLER["x_stabs"]) == [8, 8, 8, 8]
    assert sorted(len(s) for s in REED_MULLER["z_stabs"]) == [
        4,
        4,
        4,
        4,
        4,
        4,
        8,
        8,
        8,
        8,
    ]


def test_phi_zero_is_plus_logical():
    for code in (STEANE, REED_MULLER):
        logical, stab = _expectations(code, 0.0)
        assert np.isclose(logical, 1.0, atol=1e-6)
        assert np.isclose(stab, 1.0, atol=1e-6)


def test_hypercube_matches_stabilizer_encoder():
    # The literal hypercube circuit prepares the same |+_L> state, exactly.
    steane_sv = _plus_logical_statevector(STEANE)
    rm_sv = _plus_logical_statevector(REED_MULLER)
    assert abs(np.vdot(_hypercube_plus_logical(3, 1, 1), steane_sv)) > 1 - 1e-6
    assert abs(np.vdot(_hypercube_plus_logical(4, 1, 2), rm_sv)) > 1 - 1e-6


def test_reed_muller_has_transversal_t_plateau():
    # 3D code: at 45 degrees the global rotation is a logical T and the
    # stabilizers revive (transversal T). 2D Steane code: they do not.
    rm_logical, rm_stab = _expectations(REED_MULLER, np.deg2rad(45))
    assert np.isclose(rm_logical, 1 / np.sqrt(2), atol=1e-6)
    assert np.isclose(rm_stab, 1.0, atol=1e-6)
    _, steane_stab = _expectations(STEANE, np.deg2rad(45))
    assert steane_stab < 0.9


def test_negative_stabilizer_removes_plateau():
    # Flipping the four tetrahedron corners destroys the 45-degree plateau.
    logical, stab = _expectations(
        REED_MULLER, np.deg2rad(45), corner_flips=REED_MULLER["corners"]
    )
    assert abs(logical) < 1e-6
    assert stab < 1e-6


def test_tsim_sampler_matches_exact_on_3d_code():
    # tsim's stabilizer-rank sampler runs the 15-rotation 3D code.
    n = REED_MULLER["n"]
    text = str(_encoder(REED_MULLER)) + "\n"
    text += "R_Z(0.25) " + " ".join(map(str, range(n))) + "\n"
    text += "MPP " + "*".join(f"X{q}" for q in REED_MULLER["x_logical"]) + "\n"
    text += "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    sampler = tsim.Circuit(text).compile_detector_sampler(seed=1)
    obs = sampler.sample(shots=4000, separate_observables=True)[1]
    assert abs((1.0 - 2.0 * obs.mean()) - 1 / np.sqrt(2)) < 0.05
