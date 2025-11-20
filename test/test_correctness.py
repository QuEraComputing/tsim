import random

import numpy as np
import pytest
import stim

import tsim.external.pyzx as zx
from tsim.circuit import Circuit
from tsim.external.vec_sim.vec_sampler import VecSampler
from tsim.sampler import CompiledProbSampler


def random_stim_circuit(
    qubits: int,
    depth: int,
    p_t: float = 1,
    p_s: float = 1,
    p_hsh: float = 1,
    p_cnot: float = 1,
    p_x: float = 1,
    p_z: float = 1,
    p_h: float = 1,
    seed: int | None = None,
) -> stim.Circuit:

    if seed is not None:
        random.seed(seed)

    # Set default equal probabilities if not specified
    gate_probs = []
    gate_types = []

    if p_t > 0:
        gate_probs.append(p_t)
        gate_types.append("t")
    if p_s > 0:
        gate_probs.append(p_s)
        gate_types.append("s")
    if p_hsh > 0:
        gate_probs.append(p_hsh)
        gate_types.append("hsh")
    if p_cnot > 0:
        gate_probs.append(p_cnot)
        gate_types.append("cnot")
    if p_x > 0:
        gate_probs.append(p_x)
        gate_types.append("x")
    if p_z > 0:
        gate_probs.append(p_z)
        gate_types.append("z")
    if p_h > 0:
        gate_probs.append(p_h)
        gate_types.append("h")

    # Normalize probabilities
    total = sum(gate_probs)
    gate_probs = [p / total for p in gate_probs]

    circ = stim.Circuit()

    for q in range(qubits):
        circ.append_from_stim_program_text(f"R {q}")

    for _ in range(depth):
        gate_type = random.choices(gate_types, weights=gate_probs, k=1)[0]

        if gate_type == "t":
            q = random.randint(0, qubits - 1)
            circ.append_from_stim_program_text(f"S[T] {q}")

        elif gate_type == "s":
            q = random.randint(0, qubits - 1)
            circ.append_from_stim_program_text(f"S {q}")

        elif gate_type == "hsh":
            q = random.randint(0, qubits - 1)
            gate = random.choice(["H", "SQRT_X", "SQRT_Y"])
            circ.append_from_stim_program_text(f"{gate} {q}")

        elif gate_type == "cnot":
            if qubits < 2:
                continue
            q1, q2 = random.sample(range(qubits), 2)
            circ.append_from_stim_program_text(f"CNOT {q1} {q2}")

        elif gate_type == "x":
            q = random.randint(0, qubits - 1)
            circ.append_from_stim_program_text(f"X {q}")

        elif gate_type == "z":
            q = random.randint(0, qubits - 1)
            circ.append_from_stim_program_text(f"Z {q}")

        elif gate_type == "h":
            q = random.randint(0, qubits - 1)
            circ.append_from_stim_program_text(f"H {q}")

    return circ


def samples_to_ints(samples: np.ndarray) -> np.ndarray:
    """Convert binary samples to integers.

    Args:
        samples: Array of shape (n_samples, n_qubits) with binary values

    Returns:
        Array of integers representing the binary samples
    """
    # Convert each binary sample to an integer
    # e.g., [1, 0, 1] -> 1*2^2 + 0*2^1 + 1*2^0 = 5
    powers = 2 ** np.arange(samples.shape[1])[::-1]
    return samples @ powers


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sampler(seed):
    num_qubits = 3
    stim_circuit = random_stim_circuit(num_qubits, 100, p_hsh=0, seed=seed)
    stim_circuit.append_from_stim_program_text(
        "M " + " ".join([str(i) for i in range(num_qubits)])
    )
    circuit = Circuit.from_stim_program(stim_circuit)

    batch_size = 512
    n_samples = batch_size

    # Sample from both simulators
    stim_sampler = VecSampler(stim_circuit, False)
    stim_samples, _, _ = stim_sampler.sample(n_samples)

    sampler = circuit.compile_sampler()
    tsim_samples = sampler.sample(n_samples, batch_size=batch_size)

    # Convert samples to integers
    stim_ints = samples_to_ints(stim_samples)
    tsim_ints = samples_to_ints(tsim_samples)

    # Calculate bin edges to ensure same binning for both
    max_val = 2**num_qubits

    # Count occurrences of each state
    stim_counts = np.bincount(stim_ints.astype(int), minlength=max_val)
    tsim_counts = np.bincount(tsim_ints.astype(int), minlength=max_val)

    # Calculate sampling errors (binomial standard deviation)
    stim_err = np.sqrt(stim_counts * (1 - stim_counts / n_samples))
    tsim_err = np.sqrt(tsim_counts * (1 - tsim_counts / n_samples))

    g = circuit.without_annotations().g
    g.normalize()
    zx.full_reduce(g, paramSafe=True)
    exact_state = np.abs(g.to_matrix()[:, 0]) ** 2
    exact_state /= np.sum(exact_state)
    exact_counts = exact_state * n_samples

    # assert that stim_counts is within 4-sigma of exact_counts
    assert np.all(np.abs(stim_counts - exact_counts) <= 3 * stim_err + 1e-12)
    assert np.all(np.abs(tsim_counts - exact_counts) <= 3 * tsim_err + 1e-12)


@pytest.mark.parametrize("num_qubits", [3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_statevector_simulator(num_qubits, seed):
    stim_circuit = random_stim_circuit(num_qubits, 100, p_hsh=0, seed=seed)  # seed 5

    # Statevector simulator
    stim_sampler = VecSampler(stim_circuit)
    stim_state = np.abs(stim_sampler.state_vector().reshape((-1,))) ** 2

    # tsim statevector simulator
    circuit = Circuit.from_stim_program(stim_circuit)
    circuit.m(range(num_qubits))
    prob_sampler = CompiledProbSampler(circuit)

    sv = []
    for i in range(2**num_qubits):
        state = np.zeros(num_qubits, dtype=np.bool_)
        for j in range(num_qubits):
            state[j] = (i >> j) & 1
        state = state[::-1]
        sv.append(prob_sampler.probabilities(state, batch_size=1)[0])
    tsim_state = np.array(sv)

    # pyzx tensor contraction
    g = circuit.without_annotations().g
    g.normalize()
    zx.full_reduce(g, paramSafe=True)
    exact_state = np.abs(g.to_matrix()[:, 0]) ** 2
    exact_state /= np.sum(exact_state)

    assert np.allclose(tsim_state, exact_state, rtol=1e-6, atol=1e-6)
    assert np.allclose(stim_state, exact_state, rtol=1e-6, atol=1e-6)
