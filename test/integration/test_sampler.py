import os
import sys

import pymatching
from tqdm import tqdm

import tsim.external.pyzx as zx
from tsim.sampler import CompiledStateProbs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from test.helpers.gen import gen_stim_circuit

import numpy as np
import pytest
import stim

from tsim.circuit import Circuit
from tsim.external.vec_sim.vec_sampler import VecSampler


def bitstring_to_int(samples: np.ndarray) -> np.ndarray:
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


def assert_samples_match(samples1: np.ndarray, samples2: np.ndarray):
    samples1_ints = bitstring_to_int(samples1)
    samples2_ints = bitstring_to_int(samples2)

    # Count occurrences of each state
    max_val = 2 ** samples1.shape[1]
    stim_counts = np.bincount(samples1_ints.astype(int), minlength=max_val)
    tsim_counts = np.bincount(samples2_ints.astype(int), minlength=max_val)

    # Calculate sampling errors (binomial standard deviation)
    n_samples = samples1.shape[0]
    stim_err = np.sqrt(stim_counts * (1 - stim_counts / n_samples))
    tsim_err = np.sqrt(tsim_counts * (1 - tsim_counts / n_samples))
    err = np.max([stim_err, tsim_err], axis=0)

    # assert that stim_counts is within 4-sigma of exact_counts plus small constant
    # to handle cases where counts are close to 0
    assert np.all(np.abs(stim_counts - tsim_counts) <= 4 * err + 3)


@pytest.mark.parametrize(
    "code_task",
    [
        "repetition_code:memory",
        "surface_code:rotated_memory_x",
        "surface_code:rotated_memory_z",
        "surface_code:unrotated_memory_x",
        "surface_code:unrotated_memory_z",
        "color_code:memory_xyz",
    ],
)
def test_quantum_memory_codes_without_noise(code_task: str):

    circ = stim.Circuit.generated(
        code_task,
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.0,
        before_measure_flip_probability=0.0,
        before_round_data_depolarization=0.0,
        after_reset_flip_probability=0.0,
    )
    c = Circuit.from_stim_program(circ)
    sampler = c.compile_detector_sampler(seed=0)
    samples = sampler.sample(10)
    assert not np.any(samples)


@pytest.mark.parametrize("seed", [1, 2, 42])
def test_sampler(seed):
    num_qubits = 3
    stim_circuit = gen_stim_circuit(
        num_qubits, 100, include_measurements=True, seed=seed
    )
    circuit = Circuit.from_stim_program(stim_circuit)

    n_samples = 512
    batch_size = n_samples

    # Sample from both simulators
    stim_sampler = VecSampler(stim_circuit)
    sampler = circuit.compile_sampler(seed=seed)

    stim_samples, _, _ = stim_sampler.sample(n_samples)
    tsim_samples = sampler.sample(n_samples, batch_size=batch_size)

    assert_samples_match(stim_samples, tsim_samples)


@pytest.mark.parametrize(
    "code_task",
    [
        "repetition_code:memory",
        "surface_code:rotated_memory_x",
        "surface_code:rotated_memory_z",
        "surface_code:unrotated_memory_x",
        "surface_code:unrotated_memory_z",
    ],
)
def test_memory_error_correction_and_compare_to_stim(code_task: str):
    p = 0.01
    circ = stim.Circuit.generated(
        code_task,
        distance=3,
        rounds=2,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p * 1.2,
        before_round_data_depolarization=p * 0.8,
        after_reset_flip_probability=p * 0.9,
    )
    error_count = []
    error_count_after_correction = []

    for c in [circ, Circuit.from_stim_program(circ)]:
        sampler = c.compile_detector_sampler(seed=0)
        if isinstance(c, Circuit):
            detection_events, observable_flips = sampler.sample(
                10_000, batch_size=10_000, separate_observables=True
            )
        else:
            detection_events, observable_flips = sampler.sample(
                10_000, separate_observables=True
            )

        detector_error_model = circ.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

        predictions = matcher.decode_batch(detection_events)

        num_errors = np.count_nonzero(observable_flips)
        num_errors_after_correction = np.count_nonzero(
            np.logical_xor(observable_flips, predictions)
        )
        error_count.append(num_errors)
        error_count_after_correction.append(num_errors_after_correction)
        assert num_errors_after_correction <= num_errors

    stim_errors = error_count[0]
    tsim_errors = error_count[1]
    stim_errors_after_correction = error_count_after_correction[0]
    tsim_errors_after_correction = error_count_after_correction[1]

    assert np.abs(stim_errors - tsim_errors) / stim_errors <= 0.1
    assert (
        np.abs(stim_errors_after_correction - tsim_errors_after_correction)
        / stim_errors_after_correction
        <= 0.3
    )


def simulate_with_vec_sampler(stim_circuit: stim.Circuit) -> np.ndarray:
    """Compute state probabilities using VecSampler statevector simulator.

    Args:
        stim_circuit: The stim circuit (with tags) to simulate. Should not include measurements.

    Returns:
        Array of probabilities for each computational basis state.
    """
    sampler = VecSampler(stim_circuit)
    state_vector = sampler.state_vector().reshape((-1,))
    return np.abs(state_vector) ** 2


def simulate_with_tsim(stim_circuit: stim.Circuit) -> np.ndarray:
    """Compute state probabilities using tsim's CompiledStateProbs.

    Args:
        stim_circuit: The stim circuit (with tags) to simulate. Should not include measurements.

    Returns:
        Array of probabilities for each computational basis state.
    """
    num_qubits = stim_circuit.num_qubits
    stim_circuit_with_m = stim_circuit.copy()
    stim_circuit_with_m.append_from_stim_program_text(
        "M " + " ".join([str(i) for i in range(stim_circuit.num_qubits)])
    )
    circuit = Circuit.from_stim_program(stim_circuit_with_m)
    prob_sampler = CompiledStateProbs(circuit)

    probabilities = []
    for i in range(2**num_qubits):
        state = np.zeros(num_qubits, dtype=np.bool_)
        for j in range(num_qubits):
            state[j] = (i >> j) & 1
        state = state[::-1]
        probabilities.append(prob_sampler.probability_of(state, batch_size=1)[0])

    return np.array(probabilities)


def simulate_with_pyzx_tensor(stim_circuit: stim.Circuit) -> np.ndarray:
    """Compute state probabilities using pyzx tensor contraction.

    Args:
        stim_circuit: The stim circuit (with tags) to simulate. Should not include measurements.

    Returns:
        Array of normalized probabilities for each computational basis state.
    """
    stim_circuit_without_noise = stim.Circuit(
        str(stim_circuit).replace("X_ERROR(1)", "X").replace("Z_ERROR(1)", "Z")
        + "\nM "
        + " ".join([str(i) for i in range(stim_circuit.num_qubits)])
    )
    c = Circuit.from_stim_program(stim_circuit_without_noise)
    g = c.get_sampling_graph()
    state_probs = g.to_tensor().reshape((-1,))
    return state_probs / np.sum(state_probs)


@pytest.mark.parametrize("num_qubits", [3, 4])
@pytest.mark.parametrize("seed", [1, 2])
def test_compare_to_statevector_simulator_and_pyzx_tensor(num_qubits, seed):
    stim_circuit = gen_stim_circuit(
        num_qubits,
        100,
        include_measurements=False,
        seed=seed,
    )
    tsim_state_vector = simulate_with_tsim(stim_circuit)
    pyzx_state_vector = simulate_with_pyzx_tensor(stim_circuit)
    stim_state_vector = simulate_with_vec_sampler(stim_circuit)

    assert np.allclose(stim_state_vector, pyzx_state_vector)
    assert np.allclose(tsim_state_vector, pyzx_state_vector)


@pytest.mark.parametrize("num_qubits", [3, 4])
@pytest.mark.parametrize("seed", [2, 42])
def test_compare_to_statevector_simulator_and_pyzx_tensor_with_arbitrary_rotations(
    num_qubits, seed
):
    stim_circuit = gen_stim_circuit(
        num_qubits,
        20,
        include_measurements=False,
        seed=seed,
        p_r_x=1,
        p_r_y=1,
        p_r_z=1,
        p_u3=1,
    )
    c = Circuit.from_stim_program(stim_circuit)
    assert zx.simplify.u3_count(c.get_graph()) > 0

    tsim_state_vector = simulate_with_tsim(stim_circuit)
    pyzx_state_vector = simulate_with_pyzx_tensor(stim_circuit)
    stim_state_vector = simulate_with_vec_sampler(stim_circuit)

    tol = 1e-6
    assert np.allclose(stim_state_vector, pyzx_state_vector, atol=tol, rtol=tol)
    assert np.allclose(tsim_state_vector, pyzx_state_vector, atol=tol, rtol=tol)


if __name__ == "__main__":
    # Debugging code...
    import random
    from test.helpers.util import plot_comparison

    initial_seed = random.randint(0, int(1e10))
    for i in tqdm(range(100_000)):
        seed = initial_seed + i
        random.seed(seed)
        stim_circuit = gen_stim_circuit(
            qubits=3,
            depth=10,  # reduce depth when using U3 gates
            include_measurements=False,
            seed=seed,
            p_r_x=1,
            p_r_y=1,
            p_r_z=1,
            p_u3=1,
        )
        tsim_state_vector = simulate_with_tsim(stim_circuit)
        pyzx_state_vector = simulate_with_pyzx_tensor(stim_circuit)
        stim_state_vector = simulate_with_vec_sampler(stim_circuit)

        tol = 1e-6  # reduce tolerance when using U3 gates, since it uses fp32
        if not np.allclose(
            tsim_state_vector, pyzx_state_vector, atol=tol, rtol=tol
        ) or not np.allclose(stim_state_vector, pyzx_state_vector, atol=tol, rtol=tol):
            c = Circuit.from_stim_program(stim_circuit)
            from IPython.display import display

            display(c.diagram("timeline-svg"))

            plot_comparison(
                tsim_state_vector,
                tsim_state_vector,
                pyzx_state_vector,
                plot_difference=False,
            )

            assert np.allclose(
                stim_state_vector, pyzx_state_vector, atol=tol, rtol=tol
            ), f"Seed: {seed}"
            assert np.allclose(
                tsim_state_vector, pyzx_state_vector, atol=tol, rtol=tol
            ), f"Seed: {seed}"
