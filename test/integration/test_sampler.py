from test.helpers.gen import gen_stim_circuit

import numpy as np
import pymatching
import pytest
import pyzx as zx
import stim
from tqdm import tqdm

from tsim.circuit import Circuit
from tsim.external.vec_sim.vec_sampler import VecSampler
from tsim.sampler import CompiledStateProbs


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
                30_000, batch_size=30_000, separate_observables=True
            )
        else:
            detection_events, observable_flips = sampler.sample(
                30_000, separate_observables=True
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


@pytest.mark.parametrize(
    "stim_program",
    [
        """
        CORRELATED_ERROR(0.1) X0
        M 0
        """,
        """
        CORRELATED_ERROR(0.1) X0
        ELSE_CORRELATED_ERROR(0.2) Y1
        CORRELATED_ERROR(0.3) X2
        H 3
        CORRELATED_ERROR(0.5) Z3
        H 3
        M 0 1 2 3
        """,
        """
        CORRELATED_ERROR(0.1) X0
        CORRELATED_ERROR(0.2) X1
        M 0 1
        """,
        """
        CORRELATED_ERROR(0.1) X0
        ELSE_CORRELATED_ERROR(0.3) Y2
        CORRELATED_ERROR(0.2) X1
        ELSE_CORRELATED_ERROR(0.3) Y2
        M 0 1 2
        """,
    ],
    ids=[
        "single_correlated_error",
        "mixed_else_and_h",
        "two_correlated_errors",
        "two_correlated_errors_with_else",
    ],
)
def test_correlated_error(stim_program: str):
    n_samples = 10_000
    stim_samples = stim.Circuit(stim_program).compile_sampler(seed=0).sample(n_samples)
    tsim_samples = Circuit(stim_program).compile_sampler(seed=0).sample(n_samples)
    assert_samples_match(stim_samples, tsim_samples)


@pytest.mark.parametrize(
    "stim_program",
    [
        # Single-qubit MPP
        """
        H 0
        MPP X0
        """,
        # Two-qubit product
        """
        H 0 1
        CZ 0 1
        MPP X0*Z1
        """,
        # Inversion on first Pauli
        """
        H 0 1
        CZ 0 1
        MPP !X0*Z1
        """,
        # Inversion on second Pauli (should be equivalent to first)
        """
        H 0 1
        CZ 0 1
        MPP X0*!Z1
        """,
        # Double inversion (cancels out)
        """
        H 0 1
        CZ 0 1
        MPP !X0*!Z1
        """,
        # Three-qubit product with single inversion
        """
        H 0 1 2
        CZ 0 1
        CZ 0 2
        MPP !X0*Z1*Z2
        """,
        # Three-qubit product with inversion on middle
        """
        H 0 1 2
        CZ 0 1
        CZ 0 2
        MPP X0*!Z1*Z2
        """,
        # Three-qubit product with triple inversion (odd = inverted) and noise
        """
        H 0 1 2
        CZ 0 1
        CZ 0 2
        Z_ERROR(0.1) 0
        Y_ERROR(0.1) 1 2
        MPP !X0*!Z1*!Z2
        """,
        # Multiple products with different inversions
        """
        H 0 1 2
        CZ 0 1
        CZ 0 2
        MPP X0*Z1*Z2 X0*Z1*!Z2 X0*!Z1*!Z2 !X0*!Z1*!Z2
        """,
    ],
    ids=[
        "single_qubit",
        "two_qubit_product",
        "invert_first",
        "invert_second",
        "invert_both_cancels",
        "three_qubit_invert_first",
        "three_qubit_invert_middle",
        "three_qubit_triple_invert_and_noise",
        "multiple_products",
    ],
)
def test_mpp_inversion_parity(stim_program: str):
    """Test that MPP correctly handles inversions on any Pauli (only parity matters)."""
    n_samples = 10_000
    stim_samples = stim.Circuit(stim_program).compile_sampler(seed=0).sample(n_samples)
    tsim_samples = Circuit(stim_program).compile_sampler(seed=0).sample(n_samples)
    assert_samples_match(stim_samples, tsim_samples)


def test_channel_simplification():
    c = Circuit(
        """
        X_ERROR(0.1) 0
        X_ERROR(0.1) 0
        X_ERROR(0.1) 0
        M 0
        """
    )
    sampler = c.compile_sampler(seed=42)

    # Check that channels were simplified into a single equivalent channel
    assert len(sampler._channel_sampler.channels) == 1

    n_samples = 100_000
    samples = sampler.sample(n_samples)

    # Three independent X_ERROR(0.1) channels combine via XOR convolution
    p = 0.1
    p_combined = 3 * p * (1 - p) ** 2 + p**3

    # Count how many samples have measurement result 1
    measured_ones = np.sum(samples[:, 0])
    observed_rate = measured_ones / n_samples
    std_err = np.sqrt(p_combined * (1 - p_combined) / n_samples)

    assert (
        abs(observed_rate - p_combined) < 4 * std_err
    ), f"Expected rate {p_combined:.4f}, got {observed_rate:.4f}, std_err={std_err:.4f}"


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
