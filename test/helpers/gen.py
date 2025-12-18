import random

import stim


def gen_stim_circuit(
    qubits: int,
    depth: int,
    include_measurements: bool = True,
    p_t: float = 1,
    p_h: float = 1,
    p_s: float = 1,
    p_sqrt_x: float = 1,
    p_sqrt_y: float = 1,
    p_cnot: float = 1,
    p_x: float = 1,
    p_z: float = 1,
    p_y: float = 1,
    p_x_error: float = 0,
    p_z_error: float = 0,
    p_y_error: float = 0,
    p_depolarize1: float = 0,
    p_depolarize2: float = 0,
    p_pauli_channel_1: float = 0,
    p_pauli_channel_2: float = 0,
    seed: int | None = None,
) -> stim.Circuit:
    """Generate a random stim circuit with specified gate probabilities.

    Args:
        qubits: Number of qubits in the circuit.
        depth: Number of random gates to add.
        include_measurements: Whether to include measurements at the end of the circuit.
        p_*: Relative probability for each gate type (0 to disable).
        seed: Random seed.

    Returns:
        A stim.Circuit with random gates.
    """
    if seed is not None:
        random.seed(seed)

    # Gate specs: (stim_cmd, num_qubits, probability)
    gates = [
        ("S[T]", 1, p_t),
        ("S", 1, p_s),
        ("H", 1, p_h),
        ("SQRT_X", 1, p_sqrt_x),
        ("SQRT_Y", 1, p_sqrt_y),
        ("CNOT", 2, p_cnot),
        ("X_ERROR(1)", 1, p_x),
        ("Z_ERROR(1)", 1, p_z),
        ("Y", 1, p_y),
        ("X_ERROR(0.4)", 1, p_x_error),
        ("Z_ERROR(0.4)", 1, p_z_error),
        ("Y_ERROR(0.4)", 1, p_y_error),
        ("DEPOLARIZE1(0.4)", 1, p_depolarize1),
        ("DEPOLARIZE2(0.5)", 2, p_depolarize2),
        ("PAULI_CHANNEL_1(0.3, 0.2, 0.1)", 1, p_pauli_channel_1),
        (
            "PAULI_CHANNEL_2(0.1, 0.12, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0002)",
            2,
            p_pauli_channel_2,
        ),
    ]

    # Filter to gates with non-zero probability
    gates = [(cmd, nq, p) for cmd, nq, p in gates if p > 0]
    if not gates:
        raise ValueError("At least one gate must have non-zero probability")

    total = sum(p for _, _, p in gates)
    probs = [p / total for _, _, p in gates]

    circ = stim.Circuit()
    for q in range(qubits):
        circ.append_from_stim_program_text(f"R {q}")

    for _ in range(depth):
        cmd, num_qubits, _ = random.choices(gates, weights=probs, k=1)[0]

        if num_qubits > qubits:
            continue

        targets = random.sample(range(qubits), num_qubits)
        circ.append_from_stim_program_text(f"{cmd} {' '.join(map(str, targets))}")

    if include_measurements:
        circ.append_from_stim_program_text(
            "M " + " ".join([str(i) for i in range(qubits)])
        )

    return circ
