"""Circuit generators for benchmarking.

Each generator returns a tsim.Circuit. Add new generator functions here
to benchmark different circuit families.
"""

import random

from tsim import Circuit


def exponentiated_pauli(
    qubits: list[int],
    paulis: list[str],
    angle: float,
) -> list[str]:
    """Build gate lines for a single exponentiated Pauli unitary.

    Implements ``exp(-i * angle * pi * P)`` where *P* is the Pauli string
    defined by *paulis* acting on *qubits*.

    The decomposition uses basis-change gates, a CNOT ladder to compute
    parity into the last qubit, and an R_Z rotation.

    Args:
        qubits: Qubit indices the Pauli string acts on.
        paulis: Pauli operators per qubit (each ``"X"``, ``"Y"``, or ``"Z"``).
        angle: Rotation angle in units of pi.

    Returns:
        A list of instruction strings (one per line).

    """
    if len(qubits) != len(paulis):
        raise ValueError("qubits and paulis must have the same length")
    if len(qubits) == 0:
        return []

    lines: list[str] = []

    # 1. Basis change to Z-basis
    for q, p in zip(qubits, paulis):
        if p == "X":
            lines.append(f"H {q}")
        elif p == "Y":
            lines.append(f"S_DAG {q}")
            lines.append(f"H {q}")

    # 2. CNOT ladder to compute parity into the last qubit
    target = qubits[-1]
    for q in qubits[:-1]:
        lines.append(f"CNOT {q} {target}")

    # 3. R_Z rotation on target
    if angle == 0.25:
        lines.append(f"T {target}")
    elif angle == 0.75:
        lines.append(f"T {target}")
        lines.append(f"S {target}")
    elif angle == 1.25:
        lines.append(f"T {target}")
        lines.append(f"Z {target}")
    elif angle == 1.75:
        lines.append(f"T {target}")
        lines.append(f"S {target}")
        lines.append(f"Z {target}")
    else:
        lines.append(f"R_Z({angle}) {target}")

    # 4. Reverse CNOT ladder
    for q in reversed(qubits[:-1]):
        lines.append(f"CNOT {q} {target}")

    # 5. Reverse basis change
    for q, p in zip(qubits, paulis):
        if p == "X":
            lines.append(f"H {q}")
        elif p == "Y":
            lines.append(f"H {q}")
            lines.append(f"S {q}")

    return lines


def random_clifford_t(
    n_qubits: int,
    t_count: int,
    seed: int | None = None,
) -> Circuit:
    """Generate a random Clifford+T circuit via exponentiated Pauli unitaries.

    Constructs circuits by composing ``t_count`` exponentiated Pauli unitaries
    of the form ``exp(-i (2k+1) pi/4  P)`` where *P* is a random Pauli string.
    The weight of each Pauli string (number of non-identity operators) is chosen
    uniformly between 2 and min(4, n_qubits), mimicking quantum-chemistry
    Hamiltonians.

    Reference: Section 5.1 of arXiv:2202.09202.

    Args:
        n_qubits: Number of qubits in the circuit.
        t_count: Number of exponentiated Pauli unitaries (equals the T-count).
        seed: Random seed for reproducibility.

    Returns:
        A :class:`tsim.Circuit` with the specified T-count.

    """
    if seed is not None:
        random.seed(seed)

    idx_str = " ".join(map(str, range(n_qubits)))
    lines: list[str] = [f"RX {idx_str}"]

    for _ in range(t_count):
        weight = random.randint(2, min(4, n_qubits))
        qubits = random.sample(range(n_qubits), weight)
        paulis = [random.choice(["X", "Y", "Z"]) for _ in range(weight)]

        # Random angle (2k+1)pi/4 with k in {0,1,2,3}
        k = random.randint(0, 3)
        angle = (2 * k + 1) / 4  # in units of pi

        lines.extend(exponentiated_pauli(qubits, paulis, angle))

    lines.append(f"MX {idx_str}")
    return Circuit("\n".join(lines))
