r"""Compute exact expectation values for Pauli strings using ZX-calculus.

This module provides functionality to compute Pauli expectation values ⟨ψ|P|ψ⟩
for quantum circuits without measurements or noise, where |ψ⟩ = U|0⟩ and P is
a Pauli string.

The implementation uses ZX-calculus to construct a diagram representing
⟨0|U† P U|0⟩, where U is a quantum circuit without measurements or noise,
which is then compiled and evaluated efficiently for various Pauli strings P.

Example:
    >>> from tsim import Circuit
    >>> from tsim.paulistrings import PauliStrings
    >>> import numpy as np
    >>>
    >>> # Create a Bell state circuit
    >>> circuit = Circuit("H 0\\nCNOT 0 1")
    >>> ps = PauliStrings(circuit)
    >>>
    >>> # Evaluate ZZ expectation (should be +1 for Bell state)
    >>> # Format: [x0, x1, z0, z1]
    >>> paulis = np.array([[0, 0, 1, 1]])  # ZZ
    >>> result = ps.evaluate(paulis)

"""

import jax.numpy as jnp
import numpy as np
import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType

from tsim.circuit import Circuit
from tsim.compile.compile import compile_scalar_graphs
from tsim.compile.evaluate import evaluate_batch
from tsim.compile.stabrank import find_stab
from tsim.core.parse import parse_stim_circuit

_UNSUPPORTED_INSTRUCTIONS = frozenset(
    [
        "M",
        "MX",
        "MY",
        "MZ",
        "MPP",
        "MR",
        "MRX",
        "MRY",
        "MRZ",
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
    ]
)


def _get_graph(circuit: Circuit) -> BaseGraph:
    """Convert a circuit to a ZX graph with |0⟩ input states.

    Parses the circuit into a ZX graph representation and initializes
    all non-initialized input qubits to the |0⟩ state (X spider with no phase).

    Args:
        circuit: The quantum circuit to convert.

    Returns:
        A PyZX graph representing the circuit with normalized vertex positions.

    """
    built = parse_stim_circuit(circuit._stim_circ)
    g = built.graph.copy()
    # Initialize un-initialized first vertices to the 0 state
    for v in built.first_vertex.values():
        if g.type(v) == VertexType.BOUNDARY:
            g.set_type(v, VertexType.X)

    # Clean up last row
    if built.last_vertex:
        max_row = max(g.row(v) for v in built.last_vertex.values())
        for q in built.last_vertex:
            g.set_row(built.last_vertex[q], max_row)

    g.normalize()
    return g


class PauliStrings:
    r"""Compute exact Pauli expectation values for a quantum circuit.

    This class compiles a circuit into an efficient representation for
    evaluating expectation values ⟨ψ|P|ψ⟩ where |ψ⟩ = U|0⟩ and P is any
    Pauli string (tensor product of Pauli operators).


    Attributes:
        compiled: The compiled scalar graphs for evaluation.
        num_qubits: Number of qubits in the circuit.

    Example:
        >>> circuit = Circuit("H 0\\nCNOT 0 1")  # Bell state
        >>> ps = PauliStrings(circuit)
        >>>
        >>> # Pauli format: [x0, x1, ..., xn, z0, z1, ..., zn]
        >>> paulis = np.array([
        ...     [0, 0, 0, 0],  # II (identity)
        ...     [0, 0, 1, 1],  # ZZ
        ...     [1, 1, 0, 0],  # XX
        ...     [1, 1, 1, 1],  # YY
        ... ])
        >>> values = ps.evaluate(paulis)  # Returns [1.0, 1.0, 1.0, -1.0]

    """

    def __init__(self, circuit: Circuit):
        """Compile a circuit for Pauli expectation value evaluation.

        Args:
            circuit: A quantum circuit without measurements or noise channels.
                Must not contain M, MX, MY, MZ, MPP, MR, MRX, MRY, MRZ,
                DETECTOR, OBSERVABLE_INCLUDE, or any error instructions.

        Raises:
            ValueError: If the circuit contains measurements or noise instructions.

        """
        for instruction in circuit._stim_circ:
            if instruction.name in _UNSUPPORTED_INSTRUCTIONS:
                raise ValueError(
                    f"Circuit must not contain {instruction.name} instructions."
                )

        n = circuit.num_qubits
        qubits = " ".join(map(str, range(n)))

        circuit_with_paulis = circuit.copy()
        circuit_with_paulis.append_from_stim_program_text(f"X_ERROR(1) {qubits}")
        circuit_with_paulis.append_from_stim_program_text(f"Z_ERROR(1) {qubits}")

        g = _get_graph(circuit_with_paulis)
        g.compose(_get_graph(circuit).adjoint())
        zx.full_reduce(g, paramSafe=True)
        g.scalar.power2 -= 2 * circuit.num_qubits

        g_list = find_stab(g)
        self.compiled = compile_scalar_graphs(g_list, [f"e{i}" for i in range(2 * n)])
        self.num_qubits = n

    def evaluate(self, paulis: np.ndarray) -> np.ndarray:
        """Evaluate expectation values for a batch of Pauli strings.

        Computes ⟨ψ|P|ψ⟩ for each Pauli string P in the input batch,
        where |ψ⟩ is the state prepared by the circuit.

        Args:
            paulis: Array of shape (num_paulis, 2 * num_qubits) specifying
                Pauli strings in binary symplectic form [x | z], where:
                - First num_qubits columns are X bits (1 = X operator on that qubit)
                - Last num_qubits columns are Z bits (1 = Z operator on that qubit)
                - Both bits set (x=1, z=1) represents Y on that qubit

                Examples for 2 qubits [x0, x1, z0, z1]:
                - [0, 0, 0, 0] = I ⊗ I (identity)
                - [0, 0, 1, 0] = Z ⊗ I
                - [1, 0, 0, 0] = X ⊗ I
                - [1, 0, 1, 0] = Y ⊗ I
                - [0, 0, 1, 1] = Z ⊗ Z
                - [1, 1, 0, 0] = X ⊗ X
                - [1, 1, 1, 1] = Y ⊗ Y

        Returns:
            Array of shape (num_paulis,) containing the real expectation values.

        Raises:
            ValueError: If paulis has incorrect number of columns.

        """
        if not paulis.shape[1] == 2 * self.num_qubits:
            raise ValueError(
                f"Pauli strings must be on {self.num_qubits} qubits. "
                f"Expected {2 * self.num_qubits} columns, got {paulis.shape[1]}."
            )

        n = self.num_qubits
        x_part = paulis[:, :n]
        z_part = paulis[:, n:]

        # Count Y operators (where both x and z bits are set) per Pauli string
        # XZ = iY, so we need to multiply by i for each Y to get correct result
        y_counts = np.sum(x_part & z_part, axis=1)

        pp = jnp.vstack([np.zeros((1, 2 * self.num_qubits), dtype=np.uint8), paulis])
        # first row of pp is all zeros to compute the normalization factor

        res = np.array(evaluate_batch(self.compiled, pp))
        normalized = res[1:] / res[0]

        # Apply (-i)^y_count correction for Y operators
        correction = (-1j) ** y_counts
        return np.real(normalized * correction)
