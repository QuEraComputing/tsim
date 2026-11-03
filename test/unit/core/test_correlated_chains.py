"""Encoding of E/ELSE_CORRELATED_ERROR chains as noise channels."""

import itertools

import numpy as np
import pytest
import stim
from numpy.testing import assert_allclose

from tsim.circuit import Circuit
from tsim.core.builder import MAX_CORRELATED_ERROR_BITS, GraphBuilder
from tsim.core.parse import parse_stim_circuit
from tsim.noise.channels import correlated_error_probs


def _average_channel(circuit: Circuit) -> np.ndarray:
    """Return the circuit's channel, averaged over all its error outcomes."""
    builder = parse_stim_circuit(circuit.stim_circuit)
    graph = builder.build(closed=False, outputs="none")
    supports = [
        [(outcome, p) for outcome, p in enumerate(probs) if p > 0]
        for probs in builder.channel_probs
    ]
    first_bits = np.cumsum(
        [0] + [len(probs).bit_length() - 1 for probs in builder.channel_probs]
    )
    tensors = []
    for outcomes in itertools.product(*supports):
        weight = 1.0
        bindings = {}
        for first, num_bits, (outcome, p) in zip(
            first_bits[:-1], np.diff(first_bits), outcomes, strict=True
        ):
            weight *= p
            for b in range(num_bits):
                bindings[int(first + b)] = (outcome >> b) & 1
        tensors.append(weight * np.asarray(graph.to_tensor(bindings)))
    return np.sum(tensors, axis=0)


def _reference_channel(circuit: Circuit) -> np.ndarray:
    """Return the circuit's channel, firing the links of its chains one at a time.

    Each outcome of each chain (no link, or link ``j``) is a circuit in which the
    fired link is a layer of Pauli gates at its own position, so no chain
    encoding is involved; its other noise is averaged as in ``_average_channel``.
    """
    chains: list[list[tuple[float, stim.CircuitInstruction]]] = []
    program: list[stim.CircuitInstruction | tuple[int, int]] = []
    for instruction in circuit.stim_circuit:
        assert isinstance(instruction, stim.CircuitInstruction)
        if instruction.name == "E":
            chains.append([])
        if instruction.name in ("E", "ELSE_CORRELATED_ERROR"):
            program.append((len(chains) - 1, len(chains[-1])))
            chains[-1].append((instruction.gate_args_copy()[0], instruction))
        else:
            program.append(instruction)

    def outcomes(chain):
        none = 1.0
        for j, (p, _) in enumerate(chain):
            yield j, none * p
            none *= 1 - p
        yield None, none

    tensors = []
    for fired in itertools.product(*(list(outcomes(chain)) for chain in chains)):
        weight = float(np.prod([p for _, p in fired]))
        fired_circuit = stim.Circuit()
        for entry in program:
            if isinstance(entry, stim.CircuitInstruction):
                fired_circuit.append(entry)
            elif fired[entry[0]][0] == entry[1]:
                for target in chains[entry[0]][entry[1]][1].targets_copy():
                    fired_circuit.append(
                        stim.CircuitInstruction(target.pauli_type, [target.value])
                    )
        fired_tensor = _average_channel(Circuit.from_stim_program(fired_circuit))
        tensors.append(weight * fired_tensor)
    return np.sum(tensors, axis=0)


CHAINS = {
    "repeated_paulis": """
        E(0.1) X0
        ELSE_CORRELATED_ERROR(0.2) X0
        ELSE_CORRELATED_ERROR(0.3) Y0
        ELSE_CORRELATED_ERROR(0.15) Z0
    """,
    "x_and_z_interleaved": """
        E(0.1) X0
        ELSE_CORRELATED_ERROR(0.2) Z0
        ELSE_CORRELATED_ERROR(0.3) X0 Z0
        ELSE_CORRELATED_ERROR(0.25) Y0
    """,
    "h_between_links": """
        E(0.1) X0
        H 0
        ELSE_CORRELATED_ERROR(0.2) X0
        ELSE_CORRELATED_ERROR(0.3) Z0
        ELSE_CORRELATED_ERROR(0.25) X0
    """,
    "h_between_x_and_z": """
        E(0.1) X0
        H 0
        ELSE_CORRELATED_ERROR(0.2) Z0
        ELSE_CORRELATED_ERROR(0.3) X0
        ELSE_CORRELATED_ERROR(0.25) Z0
    """,
    "h_h_between_links": """
        E(0.1) X0
        H 0
        H 0
        ELSE_CORRELATED_ERROR(0.2) X0
        ELSE_CORRELATED_ERROR(0.3) Y0
    """,
    "swap_and_back": """
        E(0.1) X0 Z1
        SWAP 0 1
        ELSE_CORRELATED_ERROR(0.2) Z0
        SWAP 0 1
        ELSE_CORRELATED_ERROR(0.3) X0 Z1
        ELSE_CORRELATED_ERROR(0.25) Z0
    """,
    "swap_h_swap": """
        E(0.1) X0
        SWAP 0 1
        H 1
        SWAP 0 1
        ELSE_CORRELATED_ERROR(0.2) X0
        ELSE_CORRELATED_ERROR(0.3) X0
    """,
    "gates_between_links": """
        H 0
        CX 0 1
        E(0.1) X0 Z1
        S 0
        ELSE_CORRELATED_ERROR(0.2) Y0 Y1
        SQRT_X 1
        ELSE_CORRELATED_ERROR(0.15) Z1
        T 1
        ELSE_CORRELATED_ERROR(0.25) X0 Z1
        CZ 0 1
        ELSE_CORRELATED_ERROR(0.2) X0 Z1
        ELSE_CORRELATED_ERROR(0.1) Y0
    """,
    "measure_and_reset_between_links": """
        H 0
        E(0.1) Z0
        M 0
        ELSE_CORRELATED_ERROR(0.2) Z0
        ELSE_CORRELATED_ERROR(0.3) X0
        R 0
        ELSE_CORRELATED_ERROR(0.25) X0
        ELSE_CORRELATED_ERROR(0.2) X0
    """,
    "other_noise_between_links": """
        E(0.1) X0
        X_ERROR(0.2) 0
        ELSE_CORRELATED_ERROR(0.2) X0
        ELSE_CORRELATED_ERROR(0.3) X0
    """,
    "empty_and_cancelling_links": """
        E(0.2) X0 X0
        ELSE_CORRELATED_ERROR(0.3)
        ELSE_CORRELATED_ERROR(0.4) Y0 Z0
        ELSE_CORRELATED_ERROR(0.1) X0 Z1
    """,
    "consecutive_chains_on_one_qubit": """
        E(0.3) X0
        E(0.4) X0
        ELSE_CORRELATED_ERROR(0.2) X0
    """,
    "disjoint_components": """
        H 0
        E(0.1) X0 Z1
        ELSE_CORRELATED_ERROR(0.2) Y2
        ELSE_CORRELATED_ERROR(0.3) X1
    """,
}


@pytest.mark.parametrize("body", list(CHAINS.values()), ids=list(CHAINS))
def test_chain_channel_matches_firing_one_link_at_a_time(body: str):
    circuit = Circuit("I 0 1 2\n" + body)
    assert_allclose(_average_channel(circuit), _reference_channel(circuit), atol=1e-12)


def _three_qubit_pauli_channel(probability) -> str:
    paulis = ["".join(p) for p in itertools.product("IXYZ", repeat=3)][1:]
    lines = []
    for j, pauli in enumerate(paulis):
        targets = " ".join(f"{c}{q}" for q, c in enumerate(pauli) if c != "I")
        name = "E" if j == 0 else "ELSE_CORRELATED_ERROR"
        lines.append(f"{name}({probability(j)}) {targets}")
    return "\n".join(lines)


def test_three_qubit_pauli_channel_is_one_six_bit_channel():
    circuit = Circuit(
        "I 0 1 2\nH 0\nT 0\nCX 0 1\n"
        + _three_qubit_pauli_channel(lambda j: 0.002 + j / 1000)
    )
    builder = parse_stim_circuit(circuit.stim_circuit)
    assert builder.num_error_bits == 6
    assert [probs.shape for probs in builder.channel_probs] == [(64,)]
    assert_allclose(builder.channel_probs[0].sum(), 1.0, atol=1e-12)
    assert_allclose(_average_channel(circuit), _reference_channel(circuit), atol=1e-12)


def test_repeated_paulis_add_up():
    builder = parse_stim_circuit(
        stim.Circuit(
            "E(0.1) X0\nELSE_CORRELATED_ERROR(0.2) Z0\nELSE_CORRELATED_ERROR(0.5) X0"
        )
    )
    # NOTE: bit 0 is the X spider and bit 1 the Z spider, in creation order.
    assert_allclose(builder.channel_probs[0], [0.36, 0.1 + 0.36, 0.18, 0.0])


def test_disjoint_components_keep_one_bit_per_link():
    builder = parse_stim_circuit(
        stim.Circuit(
            "E(0.1) X0 Z1\nELSE_CORRELATED_ERROR(0.2) Y2\nELSE_CORRELATED_ERROR(0.3) X1"
        )
    )
    assert len(builder.channel_probs) == 1
    assert_allclose(builder.channel_probs[0], correlated_error_probs([0.1, 0.2, 0.3]))
    g = builder.graph
    links = sorted(
        (g.row(v), g.vertex_type(v).name, sorted(g.phase(v).parameters))
        for v in g.vertices()
        if g.phase(v).parameters
    )
    assert links == [
        (0, "X", [0]),
        (1, "X", [2]),
        (1, "Z", [0]),
        (2, "X", [1]),
        (2, "Z", [1]),
    ]


def test_chain_that_flips_nothing_has_no_channel():
    builder = parse_stim_circuit(
        stim.Circuit("E(0.1) X0 X0\nELSE_CORRELATED_ERROR(0.2)")
    )
    assert builder.channel_probs == []
    assert builder.num_error_bits == 0


@pytest.mark.parametrize(
    "text, num_bits",
    [
        # NOTE: SWAP moves the wire, and its spider, to qubit 1.
        ("E(0.1) X0\nSWAP 0 1\nELSE_CORRELATED_ERROR(0.2) X1", 1),
        # NOTE: the X spider of the first link is dead and gets no bit.
        (
            "E(0.1) X0 X0\nELSE_CORRELATED_ERROR(0.2) X1\nELSE_CORRELATED_ERROR(0.3) X1",
            1,
        ),
        (
            "E(0.1) X0 Z0\nELSE_CORRELATED_ERROR(0.2) Z0\n"
            "ELSE_CORRELATED_ERROR(0.3) X0\nELSE_CORRELATED_ERROR(0.4) Y0",
            2,
        ),
    ],
    ids=["across_swap", "after_dead_spider", "x_and_z"],
)
def test_links_share_spiders(text: str, num_bits: int):
    builder = parse_stim_circuit(stim.Circuit(text))
    assert builder.num_error_bits == num_bits


def test_wide_link_has_one_bit():
    targets = " ".join(f"X{q}" for q in range(MAX_CORRELATED_ERROR_BITS + 5))
    builder = parse_stim_circuit(stim.Circuit(f"E(0.1) {targets}"))
    assert builder.num_error_bits == 1


@pytest.mark.parametrize(
    "links",
    [
        # NOTE: one bit per link, for more links than the limit.
        [f"X{q}" for q in range(MAX_CORRELATED_ERROR_BITS + 1)],
        # NOTE: one bit per component, for more components than the limit.
        [f"X{q}" for q in range(MAX_CORRELATED_ERROR_BITS + 1)] + ["X0"],
    ],
)
def test_too_many_chain_bits_raise_and_drop_the_chain(links: list[str]):
    builder = GraphBuilder()
    for link in links:
        builder.correlated_error([int(link[1:])], ["X"], 0.01)
    bits = MAX_CORRELATED_ERROR_BITS + 1
    with pytest.raises(ValueError, match=f"needs 2\\^{bits} outcomes"):
        builder.finalize_correlated_error()
    assert builder.num_error_bits == 0
    builder.correlated_error([0], ["Z"], 0.1)
    builder.finalize_correlated_error()
    assert builder.num_error_bits == 1
