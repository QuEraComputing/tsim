"""Stabilizer flows of circuits with non-Clifford gates.

A flow ``P -> Q xor rec[S]`` says that the Pauli ``P`` before the circuit is
the Pauli ``Q`` after it times the measurements ``S``: every Kraus operator
``K`` of the circuit, with record ``m``, has ``K P = (-1)^(S·m) Q K``. Stim
finds the flows of a Clifford circuit with a tableau (``stim.Circuit.has_flow``,
``has_all_flows`` and ``flow_generators``); this module finds them for any
circuit that Tsim samples, as determined parities (``tsim.determined``) of two
circuits.

- The Choi circuit runs the circuit on halves of Bell pairs. A Pauli on the
  other halves, the references, then acts as its transpose before the circuit.
  So the flow holds, up to sign, when the parity of ``S`` and a measurement of
  ``P`` on the references times ``Q`` on the qubits is determined, and it
  holds with its sign when that parity is 0, or 1 if ``P`` has an odd number
  of ``Y``, which the transpose negates. ``has_all_flows`` measures every flow
  so.
- That takes a known flow. To find the flows, ``flow_generators`` runs the Choi
  circuit twice and measures ``XX`` and ``ZZ`` of each qubit and its copy, a
  Bell measurement [1]. A Pauli stabilizes a state, up to sign, exactly when
  the parity of the ``XX`` outcomes of its ``X`` parts and the ``ZZ`` outcomes
  of its ``Z`` parts is determined. A flow reads a measurement of the circuit
  as a ``Z``, whose ``ZZ`` outcome is the XOR of the two copies' records. The
  flows are therefore the determined parities of these outcomes.

The two copies double the qubits that Tsim's statevector holds at once, which
bounds the circuits that ``flow_generators`` takes. ``has_all_flows`` needs
one copy.

[1] A. Montanaro, "Learning stabilizer states by Bell sampling",
    arXiv:1707.04012.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import stim

from tsim.circuit import Circuit
from tsim.determined import _directions, nondeterministic_detectors
from tsim.utils.gf2_span import Span, unpack


def flow_generators(
    circuit: Circuit, *, shots: int | None = None, seed: int | None = None
) -> list[stim.Flow]:
    """Return flows that generate every stabilizer flow of the circuit.

    As ``stim.Circuit.flow_generators``, for any circuit: every flow of the
    circuit is a product of the returned ones, which are independent. They are
    Stim's generators if the circuit is Clifford. A non-Clifford gate leaves
    fewer: ``T 0`` has ``Z -> Z`` alone, where ``S 0`` also has ``X -> Y``.

    Args:
        circuit: The circuit. Its noise and annotations are ignored.
        shots: If given, find the flows from this many noiseless shots instead
            of exactly, as ``count_determined_measurements`` does. Too few
            shots return flows that do not hold.
        seed: The seed of the shots, if any.

    Returns:
        The generators, in Stim's order.

    Raises:
        ValueError: If the exact walk follows more than
            ``tsim.statevector.branches.MAX_BRANCHES`` branches, which
            sampling avoids.

    """
    n = circuit.num_qubits
    choi = _choi(circuit, n)
    num_m = choi.num_measurements
    twice = choi.copy()
    twice.append("SWAP", [t for q in range(2 * n) for t in (q, q + 2 * n)])
    twice += choi
    # NOTE: inputs before outputs and X before Z, the order of Stim's basis.
    for q in (*range(n, 2 * n), *range(n)):
        copy = q + 2 * n
        twice.append_from_stim_program_text(f"MPP X{q}*X{copy} Z{q}*Z{copy}")

    directions = _directions(Circuit(twice), False, shots, seed)
    bits = unpack(directions.basis(), twice.num_measurements)
    first, second, bell = np.split(bits, [num_m, 2 * num_m], axis=1)
    outcomes = Span(4 * n + num_m)
    # NOTE: the annihilator's rows end at distinct columns. Reversed, they
    # start at distinct ones, as the rows of Stim's basis do.
    outcomes.insert_bits(np.concatenate([bell, first ^ second], axis=1)[:, ::-1])
    rows = unpack(outcomes.annihilator(), outcomes.width)[:, ::-1].astype(bool)

    unsigned = []
    for row in rows:
        paulis = stim.PauliString.from_numpy(
            xs=row[0 : 4 * n : 2], zs=row[1 : 4 * n : 2]
        )
        unsigned.append(
            stim.Flow(
                input=_stim_pauli(paulis[:n]),
                output=_stim_pauli(paulis[n:]),
                measurements=np.flatnonzero(row[4 * n :]).tolist(),
            )
        )
    probe, flipped = _probe(circuit, unsigned)
    negative = probe.compile_detector_sampler().sample(1)[0] ^ flipped
    minus = stim.Flow("1 -> -1")
    flows = [
        flow * minus if sign else flow
        for flow, sign in zip(unsigned, negative, strict=True)
    ]
    return sorted(flows, key=_stim_order)


def has_flow(
    circuit: Circuit,
    flow: stim.Flow,
    *,
    unsigned: bool = False,
    shots: int | None = None,
    seed: int | None = None,
) -> bool:
    """Return whether the circuit has the stabilizer flow.

    As ``stim.Circuit.has_flow``, for any circuit; see ``has_all_flows``.
    """
    return has_all_flows(circuit, [flow], unsigned=unsigned, shots=shots, seed=seed)


def has_all_flows(
    circuit: Circuit,
    flows: Iterable[stim.Flow],
    *,
    unsigned: bool = False,
    shots: int | None = None,
    seed: int | None = None,
) -> bool:
    """Return whether the circuit has all the stabilizer flows.

    As ``stim.Circuit.has_all_flows``, for any circuit. Unlike Stim, the
    result is exact with signs too, unless ``shots`` is given.

    Args:
        circuit: The circuit. Its noise and annotations are ignored.
        flows: The flows. They may name qubits past the circuit's.
        unsigned: If True, a flow may hold with the opposite sign.
        shots: If given, decide from this many noiseless shots instead of
            exactly, as ``count_determined_measurements`` does. Then False is
            certain, but True may rarely be wrong.
        seed: The seed of the shots, if any.

    Returns:
        Whether every flow holds.

    Raises:
        ValueError: If a flow includes an observable or a measurement that the
            circuit lacks, or if the exact walk follows more than
            ``tsim.statevector.branches.MAX_BRANCHES`` branches, which
            sampling avoids.

    """
    probe, values = _probe(circuit, list(flows))
    if nondeterministic_detectors(probe, shots=shots, seed=seed):
        return False
    return unsigned or np.array_equal(
        probe.compile_detector_sampler().sample(1)[0], values
    )


def _choi(circuit: Circuit, n: int) -> stim.Circuit:
    """Return the noiseless circuit on halves of Bell pairs of ``n`` qubits.

    The other half of qubit ``q`` is the reference qubit ``q + n``.
    """
    choi = stim.Circuit()
    choi.append("H", range(n, 2 * n))
    choi.append("CX", [t for q in range(n) for t in (q + n, q)])
    return choi + circuit.without_noise().without_annotations().stim_circuit


def _probe(circuit: Circuit, flows: Sequence[stim.Flow]) -> tuple[Circuit, np.ndarray]:
    """Return the Choi circuit with a detector for each flow, and their values.

    A flow holds up to sign when its detector is determined, and with its sign
    when the detector also has the returned value.
    """
    sizes = [len(p) for flow in flows for p in (flow.input_copy(), flow.output_copy())]
    n = max([circuit.num_qubits, *sizes])
    probe = _choi(circuit, n)
    num_m = probe.num_measurements
    values = []
    for flow in flows:
        if flow.included_observables_copy():
            raise ValueError(f"flows of observables are not supported: {flow}")
        before, after = flow.input_copy(), flow.output_copy()
        product = after + stim.PauliString(n - len(after)) + before
        records = []
        if product.weight:
            probe.append("MPP", [product])
            records.append(-1)
        for m in flow.measurements_copy():
            if not -num_m <= m < num_m:
                raise ValueError(
                    f"the circuit has {num_m} measurements, so none is rec[{m}]: {flow}"
                )
            records.append(m % num_m - probe.num_measurements)
        probe.append("DETECTOR", [stim.target_rec(r) for r in records])
        values.append((product.sign == -1) ^ (len(before.pauli_indices("Y")) % 2 == 1))
    return Circuit(probe), np.array(values, dtype=np.bool_)


def _stim_pauli(pauli: stim.PauliString) -> stim.PauliString:
    """Return the Pauli as Stim's generators write it: an identity has no qubits."""
    return pauli if pauli.weight else stim.PauliString(0)


def _stim_order(flow: stim.Flow) -> tuple[list[int], list[int], bool, list[int]]:
    """Return a key that sorts flows as Stim sorts its generators."""

    def paulis(string: stim.PauliString) -> list[int]:
        return [string[q] for q in range(len(string))]

    before, after = flow.input_copy(), flow.output_copy()
    return paulis(before), paulis(after), after.sign == -1, flow.measurements_copy()
