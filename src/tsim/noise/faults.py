"""Exact search for the smallest sets of circuit errors that cause a logical error.

Tsim compiles a noisy circuit to error-basis bits ``f = T e`` and a program over
them. Nothing else about the noise reaches the outputs, so circuit errors with
the same f-bits are one fault, as error mechanisms with the same symptoms are
in a detector error model. Each direct detector is one f-bit, which a set of
faults flips or does not; the other detectors and the observables come out of
the statevector components, where ``Branches`` gives a set's exact chance to
flip no detector and to flip an observable.

``shortest_errors`` tries every set of ``w`` faults for ``w = 1, 2, …``. It
finds the sets whose direct detectors cancel by joining two tables of smaller
sets on a hash of those detectors, and walks the tapes for these alone.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING

import numpy as np
import stim

from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.core.parse import parse_stim_circuit
from tsim.noise.channels import channel_num_bits
from tsim.statevector.branches import Branches, unique_rows
from tsim.statevector.program import StatevectorComponent

if TYPE_CHECKING:
    from tsim.circuit import Circuit

MIN_PROBABILITY = 1e-9
"""Smallest chance to flip an observable undetected that counts as a logical error."""

MAX_COMBINATIONS = 30_000_000
"""Most sets of faults that one side of the join may hold."""

_BATCH = 1 << 20
_HASH_BITS = 40
_INDEX_BITS = 24
_PAULI = {1: stim.target_z, 2: stim.target_x, 3: stim.target_y}
_MEASURED = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}
_MEASURED_PAIR = {"MXX": "X", "MYY": "Y", "MZZ": "Z"}


@dataclass(frozen=True)
class Fault:
    """One way for a circuit to go wrong: circuit errors with the same effect.

    Attributes:
        circuit_error_locations: The circuit errors that are this fault, as in
            ``stim.ExplainedError``. Their instruction offsets index
            ``circuit.flattened()``.
        probability: The sum of their probabilities.

    """

    circuit_error_locations: tuple[stim.CircuitErrorLocation, ...]
    probability: float


@dataclass(frozen=True)
class LogicalError:
    """Faults that together can flip an observable without flipping a detector.

    ``len(error)`` is the number of faults and iterating yields them, as for the
    list that ``stim.Circuit.shortest_graphlike_error`` returns.

    Attributes:
        faults: The faults.
        undetected_probability: The probability that no detector flips when
            exactly these faults happen.
        logical_flip_probability: The probability that no detector flips and an
            observable does.

    """

    faults: tuple[Fault, ...]
    undetected_probability: float
    logical_flip_probability: float

    def __len__(self) -> int:
        """Return the number of faults."""
        return len(self.faults)

    def __iter__(self) -> Iterator[Fault]:
        """Iterate over the faults."""
        return iter(self.faults)


def shortest_errors(circuit: Circuit, *, max_weight: int) -> list[LogicalError]:
    """Return every smallest set of faults that causes a logical error.

    A fault is a circuit error, such as one Pauli of a ``DEPOLARIZE2``, together
    with the circuit errors that act the same. A set of faults is a logical
    error if it can flip an observable while every detector keeps its noiseless
    value. With non-Clifford gates that happens with a probability, which the
    result reports; in a Clifford circuit it is 0 or 1.

    The search is exhaustive, so the number of faults in the result is the
    circuit's fault distance, and an empty result proves that the distance is
    above ``max_weight``. Its cost grows as (number of faults)^(max_weight / 2)
    and with the number of measurement branches of the circuit's statevector
    components.

    Args:
        circuit: The circuit, with its noise, detectors and observables.
        max_weight: The largest number of faults to try.

    Returns:
        The logical errors with the fewest faults, most likely first, or an
        empty list if up to ``max_weight`` faults cause none.

    Raises:
        ValueError: If a detector or an observable is random without noise, or
            the search would need more than ``MAX_COMBINATIONS`` sets of faults
            at once.

    """
    if max_weight < 1:
        raise ValueError(f"max_weight must be at least 1, not {max_weight}")
    if circuit.num_observables == 0:
        return []
    search = _Search(circuit)
    for weight in range(1, max_weight + 1):
        found = search.logical_errors(weight)
        if found:
            return found
    return []


class _Search:
    """A circuit's faults and the program that decides what they do."""

    def __init__(self, circuit: Circuit) -> None:
        prepared = prepare_graph(circuit, sample_detectors=True)
        program = compile_program(
            prepared, mode="sequential", strategy="active_statevector"
        )
        num_f = prepared.error_transform.shape[0]
        self._stim_circuit = circuit._stim_circ
        self._locations: _Locations | None = None

        # NOTE: f-vectors are Python ints here: bit j is f-bit j.
        bit_f = [
            sum(1 << row for row in column)
            for column in prepared.error_transform.columns()
        ]
        groups: dict[int, list[tuple[int, int, float]]] = {}
        first = 0
        for channel, probs in enumerate(prepared.channel_probs):
            num_bits = channel_num_bits(probs)
            for outcome in (np.flatnonzero(probs[1:]) + 1).tolist():
                f = 0
                for bit in range(num_bits):
                    if outcome >> bit & 1:
                        f ^= bit_f[first + bit]
                if f:
                    event = (channel, outcome, float(probs[outcome]))
                    groups.setdefault(f, []).append(event)
            first += num_bits
        self._events = list(groups.values())
        num_bytes = max(1, -(-num_f // 8))
        self._f = np.frombuffer(
            b"".join(f.to_bytes(num_bytes, "little") for f in groups), dtype=np.uint8
        ).reshape(len(groups), num_bytes)
        if len(groups) >= 1 << _INDEX_BITS:
            raise ValueError(
                f"the circuit has {len(groups)} faults, too many to search"
            )

        num_direct = len(program.direct_f_indices)
        direct = program.direct_f_indices
        is_detector = program.output_order[:num_direct] < prepared.num_detectors
        detector_bits = np.unique(direct[is_detector & (direct >= 0)])
        self._observable_bits = np.unique(direct[~is_detector & (direct >= 0)])
        mask = np.zeros(8 * num_bytes, dtype=np.uint8)
        mask[detector_bits] = 1
        self._detector_mask = np.packbits(mask, bitorder="little")
        self._branches = []
        for component in program.components:
            assert isinstance(component, StatevectorComponent)
            self._branches.append(Branches(component, prepared.num_detectors))
        # NOTE: this raises if a detector or an observable is random without noise.
        self._probabilities(np.zeros((1, num_bytes), dtype=np.uint8))

        # NOTE: a GF(2)-linear hash of the direct detectors that a fault flips,
        # so that a set's hash is the XOR of its faults' hashes.
        bits = np.unpackbits(self._f, axis=1, bitorder="little")[:, detector_bits]
        rng = np.random.default_rng(0)
        mixer = rng.integers(0, 2, size=(len(detector_bits), _HASH_BITS))
        mixed = (bits.astype(np.float32) @ mixer.astype(np.float32)).astype(np.int64)
        self._hash = (
            ((mixed & 1) << np.arange(_HASH_BITS)).sum(axis=1).astype(np.uint64)
        )

    def logical_errors(self, weight: int) -> list[LogicalError]:
        """Return the logical errors of exactly ``weight`` faults."""
        found = []
        for sets, f in self._undetected_by_direct(weight):
            one, group = unique_rows(f)
            undetected, flip = self._probabilities(f[one])
            for i in np.flatnonzero(flip[group] > MIN_PROBABILITY).tolist():
                faults = [int(j) for j in sets[i]]
                row = group[i]
                # NOTE: two circuit errors of one channel exclude each other.
                if _distinct([{e[0] for e in self._events[j]} for j in faults]):
                    error = LogicalError(
                        tuple(self._fault(j) for j in faults),
                        float(undetected[row]),
                        float(flip[row]),
                    )
                    found.append(error)

        def likelihood(error: LogicalError) -> float:
            product = error.logical_flip_probability
            for fault in error.faults:
                product *= fault.probability
            return product

        return sorted(found, key=likelihood, reverse=True)

    def _probabilities(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the chance that no detector flips, and that an observable does.

        Args:
            f: Packed f-vectors that flip no direct detector, one per row.

        """
        bits = np.unpackbits(f, axis=1, bitorder="little")
        undetected = np.ones(len(f))
        unflipped = np.ones(len(f))
        for branches in self._branches:
            passed, same = branches.probabilities(bits[:, branches.f_selection])
            undetected *= passed
            unflipped *= same
        unflipped[bits[:, self._observable_bits].any(axis=1)] = 0.0
        return undetected, undetected - unflipped

    def _undetected_by_direct(
        self, weight: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield the sets of ``weight`` faults that flip no direct detector.

        Each batch is the sets, shape ``(sets, weight)`` with the faults of a
        set ascending, and their packed f-vectors. A set is the first ``weight // 2`` of its faults joined to
        the others, so both sides have equal hashes.
        """
        n = len(self._events)
        small = weight // 2
        for size in (small, weight - small):
            if comb(n, size) > MAX_COMBINATIONS:
                raise ValueError(
                    f"sets of {weight} out of {n} faults need {comb(n, size)} "
                    f"combinations at once, more than {MAX_COMBINATIONS}; lower "
                    "max_weight or remove noise from the circuit"
                )
        left, right = _combinations(n, small), _combinations(n, weight - small)
        shift = np.uint64(_INDEX_BITS)
        last = left[:, -1].astype(np.uint64) + np.uint64(1) if small else np.uint64(0)
        left_key = self._set_hash(left) << shift
        # NOTE: a key is the hash, then the first fault, so the right sides of a
        # left side are a run of the sorted keys: those of its hash whose first
        # fault comes after its last.
        keys = self._set_hash(right) << shift | right[:, 0].astype(np.uint64)
        order = np.argsort(keys)
        keys = keys[order]
        lo = np.searchsorted(keys, left_key | last)
        hi = np.searchsorted(
            keys, left_key | np.uint64((1 << _INDEX_BITS) - 1), "right"
        )
        counts = hi - lo
        ends = np.cumsum(counts)
        start = 0
        while start < len(left):
            done = ends[start - 1] if start else 0
            stop = max(start + 1, int(np.searchsorted(ends, done + _BATCH, "right")))
            count = counts[start:stop]
            parent = np.repeat(np.arange(start, stop), count)
            offset = np.arange(len(parent)) - np.repeat(np.cumsum(count) - count, count)
            match = order[np.repeat(lo[start:stop], count) + offset]
            sets = np.column_stack([left[parent], right[match]])
            # NOTE: equal hashes do not prove that the direct detectors cancel.
            f = np.bitwise_xor.reduce(self._f[sets], axis=1)
            quiet = ~(f & self._detector_mask).any(axis=1)
            yield sets[quiet], f[quiet]
            start = stop

    def _set_hash(self, sets: np.ndarray) -> np.ndarray:
        """Return the hash of each set of faults."""
        if sets.shape[1] == 0:
            return np.zeros(len(sets), dtype=np.uint64)
        return np.bitwise_xor.reduce(self._hash[sets], axis=1)

    def _fault(self, index: int) -> Fault:
        """Return fault ``index`` with its circuit errors explained."""
        if self._locations is None:
            self._locations = _Locations(self._stim_circuit)
        events = self._events[index]
        return Fault(
            tuple(
                self._locations.of(channel, outcome) for channel, outcome, _ in events
            ),
            sum(p for _, _, p in events),
        )


def _combinations(n: int, size: int) -> np.ndarray:
    """Return the ascending tuples of ``size`` out of ``n`` indices, one per row."""
    sets = np.zeros((1, 0), dtype=np.int32)
    for _ in range(size):
        last = sets[:, -1] if sets.shape[1] else np.full(len(sets), -1, dtype=np.int32)
        count = n - 1 - last
        parent = np.repeat(np.arange(len(sets)), count)
        offset = np.arange(len(parent)) - np.repeat(np.cumsum(count) - count, count)
        nxt = (last[parent] + 1 + offset).astype(np.int32)
        sets = np.column_stack([sets[parent], nxt])
    return sets


def _distinct(choices: Sequence[set[int]]) -> bool:
    """Return whether one item can be taken from each set, all different."""
    if not choices:
        return True
    return any(
        _distinct([other - {item} for other in choices[1:]]) for item in choices[0]
    )


class _Locations:
    """Where each noise channel outcome happens in the flattened circuit."""

    def __init__(self, stim_circuit: stim.Circuit) -> None:
        builder = parse_stim_circuit(stim_circuit)
        self._sites = builder.channel_sites
        self._links = builder.chain_links
        self._flat = stim_circuit.flattened()
        self._coords = self._flat.get_final_qubit_coordinates()
        self._ticks = np.cumsum([op.name == "TICK" for op in self._flat])

    def _target(self, target: stim.GateTarget) -> stim.GateTargetWithCoords:
        coords = [] if target.is_combiner else self._coords.get(target.value, [])
        return stim.GateTargetWithCoords(target, coords)

    def of(self, channel: int, outcome: int) -> stim.CircuitErrorLocation:
        """Return the circuit error of ``outcome`` of noise channel ``channel``."""
        site = self._sites[channel]
        for mask, link in self._links.get(channel, ()):
            if mask == outcome:
                site = link
        op = self._flat[site.instruction]
        assert isinstance(op, stim.CircuitInstruction)
        name = op.name
        targets = op.targets_copy()[site.start : site.end]
        qubits = [t.value for t in targets if not t.is_combiner]

        def paulis(codes: Sequence[int]) -> list[stim.GateTarget]:
            pairs = zip(codes, qubits, strict=True)
            return [_PAULI[code](qubit) for code, qubit in pairs if code]

        product: list[stim.GateTarget] = []
        measured: list[stim.GateTarget] | None = None
        if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            product = [stim.target_pauli(qubits[0], name[0])]
        elif name in ("DEPOLARIZE1", "PAULI_CHANNEL_1"):
            product = paulis([outcome])
        elif name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
            product = paulis([outcome & 3, outcome >> 2])
        elif name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
            product, measured = paulis([outcome >> 1]), []
        elif name in ("E", "ELSE_CORRELATED_ERROR"):
            product = targets
        elif name in _MEASURED:
            measured = [stim.target_pauli(qubits[0], _MEASURED[name])]
        elif name in _MEASURED_PAIR:
            measured = [stim.target_pauli(q, _MEASURED_PAIR[name]) for q in qubits]
        elif name == "MPP":
            measured = [
                stim.target_pauli(t.value, t.pauli_type)
                for t in targets
                if not t.is_combiner
            ]
        elif name == "MPAD":
            measured = []
        else:
            raise NotImplementedError(f"no explanation for noise of {name}")
        # NOTE: Stim reads a FlippedMeasurement without a record as no flip. Its
        # type stub names the keywords differently from the module.
        flip: dict = {
            "record_index": None if measured is None else site.record,
            "observable": [self._target(t) for t in measured or ()],
        }
        flipped = stim.FlippedMeasurement(**flip)
        return stim.CircuitErrorLocation(
            tick_offset=int(self._ticks[site.instruction]),
            flipped_pauli_product=[self._target(t) for t in product],
            flipped_measurement=flipped,
            instruction_targets=stim.CircuitTargetsInsideInstruction(
                gate=name,
                tag=op.tag,
                args=op.gate_args_copy(),
                target_range_start=site.start,
                target_range_end=site.end,
                targets_in_range=[self._target(t) for t in targets],
            ),
            stack_frames=[
                stim.CircuitErrorLocationStackFrame(
                    instruction_offset=site.instruction,
                    iteration_index=0,
                    instruction_repetitions_arg=0,
                )
            ],
            noise_tag=op.tag,
        )
