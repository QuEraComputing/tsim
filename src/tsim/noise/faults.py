"""Exact search for the smallest sets of circuit errors that cause a logical error.

Tsim compiles a noisy circuit to error-basis bits ``f = T e`` and a program over
them. Nothing else about the noise reaches the outputs, so circuit errors with
the same f-bits are one fault, as error mechanisms with the same symptoms are
in a detector error model.

A set of faults acts through three GF(2)-linear images of its f-bits: the
direct detectors it flips, the rotation signs of the statevector components it
flips, and the flips of their outputs and of the direct observables. Only the
rotation signs act nonlinearly, and few distinct ones occur, so ``Branches``
walks each component once per distinct signs for the exact distribution of its
outputs; the output flips then just shift that distribution.

``shortest_errors`` tries every set of ``w`` faults for ``w = 1, 2, …``. A
numba join (``fault_join``) streams the sets whose direct detectors cancel, a
stored half against a streamed half on a hash of those detectors: a first pass
collects their distinct rotation signs, a second keeps the sets whose signs and
output flips can flip an observable undetected.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING

import numba
import numpy as np
import stim

from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.core.parse import parse_stim_circuit
from tsim.noise import fault_join
from tsim.noise.channels import channel_num_bits
from tsim.statevector.branches import Branches, gf2_product
from tsim.statevector.program import StatevectorComponent

if TYPE_CHECKING:
    from tsim.circuit import Circuit

MIN_PROBABILITY = 1e-9
"""Smallest chance to flip an observable undetected that counts as a logical error."""

MAX_COMBINATIONS = 30_000_000
"""Most sets of faults that the stored side of the join may hold."""

MAX_SCANNED = 100_000_000_000
"""Most sets of faults that the streamed side of the join may go through."""

FIRST_CAPACITY = 1 << 10
"""Sets that each lane of the join can hold before its buffers grow."""

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


def shortest_errors(
    circuit: Circuit, *, max_weight: int, max_errors: int | None = None
) -> list[LogicalError]:
    """Return every smallest set of faults that causes a logical error.

    A fault is a circuit error, such as one Pauli of a ``DEPOLARIZE2``, together
    with the circuit errors that act the same. A set of faults is a logical
    error if it can flip an observable while every detector keeps its noiseless
    value. With non-Clifford gates that happens with a probability, which the
    result reports; in a Clifford circuit it is 0 or 1.

    The search is exhaustive, so the number of faults in the result is the
    circuit's fault distance, and an empty result proves that the distance is
    above ``max_weight``. It stores the sets of ``max_weight // 2`` faults and
    streams those of the other faults, so its time grows as (number of
    faults)^(max_weight - max_weight // 2). It supports up to 6 faults.

    Args:
        circuit: The circuit, with its noise, detectors and observables.
        max_weight: The largest number of faults to try.
        max_errors: If given, return only this many of the logical errors, the
            most likely ones.

    Returns:
        The logical errors with the fewest faults, most likely first, or an
        empty list if up to ``max_weight`` faults cause none. A logical error's
        likelihood is its ``logical_flip_probability`` times the probabilities
        of its faults.

    Raises:
        ValueError: If a detector or an observable is random without noise,
            ``max_weight`` is above 6, or the search would need more than
            ``MAX_COMBINATIONS`` stored or ``MAX_SCANNED`` streamed sets of
            faults.

    """
    if max_weight < 1:
        raise ValueError(f"max_weight must be at least 1, not {max_weight}")
    if circuit.num_observables == 0:
        return []
    search = _Search(circuit)
    for weight in range(1, max_weight + 1):
        found = search.logical_errors(weight, max_errors=max_errors)
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
        self._explained: dict[int, Fault] = {}

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
        self._fault_probability = np.array(
            [sum(p for _, _, p in events) for events in self._events]
        )
        # NOTE: the channel of each fault whose circuit errors share one, else -1.
        self._single_channel = np.array(
            [
                events[0][0] if len({e[0] for e in events}) == 1 else -1
                for events in self._events
            ],
            dtype=np.int64,
        )
        num_bytes = max(1, -(-num_f // 8))
        packed = np.frombuffer(
            b"".join(f.to_bytes(num_bytes, "little") for f in groups), dtype=np.uint8
        ).reshape(len(groups), num_bytes)
        self._bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :num_f]

        num_direct = len(program.direct_f_indices)
        direct = program.direct_f_indices
        is_detector = program.output_order[:num_direct] < prepared.num_detectors
        detector_bits = np.unique(direct[is_detector & (direct >= 0)])
        observable_bits = np.unique(direct[~is_detector & (direct >= 0)])
        self._branches: list[Branches] = []
        for component in program.components:
            assert isinstance(component, StatevectorComponent)
            branches = Branches(component, prepared.num_detectors)
            # NOTE: this raises if an output is random without noise.
            assert branches.reference is not None
            self._branches.append(branches)

        # NOTE: each fault is three rows of bits: the direct detectors that it
        # flips, the rotation signs that it flips, and the output flips of the
        # components followed by the direct observables. A set's rows are the
        # XOR of its faults' rows.
        signs = [np.zeros((len(groups), 0), dtype=np.uint8)]
        flips = []
        self._slots: list[np.ndarray] = []
        observable: list[np.ndarray] = []
        for branches in self._branches:
            local = self._bits[:, branches.f_selection]
            signs.append(gf2_product(local, branches.sign_rows))
            flips.append(gf2_product(local, branches.output_rows))
            start = sum(len(slots) for slots in self._slots)
            self._slots.append(np.arange(start, start + len(branches.observable)))
            observable.append(branches.observable)
        flips.append(self._bits[:, observable_bits])
        observable.append(np.ones(len(observable_bits), dtype=bool))
        is_observable = np.concatenate(observable)
        self._direct = np.arange(
            len(is_observable) - len(observable_bits), len(is_observable)
        )
        self._d = _words(self._bits[:, detector_bits])
        self._s = _words(np.concatenate(signs, axis=1))
        self._o = _words(np.concatenate(flips, axis=1))
        self._det_mask = _words(~is_observable[None, :])[0]
        self._obs_mask = _words(is_observable[None, :])[0]
        # NOTE: a GF(2)-linear hash of the direct detectors, so that a set's hash
        # is the XOR of its faults' hashes.
        rng = np.random.default_rng(0)
        mixer = rng.integers(0, 2**63, size=len(detector_bits), dtype=np.uint64)
        chosen = self._bits[:, detector_bits].astype(bool)
        self._hash = np.bitwise_xor.reduce(
            np.where(chosen, mixer, np.uint64(0)), axis=1, initial=np.uint64(0)
        )

    def logical_errors(
        self, weight: int, *, max_errors: int | None = None
    ) -> list[LogicalError]:
        """Return the logical errors of exactly ``weight`` faults."""
        n = len(self._events)
        small, large = weight // 2, weight - weight // 2
        if large > 3:
            raise ValueError(f"the search supports up to 6 faults, not {weight}")
        if comb(n, small) > MAX_COMBINATIONS or comb(n, large) > MAX_SCANNED:
            raise ValueError(
                f"sets of {weight} out of {n} faults are too many to search; lower "
                "max_weight or remove noise from the circuit"
            )
        join = _Join(self, weight)

        # NOTE: one walk per distinct rotation signs serves every set with them.
        signs, sets = join.collect()
        if len(signs) == 0:
            return []
        reps = np.bitwise_xor.reduce(self._bits[sets], axis=1)
        distributions = [
            branches.outcomes(reps[:, branches.f_selection])
            for branches in self._branches
        ]
        sets = join.emit(self._table(signs, distributions))
        undetected, flip = self._probabilities(signs, distributions, sets)
        keep = (flip > MIN_PROBABILITY) & self._distinct_channels(sets)
        sets, undetected, flip = sets[keep], undetected[keep], flip[keep]
        likelihood = flip * np.prod(self._fault_probability[sets], axis=1)
        order = np.argsort(-likelihood, kind="stable")[:max_errors]
        return [
            LogicalError(
                tuple(self._fault(j) for j in sets[i].tolist()),
                float(undetected[i]),
                float(flip[i]),
            )
            for i in order.tolist()
        ]

    def _distinct_channels(self, sets: np.ndarray) -> np.ndarray:
        """Return whether each set can take its circuit errors from distinct channels.

        Two circuit errors of one channel exclude each other.
        """
        channel = self._single_channel[sets]
        simple = (channel >= 0).all(axis=1)
        ordered = np.sort(channel, axis=1)
        ok = ~(ordered[:, 1:] == ordered[:, :-1]).any(axis=1)
        for i in np.flatnonzero(~simple).tolist():
            ok[i] = _distinct([{e[0] for e in self._events[j]} for j in sets[i]])
        return ok

    def _table(self, signs: np.ndarray, distributions: list[list]) -> tuple:
        """Return ``fault_join.scan``'s table of the logical errors of ``signs``.

        Its rows are the signs and detector deviations that can happen together.
        A set with signs ``s`` and output flips ``o`` flips no detector exactly
        when ``(s, o)`` is a row, and can then flip an observable unless the row
        has one observable deviation only, equal to ``o``'s.
        """
        width = sum(len(slots) for slots in self._slots) + len(self._direct)
        keys, masked, multi, observables = [], [], [], []
        for i, key in enumerate(signs):
            # NOTE: the components are independent, so any deviations of theirs
            # can happen together.
            rows = [np.zeros(width, dtype=bool)]
            for slots, component in zip(self._slots, distributions, strict=True):
                deviations, _ = component[i]
                grown = []
                for row in rows:
                    for deviation in deviations:
                        new = row.copy()
                        new[slots] = deviation.astype(bool)
                        grown.append(new)
                rows = grown
            options: dict[bytes, set[bytes]] = {}
            for row in rows:
                words = _words(row[None, :])[0]
                det = (words & self._det_mask).tobytes()
                options.setdefault(det, set()).add((words & self._obs_mask).tobytes())
            for det, obs in options.items():
                keys.append(key)
                masked.append(np.frombuffer(det, dtype=np.uint64))
                multi.append(len(obs) > 1)
                observables.append(np.frombuffer(next(iter(obs)), dtype=np.uint64))
        keys_array = np.array(keys, dtype=np.uint64).reshape(len(keys), -1)
        masked_array = np.array(masked, dtype=np.uint64).reshape(len(keys), -1)
        index = fault_join.build_word_index(
            keys_array, masked_array, self._det_mask, _capacity(len(keys))
        )
        return (
            keys_array,
            masked_array,
            self._det_mask,
            self._obs_mask,
            np.array(multi, dtype=np.bool_),
            np.array(observables, dtype=np.uint64).reshape(len(keys), -1),
            index,
        )

    def _probabilities(
        self, signs: np.ndarray, distributions: list[list], sets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return each set's chance that no detector flips, and that an observable does.

        ``distributions[j][i]`` is component ``j``'s distribution for ``signs[i]``,
        which must hold the signs of every set.
        """
        row_s = np.bitwise_xor.reduce(self._s[sets], axis=1)
        index = {key.tobytes(): i for i, key in enumerate(signs)}
        group = np.array([index[row.tobytes()] for row in row_s], dtype=np.int64)
        flips = np.unpackbits(
            np.bitwise_xor.reduce(self._o[sets], axis=1).view(np.uint8),
            axis=1,
            bitorder="little",
        ).astype(bool)
        undetected = np.ones(len(sets))
        unflipped = np.ones(len(sets))
        order = np.argsort(group, kind="stable")
        bounds = np.searchsorted(group[order], np.arange(len(signs) + 1))
        for i in np.flatnonzero(np.diff(bounds)).tolist():
            rows = order[bounds[i] : bounds[i + 1]]
            for branches, slots, component in zip(
                self._branches, self._slots, distributions, strict=True
            ):
                deviations, probability = component[i]
                same = (
                    deviations.astype(bool)[None, :, :] == flips[rows][:, None, slots]
                )
                passed = same[:, :, ~branches.observable].all(axis=2)
                undetected[rows] *= passed @ probability
                unflipped[rows] *= same.all(axis=2) @ probability
        unflipped[flips[:, self._direct].any(axis=1)] = 0.0
        return undetected, undetected - unflipped

    def _fault(self, index: int) -> Fault:
        """Return fault ``index`` with its circuit errors explained."""
        if index in self._explained:
            return self._explained[index]
        if self._locations is None:
            self._locations = _Locations(self._stim_circuit)
        events = self._events[index]
        self._explained[index] = Fault(
            tuple(
                self._locations.of(channel, outcome) for channel, outcome, _ in events
            ),
            sum(p for _, _, p in events),
        )
        return self._explained[index]


class _Join:
    """The sets of ``weight`` faults whose direct detectors cancel.

    A set is a stored left set of its first ``weight // 2`` faults and a right
    set of the others, which ``fault_join.scan`` streams.
    """

    def __init__(self, search: _Search, weight: int, small: int | None = None) -> None:
        small = weight // 2 if small is None else small
        self._weight = weight
        self._large = weight - small
        self._search = search
        self._faults = (search._d, search._s, search._o, search._hash)
        sets = _combinations(len(search._events), small)
        last = sets[:, -1].astype(np.int64) if small else np.full(len(sets), -1)
        hashes = np.bitwise_xor.reduce(search._hash[sets], axis=1, initial=np.uint64(0))
        order = np.lexsort((last, hashes))
        sets, last, hashes = sets[order], last[order], hashes[order]
        words = (
            np.bitwise_xor.reduce(rows[sets], axis=1, initial=np.uint64(0))
            for rows in self._faults[:3]
        )
        self._left = (sets, last, *words)
        keys, start = np.unique(hashes, return_index=True)
        end = np.append(start[1:], len(hashes))
        table = fault_join.build_index(keys, _capacity(len(keys)))
        self._runs = (keys, start.astype(np.int64), end.astype(np.int64), table)

    def collect(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the distinct signs of the sets, and one set of each."""
        signs, sets = self._scan(fault_join.COLLECT, None)
        _, first = np.unique(signs, axis=0, return_index=True)
        return signs[first], sets[first]

    def emit(self, table: tuple) -> np.ndarray:
        """Return the sets whose signs and output flips ``table`` marks."""
        return self._scan(fault_join.EMIT, table)[1]

    def _scan(self, mode: int, table: tuple | None) -> tuple[np.ndarray, np.ndarray]:
        """Run ``fault_join.scan``, growing its buffers until they fit."""
        lanes = numba.get_num_threads()
        width_s, width_o = self._search._s.shape[1], self._search._o.shape[1]
        if table is None:
            table = (
                np.zeros((0, width_s), np.uint64),
                np.zeros((0, width_o), np.uint64),
                np.zeros(width_o, np.uint64),
                np.zeros(width_o, np.uint64),
                np.zeros(0, np.bool_),
                np.zeros((0, width_o), np.uint64),
                np.full(1, -1, np.int64),
            )
        collect = mode == fault_join.COLLECT
        capacity = FIRST_CAPACITY
        work = comb(len(self._search._events), self._large)
        while True:
            sets = np.zeros((lanes, capacity, self._weight), np.int32)
            count = np.zeros(lanes, np.int64)
            store = (
                np.zeros((lanes, capacity if collect else 0, width_s), np.uint64),
                np.full((lanes, _capacity(capacity) if collect else 1), -1, np.int64),
                sets,
                count,
            )
            found = (*table, sets, count)
            args = (mode, self._large, self._faults, self._left, self._runs)
            if not fault_join.scan(work, *args, store, found):
                break
            capacity *= 8
        sets = np.concatenate([sets[i, : count[i]] for i in range(lanes)])
        signs = np.concatenate([store[0][i, : count[i]] for i in range(lanes)])
        return signs, sets


def _words(bits: np.ndarray) -> np.ndarray:
    """Return rows of bits as rows of uint64 words, at least one word per row."""
    width = max(1, -(-bits.shape[1] // 64))
    padded = np.zeros((len(bits), 64 * width), dtype=np.uint8)
    padded[:, : bits.shape[1]] = bits
    return np.packbits(padded, axis=1, bitorder="little").view(np.uint64)


def _capacity(count: int) -> int:
    """Return the power of 2 at least twice ``count``, for an open-addressing table."""
    return 1 << max(1, (2 * count).bit_length())


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
