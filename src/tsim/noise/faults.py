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
outputs; the output flips then just shift that distribution. The signs and
output flips of a set are its value.

``shortest_errors`` tries ``w = 1, 2, …`` faults, after Cheng, Guo and Pan's
distance certifier (arXiv:2611.08247): the fault distance is the fewest faults
of a set whose direct detectors cancel and whose value is a logical error.

- ``fault_pieces`` lists the pieces of ``w`` faults, the sets whose direct
  detectors cancel and that hold no smaller such set, by canonical branching.
- Every set whose direct detectors cancel is a disjoint union of pieces, so the
  values of those of ``w`` faults are those of the pieces of ``w`` faults and
  the sums of the values of pieces of ``j`` faults and of sets of ``w - j``.
  These sums hold far fewer distinct values than there are sets.
- Each distinct sign is walked once, and the values that can flip an
  observable undetected are kept (``fault_values``), without listing the sums.
- Only the sets with those values are then listed, and their probabilities
  computed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import stim

from tsim.compile.pipeline import compile_program
from tsim.core.builder import Site
from tsim.core.graph import prepare_graph
from tsim.core.parse import parse_stim_circuit
from tsim.noise import fault_values
from tsim.noise.channels import channel_num_bits
from tsim.noise.fault_pieces import Pieces
from tsim.noise.fault_values import pack
from tsim.statevector.branches import Branches, gf2_product
from tsim.statevector.program import StatevectorComponent

if TYPE_CHECKING:
    from tsim.circuit import Circuit

MIN_PROBABILITY = 1e-9
"""Smallest chance to flip an observable undetected that counts as a logical error."""

MAX_VALUE_BYTES = 1 << 34
"""Most bytes that one set of distinct values of sets of faults may take, with its
hash table; the search as a whole can take about twice as much."""

MAX_PIECES = 10**10
"""Most pieces of one weight that the search lists."""

_PAULI = {1: stim.target_z, 2: stim.target_x, 3: stim.target_y}
_MEASURED = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}
_MEASURED |= {"MXX": "X", "MYY": "Y", "MZZ": "Z"}


@dataclass(frozen=True)
class Fault:
    """One way for a circuit to go wrong: circuit errors with the same effect.

    Its first two attributes are those of ``stim.ExplainedError``, which
    ``to_explained_error`` returns.

    Attributes:
        dem_error_terms: The detectors and then the observables that the fault
            flips in every shot when it happens alone. In a Clifford circuit
            that is every one it flips; with non-Clifford gates it can also flip
            some in only part of the shots, which are not listed.
        circuit_error_locations: The circuit errors that are this fault. Their
            instruction offsets index ``circuit.flattened()``, so for a circuit
            with ``REPEAT`` blocks they have one stack frame, not Stim's nested
            ones.
        probability: The probability that the fault happens: that an odd number
            of its circuit errors do, those of one noise channel excluding each
            other and those of different channels being independent. This is
            exact; Stim's detector error model differs slightly, as it splits
            ``DEPOLARIZE`` and ``PAULI_CHANNEL`` instructions into independent
            errors.

    """

    dem_error_terms: tuple[stim.DemTargetWithCoords, ...]
    circuit_error_locations: tuple[stim.CircuitErrorLocation, ...]
    probability: float

    def to_explained_error(self) -> stim.ExplainedError:
        """Return the fault as a ``stim.ExplainedError``, without its probability."""
        return stim.ExplainedError(
            dem_error_terms=list(self.dem_error_terms),
            circuit_error_locations=list(self.circuit_error_locations),
        )


@dataclass(frozen=True)
class LogicalError:
    """Faults that together can flip an observable without flipping a detector.

    ``len(error)`` is the number of faults and iterating yields them, as for the
    list of ``stim.ExplainedError`` that Stim's search methods return.

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
    above ``max_weight``. A set counts only if it flips an observable undetected
    with a probability above ``MIN_PROBABILITY`` (1e-9). Its cost grows with
    the number of pieces of up to ``max_weight`` faults (sets whose direct
    detectors cancel and that hold no smaller such set), with the number of
    distinct values that sets of them have, and with the number of distinct
    rotation signs among those, each of which takes a walk of the state
    vector.

    On a Clifford circuit it finds what
    ``stim.Circuit.search_for_undetectable_logical_errors`` finds without
    limits, not only graphlike errors as ``shortest_graphlike_error`` does, but
    every logical error of the smallest weight rather than one, and it returns
    an empty list where Stim raises because there is none. Unlike Stim, it never
    combines two circuit errors of one noise channel, such as the X and Y of a
    ``PAULI_CHANNEL_1`` or two links of an ``ELSE_CORRELATED_ERROR`` chain:
    they exclude each other, while Stim's detector error model treats them as
    independent and can report a smaller distance.

    Args:
        circuit: The circuit, with its noise, detectors and observables.
        max_weight: The largest number of faults to try.
        max_errors: If given, at least 1: return only this many of the logical
            errors, the most likely ones.

    Returns:
        The logical errors with the fewest faults, most likely first, or an
        empty list if up to ``max_weight`` faults cause none. A logical error's
        likelihood is its ``logical_flip_probability`` times the probabilities
        of its faults.

    Raises:
        ValueError: If a detector or an observable is random without noise,
            ``max_weight`` or ``max_errors`` is below 1, or the search would
            list more than ``MAX_PIECES`` pieces of one weight or hold more
            than ``MAX_VALUE_BYTES`` in one set of distinct values.

    """
    if max_weight < 1:
        raise ValueError(f"max_weight must be at least 1, not {max_weight}")
    if max_errors is not None and max_errors < 1:
        raise ValueError(f"max_errors must be at least 1, not {max_errors}")
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
            [_probability(events) for events in self._events]
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
        self._num_detectors = prepared.num_detectors
        self._num_outputs = prepared.num_outputs
        self._direct_outputs = program.output_order[:num_direct]
        self._direct_bits = direct
        is_detector = self._direct_outputs < prepared.num_detectors
        detector_bits = np.unique(direct[is_detector & (direct >= 0)])
        observable_bits = np.unique(direct[~is_detector & (direct >= 0)])
        self._branches: list[Branches] = []
        self._outputs: list[np.ndarray] = []
        for component in program.components:
            assert isinstance(component, StatevectorComponent)
            branches = Branches(component, prepared.num_detectors)
            # NOTE: this raises if an output is random without noise.
            assert branches.reference is not None
            self._branches.append(branches)
            self._outputs.append(np.asarray(component.output_indices, np.int64))

        # NOTE: each fault is three rows of bits: the direct detectors that it
        # flips, the rotation signs that it flips, and the output flips of the
        # components followed by the direct observables. A set's rows are the
        # XOR of its faults' rows; the last two are its value.
        signs = [np.zeros((len(groups), 0), dtype=np.uint8)]
        flips = []
        self._sign_columns: list[slice] = []
        self._slots: list[np.ndarray] = []
        observable: list[np.ndarray] = []
        for branches in self._branches:
            local = self._bits[:, branches.f_selection]
            column = sum(rows.shape[1] for rows in signs)
            self._sign_columns.append(slice(column, column + len(branches.sign_rows)))
            signs.append(gf2_product(local, branches.sign_rows))
            flips.append(gf2_product(local, branches.output_rows))
            start = sum(len(slots) for slots in self._slots)
            self._slots.append(np.arange(start, start + len(branches.observable)))
            observable.append(branches.observable)
        flips.append(self._bits[:, observable_bits])
        observable.append(np.ones(len(observable_bits), dtype=bool))
        is_observable = np.concatenate(observable)
        sign_words = pack(np.concatenate(signs, axis=1))
        self._ws = sign_words.shape[1]
        self._values = np.hstack([sign_words, pack(np.concatenate(flips, axis=1))])
        self._det_mask = pack(~is_observable[None, :])[0]
        self._obs_mask = pack(is_observable[None, :])[0]
        self._pieces = Pieces(self._bits[:, detector_bits], self._values)
        self._piece_values: dict[int, _Values] = {}
        self._closures: dict[int, _Values] = {}
        # NOTE: the table of the walked signs and their deviations; see
        # ``fault_values``.
        width = self._values.shape[1] - self._ws
        self._signs = np.zeros((0, self._ws), dtype=np.uint64)
        self._sign_index = fault_values.index(self._signs)
        self._start = np.zeros(1, dtype=np.int64)
        self._deviations = np.zeros((0, width), dtype=np.uint64)
        self._deviation_probability = np.zeros(0)
        # NOTE: each component's deviations by its local signs, for components
        # that show up with the same local signs in several signs.
        self._component_deviations: list[dict[bytes, tuple]] = [
            {} for _ in self._branches
        ]

    def logical_errors(
        self, weight: int, *, max_errors: int | None = None
    ) -> list[LogicalError]:
        """Return the logical errors of exactly ``weight`` faults.

        It leaves out those that hold a piece whose value is 0 (see
        ``_pieces_of``): each holds a lighter logical error.
        """
        accepted = self._accepted(weight)
        if not len(accepted):
            return []
        sets = self._sets(weight, accepted)
        sets = sets[np.lexsort(sets.T[::-1])]
        undetected, flip = self._probabilities(sets)
        keep = (flip > MIN_PROBABILITY) & self._distinct_channels(sets)
        sets, undetected, flip = sets[keep], undetected[keep], flip[keep]
        likelihood = flip * np.prod(self._fault_probability[sets], axis=1)
        order = np.argsort(-likelihood, kind="stable")[:max_errors]
        self._explain(np.unique(sets[order]).tolist())
        return [
            LogicalError(
                tuple(self._fault(j) for j in sets[i].tolist()),
                float(undetected[i]),
                float(flip[i]),
            )
            for i in order.tolist()
        ]

    def _accepted(self, weight: int) -> np.ndarray:
        """Return the values of sets of ``weight`` faults that are logical errors.

        The sets are those whose direct detectors cancel, and the values are
        distinct. As in ``_closure``, some may be values of lighter sets only.
        """
        pieces = self._pieces_of(weight)
        parts = [
            (self._pieces_of(j), self._closure(weight - j)) for j in range(1, weight)
        ]
        zero = np.zeros((1, self._ws), dtype=np.uint64)
        signs = [(pieces.signs, zero)] + [(x.signs, y.signs) for x, y in parts]
        self._add_deviations(self._union(signs, self._ws))
        found = [pieces.rows[self._accepts(pieces.rows)]]
        found += [self._accepted_sums(x, y) for x, y in parts]
        return _unique(np.concatenate(found))

    def _pieces_of(self, weight: int) -> _Values:
        """Return the values of the pieces of ``weight`` faults, but 0.

        A piece is a set whose direct detectors cancel and that holds no
        smaller one. A piece whose value is 0 changes no value it joins, and a
        logical error that holds one is a lighter logical error with it, so
        ``shortest_errors`` finds the lighter one first.
        """
        if weight not in self._piece_values:
            if self._pieces.counts(weight, MAX_PIECES) is None:
                raise ValueError(
                    f"the search would list more than {MAX_PIECES} sets of {weight} "
                    "faults; lower max_weight or remove noise from the circuit"
                )
            width = self._values.shape[1]
            zero = np.zeros((1, width), dtype=np.uint64)
            batches = self._pieces.values(weight)
            values = self._union(((batch, zero) for batch in batches), width)
            self._piece_values[weight] = self._value_set(values[values.any(axis=1)])
        return self._piece_values[weight]

    def _closure(self, weight: int) -> _Values:
        """Return the values of the sets of ``weight`` faults whose detectors cancel.

        Such a set is a piece, or a piece of some ``j`` faults and such a set
        of ``weight - j`` faults, disjoint. Sums of the values of overlapping
        ones add values of lighter sets, which do no harm: they are logical
        errors only if those lighter sets are.
        """
        if weight not in self._closures:
            width = self._values.shape[1]
            parts = [(self._pieces_of(weight).rows, np.zeros((1, width), np.uint64))]
            parts += [
                (self._pieces_of(j).rows, self._closure(weight - j).rows)
                for j in range(1, weight)
            ]
            self._closures[weight] = self._value_set(self._union(parts, width))
        return self._closures[weight]

    def _value_set(self, values: np.ndarray) -> _Values:
        """Return the distinct ``values`` with their distinct signs."""
        mask = np.zeros(values.shape[1], dtype=np.uint64)
        mask[: self._ws] = ~np.uint64(0)
        first, _ = fault_values.distinct(values, mask, np.zeros(0, dtype=np.int64))
        return _Values(values, np.ascontiguousarray(values[first, : self._ws]))

    @staticmethod
    def _union(
        pairs: Iterable[tuple[np.ndarray, np.ndarray]], width: int
    ) -> np.ndarray:
        """Return the distinct XORs of a row of ``x`` and one of ``y``, for all ``pairs``.

        Raises:
            ValueError: If they and their hash table take more than
                ``MAX_VALUE_BYTES``.

        """
        rows = np.empty((1 << 10, width), dtype=np.uint64)
        count, table = 0, np.zeros(0, dtype=np.int64)
        for x, y in pairs:
            rows, count, table = fault_values.add_sums(
                rows, count, table, x, y, MAX_VALUE_BYTES
            )
            if count < 0:
                raise _too_large()
        del table
        return rows[:count].copy()

    def _add_deviations(self, signs: np.ndarray) -> None:
        """Walk those of the distinct ``signs`` that are not in the table yet."""
        known = np.empty(len(signs), dtype=np.int64)
        fault_values.find_all(len(signs), self._sign_index, self._signs, signs, known)
        new = signs[known < 0]
        if not len(new):
            return
        bits = np.unpackbits(new.view(np.uint8), axis=1, bitorder="little")
        width = self._deviations.shape[1]
        # NOTE: the components are independent, so a sign's deviations are
        # those of each component together, with the product of their
        # probabilities. A component whose local signs are 0 deviates by 0.
        rows: list[np.ndarray] = [np.zeros((1, width), dtype=np.uint64)] * len(new)
        probability: list[np.ndarray] = [np.ones(1)] * len(new)
        for branches, columns, slots, walked in zip(
            self._branches,
            self._sign_columns,
            self._slots,
            self._component_deviations,
            strict=True,
        ):
            local = bits[:, columns]
            touched = np.flatnonzero(local.any(axis=1)).tolist()
            keys = [row.tobytes() for row in np.packbits(local[touched], axis=1)]
            missing: dict[bytes, int] = {}
            for i, key in zip(touched, keys, strict=True):
                if key not in walked:
                    missing.setdefault(key, i)
            walks = (
                branches.deviations(local[list(missing.values())]) if missing else []
            )
            for key, (deviations, p) in zip(missing, walks, strict=True):
                placed = np.zeros((len(deviations), 64 * width), dtype=np.uint8)
                placed[:, slots] = deviations
                walked[key] = (pack(placed), p)
            for i, key in zip(touched, keys, strict=True):
                deviations, p = walked[key]
                rows[i] = (rows[i][:, None] ^ deviations[None]).reshape(-1, width)
                probability[i] = np.outer(probability[i], p).ravel()
        # NOTE: ``accepted_sums`` needs the deviations of a sign that agree on
        # the detectors next to each other.
        for i, deviations in enumerate(rows):
            if len(deviations) > 1:
                order = np.lexsort((deviations & self._det_mask).T[::-1])
                rows[i], probability[i] = deviations[order], probability[i][order]
        counts = np.array([len(r) for r in rows], dtype=np.int64)
        self._signs = np.concatenate([self._signs, new])
        self._sign_index = fault_values.index(self._signs)
        self._start = np.concatenate([self._start, self._start[-1] + np.cumsum(counts)])
        self._deviations = np.concatenate([self._deviations, *rows])
        self._deviation_probability = np.concatenate(
            [self._deviation_probability, *probability]
        )

    def _table(self) -> tuple:
        """Return the table of the walked signs and their deviations."""
        return (
            self._sign_index,
            self._signs,
            self._start,
            self._deviations,
            self._deviation_probability,
            self._det_mask,
            self._obs_mask,
        )

    def _accepts(self, values: np.ndarray) -> np.ndarray:
        """Return whether each value, whose signs were walked, is a logical error."""
        out = np.empty(len(values), dtype=np.int8)
        fault_values.accepted(len(values), values, self._ws, self._table(), out)
        assert (out >= 0).all()
        return out == 1

    def _accepted_sums(self, x: _Values, y: _Values) -> np.ndarray:
        """Return the distinct XORs of a value of ``x`` and one of ``y`` that are errors.

        Their signs must have been walked. The values of ``y`` are grouped by
        their signs and their output flips on detectors, so that each value of
        ``x`` and each deviation looks up the ones that flip no detector.
        """
        # NOTE: the work is about a lookup per value of x and sign of y.
        if len(x.rows) * len(y.signs) > len(y.rows) * len(x.signs):
            x, y = y, x
        mask = np.concatenate([~np.zeros(self._ws, dtype=np.uint64), self._det_mask])
        inverse = np.empty(len(y.rows), dtype=np.int64)
        first, key_index = fault_values.distinct(y.rows, mask, inverse)
        keys = y.rows[first] & mask
        start, order = fault_values.group(inverse, len(first))
        groups = (keys, key_index, start, order)
        args = (x.rows, y.rows, self._ws, y.signs, groups)
        work = len(x.rows) * len(y.signs)

        def run(*out: np.ndarray) -> None:
            fault_values.accepted_sums(work, *args, self._table(), *out)

        return _unique(_twice(run, len(x.rows), x.rows.shape[1], np.uint64))

    def _sets(self, weight: int, targets: np.ndarray) -> np.ndarray:
        """Return the sets of ``weight`` faults whose detectors cancel and value is a target.

        Each set is a sorted row of fault indices. Such a set is a piece, or a
        piece of some ``j`` faults and a set of ``weight - j`` faults whose
        values add up to a target. A pass down the weights finds the values
        that the lighter sets and the pieces need, and a pass up lists those
        sets, from the pieces with those values.
        """
        wanted = {weight: targets}
        joining: dict[int, list[np.ndarray]] = {}
        lighter: dict[int, list[np.ndarray]] = {}
        for w in range(weight, 0, -1):
            if w < weight:
                if w not in lighter:
                    continue
                wanted[w] = _unique(np.concatenate(lighter[w]))
            for j in range(1, w):
                piece_values = self._pieces_of(j).rows
                set_values = self._closure(w - j).rows
                _, used, needed = _xor_triples(wanted[w], piece_values, set_values)
                if len(used):
                    joining.setdefault(j, []).append(piece_values[used])
                    lighter.setdefault(w - j, []).append(set_values[needed])
        pieces: dict[int, np.ndarray] = {}
        found: dict[int, np.ndarray] = {}
        for w in range(1, weight + 1):
            values = joining.get(w, []) + ([wanted[w]] if w in wanted else [])
            if values:
                pieces[w] = self._pieces.matching(w, _unique(np.concatenate(values)))
            if w not in wanted:
                continue
            index = fault_values.index(wanted[w])
            parts = [pieces[w][_among(self._values_of(pieces[w]), wanted[w], index)]]
            for j in range(1, w):
                if j in pieces and w - j in found:
                    parts.append(self._unions(pieces[j], found[w - j], wanted[w]))
            found[w] = _unique(np.concatenate(parts))
        return found[weight]

    def _unions(
        self, pieces: np.ndarray, sets: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Return the disjoint unions of a piece and a set whose value is a target.

        The unions are sorted rows, those whose piece holds their lowest fault;
        one may come once for each such piece it holds. ``targets`` are
        distinct.
        """
        piece_first, piece_groups = _grouped(self._values_of(pieces))
        set_first, set_groups = _grouped(self._values_of(sets))
        _, a, b = _xor_triples(
            targets,
            self._values_of(pieces[piece_first]),
            self._values_of(sets[set_first]),
        )
        pairs = np.stack([a, b], axis=1)
        work = len(pieces) * len(sets)

        def run(*out: np.ndarray) -> None:
            fault_values.join(
                work, pieces, *piece_groups, sets, *set_groups, pairs, *out
            )

        return _twice(run, len(pairs), pieces.shape[1] + sets.shape[1], np.int32)

    def _values_of(self, sets: np.ndarray) -> np.ndarray:
        """Return the value of each set of faults, a row of fault indices."""
        out = np.empty((len(sets), self._values.shape[1]), dtype=np.uint64)
        fault_values.xor_sets(sets.size, sets, self._values, out)
        return out

    def _probabilities(self, sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return each set's chance that no detector flips, and that an observable does.

        The signs of the sets must have been walked.
        """
        undetected = np.empty(len(sets))
        flip = np.empty(len(sets))
        values = self._values_of(sets)
        table = self._table()
        fault_values.probabilities(len(sets), values, self._ws, table, undetected, flip)
        assert (undetected >= 0).all()
        return undetected, flip

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

    def _fault(self, index: int) -> Fault:
        """Return fault ``index`` with its circuit errors explained."""
        self._explain([index])
        return self._explained[index]

    def _explain(self, indices: list[int]) -> None:
        """Explain the faults ``indices``, with one walk per component for all."""
        indices = [i for i in indices if i not in self._explained]
        if not indices:
            return
        if self._locations is None:
            self._locations = _Locations(self._stim_circuit)
        bits = self._bits[indices]
        flipped = np.zeros((len(indices), self._num_outputs), dtype=bool)
        direct = self._direct_bits >= 0
        flipped[:, self._direct_outputs[direct]] = bits[
            :, self._direct_bits[direct]
        ].astype(bool)
        for branches, outputs in zip(self._branches, self._outputs, strict=True):
            local = bits[:, branches.f_selection]
            shifts = gf2_product(local, branches.output_rows).astype(bool)
            flipped[:, outputs] = shifts
            # NOTE: a fault that flips no rotation sign shifts the outputs, which
            # are deterministic without noise; the others need the walk.
            walked = np.flatnonzero(gf2_product(local, branches.sign_rows).any(axis=1))
            if not len(walked):
                continue
            for row, (deviations, probability) in zip(
                walked.tolist(), branches.outcomes(local[walked]), strict=True
            ):
                flips = deviations.astype(bool) ^ shifts[row]
                certain = flips.T.astype(float) @ probability > 1 - MIN_PROBABILITY
                flipped[row, outputs] = certain
        for index, row in zip(indices, flipped, strict=True):
            events = self._events[index]
            self._explained[index] = Fault(
                dem_error_terms=self._locations.terms(
                    np.flatnonzero(row).tolist(), self._num_detectors
                ),
                circuit_error_locations=tuple(
                    location
                    for channel, outcome, _ in events
                    for location in self._locations.of(channel, outcome)
                ),
                probability=float(self._fault_probability[index]),
            )


class _Values(NamedTuple):
    """Distinct values of sets of faults, and their distinct signs."""

    rows: np.ndarray
    signs: np.ndarray


def _too_large() -> ValueError:
    """Return the error for a search that would need too much memory."""
    return ValueError(
        f"the search would hold more than {MAX_VALUE_BYTES} bytes of values of "
        "sets of faults; lower max_weight or remove noise from the circuit"
    )


def _unique(rows: np.ndarray) -> np.ndarray:
    """Return the distinct rows, in the order they first appear."""
    rows = np.ascontiguousarray(rows)
    mask = ~np.zeros(rows.shape[1], dtype=rows.dtype)
    first, _ = fault_values.distinct(rows, mask, np.zeros(0, dtype=np.int64))
    return rows[first]


def _grouped(rows: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return the first index of each distinct row, and the rows grouped by them."""
    rows = np.ascontiguousarray(rows)
    inverse = np.empty(len(rows), dtype=np.int64)
    mask = ~np.zeros(rows.shape[1], dtype=rows.dtype)
    first, _ = fault_values.distinct(rows, mask, inverse)
    return first, fault_values.group(inverse, len(first))


def _among(values: np.ndarray, targets: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return whether each value is one of ``targets``, whose table is ``index``."""
    found = np.empty(len(values), dtype=np.int64)
    fault_values.find_all(len(values), index, targets, values, found)
    return found >= 0


def _xor_triples(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the indices ``(i, j, t)`` with ``a[i] ^ b[j] == c[t]``; ``c`` is distinct.

    All three must be distinct rows.
    """
    rows = (a, b, c)
    # NOTE: a ^ b == c holds for the three in any order; look up the largest.
    big = max(range(3), key=lambda k: len(rows[k]))
    first, second = (k for k in range(3) if k != big)
    x, y, z = rows[first], rows[second], rows[big]
    index = fault_values.index(z)

    def run(*out: np.ndarray) -> None:
        fault_values.triples(len(x) * len(y), x, y, index, z, *out)

    found = _twice(run, len(x), 3, np.int64)
    result = [found[:, 0]] * 3
    result[first], result[second], result[big] = found[:, 0], found[:, 1], found[:, 2]
    return result[0], result[1], result[2]


def _twice(run: Callable[..., None], tasks: int, width: int, dtype) -> np.ndarray:
    """Return the rows of a kernel of ``fault_values`` that counts, then writes them.

    ``run(counts, offsets, rows)`` calls the kernel.
    """
    counts = np.zeros(tasks, dtype=np.int64)
    run(counts, np.zeros(tasks, dtype=np.int64), np.zeros((0, width), dtype=dtype))
    # NOTE: -1 marks a sum whose signs were not walked; ``_accepted`` walks them.
    assert (counts >= 0).all()
    rows = np.empty((int(counts.sum()), width), dtype=dtype)
    if len(rows):
        written = np.zeros(tasks, dtype=np.int64)
        run(written, np.cumsum(counts) - counts, rows)
        assert np.array_equal(written, counts)
    return rows


def _probability(events: list[tuple[int, int, float]]) -> float:
    """Return the probability that an odd number of ``events`` happen.

    The events of one channel exclude each other; those of different channels
    are independent.
    """
    by_channel: dict[int, float] = {}
    for channel, _, p in events:
        by_channel[channel] = by_channel.get(channel, 0.0) + p
    odd = 0.0
    for p in by_channel.values():
        odd += p - 2 * odd * p
    return odd


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
        self._detector_coords = stim_circuit.get_detector_coordinates()

    def terms(
        self, outputs: list[int], num_detectors: int
    ) -> tuple[stim.DemTargetWithCoords, ...]:
        """Return outputs, detectors first, as Stim's terms of an explained error."""
        return tuple(
            (
                stim.DemTargetWithCoords(
                    stim.target_relative_detector_id(output),
                    self._detector_coords.get(output, []),
                )
                if output < num_detectors
                else stim.DemTargetWithCoords(
                    stim.target_logical_observable_id(output - num_detectors), []
                )
            )
            for output in sorted(outputs)
        )

    def _target(self, target: stim.GateTarget) -> stim.GateTargetWithCoords:
        coords = [] if target.is_combiner else self._coords.get(target.value, [])
        return stim.GateTargetWithCoords(target, coords)

    def of(self, channel: int, outcome: int) -> list[stim.CircuitErrorLocation]:
        """Return the circuit errors of ``outcome`` of noise channel ``channel``.

        Links of a correlated-error chain with the same effect share an outcome.
        """
        links = self._links.get(channel)
        if links is None:
            return [self._location(self._sites[channel], outcome)]
        return [
            self._location(site, outcome) for mask, site in links if mask == outcome
        ]

    def _location(self, site: Site, outcome: int) -> stim.CircuitErrorLocation:
        """Return the circuit error of ``outcome`` of the noise channel at ``site``."""
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
        elif name in _MEASURED or name == "MPP":
            # NOTE: Stim marks an inverted measurement on its measured observable.
            measured = [
                stim.target_pauli(
                    t.value,
                    t.pauli_type if name == "MPP" else _MEASURED[name],
                    t.is_inverted_result_target,
                )
                for t in targets
                if not t.is_combiner
            ]
        elif name == "MPAD":
            measured = []
        else:  # pragma: no cover - every noisy gate of Stim is above
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
