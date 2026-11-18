"""Parities of measurements that every noiseless shot gives the same value.

These are the parities that can be detectors or observables. Stim finds them
for Clifford circuits with a tableau (``stim.Circuit.count_determined_measurements``
and ``stim.Circuit.missing_detectors``); this module finds them for any circuit
that Tsim samples, in two ways.

A parity ``c`` is determined when ``c · m`` is the same for every record ``m``
that a noiseless shot can give, that is when ``c`` is orthogonal to the
directions ``m + m₀`` of the affine span of those records. The determined
parities are therefore a linear space, of dimension the number of measurements
minus that of the directions.

- Exactly (``shots=None``): Tsim compiles the noiseless circuit to direct
  outputs, which are constants, and statevector tapes, which are independent.
  ``Branches.support`` walks both outcomes of each measurement of a tape, but
  for fair coins that no rotation reads, which only add their output flips as
  directions. A measurement outcome less likely than ``IMPOSSIBLE`` (1e-12)
  given the ones before counts as impossible.
- From samples (``shots=n``): the directions of ``n`` noiseless shots. A parity
  that flips in a fraction ``q`` of shots survives ``n`` shots with probability
  ``(1 - q)^n``, so this is right with high confidence when ``n ≫ 1/q``, and it
  needs no walk.

With ``unknown_input=True`` the qubits start maximally mixed, which is a random
computational basis state: an ``X_ERROR(0.5)`` on every qubit at the start.
Its f-bits flip outputs (``A``) and rotation signs (``B``). Those in the kernel
of ``B`` only add their columns of ``A``; the exact path tries every
combination of the others and walks each tape once per distinct signs.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import TYPE_CHECKING

import numba
import numpy as np
import stim

from tsim.compile.pipeline import compile_program
from tsim.core.graph import prepare_graph
from tsim.statevector.branches import Branches, gf2_product
from tsim.statevector.program import StatevectorComponent
from tsim.utils.gf2_span import Span, num_words, pack, pack_bytes, unpack

if TYPE_CHECKING:
    from tsim.circuit import Circuit

MAX_SIGN_BITS = 16
"""Most independent rotation signs that unknown inputs may set, exactly."""

BATCH = 1 << 16
"""Shots drawn, or combinations of unknown inputs tried, at a time."""


def count_determined_measurements(
    circuit: Circuit,
    *,
    unknown_input: bool = False,
    shots: int | None = None,
    seed: int | None = None,
) -> int:
    """Count the measurements that earlier measurements predict, without noise.

    As ``stim.Circuit.count_determined_measurements``, for any circuit: the
    number of independent parities of measurements that every noiseless shot
    gives the same value. In a Clifford circuit that is the number of
    measurements that are a parity of earlier ones, as in Stim. With
    non-Clifford gates a measurement can also be a nonlinear function of
    earlier ones, which this does not count, since no detector can use it.

    If ``circuit.num_detectors + circuit.num_observables`` is below the count,
    some detectors may be missing (``missing_detectors``); if some declared
    detectors are random, ``nondeterministic_detectors`` lists them.

    Args:
        circuit: The circuit. Its noise is ignored.
        unknown_input: If True, the qubits start in unknown random states
            rather than in ``|0⟩``.
        shots: If given, estimate from this many noiseless shots instead of
            exactly; see the module docstring.
        seed: The seed of the shots, if any.

    Returns:
        The number of determined measurements.

    Raises:
        ValueError: If the exact walk follows more than
            ``Branches.MAX_BRANCHES`` branches per combination of inputs, or
            unknown inputs set more than ``MAX_SIGN_BITS`` independent
            rotation signs. Sampling avoids both.

    """
    directions = _directions(circuit, unknown_input, shots, seed)
    return circuit.num_measurements - directions.rank


def missing_detectors(
    circuit: Circuit,
    *,
    unknown_input: bool = False,
    shots: int | None = None,
    seed: int | None = None,
) -> stim.Circuit:
    """Return detectors for the determined parities that no declaration covers.

    As ``stim.Circuit.missing_detectors``, for any circuit: the returned
    ``DETECTOR`` instructions can be appended to ``circuit``, and then every
    determined parity (``count_determined_measurements``) is a parity of its
    determined detectors and observables. A random declared detector
    (``nondeterministic_detectors``) covers nothing, but parities of random
    detectors that are determined come first in the result, so it heals them:
    if S gates became T gates and randomized detectors ``D1`` and ``D2`` while
    ``D1 ⊕ D2`` stayed determined, that is a returned detector. Like Stim's, the
    result is no particularly good basis otherwise.

    A Tsim detector is its raw parity, so a returned one whose noiseless value
    is 1 fires in every shot unless the sampler XORs in a reference sample
    (``use_detector_reference_sample=True``).

    Args:
        circuit: The circuit. Its noise is ignored.
        unknown_input: As in ``count_determined_measurements``.
        shots: As in ``count_determined_measurements``.
        seed: As in ``count_determined_measurements``.

    Returns:
        A circuit of ``DETECTOR`` instructions.

    """
    num_m = circuit.num_measurements
    directions = _directions(circuit, unknown_input, shots, seed)
    declared = _Declared(circuit.stim_circuit)
    overlaps = declared.overlaps(directions.basis())
    random = np.flatnonzero(overlaps.any(axis=1))
    covered = Span(num_m)
    covered.insert(np.delete(declared.words, random, axis=0))
    # NOTE: a parity of random declarations is determined when it overlaps no
    # direction; the kernel's rows end at distinct declarations, so they pair
    # each random one with earlier ones where they can.
    kernel = Span(len(random))
    kernel.insert_bits(unpack(overlaps[random], directions.rank).T)
    combinations = unpack(kernel.annihilator(), len(random)).astype(bool)
    healed = np.zeros((len(combinations), num_words(num_m)), dtype=np.uint64)
    for i, combination in enumerate(combinations):
        healed[i] = np.bitwise_xor.reduce(declared.words[random[combination]], axis=0)
    candidates = np.concatenate([healed, directions.annihilator()])
    result = stim.Circuit()
    for words in candidates[covered.insert(candidates)]:
        records = np.flatnonzero(unpack(words[None], num_m)[0])
        result.append("DETECTOR", [stim.target_rec(int(m) - num_m) for m in records])
    return result


def nondeterministic_detectors(
    circuit: Circuit,
    *,
    unknown_input: bool = False,
    shots: int | None = None,
    seed: int | None = None,
) -> list[stim.DemTarget]:
    """Return the declared detectors and observables that are random without noise.

    Stim refuses such a circuit when it makes a detector error model. With
    non-Clifford gates they are common: replacing the S gates of a Clifford
    proxy by T gates randomizes some of its detectors. A declared parity is
    random when it is not a determined parity (``count_determined_measurements``).

    Args:
        circuit: The circuit. Its noise is ignored.
        unknown_input: As in ``count_determined_measurements``.
        shots: As in ``count_determined_measurements``; then a listed detector
            is certainly random, but an unlisted one may be random rarely.
        seed: As in ``count_determined_measurements``.

    Returns:
        The random detectors (``stim.target_relative_detector_id``) and then
        the random observables (``stim.target_logical_observable_id``).

    """
    directions = _directions(circuit, unknown_input, shots, seed)
    overlaps = _Declared(circuit.stim_circuit).overlaps(directions.basis())
    num_detectors = circuit.num_detectors
    return [
        (
            stim.target_relative_detector_id(index)
            if index < num_detectors
            else stim.target_logical_observable_id(index - num_detectors)
        )
        for index in np.flatnonzero(overlaps.any(axis=1)).tolist()
    ]


class _Declared:
    """The measurements of each detector and then each observable.

    Attributes:
        starts: Where each declaration's measurements start in ``records``,
            and the end.
        records: The measurements of each declaration, sorted.
        words: Each declaration as a packed vector over the measurements.

    """

    def __init__(self, circuit: stim.Circuit) -> None:
        num_m = circuit.num_measurements
        num_detectors = circuit.num_detectors
        rows, records = [], []
        measured = detector = 0
        for instruction in circuit.flattened():
            assert isinstance(instruction, stim.CircuitInstruction)
            if instruction.name == "DETECTOR":
                row = detector
                detector += 1
            elif instruction.name == "OBSERVABLE_INCLUDE":
                row = num_detectors + int(instruction.gate_args_copy()[0])
            else:
                measured += instruction.num_measurements
                continue
            for target in instruction.targets_copy():
                if not target.is_measurement_record_target:
                    raise ValueError(
                        f"{instruction.name} with a non-record target ({target}) "
                        "is not supported"
                    )
                rows.append(row)
                records.append(measured + target.value)
        # NOTE: a measurement that a declaration names twice cancels.
        keys, counts = np.unique(
            np.array(rows, dtype=np.int64) * num_m + np.array(records, dtype=np.int64),
            return_counts=True,
        )
        keys = keys[counts % 2 == 1]
        row, self.records = np.divmod(keys, max(num_m, 1))
        count = num_detectors + circuit.num_observables
        self.starts = np.searchsorted(row, np.arange(count + 1))
        self.words = np.zeros((count, num_words(num_m)), dtype=np.uint64)
        np.bitwise_or.at(
            self.words,
            (row, self.records // 64),
            np.uint64(1) << (self.records % 64).astype(np.uint64),
        )

    def overlaps(self, directions: np.ndarray) -> np.ndarray:
        """Return, packed, which of the packed ``directions`` each declaration flips."""
        return _overlaps(np.ascontiguousarray(directions), self.starts, self.records)


@numba.njit(cache=True)
def _overlaps(
    directions: np.ndarray, starts: np.ndarray, records: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(starts) - 1, max(1, -(-len(directions) // 64))), np.uint64)
    for d in range(len(directions)):
        word, bit = d // 64, np.uint64(1) << np.uint64(d % 64)
        for k in range(len(starts) - 1):
            parity = np.uint64(0)
            for m in records[starts[k] : starts[k + 1]]:
                parity ^= directions[d, m // 64] >> np.uint64(m % 64)
            if parity & np.uint64(1):
                result[k, word] |= bit
    return result


def _directions(
    circuit: Circuit, unknown_input: bool, shots: int | None, seed: int | None
) -> Span:
    """Return the span of ``m + m₀`` over the noiseless records ``m``."""
    from tsim.circuit import Circuit

    noiseless = circuit.without_noise().without_annotations()
    if unknown_input and noiseless.num_qubits:
        mixed = stim.Circuit()
        mixed.append("X_ERROR", range(noiseless.num_qubits), 0.5)
        noiseless = Circuit.from_stim_program(mixed) + noiseless
    if shots is None:
        return _exact_directions(noiseless)
    if shots < 1:
        raise ValueError(f"shots must be at least 1, not {shots}")
    return _sampled_directions(noiseless, shots, seed)


def _sampled_directions(circuit: Circuit, shots: int, seed: int | None) -> Span:
    """Return the span of ``m + m₀`` over ``shots`` noiseless shots ``m``."""
    num_m = circuit.num_measurements
    span = Span(num_m)
    sampler = circuit.compile_sampler(seed=seed)
    reference = None
    last_growth = 0
    for done in range(0, shots, BATCH):
        count = min(BATCH, shots - done)
        words = pack_bytes(sampler.sample(count, bit_packed=True), num_m)
        if reference is None:
            reference = words[0].copy()
        rows, first = np.unique(words ^ reference, axis=0, return_index=True)
        # NOTE: in shot order, so that ``last_growth`` is the last shot that grew it.
        order = np.argsort(first)
        grew = span.insert(rows[order])
        if grew.any():
            last_growth = done + int(first[order][grew].max()) + 1
    if last_growth > shots // 2:
        warnings.warn(
            f"shot {last_growth} of {shots} still found a new direction, so some "
            "parities that look determined may be random; use more shots",
            stacklevel=4,
        )
    return span


def _exact_directions(circuit: Circuit) -> Span:
    """Return the span of ``m + m₀`` over every noiseless record ``m``."""
    prepared = prepare_graph(circuit, sample_detectors=False)
    program = compile_program(
        prepared, mode="sequential", strategy="active_statevector"
    )
    num_m = prepared.num_outputs
    num_f = prepared.error_transform.shape[0]
    span = Span(num_m)

    # NOTE: the f-bits, set only by unknown inputs, are fair and independent.
    output_map = np.zeros((num_m, num_f), dtype=np.uint8)
    constant = np.zeros(num_m, dtype=np.uint8)
    direct = program.output_order[: len(program.direct_f_indices)]
    constant[direct] = program.direct_flips
    reads = program.direct_f_indices >= 0
    output_map[direct[reads], program.direct_f_indices[reads]] = 1
    tapes: list[tuple[Branches, np.ndarray]] = []
    sign_maps = [np.zeros((0, num_f), dtype=np.uint8)]
    for component in program.components:
        assert isinstance(component, StatevectorComponent)
        branches = Branches(component, num_detectors=0)
        outputs = np.asarray(component.output_indices, dtype=np.int64)
        tapes.append((branches, outputs))
        output_map[np.ix_(outputs, branches.f_selection)] ^= branches.output_rows
        sign_map = np.zeros((len(branches.sign_rows), num_f), dtype=np.uint8)
        sign_map[:, branches.f_selection] = branches.sign_rows
        sign_maps.append(sign_map)
        span.insert_bits(_embed(branches.coin_rows, outputs, num_m))
    sign_map = np.concatenate(sign_maps)
    span.insert_bits(gf2_product(_kernel(sign_map), output_map))

    choices = _independent_columns(sign_map)
    if len(choices) > MAX_SIGN_BITS:
        raise ValueError(
            f"unknown inputs set {len(choices)} independent rotation signs, more "
            f"than {MAX_SIGN_BITS}; pass shots to sample instead"
        )
    f = np.zeros((1 << len(choices), num_f), dtype=np.uint8)
    f[:, choices] = list(product((0, 1), repeat=len(choices)))

    # NOTE: a tape's outputs are its f-bits' flips XOR a value that depends
    # only on its rotation signs: walk each distinct signs once.
    bases = []
    for branches, outputs in tapes:
        local = f[:, branches.f_selection]
        first, inverse = _groups(gf2_product(local, branches.sign_rows))
        row, values = branches.support(local[first])
        firsts = values[np.searchsorted(row, np.arange(len(first)))]
        within = Span(len(outputs))
        within.insert_bits(values ^ firsts[row])
        span.insert_bits(_embed(unpack(within.basis(), len(outputs)), outputs, num_m))
        base = firsts ^ gf2_product(local[first], branches.output_rows)
        bases.append((inverse, base, outputs))
    reference = None
    for start in range(0, len(f), BATCH):
        chunk = slice(start, start + BATCH)
        points = gf2_product(f[chunk], output_map) ^ constant
        for inverse, base, outputs in bases:
            points[:, outputs] ^= base[inverse[chunk]]
        if reference is None:
            reference = points[0].copy()
        span.insert_bits(points ^ reference)
    return span


def _groups(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the first index of each distinct row, and each row's group."""
    if rows.shape[1] == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(len(rows), dtype=np.int64)
    _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    return first, inverse.ravel()


def _embed(local: np.ndarray, outputs: np.ndarray, num_m: int) -> np.ndarray:
    """Return rows over a tape's outputs as rows over all ``num_m`` measurements."""
    rows = np.zeros((len(local), num_m), dtype=np.uint8)
    rows[:, outputs] = local
    return rows


def _kernel(matrix: np.ndarray) -> np.ndarray:
    """Return a basis of the vectors ``v`` with ``matrix @ v = 0``, as 0/1 rows."""
    span = Span(matrix.shape[1])
    span.insert_bits(matrix)
    return unpack(span.annihilator(), matrix.shape[1])


def _independent_columns(matrix: np.ndarray) -> np.ndarray:
    """Return columns of ``matrix`` that form a basis of its column space."""
    span = Span(matrix.shape[0])
    return np.flatnonzero(span.insert(pack(matrix.T)))


__all__ = [
    "count_determined_measurements",
    "missing_detectors",
    "nondeterministic_detectors",
]
