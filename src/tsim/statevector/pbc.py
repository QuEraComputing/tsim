"""Pauli-based computation (PBC) read off a reduced ZX component.

Here a PBC is a sequence of Pauli product rotations and Pauli product
measurements on ``|0…0⟩`` [1], held in ``PauliOps``. Unlike the PBC of [2], it
keeps the rotations instead of consuming magic states.

The quantum part of a reduced component is graph-like [3]: a graph state [4] of
its quantum spiders, followed by one rotation and one single-qubit measurement
per spider. The phase ``π/2·q + 2θ`` of spider ``v`` rotates its qubit by
``exp(-iθ Z_v)`` and then measures ``m_v = S^-q X S^q`` (a signed X or Y), with
outcome bit ``y_v``. The classical spiders only relate the bits ``y`` to the
outputs. Eliminating them over GF(2) (``_Relations``) leaves each output as a
parity of bits ``y``, f-bits and fair coins, plus parities of the bits ``y``
that the diagram postselects.

A sampler cannot postselect, so the postselections are moved to the start,
where they act on the graph state. For that, every rotation must commute with
them. Let ``K_T`` be the product of the graph-state stabilizers
``K_u = X_u Z_N(u)`` over the spiders ``u`` in ``T``. If ``K_T`` commutes with
``Z_v`` and stabilizes the current state, ``exp(-iθ Z_v)`` acts on that state
like ``exp(-iθ Z_v K_T)``. Both hold when no spider in ``T`` is rotated at or
before ``v``. ``_rotation_axes`` finds sets ``T`` that make every rotation
commute with every postselection, with a backward greedy search like the gflow
search of [5].

A Clifford frame then maps the postselected graph state to ``|0…0⟩``.
Noise-dependent postselection targets become f-dependent signs, lines that no
operation flips are dropped, and Paulis that commute with every operation are
measured first: they change no outcome and keep the active width small.

References:
    [1] D. Litinski, "A Game of Surface Codes: Large-Scale Quantum Computing with
        Lattice Surgery", Quantum 3, 128 (2019).
    [2] S. Bravyi, G. Smith, J. A. Smolin, "Trading classical and quantum
        computational resources", Phys. Rev. X 6, 021043 (2016).
    [3] R. Duncan, A. Kissinger, S. Perdrix, J. van de Wetering, "Graph-theoretic
        simplification of quantum circuits with the ZX-calculus", Quantum 4, 279
        (2020).
    [4] M. Hein, J. Eisert, H. J. Briegel, "Multiparty entanglement in graph
        states", Phys. Rev. A 69, 062311 (2004).
    [5] M. Mhalla, S. Perdrix, "Finding optimal flows efficiently", ICALP 2008,
        LNCS 5125.

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction
from math import pi

import numpy as np
import stim
from zxparam import Domain, EdgeType, Graph, Phase, VertexType

from tsim.core.params import ParamLayout

ROTATION = 0
MEASUREMENT = 1


@dataclass(frozen=True)
class ComponentLayout:
    """Where a component's parameters and outputs go.

    Attributes:
        f_selection: Global f-bit index of each component f-bit, ascending.
        f_position: Position in ``f_selection`` of each phase parameter ID.
        output_indices: Global output index of each output slot.

    """

    f_selection: tuple[int, ...]
    f_position: Mapping[int, int]
    output_indices: tuple[int, ...]

    @classmethod
    def of(
        cls, component: Graph, output_indices: Sequence[int], layout: ParamLayout
    ) -> ComponentLayout:
        """Return the layout of ``component``, whose outputs are ``output_indices``."""
        f_selection = tuple(
            sorted(layout.basis_index(p) for p in component.parameters())
        )
        f_position = {layout.basis(i): pos for pos, i in enumerate(f_selection)}
        return cls(f_selection, f_position, tuple(output_indices))


def anticommute(
    x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray
) -> np.ndarray:
    """Return which Paulis ``(x1, z1)`` anticommute with which Paulis ``(x2, z2)``.

    Each row of an argument is a Pauli; a 1-D argument is one Pauli, whose axis
    the result drops.
    """
    f = np.float32
    return (x1.astype(f) @ z2.T.astype(f) + z1.astype(f) @ x2.T.astype(f)) % 2 == 1


@dataclass(frozen=True)
class PauliOps:
    """Rotations and measurements in the frame of the initial ``|0…0⟩`` state.

    Op ``j`` is the rotation ``exp(-iθ (-1)^s P)`` or the measurement of
    ``(-1)^s P``, where ``P`` has the bit rows ``x[j]`` and ``z[j]`` (``Y``
    where both are set) and ``s`` is the parity of ``sign[j]`` against
    ``[1, f_0, f_1, …]``. A measurement records its eigenvalue bit.

    Attributes:
        kind: ``ROTATION`` or ``MEASUREMENT``, shape ``(N,)``.
        x: X part of each Pauli, shape ``(N, num_lines)``.
        z: Z part of each Pauli, shape ``(N, num_lines)``.
        angle: θ of each rotation, 0 for measurements, shape ``(N,)``.
        sign: Constant, then one column per f-bit, shape ``(N, 1 + num_f)``.
        slot: Output slot of a recorded measurement, -1 for a forgotten
            measurement or a rotation, shape ``(N,)``.

    """

    kind: np.ndarray
    x: np.ndarray
    z: np.ndarray
    angle: np.ndarray
    sign: np.ndarray
    slot: np.ndarray

    @property
    def num_ops(self) -> int:
        """Return the number of operations."""
        return len(self.kind)

    @property
    def num_lines(self) -> int:
        """Return the number of qubit lines."""
        return self.x.shape[1]

    @property
    def num_f(self) -> int:
        """Return the number of component f-bits the signs depend on."""
        return self.sign.shape[1] - 1

    @property
    def num_outputs(self) -> int:
        """Return the number of recorded measurements."""
        return int(np.count_nonzero(self.slot >= 0))

    def take(self, order: np.ndarray) -> PauliOps:
        """Return the operations in the given order.

        The result describes the same distribution when ``order`` keeps every
        pair of anticommuting operations in their original order.
        """
        return PauliOps(*(getattr(self, f.name)[order] for f in fields(self)))


def read_pauli_ops(component: Graph, layout: ComponentLayout) -> PauliOps:
    """Return the operations of a component of the reduced sampling graph.

    Forgotten measurements of Paulis that commute with every op come first,
    then the rotations, then one measurement per output slot.

    Raises:
        ValueError: If the component is not a graph-like diagram of Z spiders
            with classical outputs, its classical part does not describe
            probabilities, or no rotation order exists.

    """
    reading = _Reading(component, layout)
    n, num_f = reading.num_spiders, len(layout.f_selection)
    patterns = _patterns(reading)
    axes = _rotation_axes(reading, patterns)
    frame, targets = _frame(reading, patterns[0], num_f)
    paulis = []
    for v, t in axes:
        axis = stim.PauliString(n)
        axis[v] = "Z"
        paulis.append((axis * reading.stabilizer(t), 0, 0))
    paulis += [(reading.measured(y), coins, rhs) for y, coins, rhs in reading.outputs]
    images = [frame(pauli) for pauli, _, _ in paulis]
    flipped = np.zeros(n, dtype=bool)
    for image in images:
        flipped |= image.to_numpy()[0]
    lines = np.flatnonzero(flipped)
    x = np.zeros((len(paulis), len(lines) + reading.num_coins), dtype=bool)
    z = np.zeros_like(x)
    sign = np.zeros((len(paulis), 1 + num_f), dtype=bool)
    for j, ((_, coins, rhs), image) in enumerate(zip(paulis, images, strict=True)):
        image_x, image_z = image.to_numpy()
        x[j, : len(lines)], z[j, : len(lines)] = image_x[lines], image_z[lines]
        x[j, len(lines) :] = _unmask(coins, reading.num_coins)
        sign[j] = _unmask(rhs, 1 + num_f)
        sign[j, 0] ^= image.sign == -1
        # NOTE: postselection k sits on frame line k with the f-dependent target
        # t_k, so that line starts in |t_k⟩: an op with a Z on it flips with t_k.
        on_targets = np.flatnonzero(image_z[: len(targets)])
        sign[j, 1:] ^= targets[on_targets].sum(axis=0) % 2 == 1
    num_outputs = len(reading.outputs)
    ops = PauliOps(
        kind=np.array([ROTATION] * len(axes) + [MEASUREMENT] * num_outputs, np.uint8),
        x=x,
        z=z,
        angle=np.array([reading.angle[v] for v, _ in axes] + [0.0] * num_outputs),
        sign=sign,
        slot=np.array([-1] * len(axes) + list(range(num_outputs)), dtype=np.int32),
    )
    return _with_commutant_measurements(ops)


class _Reading:
    """A component as a graph state, rotations, postselections and outputs.

    Quantum spider ``v`` is graph-state qubit ``v``, in the order of vertex IDs.

    Attributes:
        num_spiders: Number of quantum spiders.
        neighbours: Graph-state neighbours of each spider, as a bit mask.
        letter: ``"X"`` or ``"Y"``, the axis of each spider's ``m_v``.
        negative: Mask of the spiders whose ``m_v`` is negated.
        angle: θ of each rotated spider.
        posts: ``(mask, rhs)`` per postselection: the parity of the bits ``y``
            in ``mask`` must be ``rhs`` (bit 0 a constant, bit ``k + 1`` f-bit
            ``k``).
        outputs: ``(mask, coins, rhs)`` per output slot: the output is the parity
            of the bits ``y`` in ``mask``, of the fair coins in ``coins`` and of
            ``rhs``.
        num_coins: Number of fair coins.

    """

    def __init__(self, component: Graph, layout: ComponentLayout) -> None:
        """Read ``component`` and eliminate its classical spiders."""
        quantum, classical, boundaries = [], [], 0
        for v in component.vertices():
            kind, domain = component.vertex_type(v), component.vertex_domain(v)
            if kind == VertexType.BOUNDARY and domain == Domain.CLASSICAL:
                boundaries += 1
            elif kind != VertexType.Z:
                raise ValueError(
                    f"vertex {v} is neither a Z spider nor a classical output"
                )
            else:
                (quantum if domain == Domain.QUANTUM else classical).append(v)
        outputs = list(component.outputs())
        if len(outputs) != boundaries:
            raise ValueError("the component has an input")
        n, c = len(quantum), len(classical)
        index = {
            v: i for i, v in enumerate([*sorted(quantum), *sorted(classical), *outputs])
        }

        def noise(phase: Phase) -> int:
            return sum(2 << layout.f_position[p] for p in phase.parameters)

        self.num_spiders = n
        self.neighbours = [0] * n
        self.letter = ["X"] * n
        self.negative = 0
        self.angle: dict[int, float] = {}
        relations = _Relations(n + c + len(outputs))
        y_masks, y_rhs = [1 << v for v in range(n)], [0] * n
        for vertex in quantum:
            v, phase = index[vertex], component.phase(vertex)
            quarters = round(2 * phase.value)
            if rest := phase.value - Fraction(quarters, 2):
                self.angle[v] = pi * float(rest) / 2
            self.letter[v] = "XY"[quarters % 2]
            self.negative |= (quarters % 4 in (1, 2)) << v
            y_rhs[v] = noise(phase)
        for vertex in classical:
            phase = component.phase(vertex)
            if phase.value not in (0, 1):
                raise ValueError(f"classical spider {vertex} has phase {phase}")
            relations.lin[index[vertex]] = noise(phase) | (phase.value == 1)
        for u, w in component.edges():
            a, b = sorted((index[u], index[w]))
            hadamard = component.edge_type(u, w) == EdgeType.HADAMARD
            if a < n and not hadamard:
                raise ValueError(f"the edge {u}-{w} at a quantum spider is plain")
            if b < n:
                self.neighbours[a] |= 1 << b
                self.neighbours[b] |= 1 << a
            elif a < n:
                if b >= n + c:
                    raise ValueError(f"the output {w} touches a quantum spider")
                y_masks[a] ^= 1 << b
            elif hadamard:
                relations.add_sign(a, b)
            else:
                relations.equalities.append((1 << a | 1 << b, 0))
        relations.equalities += zip(y_masks, y_rhs, strict=True)
        for x in range(n, n + c):
            relations.eliminate(x)
        output_bits = range(n + c, n + c + len(outputs))
        rows = relations.reduce([*output_bits, *range(n)])
        pivots = {p for p, _, _ in rows}
        coin = {b: k for k, b in enumerate(b for b in output_bits if b not in pivots)}
        self.num_coins = len(coin)
        self.posts = [(mask, rhs) for p, mask, rhs in rows if p < n]
        outputs_by_bit = {b: (0, 1 << k, 0) for b, k in coin.items()}
        for p, mask, rhs in rows:
            if p >= n:
                others = _bits(mask >> (n + c) << (n + c) & ~(1 << p))
                coins = sum(1 << coin[b] for b in others)
                outputs_by_bit[p] = (mask & ((1 << n) - 1), coins, rhs)
        self.outputs = [outputs_by_bit[b] for b in output_bits]
        self._stabilizers: dict[int, stim.PauliString] = {}

    def stabilizer(self, t: int) -> stim.PauliString:
        """Return ``K_T``, the product of ``K_u = X_u Z_N(u)`` over the spiders in ``t``."""
        product = stim.PauliString(self.num_spiders)
        for u in _bits(t):
            if u not in self._stabilizers:
                k_u = stim.PauliString(self.num_spiders)
                k_u[u] = "X"
                for w in _bits(self.neighbours[u]):
                    k_u[w] = "Z"
                self._stabilizers[u] = k_u
            product *= self._stabilizers[u]
        return product

    def measured(self, mask: int) -> stim.PauliString:
        """Return the product of ``m_v`` over the spiders in ``mask``."""
        pauli = stim.PauliString(self.num_spiders)
        for v in _bits(mask):
            pauli[v] = self.letter[v]
        return -pauli if (mask & self.negative).bit_count() % 2 else pauli


class _Relations:
    """Equalities and a sign over GF(2) variables, the bits of int masks.

    An equality ``(mask, rhs)`` says that the parity of the variables in ``mask``
    is the affine form ``rhs`` of ``(1, f_0, f_1, …)``. The sign is ``(-1)^Φ``,
    with ``Φ = Σ v_i·v_j`` over the pairs in ``pairs`` plus ``Σ v_i·lin[i]``.
    """

    def __init__(self, num_vars: int) -> None:
        """Start with no equality and the sign +1."""
        self.equalities: list[tuple[int, int]] = []
        self.pairs = [0] * num_vars
        self.lin = [0] * num_vars

    def add_sign(self, i: int, j: int) -> None:
        """Multiply the sign by ``(-1)^(v_i·v_j)``."""
        if i == j:
            self.lin[i] ^= 1
        else:
            self.pairs[i] ^= 1 << j
            self.pairs[j] ^= 1 << i

    def eliminate(self, v: int) -> None:
        """Remove ``v``: solve an equality for it, or sum the sign over it.

        Summing over ``v`` turns its factor ``(-1)^(v·ℓ)`` into the equality
        ``ℓ = 0`` and leaves the sign at ``v = 0``.
        """
        bit = 1 << v
        for k, (mask, rhs) in enumerate(self.equalities):
            if mask & bit:
                del self.equalities[k]
                self._substitute(v, mask ^ bit, rhs)
                return
        self.equalities.append((self.pairs[v], self.lin[v]))
        self._substitute(v, 0, 0)

    def reduce(self, order: Sequence[int]) -> list[tuple[int, int, int]]:
        """Return the equalities in reduced echelon form, as ``(pivot, mask, rhs)``.

        Pivots are taken in ``order``, then substituted into the sign.

        Raises:
            ValueError: If a sign is left, so the component does not describe
                probabilities.

        """
        # NOTE: the reduced echelon form is unique, so eliminate by leading bits,
        # with the first variable of ``order`` as the highest, then back-substitute.
        position = {v: len(order) - 1 - i for i, v in enumerate(order)}
        echelon = _Basis()
        for mask, rhs in self.equalities:
            if echelon.insert(sum(1 << position[v] for v in _bits(mask)), rhs):
                raise ValueError("the classical part of the component is inconsistent")
        reduced: dict[int, tuple[int, int]] = {}
        below = 0
        for lead in sorted(echelon.rows):
            mask, rhs = echelon.rows[lead]
            for bit in _bits(mask & below):
                mask, rhs = mask ^ reduced[bit][0], rhs ^ reduced[bit][1]
            reduced[lead] = (mask, rhs)
            below |= 1 << lead
        variable = order[::-1]
        rows = [
            (variable[lead], sum(1 << variable[b] for b in _bits(mask)), rhs)
            for lead, (mask, rhs) in sorted(reduced.items(), reverse=True)
        ]
        self.equalities = []
        for p, mask, rhs in rows:
            self._substitute(p, mask ^ 1 << p, rhs)
        if any(self.pairs) or any(self.lin):
            raise ValueError("the classical part of the component is not stochastic")
        return rows

    def _substitute(self, v: int, mask: int, rhs: int) -> None:
        """Replace ``v`` everywhere by the parity of ``mask`` plus ``rhs``."""
        bit = 1 << v
        for u in _bits(self.pairs[v]):
            self.pairs[u] ^= bit
            self.lin[u] ^= rhs
            for w in _bits(mask):
                self.add_sign(w, u)
        # NOTE: v·lin[v] also leaves rhs·lin[v], a sign of the f-bits alone,
        # which the normalization of each f-bit pattern absorbs.
        for w in _bits(mask):
            self.lin[w] ^= self.lin[v]
        self.pairs[v] = self.lin[v] = 0
        self.equalities = [
            (m ^ bit ^ mask, r ^ rhs) if m & bit else (m, r) for m, r in self.equalities
        ]
        self._drop_trivial()

    def _drop_trivial(self) -> None:
        if any(r for m, r in self.equalities if not m):
            raise ValueError("the classical part of the component is inconsistent")
        self.equalities = [(m, r) for m, r in self.equalities if m]


def _patterns(reading: _Reading) -> tuple[list[int], list[int]]:
    """Return, as masks, the postselections that anticommute with each ``K_u`` and ``Z_v``.

    ``K_u = X_u Z_N(u)`` anticommutes with ``m_w`` when ``w`` is a neighbour of
    ``u``, or when ``w = u`` and ``m_w`` is a Y; ``Z_v`` anticommutes with ``m_v``.
    """
    n = reading.num_spiders
    k_pattern, z_pattern = [0] * n, [0] * n
    for k, (mask, _) in enumerate(reading.posts):
        for w in _bits(mask):
            z_pattern[w] |= 1 << k
            for u in _bits(reading.neighbours[w]):
                k_pattern[u] ^= 1 << k
            if reading.letter[w] == "Y":
                k_pattern[w] ^= 1 << k
    return k_pattern, z_pattern


def _rotation_axes(
    reading: _Reading, patterns: tuple[list[int], list[int]]
) -> list[tuple[int, int]]:
    """Return ``(v, T)`` per rotated spider, in rotation order.

    ``Z_v K_T`` commutes with every postselection, and ``T`` holds no spider
    rotated at or before ``v``. The order is built from the back: a spider
    placed just before the placed ones may use them and the unrotated spiders
    in ``T``. That set only grows, so a spider that can be placed stays
    placeable, and the greedy fails only if no order exists.

    Raises:
        ValueError: If no order exists.

    """
    k_pattern, z_pattern = patterns
    unplaced = sorted(reading.angle, reverse=True)
    allowed = sum(1 << u for u in range(reading.num_spiders) if u not in reading.angle)
    basis = _Basis()
    for u in _bits(allowed):
        basis.insert(k_pattern[u], 1 << u)

    def witness(v: int) -> int | None:
        # NOTE: try single neighbours first, typically a phase gadget's root.
        if not z_pattern[v]:
            return 0
        for u in _bits(reading.neighbours[v] & allowed):
            if k_pattern[u] == z_pattern[v]:
                return 1 << u
        return basis.solve(z_pattern[v])

    placed = []
    while unplaced:
        found = next(
            ((k, t) for k, v in enumerate(unplaced) if (t := witness(v)) is not None),
            None,
        )
        if found is None:
            raise ValueError(
                "no rotation order makes every postselection commute "
                "(the component has no flow)"
            )
        k, t = found
        v = unplaced.pop(k)
        placed.append((v, t))
        basis.insert(k_pattern[v], 1 << v)
        allowed |= 1 << v
    return placed[::-1]


def _frame(
    reading: _Reading, k_pattern: list[int], num_f: int
) -> tuple[stim.Tableau, np.ndarray]:
    """Return the frame of the postselected graph state and the targets of its lines.

    The state is stabilized by a maximal independent set of the postselections,
    on frame lines ``0, 1, …``, and by the graph-state stabilizers that commute
    with every postselection, on the other lines. Every other postselection is
    a product of those and must hold for every f-bit pattern.

    Raises:
        ValueError: If a postselection has probability 0.

    """
    independent, kernel = _Basis(), []
    for u in range(reading.num_spiders):
        if zero := independent.insert(k_pattern[u], 1 << u):
            kernel.append(zero)
    lines = sorted(independent.rows)
    posts = [reading.measured(mask) * (-1) ** (rhs & 1) for mask, rhs in reading.posts]
    generators = [posts[k] for k in lines] + [reading.stabilizer(t) for t in kernel]
    frame = stim.Tableau.from_stabilizers(generators).inverse()
    targets = np.array(
        [_unmask(reading.posts[k][1] >> 1, num_f) for k in lines], dtype=np.uint8
    ).reshape(len(lines), num_f)
    for k, (_, rhs) in enumerate(reading.posts):
        if k in independent.rows:
            continue
        image = frame(posts[k])
        x, z = image.to_numpy()
        target = (z[: len(lines)].astype(np.uint8) @ targets) % 2
        if image.sign == -1 or x.any() or (target != _unmask(rhs >> 1, num_f)).any():
            raise ValueError("the component postselects an event of probability 0")
    return frame, targets


class _Basis:
    """GF(2) vectors with distinct leading bits, each with the combination it came from."""

    def __init__(self) -> None:
        """Start empty."""
        self.rows: dict[int, tuple[int, int]] = {}

    def insert(self, vector: int, combination: int) -> int:
        """Insert a vector; return the combination that reduces it to zero, or 0."""
        vector, combination = self._reduce(vector, combination)
        if vector:
            self.rows[vector.bit_length() - 1] = (vector, combination)
            return 0
        return combination

    def solve(self, vector: int) -> int | None:
        """Return a combination of the inserted vectors that equals ``vector``, if any."""
        rest, combination = self._reduce(vector, 0)
        return None if rest else combination

    def _reduce(self, vector: int, combination: int) -> tuple[int, int]:
        while vector and (row := self.rows.get(vector.bit_length() - 1)):
            vector ^= row[0]
            combination ^= row[1]
        return vector, combination


def _with_commutant_measurements(ops: PauliOps) -> PauliOps:
    """Prepend forgotten measurements of Paulis that commute with every op.

    Such a measurement changes no outcome, and its Pauli then stays in every
    trajectory's stabilizer group, which can only lower the active width. The
    Paulis are chosen greedily, pairwise commuting, with X parts independent of
    the ones already chosen: a Pauli whose X part is in their span measures
    deterministically.
    """
    num_lines = ops.num_lines
    # NOTE: X_l anticommutes with the ops that have a Z on line l, and Z_l with
    # those that have an X, so the Paulis commuting with every op are the zero
    # combinations of these columns.
    columns, commutant = _Basis(), []
    for bit, column in enumerate([*ops.z.T, *ops.x.T]):
        if zero := columns.insert(_mask(column), 1 << bit):
            commutant.append(zero)
    chosen, x_parts = [], _Basis()
    for q in sorted(commutant, key=int.bit_count):
        x_part = q & ((1 << num_lines) - 1)
        if any(_symplectic(q, c, num_lines) for c in chosen):
            continue
        if not x_parts.insert(x_part, 1):
            chosen.append(q)
    k = len(chosen)
    return PauliOps(
        kind=np.concatenate([np.full(k, MEASUREMENT, dtype=np.uint8), ops.kind]),
        x=np.vstack([*(_unmask(q, num_lines) for q in chosen), ops.x]),
        z=np.vstack([*(_unmask(q >> num_lines, num_lines) for q in chosen), ops.z]),
        angle=np.concatenate([np.zeros(k), ops.angle]),
        sign=np.vstack([np.zeros((k, ops.sign.shape[1]), dtype=bool), ops.sign]),
        slot=np.concatenate([np.full(k, -1, dtype=np.int32), ops.slot]),
    )


def _symplectic(a: int, b: int, num_lines: int) -> int:
    full = (1 << num_lines) - 1
    return (
        (a & full & b >> num_lines).bit_count()
        + (a >> num_lines & b & full).bit_count()
    ) % 2


def _mask(bits: np.ndarray) -> int:
    return sum(1 << int(i) for i in np.flatnonzero(bits))


def _unmask(mask: int, length: int) -> np.ndarray:
    return np.array([mask >> i & 1 for i in range(length)], dtype=bool)


def _bits(mask: int) -> Iterator[int]:
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low
