"""A component compiled for ``strategy="conditioned_statevector"``, and its chunk runs.

The component keeps its Pauli ops (in the frame of the initial ``|0...0>``) and
their static step plan. The noise reaches it in two ways, as for
``active_statevector``:

- the signs of the rotations read parities of the f-bits, the *key*, which the
  noise sampler writes into each shot's words (``KeyBasis``);
- a measurement's sign only flips its recorded bit, so its f-part and constant go
  into the output (``OutputTape``), and the runner XORs in the outcome of the
  unsigned Pauli.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np
from zxparam import Graph

from tsim.core.params import ParamLayout
from tsim.statevector.conditioned import lane, runner
from tsim.statevector.conditioned.plan import StepPlan, step_plan
from tsim.statevector.pbc import ROTATION, ComponentLayout, PauliOps, read_pauli_ops
from tsim.statevector.program import STABILIZER_RANK_HINT
from tsim.statevector.shared import KeyBasis, pack_bits
from tsim.utils.kernels import run_tasks
from tsim.utils.linalg import find_basis

# NOTE: the most active qubits a lane may hold: 2^26 amplitudes are 1 GiB.
MAX_WIDTH = 26


class OutputTape(NamedTuple):
    """What the shot words need of a component's outputs.

    Attributes:
        f_selection: Global f-bit index of each component f-bit.
        output_const: Constant of each output slot.
        output_row: f-part of each output slot, ``(num_outputs, num_f)``.

    """

    f_selection: np.ndarray
    output_const: np.ndarray
    output_row: np.ndarray


class ConditionedProgram(NamedTuple):
    """A component's per-op data for one postselection mask.

    Attributes:
        key_start: First shot word of the component's key.
        key_words: Number of words of the key.
        sign_key: Each op's sign as key bits, ``(N, key_words)``.
        out_bit: Each measurement's output bit in the shot words, or -1.
        post: Whether each measurement's output is postselected.

    """

    key_start: int
    key_words: int
    sign_key: np.ndarray
    out_bit: np.ndarray
    post: np.ndarray


@dataclass(frozen=True, eq=False)
class ConditionedComponent:
    """A component compiled for ``strategy="conditioned_statevector"``.

    Attributes:
        output_indices: Global output index of each output slot.
        ops: The component's Pauli ops.
        plan: Their static step plan.
        tape: The outputs' constants and f-parts.
        key: The key basis of the rotation signs.
        op_x, op_z: Each op's Pauli as X and Z bit words, ``(N, W)``.
        op_e: The phase exponent of each op's letters (its number of Y), mod 4.
        root_amp: Per step, the reference key's certain outcome at the root state
            (+1 or -1), or 0 (``runner.reference_prefix``).
        width: Nominal lane width, for chunk sizing only: lanes size themselves.

    """

    output_indices: tuple[int, ...]
    ops: PauliOps
    plan: StepPlan
    tape: OutputTape
    key: KeyBasis
    op_x: np.ndarray
    op_z: np.ndarray
    op_e: np.ndarray
    root_amp: np.ndarray
    width: int = 10

    norm_warning_threshold: ClassVar[float] = 1e-9

    @property
    def num_outputs(self) -> int:
        """Return the number of output slots."""
        return len(self.output_indices)

    @property
    def num_actions(self) -> int:
        """Return the number of ops."""
        return self.ops.num_ops

    @property
    def num_lines(self) -> int:
        """Return the number of qubit lines."""
        return self.ops.num_lines

    def program(self, key_start: int, postselected: np.ndarray) -> ConditionedProgram:
        """Return the per-op data for a postselection mask over the output slots."""
        key_words = -(-len(self.key.rows) // 64)
        sign_key = (
            pack_bits(self.key.sign_coords, key_words).reshape(
                self.ops.num_ops, key_words
            )
            if key_words
            else np.zeros((self.ops.num_ops, 0), dtype=np.uint64)
        )
        out_bit = np.full(self.ops.num_ops, -1, dtype=np.int64)
        post = np.zeros(self.ops.num_ops, dtype=np.bool_)
        recorded = np.flatnonzero(self.ops.slot >= 0)
        slots = self.ops.slot[recorded]
        out_bit[recorded] = np.asarray(self.output_indices, dtype=np.int64)[slots]
        post[recorded] = np.asarray(postselected, dtype=bool)[slots]
        return ConditionedProgram(key_start, key_words, sign_key, out_bit, post)


def compile_conditioned_component(
    component: Graph, output_indices: Sequence[int], layout: ParamLayout
) -> ConditionedComponent:
    """Compile one non-direct component for ``strategy="conditioned_statevector"``."""
    component_layout = ComponentLayout.of(component, output_indices, layout)
    try:
        ops = read_pauli_ops(component, component_layout)
    except ValueError as error:
        raise ValueError(
            "strategy='conditioned_statevector' could not read the component with "
            f"outputs {list(output_indices)}: {error}. {STABILIZER_RANK_HINT}"
        ) from error
    return conditioned_component(ops, component_layout)


def conditioned_component(
    ops: PauliOps, layout: ComponentLayout
) -> ConditionedComponent:
    """Return the conditioned component of ``ops``."""
    num_f = ops.num_f
    sign_f = ops.sign[:, 1:].astype(bool)
    num_slots = len(layout.output_indices)
    output_const = np.zeros(num_slots, dtype=bool)
    output_row = np.zeros((num_slots, num_f), dtype=bool)
    for j in np.flatnonzero(ops.slot >= 0):
        output_const[ops.slot[j]] = bool(ops.sign[j, 0])
        output_row[ops.slot[j]] = sign_f[j]
    assert np.count_nonzero(ops.slot >= 0) == num_slots
    tape = OutputTape(
        np.array(layout.f_selection, dtype=np.int32), output_const, output_row
    )
    words = -(-ops.num_lines // 64)
    plan = step_plan(ops)
    op_x = pack_bits(ops.x, words)
    op_z = pack_bits(ops.z, words)
    op_e = (np.count_nonzero(ops.x & ops.z, axis=1) % 4).astype(np.int64)
    root_amp = runner.reference_prefix(
        ops.num_lines,
        op_x,
        op_z,
        op_e,
        ops.angle.astype(np.float64),
        ops.sign[:, 0].astype(np.int64),
        plan.measure,
        plan.apply_start,
        plan.layer_start,
        plan.layer_ops,
    )
    return ConditionedComponent(
        output_indices=tuple(layout.output_indices),
        ops=ops,
        plan=plan,
        tape=tape,
        key=_key_basis(ops),
        op_x=op_x,
        op_z=op_z,
        op_e=op_e,
        root_amp=root_amp,
    )


def _key_basis(ops: PauliOps) -> KeyBasis:
    """Return a basis of the f-parts of the rotation signs, and every op's coordinates in it.

    Measurement signs only flip outputs, so their coordinates are zero.
    """
    num_f = ops.num_f
    signs = ops.sign[:, 1:].astype(bool) & (ops.kind == ROTATION)[:, None]
    nonzero = signs.any(axis=1)
    if not nonzero.any():
        return KeyBasis(np.zeros((0, num_f), bool), np.zeros((ops.num_ops, 0), bool))
    rows, _ = find_basis(signs[nonzero].astype(np.uint8))
    rows = np.asarray(rows, dtype=bool)
    _, transform = find_basis(np.vstack([rows, signs]).astype(np.uint8))
    coords = np.asarray(transform, dtype=bool)[len(rows) :]
    return KeyBasis(rows, coords)


def run_chunk(
    component: ConditionedComponent,
    program: ConditionedProgram,
    rows: np.ndarray,
    alive: np.ndarray,
    first_shot: int,
    seed: np.uint64,
) -> tuple:
    """Run the alive shots of ``rows`` through the component; return its stats.

    In the layout of ``shared.run_lanes``' stats: the normalization deviation
    (projections renormalize, so 0), then the lanes, the widest lane, the splits
    and the steps run.
    """
    stats = np.zeros(runner.NUM_STATS, dtype=np.int64)
    order, group_start = runner.group_by_key(
        rows, alive, program.key_start, program.key_words
    )
    if len(order) == 0:
        return _stats(stats)
    try:
        _run_chunk_kernel(
            component, program, rows, alive, first_shot, seed, order, group_start, stats
        )
    except lane.TooWide:
        raise ValueError(
            _TOO_WIDE.format(outputs=list(component.output_indices))
        ) from None
    return _stats(stats)


def _run_chunk_kernel(
    component, program, rows, alive, first_shot, seed, order, group_start, stats
):
    plan = component.plan
    runner.run_chunk(
        rows,
        alive,
        first_shot,
        np.uint64(seed),
        order,
        group_start,
        program.key_start,
        program.key_words,
        component.num_lines,
        component.op_x,
        component.op_z,
        component.op_e,
        component.ops.angle.astype(np.float64),
        component.ops.sign[:, 0].astype(np.int64),
        program.sign_key,
        program.out_bit,
        program.post,
        plan.measure,
        plan.apply_start,
        plan.apply_ops,
        plan.layer_start,
        plan.layer_ops,
        component.root_amp,
        MAX_WIDTH,
        stats,
        _NO_TRACK,
        _NO_PROB,
    )


def run_part(
    component: ConditionedComponent,
    program: ConditionedProgram,
    words: np.ndarray,
    alive: np.ndarray,
    first_shot: int,
    seed: np.uint64,
    threads: int,
) -> tuple:
    """Run the alive shots of a whole part through the component, a key's shots on one thread.

    Shots are grouped by key over the part. A group larger than ``_PIECE_SHOTS``
    is cut into at most ``threads`` pieces (each builds its own lanes, which costs
    about as much as recording a few thousand shots), and the pieces are dealt to
    ``8 * threads`` tasks, largest first, each to the task with the least work so far.
    """
    order, starts = runner.group_by_key(
        words, alive, program.key_start, program.key_words
    )
    stats = np.zeros(runner.NUM_STATS, dtype=np.int64)
    if len(order) == 0:
        return _stats(stats)
    pieces = []
    for g in range(len(starts) - 1):
        lo, hi = int(starts[g]), int(starts[g + 1])
        count = min(-(-(hi - lo) // _PIECE_SHOTS), threads)
        for i in range(count):
            pieces.append(
                (lo + (hi - lo) * i // count, lo + (hi - lo) * (i + 1) // count)
            )
    sizes = np.array([hi - lo for lo, hi in pieces])
    num_tasks = max(1, min(8 * threads, len(pieces)))
    load = np.zeros(num_tasks)
    members: list[list[int]] = [[] for _ in range(num_tasks)]
    for g in np.argsort(-sizes, kind="stable"):
        t = int(np.argmin(load))
        members[t].append(int(g))
        load[t] += _GROUP_COST + sizes[g]
    plan = component.plan

    def task(groups: list[int]) -> np.ndarray:
        task_stats = np.zeros(runner.NUM_STATS, dtype=np.int64)
        if not groups:
            return task_stats
        groups = sorted(groups)
        sub_order = np.concatenate([order[pieces[g][0] : pieces[g][1]] for g in groups])
        sub_starts = np.concatenate([[0], np.cumsum(sizes[groups])]).astype(np.int64)
        runner.run_chunk(
            words,
            alive,
            first_shot,
            np.uint64(seed),
            sub_order,
            sub_starts,
            program.key_start,
            program.key_words,
            component.num_lines,
            component.op_x,
            component.op_z,
            component.op_e,
            component.ops.angle.astype(np.float64),
            component.ops.sign[:, 0].astype(np.int64),
            program.sign_key,
            program.out_bit,
            program.post,
            plan.measure,
            plan.apply_start,
            plan.apply_ops,
            plan.layer_start,
            plan.layer_ops,
            component.root_amp,
            MAX_WIDTH,
            task_stats,
            _NO_TRACK,
            _NO_PROB,
        )
        return task_stats

    try:
        results = run_tasks(task, members, threads)
    except lane.TooWide:
        raise ValueError(
            _TOO_WIDE.format(outputs=list(component.output_indices))
        ) from None
    for task_stats in results:
        stats[runner.STAT_WIDTH] = max(
            stats[runner.STAT_WIDTH], task_stats[runner.STAT_WIDTH]
        )
        for i in (
            runner.STAT_LANES,
            runner.STAT_SPLITS,
            runner.STAT_THROUGH,
            runner.STAT_STEPS,
        ):
            stats[i] += task_stats[i]
    return _stats(stats)


# NOTE: a group's fixed cost in shots, for dealing groups to tasks: a lane's
# steps cost about as much as recording a few thousand shots. A group of more
# than _PIECE_SHOTS shots is cut, so that one key cannot hold up a part; its
# pieces rebuild its lanes, a cost of the order of _GROUP_COST shots each.
_GROUP_COST = 2000
_TOO_WIDE = (
    f"strategy='conditioned_statevector' needs more than {MAX_WIDTH} active qubits for a "
    "lane of the component with outputs {outputs}. Such lanes are rare noise patterns "
    "of shots that a postselection_mask would discard; sample with a "
    "postselection_mask, or use strategy='active_statevector'."
)
_NO_TRACK = np.zeros(0, dtype=np.bool_)
_NO_PROB = np.zeros(0, dtype=np.float64)
_PIECE_SHOTS = 1 << 17


def _stats(stats: np.ndarray) -> tuple:
    return (
        0.0,
        int(stats[runner.STAT_LANES]),
        int(stats[runner.STAT_WIDTH]),
        int(stats[runner.STAT_SPLITS]),
        int(stats[runner.STAT_STEPS]),
    )
