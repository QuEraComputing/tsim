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
from tsim.statevector.conditioned import runner
from tsim.statevector.conditioned.plan import StepPlan, step_plan
from tsim.statevector.pbc import ROTATION, ComponentLayout, PauliOps, read_pauli_ops
from tsim.statevector.program import STABILIZER_RANK_HINT
from tsim.statevector.shared import KeyBasis, pack_bits
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
            pack_bits(self.key.sign_coords, key_words).reshape(self.ops.num_ops, key_words)
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


def conditioned_component(ops: PauliOps, layout: ComponentLayout) -> ConditionedComponent:
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
    tape = OutputTape(np.array(layout.f_selection, dtype=np.int32), output_const, output_row)
    words = -(-ops.num_lines // 64)
    return ConditionedComponent(
        output_indices=tuple(layout.output_indices),
        ops=ops,
        plan=step_plan(ops),
        tape=tape,
        key=_key_basis(ops),
        op_x=pack_bits(ops.x, words),
        op_z=pack_bits(ops.z, words),
        op_e=(np.count_nonzero(ops.x & ops.z, axis=1) % 4).astype(np.int64),
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
    shots = np.flatnonzero(alive)
    stats = np.zeros(runner.NUM_STATS, dtype=np.int64)
    if len(shots) == 0:
        return _stats(stats)
    keys = rows[shots, program.key_start : program.key_start + program.key_words]
    if program.key_words:
        _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        order = shots[np.argsort(group.reshape(-1), kind="stable")]
        group_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    else:
        order = shots.astype(np.int64)
        group_start = np.array([0, len(shots)], dtype=np.int64)
    plan = component.plan
    runner.run_chunk(
        rows,
        alive,
        first_shot,
        np.uint64(seed),
        order.astype(np.int64),
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
        MAX_WIDTH,
        stats,
    )
    return _stats(stats)


def _stats(stats: np.ndarray) -> tuple:
    return (
        0.0,
        int(stats[runner.STAT_LANES]),
        int(stats[runner.STAT_WIDTH]),
        int(stats[runner.STAT_SPLITS]),
        int(stats[runner.STAT_STEPS]),
    )
