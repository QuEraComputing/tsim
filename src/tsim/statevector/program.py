"""Compilation of one component for ``strategy="active_statevector"``.

``plan_component`` reads the Pauli ops off the component, schedules and
plans them; ``lower`` turns the plan into the NumPy arrays of a
``StatevectorComponent``: an action tape that ``shared.run_lanes`` runs, and the
outputs as parities of the symbols the tape leaves behind.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np
from zxparam import Graph

from tsim.core.params import ParamLayout
from tsim.statevector.pbc import ComponentLayout, read_pauli_ops
from tsim.statevector.plan import MeasureActive, Plan, Rotate, plan
from tsim.statevector.schedule import schedule

ROTATE_DIAG = 0
ROTATE = 1
MEASURE_DIAG = 2
MEASURE = 3
MEASURE_DORMANT = 4

MAX_WIDTH = 30

STABILIZER_RANK_HINT = (
    "Use a stabilizer-rank strategy such as strategy='cat5' for this circuit."
)


class Tape(NamedTuple):
    """A component's actions and outputs as NumPy arrays, one row per action.

    Attributes:
        f_selection: Global f-bit index of each component f-bit.
        kind: Kernel of each action (``ROTATE_DIAG`` … ``MEASURE_DORMANT``).
        x_mask: X mask of the action's Pauli on the array axes.
        z_mask: Z mask of the action's Pauli on the array axes.
        cos: cos θ of each rotation.
        sin: sin θ of each rotation.
        pivot: Axis an active measurement resets to ``|0⟩``.
        extent: One plus the highest axis the action can touch: amplitudes at
            ``2^extent`` and above are zero before and after it.
        branch: Symbol a measurement sets, -1 for a rotation.
        sign_const: Constant of each rotation's sign.
        sign_row: Symbols of each rotation's sign, shape ``(A, num_symbols)``.
        output_const: Constant of each output, shape ``(num_outputs,)``.
        output_row: Symbols of each output, shape ``(num_outputs, num_symbols)``.

    """

    f_selection: np.ndarray
    kind: np.ndarray
    x_mask: np.ndarray
    z_mask: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    pivot: np.ndarray
    extent: np.ndarray
    branch: np.ndarray
    sign_const: np.ndarray
    sign_row: np.ndarray
    output_const: np.ndarray
    output_row: np.ndarray


@dataclass(frozen=True, eq=False)
class StatevectorComponent:
    """A component compiled for ``strategy="active_statevector"``.

    Its lanes hold ``2^width`` complex128 amplitudes.

    Attributes:
        output_indices: Global output index of each output slot.
        width: Number of array axes W.
        tape: The actions and outputs.

    """

    output_indices: tuple[int, ...]
    width: int
    tape: Tape

    # NOTE: outcome probabilities are normalized by the measured weights, so
    # only drift beyond rounding warns.
    norm_warning_threshold: ClassVar[float] = 1e-9
    """Largest normalization deviation that sampling tolerates without a warning."""

    @property
    def num_actions(self) -> int:
        """Return the number of actions on the tape."""
        return int(self.tape.kind.shape[0])

    @property
    def num_symbols(self) -> int:
        """Return the number of f-bit and branch symbols the tape reads."""
        return int(self.tape.sign_row.shape[1])

    @property
    def num_outputs(self) -> int:
        """Return the number of output slots."""
        return len(self.output_indices)


def plan_component(
    component: Graph, output_indices: Sequence[int], layout: ParamLayout
) -> tuple[Plan, ComponentLayout]:
    """Read the Pauli ops of one non-direct component, schedule and plan them.

    Raises:
        ValueError: If the component cannot be read as Pauli ops or the
            plan needs more than ``MAX_WIDTH`` array axes.

    """
    component_layout = ComponentLayout.of(component, output_indices, layout)
    outputs = list(output_indices)
    try:
        ops = read_pauli_ops(component, component_layout)
    except ValueError as error:
        raise ValueError(
            "strategy='active_statevector' could not read the component with outputs "
            f"{outputs}: {error}. {STABILIZER_RANK_HINT}"
        ) from error
    scheduled = schedule(ops)
    if scheduled.width > MAX_WIDTH:
        raise ValueError(
            f"strategy='active_statevector' needs {scheduled.width} active qubits for the "
            f"component with outputs {outputs}, more than {MAX_WIDTH}. "
            f"{STABILIZER_RANK_HINT}"
        )
    return plan(ops, scheduled), component_layout


def lower(plan: Plan, layout: ComponentLayout) -> StatevectorComponent:
    """Return the action tape of ``plan``."""
    num_actions = len(plan.actions)
    kind = np.zeros(num_actions, dtype=np.int32)
    x_mask = np.zeros(num_actions, dtype=np.uint32)
    z_mask = np.zeros(num_actions, dtype=np.uint32)
    angle = np.zeros(num_actions, dtype=np.float64)
    pivot = np.zeros(num_actions, dtype=np.int32)
    extent = np.zeros(num_actions, dtype=np.int32)
    branch = np.full(num_actions, -1, dtype=np.int32)
    sign_const = np.zeros(num_actions, dtype=bool)
    sign_row = np.zeros((num_actions, plan.num_symbols), dtype=bool)
    active = 0
    for a, action in enumerate(plan.actions):
        if isinstance(action, Rotate):
            kind[a] = ROTATE if action.x else ROTATE_DIAG
            x_mask[a], z_mask[a], angle[a] = action.x, action.z, action.angle
            sign_const[a] = action.sign.constant
            sign_row[a, list(action.sign.terms)] = True
            active |= action.x | action.z
        elif isinstance(action, MeasureActive):
            kind[a] = MEASURE if action.x else MEASURE_DIAG
            x_mask[a], z_mask[a] = action.x, action.z
            pivot[a], branch[a] = action.pivot, action.branch
        else:
            kind[a] = MEASURE_DORMANT
            branch[a] = action.branch
        extent[a] = active.bit_length()
        if isinstance(action, MeasureActive):
            active &= ~(1 << action.pivot)
    output_row = np.zeros((len(plan.outputs), plan.num_symbols), dtype=bool)
    for slot, output in enumerate(plan.outputs):
        output_row[slot, list(output.terms)] = True
    tape = Tape(
        f_selection=np.array(layout.f_selection, dtype=np.int32),
        kind=kind,
        x_mask=x_mask,
        z_mask=z_mask,
        cos=np.cos(angle),
        sin=np.sin(angle),
        pivot=pivot,
        extent=extent,
        branch=branch,
        sign_const=sign_const,
        sign_row=sign_row,
        output_const=np.array([output.constant for output in plan.outputs], bool),
        output_row=output_row,
    )
    return StatevectorComponent(layout.output_indices, plan.width, tape)


def compile_statevector_component(
    component: Graph, output_indices: Sequence[int], layout: ParamLayout
) -> StatevectorComponent:
    """Compile one non-direct component for ``strategy="active_statevector"``.

    Args:
        component: A connected component of the prepared sampling graph.
        output_indices: Global output index of each of the component's outputs.
        layout: The prepared graph's parameter layout.

    """
    planned, component_layout = plan_component(component, output_indices, layout)
    return lower(planned, component_layout)
