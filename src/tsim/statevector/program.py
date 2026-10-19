"""Compilation of one component for ``strategy="active_statevector"``.

``plan_component`` reads the Pauli ops off the component, schedules and
plans them; ``lower`` turns the plan into the arrays of a ``StatevectorComponent``:
an action tape that ``execute.run`` interprets, and the outputs as parities of
the symbols the tape leaves behind.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import ceil

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
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

# NOTE: outcome probabilities are normalized by the measured weights, so only
# drift beyond rounding (~1e-7 per rotation in complex64) warns.
_NORM_WARNING_THRESHOLDS = {"complex64": 1e-3, "complex128": 1e-9}

STABILIZER_RANK_HINT = (
    "Use a stabilizer-rank strategy such as strategy='cat5' for this circuit."
)


class StatevectorComponent(eqx.Module):
    """A component compiled for ``strategy="active_statevector"``: an action tape.

    Attributes:
        output_indices: Global output index of each output slot.
        width: Number of array axes W; lanes hold ``2^W`` amplitudes.
        dtype: ``"complex64"`` or ``"complex128"``.
        f_selection: Global f-bit index of each component f-bit.
        kind: Kernel of each action (``ROTATE_DIAG`` … ``MEASURE_DORMANT``).
        x_mask: X mask of the action's Pauli on the array axes.
        z_mask: Z mask of the action's Pauli on the array axes.
        cos: cos θ of each rotation.
        sin: sin θ of each rotation.
        pivot: Axis an active measurement resets to ``|0⟩``.
        branch: Symbol a measurement sets, -1 for a rotation.
        sign_const: Constant of each rotation's sign.
        sign_row: Symbol bits of each rotation's sign, shape ``(A, num_words)``.
        output_const: Constant of each output, shape ``(num_outputs,)``.
        output_row: Symbol bits of each output, shape ``(num_outputs, num_words)``.

    """

    output_indices: tuple[int, ...] = eqx.field(static=True)
    width: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    f_selection: Array
    kind: Array
    x_mask: Array
    z_mask: Array
    cos: Array
    sin: Array
    pivot: Array
    branch: Array
    sign_const: Array
    sign_row: Array
    output_const: Array
    output_row: Array

    @property
    def num_actions(self) -> int:
        """Return the number of actions on the tape."""
        return int(self.kind.shape[0])

    @property
    def num_words(self) -> int:
        """Return the number of uint32 words that hold a lane's symbols."""
        return int(self.sign_row.shape[1])

    @property
    def num_outputs(self) -> int:
        """Return the number of output slots."""
        return len(self.output_indices)

    @property
    def norm_warning_threshold(self) -> float:
        """Return the largest normalization deviation that passes without a warning."""
        return _NORM_WARNING_THRESHOLDS[self.dtype]


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


def _pack(terms: Iterable[int], num_words: int) -> np.ndarray:
    """Return symbol bits as uint32 words: symbol s is bit s % 32 of word s // 32."""
    row = np.zeros(num_words, dtype=np.uint32)
    for term in terms:
        row[term // 32] |= np.uint32(1) << np.uint32(term % 32)
    return row


def lower(plan: Plan, layout: ComponentLayout, dtype: str) -> StatevectorComponent:
    """Return the action tape of ``plan``."""
    num_actions = len(plan.actions)
    num_words = max(1, ceil(plan.num_symbols / 32))
    kind = np.zeros(num_actions, dtype=np.int32)
    x_mask = np.zeros(num_actions, dtype=np.uint32)
    z_mask = np.zeros(num_actions, dtype=np.uint32)
    angle = np.zeros(num_actions, dtype=np.float64)
    pivot = np.zeros(num_actions, dtype=np.int32)
    branch = np.full(num_actions, -1, dtype=np.int32)
    sign_const = np.zeros(num_actions, dtype=bool)
    sign_row = np.zeros((num_actions, num_words), dtype=np.uint32)
    for a, action in enumerate(plan.actions):
        if isinstance(action, Rotate):
            kind[a] = ROTATE if action.x else ROTATE_DIAG
            x_mask[a], z_mask[a], angle[a] = action.x, action.z, action.angle
            sign_const[a] = action.sign.constant
            sign_row[a] = _pack(action.sign.terms, num_words)
        elif isinstance(action, MeasureActive):
            kind[a] = MEASURE if action.x else MEASURE_DIAG
            x_mask[a], z_mask[a] = action.x, action.z
            pivot[a], branch[a] = action.pivot, action.branch
        else:
            kind[a] = MEASURE_DORMANT
            branch[a] = action.branch
    output_row = np.zeros((len(plan.outputs), num_words), dtype=np.uint32)
    for slot, output in enumerate(plan.outputs):
        output_row[slot] = _pack(output.terms, num_words)
    real = np.finfo(dtype).dtype
    return StatevectorComponent(
        output_indices=layout.output_indices,
        width=plan.width,
        dtype=dtype,
        f_selection=jnp.asarray(np.array(layout.f_selection, dtype=np.int32)),
        kind=jnp.asarray(kind),
        x_mask=jnp.asarray(x_mask),
        z_mask=jnp.asarray(z_mask),
        cos=jnp.asarray(np.cos(angle).astype(real)),
        sin=jnp.asarray(np.sin(angle).astype(real)),
        pivot=jnp.asarray(pivot),
        branch=jnp.asarray(branch),
        sign_const=jnp.asarray(sign_const),
        sign_row=jnp.asarray(sign_row),
        output_const=jnp.asarray([output.constant for output in plan.outputs], bool),
        output_row=jnp.asarray(output_row),
    )


def compile_statevector_component(
    component: Graph, output_indices: Sequence[int], layout: ParamLayout
) -> StatevectorComponent:
    """Compile one non-direct component for ``strategy="active_statevector"``.

    Amplitudes are complex128 when JAX has 64-bit types enabled, else complex64.

    Args:
        component: A connected component of the prepared sampling graph.
        output_indices: Global output index of each of the component's outputs.
        layout: The prepared graph's parameter layout.

    """
    planned, component_layout = plan_component(component, output_indices, layout)
    dtype = jax.dtypes.canonicalize_dtype(np.complex128).name
    return lower(planned, component_layout, dtype)
