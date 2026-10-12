"""Compilation of one component for ``strategy="statevector"``.

``plan_component`` runs extraction, the Heisenberg rewrite, scheduling and
planning; ``lower`` turns the plan into the arrays of a ``StatevectorComponent``,
an action tape that ``execute.run`` interprets.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from zxparam import Graph
from zxparam.extract import extract_circuit

from tsim.core.params import ParamLayout
from tsim.statevector.heisenberg import ComponentLayout, heisenberg_ops
from tsim.statevector.plan import MeasureActive, MeasureDormant, Plan, Rotate, plan
from tsim.statevector.schedule import schedule

ROTATE_DIAG = 0
ROTATE = 1
MEASURE_DIAG = 2
MEASURE = 3
MEASURE_DORMANT = 4
RECORD = 5

MAX_WIDTH = 30

_HINT = "Use a stabilizer-rank strategy such as strategy='cat5' for this circuit."


class StatevectorComponent(eqx.Module):
    """A component compiled for ``strategy="statevector"``: an action tape.

    Attributes:
        output_indices: Global output index of each output slot.
        width: Number of array axes W; lanes hold ``2^W`` amplitudes.
        num_words: Number of uint32 words that hold a lane's symbols.
        num_outputs: Number of output slots.
        dtype: ``"complex64"`` or ``"complex128"``.
        f_selection: Global f-bit index of each component f-bit.
        kind: Kernel of each action (``ROTATE_DIAG`` … ``RECORD``).
        x_mask: X mask of the action's Pauli on the array axes.
        z_mask: Z mask of the action's Pauli on the array axes.
        cos: cos θ of each rotation.
        sin: sin θ of each rotation.
        pivot: Axis an active measurement resets to ``|0⟩``.
        branch: Symbol a measurement sets, -1 if none.
        slot: Output slot an action writes, -1 if none.
        sign_const: Constant of each action's sign.
        sign_row: Symbol bits of each action's sign, shape ``(A, num_words)``.

    """

    output_indices: tuple[int, ...] = eqx.field(static=True)
    width: int = eqx.field(static=True)
    num_words: int = eqx.field(static=True)
    num_outputs: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    f_selection: Array
    kind: Array
    x_mask: Array
    z_mask: Array
    cos: Array
    sin: Array
    pivot: Array
    branch: Array
    slot: Array
    sign_const: Array
    sign_row: Array

    @property
    def num_actions(self) -> int:
        """Return the number of actions on the tape."""
        return int(self.kind.shape[0])


def plan_component(
    component: Graph, layout: ParamLayout
) -> tuple[Plan, ComponentLayout]:
    """Extract, rewrite, schedule and plan one non-direct component.

    Raises:
        ValueError: If extraction fails or the plan needs more than
            ``MAX_WIDTH`` array axes.

    """
    component_layout = ComponentLayout.of(component, layout)
    outputs = list(component_layout.output_indices)
    try:
        circuit = extract_circuit(component)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            "strategy='statevector' could not extract a circuit from the component "
            f"with outputs {outputs}: {error}. {_HINT}"
        ) from error
    ops = heisenberg_ops(circuit, component_layout)
    scheduled = schedule(ops)
    if scheduled.width > MAX_WIDTH:
        raise ValueError(
            f"strategy='statevector' needs {scheduled.width} active qubits for the "
            f"component with outputs {outputs}, more than {MAX_WIDTH}. {_HINT}"
        )
    return plan(ops, scheduled), component_layout


def resolve_dtype(dtype: str | None) -> str:
    """Return the amplitude dtype: complex128 if JAX has 64-bit types, else complex64.

    Raises:
        ValueError: If ``dtype`` is unknown, or complex128 without
            ``jax_enable_x64``.

    """
    x64 = jax.dtypes.canonicalize_dtype(np.complex128) == np.dtype(np.complex128)
    if dtype is None:
        return "complex128" if x64 else "complex64"
    if dtype not in ("complex64", "complex128"):
        raise ValueError(f"dtype must be 'complex64' or 'complex128', got {dtype!r}")
    if dtype == "complex128" and not x64:
        raise ValueError("complex128 amplitudes need jax_enable_x64")
    return dtype


def lower(plan: Plan, layout: ComponentLayout, dtype: str) -> StatevectorComponent:
    """Return the action tape of ``plan``."""
    num_actions = len(plan.actions)
    num_words = max(1, -(-plan.num_symbols // 32))
    kind = np.zeros(num_actions, dtype=np.int32)
    x_mask = np.zeros(num_actions, dtype=np.uint32)
    z_mask = np.zeros(num_actions, dtype=np.uint32)
    angle = np.zeros(num_actions, dtype=np.float64)
    pivot = np.zeros(num_actions, dtype=np.int32)
    branch = np.full(num_actions, -1, dtype=np.int32)
    slot = np.full(num_actions, -1, dtype=np.int32)
    sign_const = np.zeros(num_actions, dtype=bool)
    sign_row = np.zeros((num_actions, num_words), dtype=np.uint32)
    for a, action in enumerate(plan.actions):
        if isinstance(action, Rotate):
            kind[a] = ROTATE if action.x else ROTATE_DIAG
            x_mask[a], z_mask[a], angle[a] = action.x, action.z, action.angle
        elif isinstance(action, MeasureActive):
            kind[a] = MEASURE if action.x else MEASURE_DIAG
            x_mask[a], z_mask[a] = action.x, action.z
            pivot[a], branch[a], slot[a] = action.pivot, action.branch, action.slot
        elif isinstance(action, MeasureDormant):
            kind[a] = MEASURE_DORMANT
            branch[a], slot[a] = action.branch, action.slot
        else:
            kind[a] = RECORD
            slot[a] = action.slot
        sign_const[a] = action.sign.constant
        for term in action.sign.terms:
            sign_row[a, term // 32] |= np.uint32(1) << np.uint32(term % 32)
    real = np.float32 if dtype == "complex64" else np.float64
    return StatevectorComponent(
        output_indices=layout.output_indices,
        width=plan.width,
        num_words=num_words,
        num_outputs=plan.num_outputs,
        dtype=dtype,
        f_selection=jnp.asarray(np.array(layout.f_selection, dtype=np.int32)),
        kind=jnp.asarray(kind),
        x_mask=jnp.asarray(x_mask),
        z_mask=jnp.asarray(z_mask),
        cos=jnp.asarray(np.cos(angle).astype(real)),
        sin=jnp.asarray(np.sin(angle).astype(real)),
        pivot=jnp.asarray(pivot),
        branch=jnp.asarray(branch),
        slot=jnp.asarray(slot),
        sign_const=jnp.asarray(sign_const),
        sign_row=jnp.asarray(sign_row),
    )


def compile_statevector_component(
    component: Graph, layout: ParamLayout, *, dtype: str | None = None
) -> StatevectorComponent:
    """Compile one non-direct component for ``strategy="statevector"``.

    Args:
        component: A connected component of the prepared sampling graph.
        layout: The prepared graph's parameter layout.
        dtype: Amplitude dtype; ``None`` picks complex128 when JAX has 64-bit
            types enabled, else complex64.

    """
    resolved = resolve_dtype(dtype)
    planned, component_layout = plan_component(component, layout)
    return lower(planned, component_layout, resolved)
