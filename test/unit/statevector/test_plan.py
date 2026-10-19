from math import pi
from test.helpers.statevector import (
    assert_distributions_close,
    component_ops,
    f_patterns,
    ops_distribution,
    plan_distribution,
)

import numpy as np
import pytest
import stim

from tsim.statevector.heisenberg import MEASUREMENT, ROTATION, HeisenbergOps
from tsim.statevector.plan import (
    AffineRow,
    MeasureActive,
    MeasureDormant,
    Rotate,
    _Planner,
    plan,
)
from tsim.statevector.schedule import Schedule, schedule


def _t_then_h():
    # |+⟩, T, H, measure Z: in the |0⟩ frame a rotation about X, then a
    # measurement of Z, which anticommute.
    return _ops([("X", pi / 8, False, -1), ("Z", None, False, 0)])


def test_t_then_h_promotes_one_axis_and_measures_it():
    ops = _t_then_h()
    planned = plan(ops, schedule(ops))
    assert planned.width == 1
    assert planned.num_branches == 1
    rotate, measure = planned.actions
    assert isinstance(rotate, Rotate)
    assert (rotate.x, rotate.z) == (1, 0)
    assert rotate.angle == pytest.approx(pi / 8)
    assert rotate.sign == AffineRow(False, frozenset())
    assert measure == MeasureActive(x=0, z=1, pivot=0, branch=1)
    assert planned.outputs == (AffineRow(False, frozenset({1})),)


def test_affine_rows_xor_and_evaluate():
    row = AffineRow(True, frozenset({0, 2})) ^ AffineRow(False, frozenset({2, 3}))
    assert row == AffineRow(True, frozenset({0, 3}))
    assert row.evaluate([1, 0, 0, 0]) == 0
    assert row.evaluate([1, 0, 1, 1]) == 1


def test_a_schedule_that_is_too_narrow_is_rejected():
    ops = _t_then_h()
    with pytest.raises(RuntimeError, match="active axes"):
        plan(ops, Schedule(order=np.arange(ops.num_ops, dtype=np.int32), width=0))


def _checked_plan(ops):
    """Plan ``ops`` and check that the plan reproduces the scheduled ops."""
    scheduled = schedule(ops)
    planned = plan(ops, scheduled)
    reordered = ops.take(scheduled.order)
    for f in f_patterns(ops.num_f):
        assert_distributions_close(
            plan_distribution(planned, f), ops_distribution(reordered, f), atol=1e-9
        )
    return planned


def test_plans_reproduce_the_scheduled_ops():
    for ops in component_ops():
        _checked_plan(ops)


def test_plans_are_well_formed():
    for ops in component_ops():
        planned = plan(ops, schedule(ops))
        assert len(planned.outputs) == ops.num_outputs
        defined = set(range(planned.num_f))
        for action in planned.actions:
            if isinstance(action, Rotate):
                assert action.sign.terms <= defined
            if isinstance(action, (Rotate, MeasureActive)):
                assert 0 <= action.x < 2**planned.width
                assert 0 <= action.z < 2**planned.width
            if isinstance(action, (MeasureActive, MeasureDormant)):
                assert action.branch not in defined
                defined.add(action.branch)
        assert defined == set(range(planned.num_symbols))
        assert all(output.terms <= defined for output in planned.outputs)


def _ops(entries):
    """Return ops from (Pauli, angle or None for a measurement, f-sign, slot)."""
    paulis = [stim.PauliString(pauli).to_numpy() for pauli, _, _, _ in entries]
    return HeisenbergOps(
        kind=np.array(
            [MEASUREMENT if angle is None else ROTATION for _, angle, _, _ in entries],
            dtype=np.uint8,
        ),
        x=np.array([x for x, _ in paulis], dtype=bool),
        z=np.array([z for _, z in paulis], dtype=bool),
        angle=np.array([angle or 0.0 for _, angle, _, _ in entries]),
        sign=np.array([[False, f_sign] for _, _, f_sign, _ in entries], dtype=bool),
        slot=np.array([slot for _, _, _, slot in entries], dtype=np.int32),
    )


def _plan_recording_gates(monkeypatch, ops):
    """Plan ``ops`` and check it; return the plan and the applied virtual gates."""
    applied = []
    apply = _Planner._apply

    def recording(self, gate, *targets):
        applied.append((gate, targets, frozenset(self.active)))
        apply(self, gate, *targets)

    monkeypatch.setattr(_Planner, "_apply", recording)
    return _checked_plan(ops), applied


def test_a_dormant_measurement_of_y_turns_its_pivot_with_s_dag(monkeypatch):
    ops = _ops(
        [
            ("YX", None, True, 0),
            ("X_", 0.3, False, -1),
            ("_Y", 0.7, True, -1),
            ("Z_", None, False, 1),
            ("_Z", None, False, 2),
        ]
    )
    planned, applied = _plan_recording_gates(monkeypatch, ops)
    assert isinstance(planned.actions[0], MeasureDormant)
    assert any(
        gate == "S_DAG" and targets[0] not in active
        for gate, targets, active in applied
    )


def test_a_forgotten_dormant_measurement_nothing_depends_on_is_skipped(monkeypatch):
    ops = _ops(
        [
            ("X_", 0.4, True, -1),
            ("_X", None, False, -1),
            ("Z_", 0.3, False, -1),
            ("Z_", 0.2, True, -1),
            ("Y_", None, False, 0),
        ]
    )
    planned, _ = _plan_recording_gates(monkeypatch, ops)
    assert not any(isinstance(action, MeasureDormant) for action in planned.actions)
    assert planned.num_branches == 1


def test_a_diagonal_measurement_on_several_active_axes(monkeypatch):
    ops = _ops(
        [
            ("X_", 0.3, True, -1),
            ("_X", 0.5, False, -1),
            ("ZZ", None, False, 0),
            ("_Y", 0.6, True, -1),
            ("_Z", None, False, 1),
        ]
    )
    planned, applied = _plan_recording_gates(monkeypatch, ops)
    assert any(
        isinstance(action, MeasureActive)
        and action.x == 0
        and action.z.bit_count() == 2
        for action in planned.actions
    )
    assert ("CX", (1, 0), frozenset({0, 1})) in applied
