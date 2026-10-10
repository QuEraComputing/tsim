from fractions import Fraction
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
from zx_param import Phase, Scalar
from zx_param.extract import ExtractedCircuit, Gate

from tsim.statevector.heisenberg import ComponentLayout, heisenberg_ops
from tsim.statevector.plan import (
    AffineRow,
    MeasureActive,
    MeasureDormant,
    Rotate,
    plan,
)
from tsim.statevector.schedule import Schedule, schedule


def _t_then_h():
    # |+⟩, T, H, measure Z: in the |0⟩ frame a rotation about X, then a
    # measurement of Z, which anticommute.
    circuit = ExtractedCircuit(
        gates=[Gate("Z", (0,), Phase(Fraction(1, 4))), Gate("H", (0,))],
        n_lines=1,
        line_start={0: None},
        line_end={0: 0},
        scalar=Scalar.one(),
        classical_ports=frozenset({0}),
    )
    return heisenberg_ops(circuit, ComponentLayout((0,), {7: 0}, (0,)))


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
    assert measure == MeasureActive(
        x=0, z=1, pivot=0, branch=1, sign=AffineRow(False, frozenset()), slot=0
    )


def test_affine_rows_xor_and_evaluate():
    row = AffineRow(True, frozenset({0, 2})) ^ AffineRow(False, frozenset({2, 3}))
    assert row == AffineRow(True, frozenset({0, 3}))
    assert row.evaluate([1, 0, 0, 0]) == 0
    assert row.evaluate([1, 0, 1, 1]) == 1


def test_a_schedule_that_is_too_narrow_is_rejected():
    ops = _t_then_h()
    with pytest.raises(RuntimeError, match="active axes"):
        plan(ops, Schedule(order=np.arange(ops.num_ops, dtype=np.int32), width=0))


def test_plans_reproduce_the_scheduled_ops():
    for ops in component_ops():
        scheduled = schedule(ops)
        planned = plan(ops, scheduled)
        reordered = ops.take(scheduled.order)
        for f in f_patterns(ops.num_f):
            assert_distributions_close(
                plan_distribution(planned, f), ops_distribution(reordered, f), atol=1e-9
            )


def test_plans_are_well_formed():
    for ops in component_ops():
        scheduled = schedule(ops)
        planned = plan(ops, scheduled)
        assert planned.width == scheduled.width
        assert planned.num_outputs == ops.num_outputs
        defined = set(range(planned.num_f))
        for action in planned.actions:
            assert action.sign.terms <= defined
            if isinstance(action, (Rotate, MeasureActive)):
                assert 0 <= action.x < 2**planned.width
                assert 0 <= action.z < 2**planned.width
            if isinstance(action, (MeasureActive, MeasureDormant)):
                assert action.branch not in defined
                defined.add(action.branch)
        assert defined == set(range(planned.num_symbols))
