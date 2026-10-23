from test.helpers.statevector import (
    assert_distributions_close,
    component_ops,
    f_patterns,
    ops_distribution,
)

import numpy as np
import pytest

from tsim.statevector.pbc import MEASUREMENT, ROTATION, PauliOps
from tsim.statevector.schedule import (
    StabilizerGroup,
    _ancestors,
    anticommutation,
    schedule,
)


def _bits(*rows):
    return np.array(rows, dtype=bool)


def _two_expansions_then_two_measurements():
    # Rotate about X0, rotate about X1, measure Z0, measure Z1.
    return PauliOps(
        kind=np.array([ROTATION, ROTATION, MEASUREMENT, MEASUREMENT], dtype=np.uint8),
        x=_bits([1, 0], [0, 1], [0, 0], [0, 0]),
        z=_bits([0, 0], [0, 0], [1, 0], [0, 1]),
        angle=np.array([0.3, 0.3, 0.0, 0.0]),
        sign=np.zeros((4, 1), dtype=bool),
        slot=np.array([-1, -1, 0, 1], dtype=np.int32),
    )


def test_stabilizer_group_tracks_the_width():
    group = StabilizerGroup(2)
    assert group.width == 0
    group.rotate(np.array([True, False]), np.array([False, False]))
    assert group.width == 1
    group.rotate(np.array([False, False]), np.array([True, False]))
    assert group.width == 1
    group.measure(np.array([True, False]), np.array([False, False]))
    assert group.width == 0
    group.measure(np.array([False, False]), np.array([True, False]))
    assert group.width == 0
    assert group.contains(np.array([False, False]), np.array([True, False]))


def test_anticommutation_finds_the_pairs():
    anti = anticommutation(_two_expansions_then_two_measurements())
    assert anti[0, 2] and anti[2, 0] and anti[1, 3] and anti[3, 1]
    assert not anti[0, 1] and not anti[0, 3] and not anti[2, 3]


def test_measurements_run_before_unrelated_expansions():
    scheduled = schedule(_two_expansions_then_two_measurements())
    assert scheduled.order.tolist() == [0, 2, 1, 3]
    assert scheduled.width == 1


def _measurement_after_two_expansions_or_after_one(lower_first: bool):
    # Rotate about X0, X1 and X2, then measure Z0 Z1 (peak 2) and Z2 (peak 1).
    rotations = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    measurements = [[0, 0, 1], [1, 1, 0]]
    if not lower_first:
        rotations, measurements = rotations[1:] + rotations[:1], measurements[::-1]
    zeros = [[0, 0, 0]] * 2
    return PauliOps(
        kind=np.array([ROTATION] * 3 + [MEASUREMENT] * 2, dtype=np.uint8),
        x=_bits(*rotations, *zeros),
        z=_bits(*[[0, 0, 0]] * 3, *measurements),
        angle=np.array([0.3, 0.3, 0.3, 0.0, 0.0]),
        sign=np.zeros((5, 1), dtype=bool),
        slot=np.array([-1, -1, -1, 0, 1], dtype=np.int32),
    )


def test_widening_runs_the_measurement_with_the_lowest_peak_first():
    lower_first = schedule(_measurement_after_two_expansions_or_after_one(True))
    assert lower_first.order.tolist() == [0, 3, 1, 2, 4]
    assert lower_first.width == 2
    lower_second = schedule(_measurement_after_two_expansions_or_after_one(False))
    assert lower_second.order.tolist() == [2, 4, 0, 1, 3]
    assert lower_second.width == 2


def test_schedules_keep_every_anticommuting_pair_in_order():
    for ops in component_ops():
        order = schedule(ops).order
        assert sorted(order.tolist()) == list(range(ops.num_ops))
        position = np.empty(ops.num_ops, dtype=int)
        position[order] = np.arange(ops.num_ops)
        first, second = np.nonzero(np.triu(anticommutation(ops), 1))
        assert np.all(position[first] < position[second])


def test_schedules_preserve_the_distribution():
    for ops in component_ops():
        reordered = ops.take(schedule(ops).order)
        for f in f_patterns(ops.num_f):
            assert_distributions_close(
                ops_distribution(reordered, f), ops_distribution(ops, f)
            )


def test_width_is_the_largest_structural_width_of_the_order():
    for ops in component_ops():
        scheduled = schedule(ops)
        group = StabilizerGroup(ops.num_lines)
        widths = [0]
        for j in scheduled.order:
            run = group.measure if ops.kind[j] == MEASUREMENT else group.rotate
            run(ops.x[j], ops.z[j])
            widths.append(group.width)
        assert scheduled.width == max(widths)


@pytest.mark.parametrize("num_ops", [0, 1, 63, 64, 65, 150])
def test_ancestors_are_the_transitive_predecessors(num_ops):
    rng = np.random.default_rng(num_ops)
    before = np.triu(rng.random((num_ops, num_ops)) < 0.05, 1)
    expected = np.zeros((num_ops, num_ops), dtype=bool)
    for j in range(num_ops):
        for i in np.flatnonzero(before[:, j]):
            expected[j] |= expected[i]
            expected[j, i] = True
    np.testing.assert_array_equal(_ancestors(before), expected)
