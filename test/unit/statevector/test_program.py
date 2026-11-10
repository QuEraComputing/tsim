from test.helpers.statevector import (
    CIRCUITS,
    assert_distributions_close,
    component_program,
    f_patterns,
    output_indices,
    plan_distribution,
    quantum_components,
    stabilizer_rank_distribution,
)

import numpy as np
import pytest

import tsim.statevector.program as program
import tsim.statevector.shared as shared
from tsim.circuit import Circuit
from tsim.compile.pipeline import classify_direct, compile_program
from tsim.core.graph import prepare_graph
from tsim.core.types import CompiledComponent
from tsim.statevector.pbc import ComponentLayout
from tsim.statevector.plan import AffineRow, MeasureActive, MeasureDormant, Plan, Rotate
from tsim.statevector.program import (
    MEASURE,
    MEASURE_DIAG,
    MEASURE_DORMANT,
    ROTATE,
    ROTATE_DIAG,
    lower,
    plan_component,
)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
@pytest.mark.parametrize("sample_detectors", [True, False])
def test_plans_match_stabilizer_rank(name, sample_detectors):
    prepared = prepare_graph(Circuit(CIRCUITS[name]), sample_detectors=sample_detectors)
    joint = compile_program(prepared, mode="joint", strategy="cat5").components
    by_outputs = {c.output_indices: c for c in joint}
    for component in prepared.graph.connected_components():
        if classify_direct(component, prepared.layout) is not None:
            continue
        indices = output_indices(component)
        planned, component_layout = plan_component(component, indices, prepared.layout)
        reference = by_outputs[tuple(indices)]
        assert isinstance(reference, CompiledComponent)
        selection = tuple(np.asarray(reference.f_selection).tolist())
        assert selection == component_layout.f_selection
        for f in f_patterns(planned.num_f):
            assert_distributions_close(
                plan_distribution(planned, f),
                stabilizer_rank_distribution(reference, f),
                atol=2e-5,
            )


def test_measurements_that_free_no_reused_axis_run_after_every_rotation():
    # NOTE: random two-qubit layers, then every qubit measured: measuring early
    # would free no axis that a later rotation reuses, so the measurements stay
    # at the end and every rotation runs before the shots split.
    rng = np.random.default_rng(0)
    lines = []
    for _ in range(8):
        perm = rng.permutation(6)
        for a, b in zip(perm[::2], perm[1::2], strict=True):
            lines += [
                f"U3({t[0]},{t[1]},{t[2]}) {q}"
                for q in (a, b)
                for t in [rng.uniform(-1, 1, 3)]
            ]
            lines.append(f"CX {a} {b}")
    lines.append("M " + " ".join(map(str, range(6))))
    (component,), layout = quantum_components("\n".join(lines), sample_detectors=False)
    planned, _ = plan_component(component, output_indices(component), layout)
    rotations = [isinstance(action, Rotate) for action in planned.actions]
    assert rotations == sorted(rotations, reverse=True)


def _work_past_the_common_prefix(planned: Plan, layout: ComponentLayout) -> int:
    """Amplitudes that the actions past ``shared.common_prefix`` touch, coins aside."""
    _, lanes = component_program(lower(planned, layout))
    start, _ = shared.common_prefix(lanes)
    coin = (lanes.kind == MEASURE_DORMANT) & ~lanes.splits
    return sum(1 << int(e) for e in lanes.extent[start:][~coin[start:]])


def test_shot_work_starts_where_the_common_prefix_ends():
    free = AffineRow(False, frozenset())
    rotations = (Rotate(1, 0, 0.3, free), Rotate(2, 1, 0.2, free))
    measure = MeasureActive(0, 1, 0, branch=1)
    outputs = (AffineRow(False, frozenset({0})), AffineRow(False, frozenset({1})))
    # NOTE: an unread dormant measurement is a fair coin, which ends no prefix.
    coin_first = Plan(2, 0, (MeasureDormant(0), *rotations, measure), outputs)
    coin_last = Plan(
        2, 0, (*rotations, MeasureActive(0, 1, 0, branch=0), MeasureDormant(1)), outputs
    )
    # NOTE: a rotation's sign reads this one, so the shots split at it.
    read = Plan(
        2,
        0,
        (
            MeasureDormant(0),
            rotations[0],
            Rotate(2, 1, 0.2, AffineRow(False, frozenset({0}))),
            measure,
        ),
        outputs,
    )
    assert program._shot_work(coin_first) == program._shot_work(coin_last) == 4
    assert program._shot_work(read) == 1 + 2 + 4 + 4
    layout = ComponentLayout((), {}, (0, 1))
    for planned in (coin_first, coin_last, read):
        work = _work_past_the_common_prefix(planned, layout)
        assert program._shot_work(planned) == work


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_shot_work_matches_the_common_prefix_of_every_plan(name):
    components, layout = quantum_components(CIRCUITS[name], sample_detectors=True)
    for component in components:
        indices = output_indices(component)
        planned, component_layout = plan_component(component, indices, layout)
        work = _work_past_the_common_prefix(planned, component_layout)
        assert program._shot_work(planned) == work


def test_lower_encodes_every_action():
    planned = Plan(
        width=2,
        num_f=1,
        actions=(
            Rotate(x=0, z=3, angle=0.25, sign=AffineRow(True, frozenset({0}))),
            Rotate(x=1, z=2, angle=-0.5, sign=AffineRow(False, frozenset())),
            MeasureActive(0, 2, 1, branch=1),
            MeasureActive(1, 0, 0, branch=2),
            MeasureDormant(branch=3),
        ),
        outputs=(
            AffineRow(False, frozenset({1, 3})),
            AffineRow(True, frozenset({0, 2})),
        ),
    )
    layout = ComponentLayout(f_selection=(4,), f_position={}, output_indices=(2, 5))
    compiled = lower(planned, layout)
    tape = compiled.tape
    assert tape.kind.tolist() == [
        ROTATE_DIAG,
        ROTATE,
        MEASURE_DIAG,
        MEASURE,
        MEASURE_DORMANT,
    ]
    assert tape.x_mask.tolist() == [0, 1, 0, 1, 0]
    assert tape.z_mask.tolist() == [3, 2, 2, 0, 0]
    np.testing.assert_allclose(tape.cos[:2], np.cos([0.25, -0.5]))
    np.testing.assert_allclose(tape.sin[:2], np.sin([0.25, -0.5]))
    assert tape.pivot.tolist()[2:4] == [1, 0]
    assert tape.branch.tolist() == [-1, -1, 1, 2, 3]
    assert tape.sign_const.tolist()[:2] == [True, False]
    assert tape.sign_row[:2].tolist() == [[True, False, False, False], [False] * 4]
    assert tape.output_const.tolist() == [False, True]
    assert np.flatnonzero(tape.output_row[0]).tolist() == [1, 3]
    assert np.flatnonzero(tape.output_row[1]).tolist() == [0, 2]
    assert compiled.num_symbols == 4
    assert tape.f_selection.tolist() == [4]
    assert compiled.output_indices == (2, 5)
    assert (compiled.width, compiled.num_outputs) == (2, 2)
    assert compiled.num_actions == 5


def test_lower_keeps_a_column_per_symbol():
    planned = Plan(
        width=0,
        num_f=40,
        actions=(),
        outputs=(AffineRow(False, frozenset({0, 33, 39})),),
    )
    compiled = lower(planned, ComponentLayout(tuple(range(40)), {}, (0,)))
    assert compiled.num_symbols == 40
    assert np.flatnonzero(compiled.tape.output_row[0]).tolist() == [0, 33, 39]


def test_too_wide_components_are_rejected(monkeypatch):
    monkeypatch.setattr(program, "MAX_WIDTH", 0)
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    with pytest.raises(ValueError, match="cat5"):
        plan_component(component, output_indices(component), layout)


def test_reading_failures_are_value_errors(monkeypatch):
    def fail(*_):
        raise ValueError("no gflow")

    monkeypatch.setattr(program, "read_pauli_ops", fail)
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    with pytest.raises(ValueError, match="no gflow"):
        plan_component(component, output_indices(component), layout)
