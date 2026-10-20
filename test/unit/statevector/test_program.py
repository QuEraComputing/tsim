from test.helpers.statevector import (
    CIRCUITS,
    assert_distributions_close,
    f_patterns,
    output_indices,
    plan_distribution,
    quantum_components,
    stabilizer_rank_distribution,
)

import numpy as np
import pytest

import tsim.statevector.program as program
from tsim.circuit import Circuit
from tsim.compile.pipeline import classify_constant, classify_direct, compile_program
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
        if (
            classify_direct(component, prepared.layout) is not None
            or classify_constant(component) is not None
        ):
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
    assert tape.sign_row[:2, 0].tolist() == [1, 0]
    assert tape.output_const.tolist() == [False, True]
    assert tape.output_row[:, 0].tolist() == [0b1010, 0b0101]
    assert compiled.num_words == 1
    assert tape.f_selection.tolist() == [4]
    assert compiled.output_indices == (2, 5)
    assert (compiled.width, compiled.num_outputs) == (2, 2)
    assert compiled.num_actions == 5


def test_lower_packs_symbols_beyond_32_into_more_words():
    planned = Plan(
        width=0,
        num_f=40,
        actions=(),
        outputs=(AffineRow(False, frozenset({0, 33, 39})),),
    )
    compiled = lower(planned, ComponentLayout(tuple(range(40)), {}, (0,)))
    assert compiled.num_words == 2
    assert compiled.tape.output_row.tolist() == [[1, (1 << 1) | (1 << 7)]]


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
