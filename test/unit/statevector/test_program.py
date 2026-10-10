from test.helpers.statevector import (
    CIRCUITS,
    assert_distributions_close,
    f_patterns,
    plan_distribution,
    quantum_components,
    stabilizer_rank_distribution,
)

import jax
import numpy as np
import pytest

import tsim.statevector.program as program
from tsim.circuit import Circuit
from tsim.compile.pipeline import classify_direct, compile_program
from tsim.core.graph import prepare_graph
from tsim.statevector.heisenberg import ComponentLayout
from tsim.statevector.plan import (
    AffineRow,
    MeasureActive,
    MeasureDormant,
    Plan,
    Record,
    Rotate,
)
from tsim.statevector.program import (
    MAX_WIDTH,
    MEASURE,
    MEASURE_DIAG,
    MEASURE_DORMANT,
    RECORD,
    ROTATE,
    ROTATE_DIAG,
    compile_statevector_component,
    lower,
    plan_component,
    resolve_dtype,
)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
@pytest.mark.parametrize("sample_detectors", [True, False])
def test_plans_match_stabilizer_rank(name, sample_detectors):
    prepared = prepare_graph(Circuit(CIRCUITS[name]), sample_detectors=sample_detectors)
    joint = compile_program(prepared, mode="joint").components
    by_outputs = {c.output_indices: c for c in joint}
    for component in prepared.graph.connected_components():
        if classify_direct(component, prepared.layout) is not None:
            continue
        planned, component_layout = plan_component(component, prepared.layout)
        reference = by_outputs[component_layout.output_indices]
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
        num_branches=3,
        num_outputs=2,
        actions=(
            Rotate(x=0, z=3, angle=0.25, sign=AffineRow(True, frozenset({0}))),
            Rotate(x=1, z=2, angle=-0.5, sign=AffineRow(False, frozenset())),
            MeasureActive(
                0, 2, 1, branch=1, sign=AffineRow(False, frozenset({0})), slot=1
            ),
            MeasureActive(
                1, 0, 0, branch=2, sign=AffineRow(True, frozenset({1})), slot=-1
            ),
            MeasureDormant(branch=3, sign=AffineRow(False, frozenset({1, 2})), slot=0),
            Record(sign=AffineRow(True, frozenset({3})), slot=1),
        ),
    )
    layout = ComponentLayout(f_selection=(4,), f_position={}, output_indices=(2, 5))
    compiled = lower(planned, layout, "complex64")
    assert compiled.kind.tolist() == [
        ROTATE_DIAG,
        ROTATE,
        MEASURE_DIAG,
        MEASURE,
        MEASURE_DORMANT,
        RECORD,
    ]
    assert compiled.x_mask.tolist() == [0, 1, 0, 1, 0, 0]
    assert compiled.z_mask.tolist() == [3, 2, 2, 0, 0, 0]
    np.testing.assert_allclose(compiled.cos[:2], np.cos([0.25, -0.5]), rtol=1e-6)
    np.testing.assert_allclose(compiled.sin[:2], np.sin([0.25, -0.5]), rtol=1e-6)
    assert compiled.pivot.tolist()[2:4] == [1, 0]
    assert compiled.branch.tolist() == [-1, -1, 1, 2, 3, -1]
    assert compiled.slot.tolist() == [-1, -1, 1, -1, 0, 1]
    assert compiled.sign_const.tolist() == [True, False, False, True, False, True]
    assert compiled.sign_row[:, 0].tolist() == [1, 0, 1, 2, 6, 8]
    assert compiled.num_words == 1
    assert compiled.f_selection.tolist() == [4]
    assert compiled.output_indices == (2, 5)
    assert (compiled.width, compiled.num_outputs, compiled.dtype) == (2, 2, "complex64")
    assert compiled.num_actions == 6


def test_lower_packs_symbols_beyond_32_into_more_words():
    planned = Plan(
        width=0,
        num_f=40,
        num_branches=0,
        num_outputs=1,
        actions=(Record(sign=AffineRow(False, frozenset({0, 33, 39})), slot=0),),
    )
    compiled = lower(planned, ComponentLayout(tuple(range(40)), {}, (0,)), "complex64")
    assert compiled.num_words == 2
    assert compiled.sign_row.tolist() == [[1, (1 << 1) | (1 << 7)]]


def test_compile_statevector_component():
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    compiled = compile_statevector_component(component, layout)
    planned, component_layout = plan_component(component, layout)
    assert compiled.width == planned.width
    assert compiled.num_actions == len(planned.actions)
    assert compiled.output_indices == component_layout.output_indices
    assert compiled.dtype == "complex64"


def test_too_wide_components_are_rejected(monkeypatch):
    assert MAX_WIDTH == 30
    monkeypatch.setattr(program, "MAX_WIDTH", 0)
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    with pytest.raises(ValueError, match="cat5"):
        plan_component(component, layout)


def test_extraction_failures_are_value_errors(monkeypatch):
    def fail(_):
        raise RuntimeError("no gflow")

    monkeypatch.setattr(program, "extract_circuit", fail)
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    with pytest.raises(ValueError, match="no gflow"):
        plan_component(component, layout)


def test_resolve_dtype():
    assert resolve_dtype(None) == "complex64"
    assert resolve_dtype("complex64") == "complex64"
    with pytest.raises(ValueError, match="jax_enable_x64"):
        resolve_dtype("complex128")
    with pytest.raises(ValueError, match="complex32"):
        resolve_dtype("complex32")
    with jax.enable_x64(True):
        assert resolve_dtype(None) == "complex128"
        assert resolve_dtype("complex128") == "complex128"
