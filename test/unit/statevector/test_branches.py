"""Tests for ``tsim.statevector.branches``: exact outcome probabilities of a tape."""

from test.helpers.statevector import (
    CIRCUITS,
    assert_distributions_close,
    f_patterns,
    ops_distribution,
    output_indices,
    quantum_components,
)

import numpy as np
import pytest

from tsim.statevector.branches import Branches, unique_rows
from tsim.statevector.pbc import ComponentLayout, read_pauli_ops
from tsim.statevector.program import compile_statevector_component

DETERMINISTIC = """
    R 0 1 2
    H 0
    CX 0 1 0 2
    T 0 1 2
    DEPOLARIZE1(0.01) 0 1 2
    T_DAG 0 1 2
    DEPOLARIZE1(0.01) 0 1 2
    MPP X0*X1*X2 Z0*Z1
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    MPP Z1*Z2
    OBSERVABLE_INCLUDE(0) rec[-1]
"""


def _components(text: str, *, sample_detectors: bool):
    """Yield each non-direct component's compiled form and its Pauli ops."""
    components, layout = quantum_components(text, sample_detectors=sample_detectors)
    for component in components:
        indices = output_indices(component)
        ops = read_pauli_ops(component, ComponentLayout.of(component, indices, layout))
        yield compile_statevector_component(component, indices, layout), ops


@pytest.mark.parametrize("name", sorted(CIRCUITS))
@pytest.mark.parametrize("sample_detectors", [True, False])
def test_the_branches_have_the_exact_output_distribution(name, sample_detectors):
    for compiled, ops in _components(CIRCUITS[name], sample_detectors=sample_detectors):
        patterns = f_patterns(ops.num_f, limit=8)
        branches = Branches(compiled, num_detectors=len(compiled.output_indices))
        got: list[dict] = [{} for _ in patterns]
        f = np.array(patterns, dtype=np.uint8).reshape(len(patterns), ops.num_f)
        for row, weight, outputs in branches._leaves(f, prune=False):
            for r, w, out in zip(
                row.tolist(), weight.tolist(), outputs.tolist(), strict=True
            ):
                got[r][tuple(out)] = got[r].get(tuple(out), 0.0) + w
        for pattern, distribution in zip(patterns, got, strict=True):
            assert_distributions_close(distribution, ops_distribution(ops, pattern))


def test_probabilities_are_those_of_the_reference_outputs():
    ((compiled, ops),) = _components(DETERMINISTIC, sample_detectors=True)
    # NOTE: outputs 0 and 1 are the detectors, output 2 is the observable.
    is_detector = np.array(compiled.output_indices) < 2
    branches = Branches(compiled, num_detectors=2)
    reference = branches.reference
    patterns = f_patterns(ops.num_f, limit=64)
    passed, unflipped = branches.probabilities(np.array(patterns, dtype=np.uint8))
    assert passed[0] == pytest.approx(1.0)
    assert unflipped[0] == pytest.approx(1.0)
    assert any(0.4 < p < 0.6 for p in passed)
    for i, pattern in enumerate(patterns):
        distribution = ops_distribution(ops, pattern)
        quiet = {
            outcome: p
            for outcome, p in distribution.items()
            if np.array_equal(np.array(outcome)[is_detector], reference[is_detector])
        }
        assert passed[i] == pytest.approx(sum(quiet.values()), abs=1e-9)
        same = quiet.get(tuple(reference.tolist()), 0.0)
        assert unflipped[i] == pytest.approx(same, abs=1e-9)


def test_many_rows_split_into_lanes(monkeypatch):
    ((compiled, ops),) = _components(DETERMINISTIC, sample_detectors=True)
    f = np.array(f_patterns(ops.num_f, limit=64), dtype=np.uint8)
    expected = Branches(compiled, num_detectors=2).probabilities(f)
    monkeypatch.setattr("tsim.statevector.branches.MAX_BYTES", 64)
    got = Branches(compiled, num_detectors=2).probabilities(f)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_a_random_noiseless_output_has_no_reference():
    text = """
        R 0
        H 0
        T 0
        H 0
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
    """
    ((compiled, _),) = _components(text, sample_detectors=True)
    with pytest.raises(ValueError, match="detector 0 is random without noise"):
        Branches(compiled, num_detectors=1).probabilities(np.zeros((1, 1)))


def test_unique_rows_groups_equal_rows():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 3, size=(500, 11), dtype=np.uint8)
    one, group = unique_rows(rows)
    assert len(one) == len(np.unique(rows, axis=0))
    assert np.array_equal(rows[one][group], rows)
    one, group = unique_rows(np.zeros((3, 0), dtype=np.uint8))
    assert one.tolist() == [0] and group.tolist() == [0, 0, 0]
