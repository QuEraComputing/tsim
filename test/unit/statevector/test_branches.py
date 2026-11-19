"""Tests for ``tsim.statevector.branches``: exact outcome distributions of a tape."""

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

from tsim.statevector.branches import Branches, gf2_product
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
@pytest.mark.parametrize("max_bytes", [None, 64])
def test_the_branches_have_the_exact_output_distribution(
    name, sample_detectors, max_bytes, monkeypatch
):
    if max_bytes is not None:
        # NOTE: tiny lanes make the walk split its branches.
        monkeypatch.setattr("tsim.statevector.branches.MAX_BYTES", max_bytes)
    for compiled, ops in _components(CIRCUITS[name], sample_detectors=sample_detectors):
        patterns = f_patterns(ops.num_f, limit=8)
        branches = Branches(compiled, num_detectors=len(compiled.output_indices))
        got: list[dict] = [{} for _ in patterns]
        f = np.array(patterns, dtype=np.uint8).reshape(len(patterns), ops.num_f)
        for row, weight, outputs in branches._leaves(f):
            for r, w, out in zip(
                row.tolist(), weight.tolist(), outputs.tolist(), strict=True
            ):
                got[r][tuple(out)] = got[r].get(tuple(out), 0.0) + w
        for pattern, distribution in zip(patterns, got, strict=True):
            assert_distributions_close(distribution, ops_distribution(ops, pattern))


def test_outcomes_are_the_deviations_from_the_reference():
    ((compiled, ops),) = _components(DETERMINISTIC, sample_detectors=True)
    branches = Branches(compiled, num_detectors=2)
    patterns = f_patterns(ops.num_f, limit=64)
    f = np.array(patterns, dtype=np.uint8)
    flips = gf2_product(f, branches.output_rows) ^ branches.reference
    for i, (deviations, probability) in enumerate(branches.outcomes(f)):
        expected = {
            tuple(int(b) for b in np.array(outcome) ^ flips[i]): p
            for outcome, p in ops_distribution(ops, patterns[i]).items()
            if p > 1e-12
        }
        got = dict(zip(map(tuple, deviations.tolist()), probability, strict=True))
        assert_distributions_close(got, expected, atol=1e-9)
    noiseless, probability = branches.outcomes(f[:1])[0]
    assert noiseless.tolist() == [[0] * len(branches.reference)]
    assert probability == pytest.approx([1.0])


def test_outcomes_depend_only_on_the_rotation_signs():
    ((compiled, ops),) = _components(DETERMINISTIC, sample_detectors=True)
    branches = Branches(compiled, num_detectors=2)
    f = np.array(f_patterns(ops.num_f, limit=64), dtype=np.uint8)
    signs = gf2_product(f, branches.sign_rows)
    outcomes = branches.outcomes(f)
    for i in range(len(f)):
        for j in np.flatnonzero((signs == signs[i]).all(axis=1)).tolist():
            assert np.array_equal(outcomes[i][0], outcomes[j][0])
            np.testing.assert_allclose(outcomes[i][1], outcomes[j][1])


def test_many_rows_split_into_lanes(monkeypatch):
    ((compiled, ops),) = _components(DETERMINISTIC, sample_detectors=True)
    f = np.array(f_patterns(ops.num_f, limit=64), dtype=np.uint8)
    expected = Branches(compiled, num_detectors=2).outcomes(f)
    monkeypatch.setattr("tsim.statevector.branches.MAX_BYTES", 64)
    got = Branches(compiled, num_detectors=2).outcomes(f)
    for (a, p), (b, q) in zip(got, expected, strict=True):
        assert np.array_equal(a, b)
        np.testing.assert_allclose(p, q, atol=1e-12)


def test_too_many_branches_are_refused(monkeypatch):
    ((compiled, ops),) = _components(DETERMINISTIC, sample_detectors=True)
    branches = Branches(compiled, num_detectors=2)
    assert branches.reference is not None
    monkeypatch.setattr("tsim.statevector.branches.MAX_BRANCHES", 0)
    with pytest.raises(ValueError, match="measurement branches"):
        branches.outcomes(np.zeros((1, ops.num_f), dtype=np.uint8))


def test_the_branches_are_limited_per_row(monkeypatch):
    # NOTE: f = 0 leaves two random MX, four branches; f = 1 undoes the T
    # gates and leaves one. Four rows have 7 branches, below 2 per row on
    # average.
    text = "RX 0 1\nT 0 1\nE(0.1) X0 X1\nT 0 1\nCX 0 1\nMX 0 1"
    ((compiled, ops),) = _components(text, sample_detectors=False)
    branches = Branches(compiled, num_detectors=len(compiled.output_indices))
    f = np.array([[0], [1], [1], [1]], dtype=np.uint8)
    assert ops.num_f == 1
    monkeypatch.setattr("tsim.statevector.branches.MAX_BRANCHES", 4)
    branches.support(f)
    monkeypatch.setattr("tsim.statevector.branches.MAX_BRANCHES", 2)
    with pytest.raises(ValueError, match="measurement branches"):
        branches.support(f)


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
        assert Branches(compiled, num_detectors=1).reference is not None


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_the_support_with_its_coins_is_every_possible_outcome(name, split, monkeypatch):
    if split:
        # NOTE: two-row lanes, so the walk splits them.
        monkeypatch.setattr("tsim.statevector.branches.MAX_BYTES", 64)
    for compiled, ops in _components(CIRCUITS[name], sample_detectors=False):
        branches = Branches(compiled, num_detectors=len(compiled.output_indices))
        f = np.array(f_patterns(ops.num_f, limit=4), dtype=np.uint8)
        f = f.reshape(len(f), ops.num_f)
        row, values = branches.support(f)
        assert np.all(np.diff(row) >= 0)
        coins = branches.coin_rows
        flips = np.array(
            [
                np.bitwise_xor.reduce(coins[list(s)], axis=0, initial=0)
                for s in _subsets(len(coins))
            ],
            dtype=np.uint8,
        ).reshape(-1, coins.shape[1])
        expected = [set() for _ in f]
        for r, _, out in branches._leaves(f):
            for i, o in zip(r.tolist(), out.tolist(), strict=True):
                expected[i].add(tuple(o))
        got = [set() for _ in f]
        for i, v in zip(row.tolist(), values, strict=True):
            got[i] |= {tuple(int(b) for b in v ^ flip) for flip in flips}
        assert got == expected


def _subsets(n: int) -> list[list[int]]:
    return [[i for i in range(n) if mask >> i & 1] for mask in range(1 << n)]
