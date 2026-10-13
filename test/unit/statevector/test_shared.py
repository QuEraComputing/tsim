from test.helpers.statevector import (
    CIRCUITS,
    assert_matches_distribution,
    f_patterns,
    output_indices,
    plan_distribution,
    quantum_components,
)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tsim.statevector.shared as shared
from tsim.statevector.program import lower, plan_component
from tsim.statevector.shared import sample_shared_component, sharing_layout

NAMES = [
    "t_ghz",
    "midcircuit_feedforward",
    "two_components",
    "clifford_component",
    "rotations",
    "injection_check",
    "t_interference",
]


def _compiled(name):
    components, layout = quantum_components(CIRCUITS[name], sample_detectors=True)
    for component in components:
        indices = output_indices(component)
        planned, component_layout = plan_component(component, indices, layout)
        yield planned, component_layout, lower(planned, component_layout, "complex64")


def _f_params(component_layout, patterns, shots_each):
    num_columns = max(component_layout.f_selection, default=-1) + 1
    f_params = np.zeros((shots_each * len(patterns), num_columns), dtype=np.uint8)
    for i, f in enumerate(patterns):
        if component_layout.f_selection:
            rows = slice(i * shots_each, (i + 1) * shots_each)
            f_params[rows, list(component_layout.f_selection)] = f
    return f_params


def _postselected_distribution(distribution, postselected, flipped=None):
    """Merge every outcome that fails postselection into one ``discarded`` key."""
    flipped = np.zeros(len(postselected), bool) if flipped is None else flipped
    merged = {}
    for key, p in distribution.items():
        failed = np.asarray(key, dtype=bool) ^ flipped
        if np.any(failed & postselected):
            key = ("discarded",)
        merged[key] = merged.get(key, 0.0) + p
    return merged


def _assert_counts(rows, discarded, expected):
    counts = {}
    for row, dropped in zip(rows.astype(int).tolist(), discarded.tolist(), strict=True):
        key = ("discarded",) if dropped else tuple(row)
        counts[key] = counts.get(key, 0) + 1
    shots = len(rows)
    for key in set(counts) | set(expected):
        p = expected.get(key, 0.0)
        sigma = np.sqrt(shots * p * (1 - p))
        assert abs(counts.get(key, 0) - shots * p) <= 5 * sigma + 3, (
            key,
            counts.get(key, 0),
            shots * p,
        )


@pytest.mark.parametrize("name", NAMES)
def test_shared_sampling_matches_the_plan_distribution(name):
    shots = 20_000
    for planned, component_layout, compiled in _compiled(name):
        layout = sharing_layout(compiled)
        for f in f_patterns(planned.num_f, limit=2):
            f_params = _f_params(component_layout, [f], shots)
            samples, _, deviation, _ = sample_shared_component(
                compiled, layout, jnp.asarray(f_params), jax.random.key(7)
            )
            assert float(deviation) < 1e-4
            assert_matches_distribution(
                np.asarray(samples), plan_distribution(planned, f)
            )


@pytest.mark.parametrize("name", NAMES)
def test_mixed_noise_patterns_share_one_batch(name):
    shots_each = 8_000
    for planned, component_layout, compiled in _compiled(name):
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(component_layout, patterns, shots_each)
        permutation = np.random.default_rng(1).permutation(len(f_params))
        samples, _, _, _ = sample_shared_component(
            compiled,
            sharing_layout(compiled),
            jnp.asarray(f_params[permutation]),
            jax.random.key(3),
        )
        samples = np.asarray(samples)[np.argsort(permutation)]
        for i, f in enumerate(patterns):
            rows = samples[i * shots_each : (i + 1) * shots_each]
            assert_matches_distribution(rows, plan_distribution(planned, f))


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_pruning_discards_exactly_the_postselected_failures(name, flip):
    shots_each = 8_000
    for planned, component_layout, compiled in _compiled(name):
        postselected = np.ones(compiled.num_outputs, dtype=bool)
        postselected[-1] = False
        flipped = np.zeros(compiled.num_outputs, dtype=bool)
        flipped[0] = flip
        layout = sharing_layout(compiled, postselected, flipped)
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(component_layout, patterns, shots_each)
        samples, _, _, _ = sample_shared_component(
            compiled, layout, jnp.asarray(f_params), jax.random.key(5)
        )
        samples = np.asarray(samples)
        for i, f in enumerate(patterns):
            rows = samples[i * shots_each : (i + 1) * shots_each]
            discarded = np.asarray(((rows ^ flipped) & postselected).any(axis=1))
            expected = _postselected_distribution(
                plan_distribution(planned, f), postselected, flipped
            )
            _assert_counts(rows, discarded, expected)


@pytest.mark.parametrize("name", NAMES)
def test_pruned_shots_report_false_after_their_pruning_action(name):
    for planned, component_layout, compiled in _compiled(name):
        postselected = np.ones(compiled.num_outputs, dtype=bool)
        layout = sharing_layout(compiled, postselected)
        f_params = _f_params(component_layout, f_patterns(planned.num_f, 3), 4_000)
        samples, _, _, _ = sample_shared_component(
            compiled, layout, jnp.asarray(f_params), jax.random.key(9)
        )
        rows = np.asarray(samples)
        ready = layout.ready
        tested = layout.prune_outputs
        prune_at = np.where(tested.any(axis=0), tested.argmax(axis=0), ready)
        pruning = np.min(np.where(rows, prune_at[None, :], np.inf), axis=1)
        late = rows & (ready[None, :] > pruning[:, None])
        assert not np.any(late), "an output set after the pruning action is 1"


@pytest.mark.parametrize("name", ["t_ghz", "midcircuit_feedforward", "injection_check"])
def test_equal_keys_give_equal_samples(name):
    for planned, component_layout, compiled in _compiled(name):
        postselected = np.zeros(compiled.num_outputs, dtype=bool)
        postselected[0] = True
        layout = sharing_layout(compiled, postselected)
        f_params = jnp.asarray(
            _f_params(component_layout, f_patterns(planned.num_f, 3), 3_000)
        )
        first, next_key, _, _ = sample_shared_component(
            compiled, layout, f_params, jax.random.key(13)
        )
        second, _, _, _ = sample_shared_component(
            compiled, layout, f_params, jax.random.key(13)
        )
        third, _, _, _ = sample_shared_component(compiled, layout, f_params, next_key)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, third)


def test_rare_outcomes_keep_their_binomial_law():
    lanes = 1 << 21
    for count, q in [(1, 1e-4), (3, 2e-3), (50, 0.3), (7, 0.5)]:
        w1 = np.full(lanes, q)
        ones = shared._draw_ones(np.full(lanes, count), 1 - w1, w1, np.uint64(count))
        p_any = 1 - (1 - q) ** count
        hits = np.count_nonzero(ones)
        assert abs(hits - lanes * p_any) <= 5 * np.sqrt(lanes * p_any) + 3
        mean, sigma = count * q, np.sqrt(count * q * (1 - q) / lanes)
        assert abs(ones.mean() - mean) <= 5 * sigma + 1e-6
        assert ones.min() >= 0 and ones.max() <= count


def test_the_common_outcome_counts_the_shots_that_miss_the_rare_one():
    lanes = 1 << 18
    count, q = 40, 0.01
    w1 = np.full(lanes, 1 - q)
    ones = shared._draw_ones(np.full(lanes, count), 1 - w1, w1, np.uint64(3))
    zeros = count - ones
    assert abs(zeros.mean() - count * q) <= 5 * np.sqrt(count * q / lanes)


def test_a_lone_shot_sees_an_outcome_of_probability_below_float32_resolution():
    q = 1e-7
    lanes = 1 << 22
    hits = 0
    for seed in range(24):
        w1 = np.full(lanes, q)
        ones = shared._draw_ones(
            np.ones(lanes, dtype=np.int64), 1 - w1, w1, np.uint64(seed)
        )
        hits += int(np.count_nonzero(ones))
    expected = 24 * lanes * q
    assert hits > 0
    assert abs(hits - expected) <= 5 * np.sqrt(expected) + 3
