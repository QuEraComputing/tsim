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
from tsim.statevector.shared import (
    Sharing,
    sample_shared_component,
    sharing_layout,
)

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
        sharing = Sharing(sharing_layout(compiled))
        for f in f_patterns(planned.num_f, limit=2):
            f_params = _f_params(component_layout, [f], shots)
            samples, _, deviation, _ = sample_shared_component(
                compiled, sharing, jnp.asarray(f_params), jax.random.key(7)
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
            Sharing(sharing_layout(compiled)),
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
        sharing = Sharing(sharing_layout(compiled, postselected, flipped))
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(component_layout, patterns, shots_each)
        samples, _, _, _ = sample_shared_component(
            compiled, sharing, jnp.asarray(f_params), jax.random.key(5)
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
        sharing = Sharing(sharing_layout(compiled, postselected))
        f_params = _f_params(component_layout, f_patterns(planned.num_f, 3), 4_000)
        samples, _, _, _ = sample_shared_component(
            compiled, sharing, jnp.asarray(f_params), jax.random.key(9)
        )
        rows = np.asarray(samples)
        ready = np.asarray(sharing.layout.ready)
        tested = np.asarray(sharing.layout.prune_outputs)
        prune_at = np.where(tested.any(axis=0), tested.argmax(axis=0), ready)
        pruning = np.min(np.where(rows, prune_at[None, :], np.inf), axis=1)
        late = rows & (ready[None, :] > pruning[:, None])
        assert not np.any(late), "an output set after the pruning action is 1"


def test_full_tables_grow_and_resume(monkeypatch):
    monkeypatch.setattr(shared, "_MIN_CAPACITY", 1)
    shots_each = 5_000
    stats = []
    for name in ["t_ghz", "midcircuit_feedforward", "injection_check"]:
        for planned, component_layout, compiled in _compiled(name):
            postselected = np.zeros(compiled.num_outputs, dtype=bool)
            postselected[0] = True
            sharing = Sharing(sharing_layout(compiled, postselected))
            patterns = f_patterns(planned.num_f, limit=3)
            f_params = _f_params(component_layout, patterns, shots_each)
            samples, _, _, stat = sample_shared_component(
                compiled, sharing, jnp.asarray(f_params), jax.random.key(11)
            )
            stats.append(stat)
            samples = np.asarray(samples)
            for i, f in enumerate(patterns):
                rows = samples[i * shots_each : (i + 1) * shots_each]
                discarded = rows[:, 0] == 1
                expected = _postselected_distribution(
                    plan_distribution(planned, f), postselected
                )
                kept = rows[~discarded]
                p_kept = 1 - expected.get(("discarded",), 0.0)
                sigma = np.sqrt(shots_each * p_kept * (1 - p_kept))
                assert abs(len(kept) - shots_each * p_kept) <= 5 * sigma + 3
    assert any(stat["resumes"] > 0 for stat in stats)


@pytest.mark.parametrize("name", ["t_ghz", "midcircuit_feedforward", "injection_check"])
def test_capacities_and_segments_do_not_change_the_samples(name, monkeypatch):
    for planned, component_layout, compiled in _compiled(name):
        postselected = np.zeros(compiled.num_outputs, dtype=bool)
        postselected[0] = True
        layout = sharing_layout(compiled, postselected)
        f_params = jnp.asarray(
            _f_params(component_layout, f_patterns(planned.num_f, 3), 3_000)
        )
        key = jax.random.key(13)
        reference, _, _, wide = sample_shared_component(
            compiled, Sharing(layout), f_params, key
        )
        segmented, _, _, _ = sample_shared_component(
            compiled, Sharing(layout), f_params, key, segment=1
        )
        monkeypatch.setattr(shared, "_MIN_CAPACITY", 1)
        tiny, _, _, narrow = sample_shared_component(
            compiled, Sharing(layout), f_params, key, segment=1
        )
        monkeypatch.undo()
        np.testing.assert_array_equal(np.asarray(segmented), np.asarray(reference))
        np.testing.assert_array_equal(np.asarray(tiny), np.asarray(reference))
        assert narrow["resumes"] > 0 or narrow["lane_capacity"] != wide["lane_capacity"]


def test_rare_outcomes_keep_their_probability():
    draw = jax.jit(shared._rare_drawn)
    for count, q in [(1, 1e-9), (1, 3e-8), (7, 1e-7), (1000, 1e-10)]:
        p_any = -np.expm1(count * np.log1p(-q))
        exact = -np.expm1(count * np.log1p(np.float32(-q)))
        assert abs(p_any - exact) <= 1e-6 * exact
    slots = 1 << 21
    for count, q in [(1, 1e-4), (3, 2e-3), (50, 0.3), (7, 0.5)]:
        rare = np.asarray(
            draw(
                jax.random.key(count),
                jnp.full(slots, count, dtype=jnp.int32),
                jnp.full(slots, q, dtype=jnp.float32),
            )
        )
        p_any = 1 - (1 - q) ** count
        hits = np.count_nonzero(rare)
        assert abs(hits - slots * p_any) <= 5 * np.sqrt(slots * p_any) + 3
        mean, sigma = count * q, np.sqrt(count * q * (1 - q) / slots)
        assert abs(rare.mean() - mean) <= 5 * sigma + 1e-6
        assert rare.min() >= 0 and rare.max() <= count


def test_a_lone_shot_sees_an_outcome_of_probability_below_float32_resolution():
    q = 1e-7
    slots = 1 << 22
    hits = 0
    draw = jax.jit(shared._rare_drawn)
    for seed in range(24):
        rare = draw(
            jax.random.key(seed),
            jnp.ones(slots, dtype=jnp.int32),
            jnp.full(slots, q, dtype=jnp.float32),
        )
        hits += int(jnp.count_nonzero(rare))
    expected = 24 * slots * q
    assert hits > 0
    assert abs(hits - expected) <= 5 * np.sqrt(expected) + 3
