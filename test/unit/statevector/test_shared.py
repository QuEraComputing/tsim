from dataclasses import replace
from test.helpers.statevector import (
    CIRCUITS,
    assert_matches_distribution,
    coherent_surface_code,
    f_patterns,
    output_indices,
    plan_distribution,
    quantum_components,
)

import numpy as np
import pytest

import tsim.statevector.shared as shared
import tsim.utils.kernels as kernels
from tsim.statevector.program import MEASURE, MEASURE_DIAG, lower, plan_component
from tsim.statevector.shared import (
    lane_bound,
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
    yield from _compiled_text(CIRCUITS[name])


def _compiled_text(text):
    components, layout = quantum_components(text, sample_detectors=True)
    for component in components:
        indices = output_indices(component)
        planned, component_layout = plan_component(component, indices, layout)
        yield planned, component_layout, lower(planned, component_layout)


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


@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("name", NAMES)
def test_sampling_matches_the_plan_distribution(name, share):
    shots = 20_000
    for planned, component_layout, compiled in _compiled(name):
        layout = sharing_layout(compiled)
        for f in f_patterns(planned.num_f, limit=2):
            f_params = _f_params(component_layout, [f], shots)
            samples, deviation, _ = sample_shared_component(
                compiled, layout, np.asarray(f_params), 7, share
            )
            assert float(deviation) < 1e-4
            distribution = plan_distribution(planned, f)
            assert_matches_distribution(np.asarray(samples), distribution)
            # NOTE: an outcome of probability 0 is never drawn, not even rarely.
            seen = {tuple(row) for row in np.asarray(samples).astype(int).tolist()}
            assert all(distribution.get(key, 0.0) > 1e-12 for key in seen)


@pytest.mark.parametrize("name", NAMES)
def test_mixed_noise_patterns_share_one_batch(name):
    shots_each = 8_000
    for planned, component_layout, compiled in _compiled(name):
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(component_layout, patterns, shots_each)
        permutation = np.random.default_rng(1).permutation(len(f_params))
        samples, _, _ = sample_shared_component(
            compiled,
            sharing_layout(compiled),
            np.asarray(f_params[permutation]),
            3,
        )
        samples = np.asarray(samples)[np.argsort(permutation)]
        for i, f in enumerate(patterns):
            rows = samples[i * shots_each : (i + 1) * shots_each]
            assert_matches_distribution(rows, plan_distribution(planned, f))


@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_pruning_discards_exactly_the_postselected_failures(name, flip, share):
    shots_each = 8_000
    for planned, component_layout, compiled in _compiled(name):
        postselected = np.ones(compiled.num_outputs, dtype=bool)
        postselected[-1] = False
        flipped = np.zeros(compiled.num_outputs, dtype=bool)
        flipped[0] = flip
        layout = sharing_layout(compiled, postselected, flipped)
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(component_layout, patterns, shots_each)
        samples, _, _ = sample_shared_component(
            compiled, layout, np.asarray(f_params), 5, share
        )
        samples = np.asarray(samples)
        for i, f in enumerate(patterns):
            rows = samples[i * shots_each : (i + 1) * shots_each]
            discarded = np.asarray((rows & postselected).any(axis=1))
            expected = _postselected_distribution(
                plan_distribution(planned, f), postselected, flipped
            )
            _assert_counts(rows ^ flipped, discarded, expected)


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_pruned_shots_report_false_after_their_pruning_action(name, flip):
    for planned, component_layout, compiled in _compiled(name):
        postselected = np.ones(compiled.num_outputs, dtype=bool)
        flipped = np.full(compiled.num_outputs, flip)
        layout = sharing_layout(compiled, postselected, flipped)
        f_params = _f_params(component_layout, f_patterns(planned.num_f, 3), 4_000)
        samples, _, _ = sample_shared_component(
            compiled, layout, np.asarray(f_params), 9
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
        f_params = np.asarray(
            _f_params(component_layout, f_patterns(planned.num_f, 3), 3_000)
        )
        first, _, _ = sample_shared_component(compiled, layout, f_params, 13)
        second, _, _ = sample_shared_component(compiled, layout, f_params, 13)
        third, _, _ = sample_shared_component(compiled, layout, f_params, 14)
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


@pytest.fixture(scope="module")
def coherent():
    """A component whose extent drops and later rises again."""
    ((planned, component_layout, compiled),) = _compiled_text(
        coherent_surface_code(5, 3)
    )
    extent = np.asarray(compiled.tape.extent)
    first_drop = np.flatnonzero(np.diff(extent) < 0)[0]
    assert np.any(np.diff(extent[first_drop:]) > 0)
    return planned, component_layout, compiled


def test_extents_and_reused_slots_do_not_change_the_samples(coherent, monkeypatch):
    planned, component_layout, compiled = coherent
    layout = sharing_layout(compiled, np.ones(compiled.num_outputs, dtype=bool))
    extent = compiled.tape.extent

    def with_extent(values):
        return replace(compiled, tape=compiled.tape._replace(extent=values))

    full = with_extent(np.full_like(extent, compiled.width))
    # NOTE: an extent is an upper bound, so raising every fifth one keeps the
    # samples and makes the extent drop and rise around splits and prunes.
    every_fifth = np.arange(len(extent)) % 5 == 0
    raised = with_extent(np.where(every_fifth, compiled.width, extent))
    f_params = np.asarray(
        _f_params(component_layout, f_patterns(planned.num_f, 7), 300)
    )
    key = 17
    reference, deviation, _ = sample_shared_component(full, layout, f_params, key)
    # NOTE: NaN fills every amplitude the executor must not read.
    monkeypatch.setattr(shared, "_POISON", True)
    for component in (compiled, raised):
        samples, other, stats = sample_shared_component(
            component, layout, f_params, key
        )
        np.testing.assert_array_equal(samples, reference)
        assert other == deviation
        assert stats["dead"] > 0


@pytest.mark.parametrize("name", [*NAMES, "coherent"])
def test_parallel_kernels_give_the_serial_samples(name, coherent, monkeypatch):
    compiled_components = [coherent] if name == "coherent" else _compiled(name)
    for planned, component_layout, compiled in compiled_components:
        postselected = np.zeros(compiled.num_outputs, dtype=bool)
        postselected[0] = True
        layout = sharing_layout(compiled, postselected)
        f_params = np.asarray(
            _f_params(component_layout, f_patterns(planned.num_f, 3), 1_000)
        )
        samples = []
        for threshold in (np.iinfo(np.int64).max, 0):
            with monkeypatch.context() as patch:
                patch.setattr(kernels, "PARALLEL_WORK", threshold)
                rows, _, _ = sample_shared_component(compiled, layout, f_params, 19)
            samples.append(rows)
        np.testing.assert_array_equal(samples[1], samples[0])


@pytest.mark.parametrize("postselect", [False, True])
def test_lanes_stay_within_the_lane_bound(postselect):
    reached = False
    for name in NAMES:
        for planned, component_layout, compiled in _compiled(name):
            postselected = np.full(compiled.num_outputs, postselect)
            layout = sharing_layout(compiled, postselected)
            f_params = np.asarray(
                _f_params(component_layout, f_patterns(planned.num_f), 500)
            )
            _, _, stats = sample_shared_component(compiled, layout, f_params, 3)
            assert stats["peak lanes"] <= lane_bound(layout)
            reached |= stats["peak lanes"] == lane_bound(layout) > 1
    assert reached


def test_shots_whose_lanes_all_die_are_still_sampled():
    all_died = False
    for name in NAMES:
        for planned, component_layout, compiled in _compiled(name):
            layout = sharing_layout(compiled, np.ones(compiled.num_outputs, dtype=bool))
            f_params = np.asarray(
                _f_params(component_layout, f_patterns(planned.num_f, 1)[:1], 1)
            )
            for seed in range(40):
                samples, _, stats = sample_shared_component(
                    compiled, layout, f_params, seed
                )
                assert samples.shape == (1, compiled.num_outputs)
                all_died |= stats["lanes"] == 0
            samples, _, _ = sample_shared_component(compiled, layout, f_params[:0], 0)
            assert samples.shape == (0, compiled.num_outputs)
    assert all_died


@pytest.mark.parametrize("name", [*NAMES, "coherent"])
def test_every_action_stays_within_its_extent(name, coherent):
    compiled_components = [coherent] if name == "coherent" else _compiled(name)
    for _, _, compiled in compiled_components:
        tape = compiled.tape
        reach = 1 << np.asarray(tape.extent, dtype=np.int64)
        support = np.asarray(tape.x_mask) | np.asarray(tape.z_mask)
        assert np.all(support < reach)
        measured = np.isin(np.asarray(tape.kind), (MEASURE, MEASURE_DIAG))
        assert np.all((1 << np.asarray(tape.pivot))[measured] < reach[measured])
