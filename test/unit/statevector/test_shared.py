from dataclasses import replace
from test.helpers.statevector import (
    CIRCUITS,
    assert_matches_distribution,
    component_program,
    f_patterns,
    output_indices,
    plan_distribution,
    quantum_components,
    sample_lanes,
)

import numpy as np
import pytest
import stim

import tsim.statevector.shared as shared
from tsim.statevector.program import MEASURE, MEASURE_DIAG, lower, plan_component

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
        for f in f_patterns(planned.num_f, limit=2):
            f_params = _f_params(component_layout, [f], shots)
            samples, deviation, _ = sample_lanes(compiled, f_params, 7, share)
            assert deviation < 1e-4
            distribution = plan_distribution(planned, f)
            assert_matches_distribution(samples, distribution)
            # NOTE: an outcome of probability 0 is never drawn, not even rarely.
            seen = {tuple(row) for row in samples.astype(int).tolist()}
            assert all(distribution.get(key, 0.0) > 1e-12 for key in seen)


@pytest.mark.parametrize("name", NAMES)
def test_mixed_noise_patterns_share_one_batch(name):
    shots_each = 8_000
    for planned, component_layout, compiled in _compiled(name):
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(component_layout, patterns, shots_each)
        permutation = np.random.default_rng(1).permutation(len(f_params))
        samples, _, _ = sample_lanes(compiled, f_params[permutation], 3)
        samples = samples[np.argsort(permutation)]
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
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(component_layout, patterns, shots_each)
        samples, _, _ = sample_lanes(
            compiled, f_params, 5, share, postselected, flipped
        )
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
        f_params = _f_params(component_layout, f_patterns(planned.num_f, 3), 4_000)
        rows, _, _ = sample_lanes(
            compiled, f_params, 9, postselected=postselected, flipped=flipped
        )
        lanes = component_program(compiled, postselected)[1]
        ready, tested = lanes.ready, lanes.prune_outputs
        prune_at = np.where(tested.any(axis=0), tested.argmax(axis=0), ready)
        pruning = np.min(np.where(rows, prune_at[None, :], np.inf), axis=1)
        late = rows & (ready[None, :] > pruning[:, None])
        assert not np.any(late), "an output set after the pruning action is 1"


@pytest.mark.parametrize("name", ["t_ghz", "midcircuit_feedforward", "injection_check"])
def test_a_seed_gives_its_samples_whatever_shares_its_lanes(name):
    for planned, component_layout, compiled in _compiled(name):
        postselected = np.zeros(compiled.num_outputs, dtype=bool)
        postselected[0] = True
        f_params = _f_params(component_layout, f_patterns(planned.num_f, 3), 3_000)
        first, _, _ = sample_lanes(compiled, f_params, 13, True, postselected)
        unshared, _, _ = sample_lanes(compiled, f_params, 13, False, postselected)
        np.testing.assert_array_equal(first, unshared)
        # NOTE: a shot's draws depend on its index, so it samples alike in any chunk.
        cut = 1_234
        head, _, _ = sample_lanes(compiled, f_params[:cut], 13, True, postselected)
        tail, _, _ = sample_lanes(
            compiled, f_params[cut:], 13, True, postselected, first_shot=cut
        )
        np.testing.assert_array_equal(np.vstack([head, tail]), first)
        other, _, _ = sample_lanes(compiled, f_params, 14, True, postselected)
        assert not np.array_equal(first, other)


@pytest.mark.parametrize(
    "p", [1e-18, 2**-60, 1e-12, 1e-7, 0.25, 0.5, 1 - 1e-9, 1 - 2**-53]
)
def test_thresholds_resolve_probabilities_to_2_to_the_minus_64(p):
    threshold = int(shared._threshold(p))
    assert 0 <= threshold < 2**64
    # NOTE: a uniform 64-bit draw is below the threshold with chance threshold / 2^64.
    assert abs(threshold / 2**64 - p) <= 2**-64 + p * 2**-52
    if p >= 2**-64:
        assert threshold > 0


@pytest.mark.parametrize("rare", [0, 1])
def test_a_rare_outcome_keeps_its_chance_either_way(rare):
    # NOTE: w1 / (w0 + w1) rounds to 1 here, but outcome 0 still has 265 draws.
    small = 1.4381067500455907e-17
    weights = (small, 1.0) if rare == 0 else (1.0, small)
    certain, threshold = shared._cut(*weights)
    assert certain == -1
    rare_draws = int(threshold) if rare else 2**64 - int(threshold)
    assert rare_draws == int(small * 2**64)
    # NOTE: a draw below the threshold is outcome 1; the rare draws sit at one end.
    lowest, highest = np.uint64(0), np.uint64(2**64 - 1)
    rare_draw, common_draw = (lowest, highest) if rare else (highest, lowest)
    assert int(rare_draw < threshold) == rare
    assert int(common_draw < threshold) == 1 - rare
    assert shared._cut(*(w if w == 1.0 else 0.0 for w in weights)) == (1 - rare, 0)
    assert shared._cut(0.0, 0.0) == (0, 0)


def _coherent_surface_code(distance, rounds, angle=0.02):
    """Return a surface-code memory with an R_Z over-rotation after every CX layer."""
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=1e-3,
    )
    lines = []
    for line in str(base.flattened()).splitlines():
        lines.append(line)
        if line.startswith("CX "):
            qubits = sorted({int(t) for t in line.split()[1:]})
            lines.append(f"R_Z({angle}) " + " ".join(map(str, qubits)))
    return "\n".join(lines)


@pytest.fixture(scope="module")
def coherent():
    """A component whose extent drops and later rises again."""
    ((planned, component_layout, compiled),) = _compiled_text(
        _coherent_surface_code(5, 3)
    )
    extent = np.asarray(compiled.tape.extent)
    first_drop = np.flatnonzero(np.diff(extent) < 0)[0]
    assert np.any(np.diff(extent[first_drop:]) > 0)
    return planned, component_layout, compiled


def test_extents_and_reused_slots_do_not_change_the_samples(coherent, monkeypatch):
    planned, component_layout, compiled = coherent
    postselected = np.ones(compiled.num_outputs, dtype=bool)
    extent = compiled.tape.extent

    def with_extent(values):
        return replace(compiled, tape=compiled.tape._replace(extent=values))

    full = with_extent(np.full_like(extent, compiled.width))
    # NOTE: an extent is an upper bound, so raising every fifth one keeps the
    # samples and makes the extent drop and rise around splits and prunes.
    every_fifth = np.arange(len(extent)) % 5 == 0
    raised = with_extent(np.where(every_fifth, compiled.width, extent))
    f_params = _f_params(component_layout, f_patterns(planned.num_f, 7), 300)
    reference, deviation, _ = sample_lanes(
        full, f_params, 17, postselected=postselected
    )
    # NOTE: NaN fills every amplitude the executor must not read.
    monkeypatch.setattr(shared, "_POISON", True)
    for component in (compiled, raised):
        samples, other, stats = sample_lanes(
            component, f_params, 17, postselected=postselected
        )
        np.testing.assert_array_equal(samples, reference)
        assert other == deviation
        assert stats["dead"] > 0


@pytest.mark.parametrize("postselect", [False, True])
def test_lanes_split_only_for_the_outcomes_their_shots_draw(postselect):
    for name in NAMES:
        for planned, component_layout, compiled in _compiled(name):
            postselected = np.full(compiled.num_outputs, postselect)
            patterns = f_patterns(planned.num_f)
            f_params = _f_params(component_layout, patterns, 500)
            _, _, stats = sample_lanes(compiled, f_params, 3, postselected=postselected)
            splits = int(component_program(compiled, postselected)[1].splits.sum())
            assert stats["groups"] <= len(patterns)
            assert stats["peak lanes"] <= min(len(f_params), stats["groups"] << splits)
            _, _, alone = sample_lanes(
                compiled, f_params, 3, share=False, postselected=postselected
            )
            assert alone["groups"] == len(f_params)


def test_shots_whose_lanes_all_die_are_still_sampled():
    all_died = False
    for name in NAMES:
        for planned, component_layout, compiled in _compiled(name):
            postselected = np.ones(compiled.num_outputs, dtype=bool)
            f_params = _f_params(component_layout, f_patterns(planned.num_f, 1)[:1], 1)
            for seed in range(40):
                samples, _, stats = sample_lanes(
                    compiled, f_params, seed, postselected=postselected
                )
                assert samples.shape == (1, compiled.num_outputs)
                all_died |= stats["lanes"] == 0
            samples, _, _ = sample_lanes(
                compiled, f_params[:0], 0, postselected=postselected
            )
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
