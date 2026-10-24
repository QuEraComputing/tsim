import dataclasses
import pickle
from test.helpers.statevector import (
    CIRCUITS,
    assert_matches_distribution,
    coherent_surface_code,
    f_patterns,
    plan_distribution,
)
from test.unit.statevector.test_tables import (
    _assert_same_rows,
    _compiled,
    _f_params,
)

import numpy as np
import pytest

import tsim.statevector.trees as trees_module
import tsim.utils.kernels as kernels
from tsim.circuit import Circuit
from tsim.statevector.program import StatevectorComponent
from tsim.statevector.shared import sample_shared_component, sharing_layout
from tsim.statevector.trees import OutcomeTrees

NAMES = sorted(CIRCUITS)


@pytest.mark.parametrize("name", NAMES)
def test_samples_follow_the_plan_distribution(name):
    shots_each = 10_000
    for planned, compiled in _compiled(name):
        trees = OutcomeTrees(compiled)
        patterns = f_patterns(planned.num_f, limit=3)
        samples, deviation, stats = trees.sample(
            _f_params(compiled, patterns, shots_each), 7, trees.layout
        )
        assert deviation < 1e-9
        assert stats["nodes built"] == stats["nodes"] == trees.num_nodes
        for i, f in enumerate(patterns):
            rows = samples[i * shots_each : (i + 1) * shots_each]
            distribution = plan_distribution(planned, f)
            assert_matches_distribution(rows, distribution)
            seen = {tuple(row) for row in rows.astype(int).tolist()}
            assert all(distribution.get(key, 0.0) > 1e-12 for key in seen)


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_discarded_shots_read_false_after_their_discarding_output(name, flip):
    for planned, compiled in _compiled(name):
        postselected = np.ones(compiled.num_outputs, dtype=bool)
        postselected[-1] = False
        flipped = np.full(compiled.num_outputs, flip)
        layout = sharing_layout(compiled, postselected, flipped)
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(compiled, patterns, 4_000)
        rows, _, _ = OutcomeTrees(compiled).sample(f_params, 9, layout)
        # NOTE: postselection tests the outputs with their reference bits.
        fired = rows & postselected
        discarding = np.min(
            np.where(fired, layout.ready[None, :], compiled.num_actions), axis=1
        )
        late = layout.ready[None, :] > discarding[:, None]
        assert not np.any(rows & late), "an output set after the discarding one is 1"
        # NOTE: the outputs up to the discarding one keep their values.
        unflipped, _, _ = OutcomeTrees(compiled).sample(
            f_params, 9, sharing_layout(compiled)
        )
        np.testing.assert_array_equal((rows ^ flipped)[~late], unflipped[~late])


@pytest.mark.parametrize("name", NAMES)
def test_samples_do_not_depend_on_what_was_built_before(name, monkeypatch):
    for planned, compiled in _compiled(name):
        patterns = f_patterns(planned.num_f, limit=6)
        f_params = _f_params(compiled, patterns, 500)
        np.random.default_rng(3).shuffle(f_params)
        fresh = OutcomeTrees(compiled)
        warmed = OutcomeTrees(compiled)
        warmed.sample(_f_params(compiled, patterns[::-1], 50), 1, warmed.layout)
        first, _, built = fresh.sample(f_params, 11, fresh.layout)
        again, _, rebuilt = fresh.sample(f_params, 11, fresh.layout)
        other, _, _ = warmed.sample(f_params, 11, warmed.layout)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(first, other)
        assert built["nodes built"] == fresh.num_nodes and rebuilt["nodes built"] == 0
        assert rebuilt["rounds"] == 0
        # NOTE: with one state slot, nearly every expansion recomputes its state.
        monkeypatch.setattr(trees_module, "_STATE_BYTES", 1)
        evicting = OutcomeTrees(compiled)
        evicted, _, _ = evicting.sample(f_params, 11, evicting.layout)
        monkeypatch.undo()
        np.testing.assert_array_equal(first, evicted)


@pytest.mark.parametrize("name", NAMES)
def test_parallel_kernels_give_the_serial_samples(name, monkeypatch):
    for planned, compiled in _compiled(name):
        f_params = _f_params(compiled, f_patterns(planned.num_f, 3), 1_000)
        samples = []
        for threshold in (np.iinfo(np.int64).max, 0):
            with monkeypatch.context() as patch:
                patch.setattr(kernels, "PARALLEL_WORK", threshold)
                trees = OutcomeTrees(compiled)
                rows, _, _ = trees.sample(f_params, 19, trees.layout)
            samples.append(rows)
        np.testing.assert_array_equal(samples[1], samples[0])


def _lanes_and_trees(text, shots, postselect=0.0, seed=0):
    sampler = Circuit(text).compile_sampler(seed=seed)
    (component,) = sampler._program.components
    assert isinstance(component, StatevectorComponent)
    f_params = sampler._channel_sampler.sample(shots)
    rng = np.random.default_rng(seed)
    postselected = rng.random(component.num_outputs) < postselect
    flipped = rng.random(component.num_outputs) < 0.5
    layout = sharing_layout(component, postselected, flipped)
    trees = OutcomeTrees(component)
    got, _, stats = trees.sample(f_params, 1, layout)
    expected, _, _ = sample_shared_component(component, layout, f_params, 2)
    return component, trees, got, expected, stats


def test_keys_wider_than_64_bits_match_lanes():
    text = coherent_surface_code(3, 3)
    sampler = Circuit(text).compile_detector_sampler(seed=0)
    (component,) = sampler._program.components
    assert isinstance(component, StatevectorComponent)
    assert len(sharing_layout(component).key_rows) > 64
    trees = OutcomeTrees(component)
    assert trees.key_words == 2
    f_params = sampler._channel_sampler.sample(1 << 13)
    postselected = np.ones(component.num_outputs, dtype=bool)
    layout = sharing_layout(component, postselected)
    got, _, stats = trees.sample(f_params, 1, layout)
    expected, _, _ = sample_shared_component(component, layout, f_params, 2)
    assert stats["nodes"] > 0
    for columns in ([0, 1, 2], [3, 4, 5], [-3, -2, -1]):
        _assert_same_rows(got[:, columns], expected[:, columns])


def test_more_than_64_outputs_and_coins_match_lanes():
    """Outputs and dormant coins span two words; trees agree with lanes."""
    coins = 70
    lines = ["R " + " ".join(map(str, range(coins + 1))), "H 0", "T 0"]
    lines += [f"CX 0 {q}" for q in range(1, coins + 1)]
    lines += ["X_ERROR(0.1) 0", "MX " + " ".join(map(str, range(1, coins + 1)))]
    lines += ["H 0", "M 0"]
    component, trees, got, expected, _ = _lanes_and_trees(
        "\n".join(lines), 1 << 15, postselect=0.05
    )
    assert component.num_outputs == coins + 1
    assert trees._coin_columns.shape == (coins, 2)
    for columns in ([0, 1, 65], [2, 69, 70], [30, 64, 70]):
        _assert_same_rows(got[:, columns], expected[:, columns])


def test_deep_trees_with_postselection_match_lanes():
    lines = ["R 0"]
    for _ in range(12):
        lines += ["R_X(0.4) 0", "DEPOLARIZE1(0.02) 0", "T 0", "R_X(0.3) 0", "M 0"]
    _, _, got, expected, stats = _lanes_and_trees(
        "\n".join(lines), 1 << 15, postselect=0.3, seed=4
    )
    assert stats["rounds"] > 3
    for columns in ([0, 1, 2], [4, 7, 11], [9, 10, 11]):
        _assert_same_rows(got[:, columns], expected[:, columns])


def test_a_nan_norm_deviation_reaches_the_caller():
    sampler = Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(seed=0)
    (component,) = sampler._program.components
    assert isinstance(component, StatevectorComponent)
    cos = component.tape.cos.copy()
    cos[np.flatnonzero(component.tape.sin)[0]] = np.nan
    broken = dataclasses.replace(component, tape=component.tape._replace(cos=cos))
    f_params = np.zeros(
        (4, sampler._channel_sampler.signature_matrix.shape[1]), np.uint8
    )
    trees = OutcomeTrees(broken)
    _, deviation, _ = trees.sample(f_params, 1, trees.layout)
    assert np.isnan(deviation)


def test_trees_pickle_with_their_nodes():
    sampler = Circuit(CIRCUITS["t_ghz"]).compile_detector_sampler(seed=0)
    (component,) = sampler._program.components
    assert isinstance(component, StatevectorComponent)
    trees = OutcomeTrees(component)
    f_params = sampler._channel_sampler.sample(2_000)
    first, _, _ = trees.sample(f_params, 5, trees.layout)
    copy = pickle.loads(pickle.dumps(trees))  # noqa: S301
    assert copy.num_nodes == trees.num_nodes > 0
    again, _, stats = copy.sample(f_params, 5, copy.layout)
    np.testing.assert_array_equal(first, again)
    assert stats["nodes built"] == 0


def test_trees_that_build_a_node_per_shot_fall_back_to_lanes(monkeypatch):
    """Without repeating keys or outcomes, a call moves the component to lanes."""
    import tsim.sampler as sampler_module

    monkeypatch.setattr(sampler_module, "table_keys", lambda *args: None)
    lines = ["R 0"]
    for _ in range(20):
        lines += ["R_X(0.6) 0", "DEPOLARIZE1(0.2) 0", "T 0", "R_X(0.6) 0", "M 0"]
    sampler = Circuit("\n".join(lines)).compile_sampler(seed=0)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    (plan,) = backend.plans()
    assert plan is not None and plan.trees is not None
    sampler.sample(4_000)
    # NOTE: the first call warms the trees up and doesn't count.
    assert backend.plans() == (plan,)
    sampler.sample(4_000)
    assert backend.plans() == (None,)
    before = plan.trees.num_nodes
    sampler.sample(4_000)
    assert plan.trees.num_nodes == before


def test_trees_that_repeat_stay():
    import tsim.sampler as sampler_module

    sampler = Circuit(coherent_surface_code(3, 3)).compile_detector_sampler(seed=0)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    for _ in range(3):
        sampler.sample(20_000)
    (plan,) = backend.plans()
    assert plan is not None and plan.tables is None and plan.trees is not None
