from test.helpers.statevector import (
    CIRCUITS,
    assert_matches_distribution,
    f_patterns,
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
        planned, component_layout = plan_component(component, layout)
        yield planned, component_layout, lower(planned, component_layout, "complex64")


def _f_params(component_layout, patterns, shots_each):
    num_columns = max(component_layout.f_selection, default=-1) + 1
    f_params = np.zeros((shots_each * len(patterns), num_columns), dtype=np.uint8)
    for i, f in enumerate(patterns):
        if component_layout.f_selection:
            rows = slice(i * shots_each, (i + 1) * shots_each)
            f_params[rows, list(component_layout.f_selection)] = f
    return f_params


def _postselected_distribution(distribution, postselected):
    """Merge every outcome with a postselected 1 into one ``discarded`` key."""
    merged = {}
    for key, p in distribution.items():
        if any(bit and keep for bit, keep in zip(key, postselected, strict=True)):
            key = ("discarded",)
        merged[key] = merged.get(key, 0.0) + p
    return merged


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


@pytest.mark.parametrize("name", NAMES)
def test_pruning_discards_exactly_the_postselected_failures(name):
    shots_each = 8_000
    for planned, component_layout, compiled in _compiled(name):
        postselected = np.ones(compiled.num_outputs, dtype=bool)
        postselected[-1] = False
        layout = sharing_layout(compiled, postselected)
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(component_layout, patterns, shots_each)
        samples, _, _, _ = sample_shared_component(
            compiled, layout, jnp.asarray(f_params), jax.random.key(5)
        )
        samples = np.asarray(samples)
        for i, f in enumerate(patterns):
            rows = samples[i * shots_each : (i + 1) * shots_each]
            discarded = np.asarray((rows & postselected).any(axis=1), dtype=bool)
            expected = _postselected_distribution(
                plan_distribution(planned, f), postselected
            )
            counts = {}
            for row, dropped in zip(
                rows.astype(int).tolist(), discarded.tolist(), strict=True
            ):
                key = ("discarded",) if dropped else tuple(row)
                counts[key] = counts.get(key, 0) + 1
            for key in set(counts) | set(expected):
                p = expected.get(key, 0.0)
                sigma = np.sqrt(shots_each * p * (1 - p))
                assert abs(counts.get(key, 0) - shots_each * p) <= 5 * sigma + 3, (
                    key,
                    counts.get(key, 0),
                    shots_each * p,
                )


def test_full_tables_grow_and_resume(monkeypatch):
    monkeypatch.setattr(shared, "_MIN_CAPACITY", 1)
    shots_each = 5_000
    stats = []
    for name in ["t_ghz", "midcircuit_feedforward", "injection_check"]:
        for planned, component_layout, compiled in _compiled(name):
            postselected = np.zeros(compiled.num_outputs, dtype=bool)
            postselected[0] = True
            layout = sharing_layout(compiled, postselected)
            patterns = f_patterns(planned.num_f, limit=3)
            f_params = _f_params(component_layout, patterns, shots_each)
            samples, _, _, stat = sample_shared_component(
                compiled, layout, jnp.asarray(f_params), jax.random.key(11)
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
