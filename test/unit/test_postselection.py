"""Unit tests for CompiledDetectorSampler.sample postselection_mask feature."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import stim

import tsim.sampler as sampler_module
from tsim.circuit import Circuit

# ────────────────────────── shared circuits ──────────────────────────────────

# Detector 0 is direct (single X_ERROR -> M -> DETECTOR).
# Detector 1 is a compiled component (DETECTOR rec[-1] rec[-1] is trivially 0
# but involves a JAX component because it entangles with the ZX diagram).
MIXED_DIRECT_CIRCUIT = """
X_ERROR(0.5) 0
R 1
H 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1] rec[-2]
"""

FULLY_DIRECT_CIRCUIT = """
X_ERROR(0.5) 0
M 0
DETECTOR rec[-1]
"""

ALWAYS_DISCARD_CIRCUIT = """
X_ERROR(1) 0
R 1
H 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1] rec[-2]
"""

ALWAYS_DISCARD_OBS_CIRCUIT = """
X_ERROR(1) 0
R 1
H 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""

# Circuit with detectors and an observable for output-layout tests.
DET_OBS_CIRCUIT = """
R 0 1 2
X 2
M 0 1 2
DETECTOR rec[-2]
DETECTOR rec[-3]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def _make(circuit_str: str, seed: int = 0):
    return Circuit(circuit_str).compile_detector_sampler(seed=seed)


def _keep(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean row mask of shots not discarded by postselection."""
    return ~np.any(samples & mask, axis=1)


# ────────────────────────── validation ───────────────────────────────────────


def test_postselection_mask_wrong_length_raises():
    sampler = _make(MIXED_DIRECT_CIRCUIT)
    with pytest.raises(ValueError, match="postselection_mask must have shape"):
        sampler.sample(1, postselection_mask=np.array([True, False, False]))


def test_postselection_mask_wrong_ndim_raises():
    sampler = _make(MIXED_DIRECT_CIRCUIT)
    with pytest.raises(ValueError, match="postselection_mask must have shape"):
        sampler.sample(1, postselection_mask=np.zeros((2, 1), dtype=np.bool_))


def test_postselection_negative_shots_raises():
    sampler = _make(MIXED_DIRECT_CIRCUIT)
    with pytest.raises(ValueError, match="shots must be non-negative"):
        sampler.sample(-1, postselection_mask=np.array([True, False]))


def test_postselection_invalid_batch_size_raises():
    sampler = _make(MIXED_DIRECT_CIRCUIT)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        sampler.sample(1, batch_size=0, postselection_mask=np.array([True, False]))


# ────────────────────────── basic shape / identity ───────────────────────────


def test_postselection_none_matches_default():
    """postselection_mask=None must be bit-identical to omitting the argument."""
    a = _make(MIXED_DIRECT_CIRCUIT, seed=5).sample(16, batch_size=4)
    b = _make(MIXED_DIRECT_CIRCUIT, seed=5).sample(
        16, batch_size=4, postselection_mask=None
    )
    assert np.array_equal(a, b)


def test_postselection_return_shape_preserved():
    """Always return exactly (shots, num_detectors)."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=0)
    mask = np.array([True, False])
    assert sampler.sample(0, postselection_mask=mask).shape == (0, 2)
    assert sampler.sample(1, postselection_mask=mask).shape == (1, 2)
    assert sampler.sample(17, batch_size=4, postselection_mask=mask).shape == (17, 2)


def test_postselection_zero_shots():
    sampler = _make(MIXED_DIRECT_CIRCUIT)
    mask = np.array([True, False])
    assert sampler.sample(0, postselection_mask=mask).shape == (0, 2)


def test_postselection_all_false_mask_matches_default():
    """All-False mask → no JAX skipped; survivors == all shots."""
    mask = np.zeros(2, dtype=np.bool_)
    a = _make(MIXED_DIRECT_CIRCUIT, seed=7).sample(20, batch_size=5)
    b = _make(MIXED_DIRECT_CIRCUIT, seed=7).sample(
        20, batch_size=5, postselection_mask=mask
    )
    assert np.array_equal(a, b)


# ────────────────────────── discard / partial-row semantics ──────────────────


def test_postselection_discarded_rows_component_cols_false():
    """Discarded rows: direct col truthful, component cols all False."""
    sampler = _make(ALWAYS_DISCARD_CIRCUIT, seed=0)
    mask = np.array([True, False])
    samples = sampler.sample(20, batch_size=4, postselection_mask=mask)

    # All shots discarded because det0 always fires.
    assert np.all(samples[:, 0])
    assert np.all(~samples[:, 1])


def test_postselection_discarded_and_surviving_rows():
    """With 50% noise, both discarded and surviving rows appear."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=2)
    mask = np.array([True, False])
    samples = sampler.sample(64, batch_size=8, postselection_mask=mask)

    discarded = samples[:, 0] & mask[0]
    assert discarded.any(), "expected some discards with 50% noise"
    assert (~discarded).any(), "expected some survivors"

    # Component col False for every discarded row.
    assert np.all(~samples[discarded, 1])


def test_postselection_direct_cols_always_equal_numpy():
    """Direct output columns match NumPy computation for every row (discard or not)."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=3)
    mask = np.array([True, False])
    drawn: list[np.ndarray] = []
    original = sampler._channel_sampler.sample

    def capture(n: int) -> np.ndarray:
        batch = original(n)
        drawn.append(batch.copy())
        return batch

    with patch.object(sampler._channel_sampler, "sample", side_effect=capture):
        samples = sampler.sample(8, batch_size=4, postselection_mask=mask)

    f_all = np.concatenate(drawn)
    expected_direct = sampler._compute_direct_outputs(f_all)
    assert np.array_equal(samples & sampler._direct_output_mask, expected_direct)


# ────────────────────────── JAX-skip behaviour ───────────────────────────────


def test_postselection_jax_never_called_for_all_direct_discards(monkeypatch):
    """When every shot is discarded by a direct detector, sample_program is never called."""
    sampler = _make(ALWAYS_DISCARD_CIRCUIT, seed=0)
    mask = np.array([True, False])
    calls: list[int] = []

    original = sampler_module.sample_program

    def spy(program, f_params, key):
        calls.append(f_params.shape[0])
        return original(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", spy)
    sampler.sample(10, batch_size=4, postselection_mask=mask)
    assert calls == []


def test_postselection_jax_rows_less_than_shots():
    """Total JAX rows < shots when some shots are direct-discarded."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=0)
    mask = np.array([True, False])
    jax_rows: list[int] = []

    original = sampler_module.sample_program

    def spy(program, f_params, key):
        jax_rows.append(f_params.shape[0])
        return original(program, f_params, key)

    with patch.object(sampler_module, "sample_program", side_effect=spy):
        samples = sampler.sample(32, batch_size=8, postselection_mask=mask)

    discarded = np.any(samples[:, : sampler._num_detectors] & mask, axis=1)
    assert sum(jax_rows) < 32
    assert sum(jax_rows) >= int((~discarded).sum())


def test_postselection_jax_batch_size_fixed(monkeypatch):
    """Every JAX call uses the same batch_size (no recompilation)."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=4)
    mask = np.array([True, False])
    seen: list[int] = []

    original = sampler_module.sample_program

    def spy(program, f_params, key):
        seen.append(f_params.shape[0])
        return original(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", spy)
    sampler.sample(10, batch_size=4, postselection_mask=mask)

    assert seen, "expected at least one JAX call for surviving shots"
    assert all(b == 4 for b in seen), f"non-uniform batch sizes: {seen}"


def test_postselection_non_direct_mask_runs_jax_for_all(monkeypatch):
    """Mask on non-direct detector only → JAX runs for every shot."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=9)
    mask = np.array([False, True])  # det1 is a component
    jax_rows: list[int] = []

    original = sampler_module.sample_program

    def spy(program, f_params, key):
        jax_rows.append(f_params.shape[0])
        return original(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", spy)
    sampler.sample(16, batch_size=8, postselection_mask=mask)
    assert sum(jax_rows) == 16


def test_postselection_mixed_mask_skips_jax_only_on_direct_discard(monkeypatch):
    """Mixed mask: JAX skipped only when a masked direct detector fires."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=1)
    mask = np.array([True, True])
    jax_rows: list[int] = []

    original = sampler_module.sample_program

    def spy(program, f_params, key):
        jax_rows.append(f_params.shape[0])
        return original(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", spy)
    samples = sampler.sample(32, batch_size=8, postselection_mask=mask)
    direct_discarded = samples[:, 0] & mask[0]
    assert direct_discarded.any()
    assert sum(jax_rows) < 32
    assert sum(jax_rows) >= int((~direct_discarded).sum())


def test_postselection_detector_reference_xor_before_discard_check(monkeypatch):
    """Discard check uses XOR'd detectors; ref fire cancels raw fire on direct det."""
    circuit = """
    X 0
    X_ERROR(0.5) 0
    R 1
    H 1
    M 0 1
    DETECTOR rec[-2]
    DETECTOR rec[-1] rec[-2]
    """
    mask = np.array([True, False])
    jax_without: list[int] = []
    jax_with: list[int] = []
    original = sampler_module.sample_program

    def make_spy(store: list[int]):
        def spy(program, f_params, key):
            store.append(f_params.shape[0])
            return original(program, f_params, key)

        return spy

    s1 = _make(circuit, seed=0)
    monkeypatch.setattr(sampler_module, "sample_program", make_spy(jax_without))
    s1.sample(32, batch_size=8, postselection_mask=mask)

    s2 = _make(circuit, seed=0)
    monkeypatch.setattr(sampler_module, "sample_program", make_spy(jax_with))
    s2.sample(
        32,
        batch_size=8,
        postselection_mask=mask,
        use_detector_reference_sample=True,
    )

    assert sum(jax_with) > sum(jax_without)


def test_postselection_observable_reference_on_jax_computed_discarded_rows():
    """Observable ref XOR applies to every row that ran JAX, including caller discards."""
    circuit = """
    X 0
    X_ERROR(0.5) 0
    R 1
    H 1
    M 0 1
    DETECTOR rec[-2]
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-1]
    """
    mask = np.array([True, True])
    kwargs = {
        "batch_size": 8,
        "use_observable_reference_sample": True,
        "separate_observables": True,
    }
    sampler = _make(circuit, seed=2)
    captured: dict[str, np.ndarray] = {}
    original = sampler._sample_batches_with_postselection

    def capture(*args, **kwargs):
        samples, reference, direct_discarded = original(*args, **kwargs)
        assert reference is not None
        captured["raw_obs"] = samples[:, sampler._num_detectors :].copy()
        captured["direct_discarded"] = direct_discarded.copy()
        captured["obs_ref"] = reference[sampler._num_detectors :].copy()
        return samples, reference, direct_discarded

    sampler._sample_batches_with_postselection = capture
    _dets, obs = sampler.sample(128, postselection_mask=mask, **kwargs)

    raw_obs = captured["raw_obs"]
    direct_discarded = captured["direct_discarded"]
    obs_ref = captured["obs_ref"]
    expected = raw_obs.copy()
    expected[~direct_discarded] ^= obs_ref

    assert np.all(obs_ref)
    assert np.array_equal(obs, expected)
    assert not np.any(obs[direct_discarded])
    ran_jax = ~direct_discarded
    assert np.any(obs[ran_jax] != raw_obs[ran_jax])


# ────────────────────────── fully-direct fast path ───────────────────────────


def test_postselection_fully_direct_no_jax(monkeypatch):
    """Fully-direct circuits never call sample_program."""
    sampler = _make(FULLY_DIRECT_CIRCUIT, seed=0)
    mask = np.array([True])
    spy = []

    def counting_sp(program, f_params, key):
        spy.append(1)
        return sampler_module.sample_program(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", counting_sp)
    result = sampler.sample(10, postselection_mask=mask)
    assert result.shape == (10, 1)
    assert spy == []


def test_postselection_fully_direct_matches_default():
    """Fully-direct + all-False mask → identical to default sampling."""
    a = _make(FULLY_DIRECT_CIRCUIT, seed=11).sample(50)
    b = _make(FULLY_DIRECT_CIRCUIT, seed=11).sample(
        50, postselection_mask=np.zeros(1, dtype=np.bool_)
    )
    assert np.array_equal(a, b)


def test_postselection_fully_direct_detector_reference():
    """Fully-direct circuits fall back to the standard path, including detector ref."""
    mask = np.array([True])
    with_mask = _make(FULLY_DIRECT_CIRCUIT, seed=0).sample(
        12, postselection_mask=mask, use_detector_reference_sample=True
    )
    without = _make(FULLY_DIRECT_CIRCUIT, seed=0).sample(
        12, use_detector_reference_sample=True
    )
    assert np.array_equal(with_mask, without)


# ────────────────────────── reference sample interaction ─────────────────────


def test_postselection_with_detector_reference_no_crash():
    """use_detector_reference_sample combined with postselection_mask must not raise."""
    sampler = _make(DET_OBS_CIRCUIT, seed=0)
    mask = np.zeros(2, dtype=np.bool_)
    result = sampler.sample(
        8, postselection_mask=mask, use_detector_reference_sample=True
    )
    assert result.shape == (8, 2)


def test_postselection_detector_reference_matches_unmasked():
    """All-false mask + detector ref must match sampling without postselection."""
    mask = np.zeros(2, dtype=np.bool_)
    kwargs = {"batch_size": 4, "use_detector_reference_sample": True}
    with_ref = _make(MIXED_DIRECT_CIRCUIT, seed=0).sample(
        24, postselection_mask=mask, **kwargs
    )
    without = _make(MIXED_DIRECT_CIRCUIT, seed=0).sample(24, **kwargs)
    assert np.array_equal(with_ref, without)


def test_postselection_detector_reference_survivors_and_discarded():
    """Detector ref XOR applies to both survivor and discarded rows."""
    mask = np.array([True, False])
    kwargs = {"batch_size": 8, "use_detector_reference_sample": True}
    samples = _make(MIXED_DIRECT_CIRCUIT, seed=3).sample(
        64, postselection_mask=mask, **kwargs
    )
    keep = _keep(samples, mask)
    assert keep.any() and (~keep).any()
    assert not np.any(samples[keep] & mask)
    assert np.all(samples[~keep, 0])
    assert np.all(~samples[~keep, 1])


def test_postselection_reference_does_not_advance_channel_rng():
    """_compute_reference_sample must not draw from channel_sampler RNG."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=0)
    original = sampler._channel_sampler.sample
    calls: list[int] = []

    def spy(n: int) -> np.ndarray:
        calls.append(n)
        return original(n)

    with patch.object(sampler._channel_sampler, "sample", side_effect=spy):
        sampler._compute_reference_sample()

    assert calls == [], (
        "_compute_reference_sample must not call channel_sampler.sample; "
        f"got calls {calls}"
    )


def test_postselection_with_observable_reference():
    sampler = _make(DET_OBS_CIRCUIT, seed=3)
    mask = np.zeros(2, dtype=np.bool_)
    dets, obs = sampler.sample(
        8,
        postselection_mask=mask,
        separate_observables=True,
        use_observable_reference_sample=True,
    )
    assert dets.shape == (8, 2)
    assert obs.shape == (8, 1)


def test_postselection_observable_reference_skipped_on_discarded():
    """Observable ref XOR must not fill discarded rows' uncomputed obs columns."""
    mask = np.array([True, False])
    dets, obs = _make(ALWAYS_DISCARD_OBS_CIRCUIT, seed=1).sample(
        16,
        batch_size=8,
        postselection_mask=mask,
        separate_observables=True,
        use_observable_reference_sample=True,
    )
    discarded = np.any(dets & mask, axis=1)
    assert np.all(discarded)
    assert not np.any(obs)


# ────────────────────────── output-layout flags ──────────────────────────────


def test_postselection_output_layout_append_observables():
    sampler = _make(DET_OBS_CIRCUIT, seed=0)
    mask = np.array([True, False])
    result = sampler.sample(4, postselection_mask=mask, append_observables=True)
    assert result.shape == (4, 3)


def test_postselection_output_layout_prepend_observables():
    sampler = _make(DET_OBS_CIRCUIT, seed=0)
    mask = np.array([True, False])
    result = sampler.sample(4, postselection_mask=mask, prepend_observables=True)
    assert result.shape == (4, 3)


def test_postselection_output_layout_separate_observables():
    sampler = _make(DET_OBS_CIRCUIT, seed=0)
    mask = np.array([True, False])
    dets, obs = sampler.sample(4, postselection_mask=mask, separate_observables=True)
    assert dets.shape == (4, 2)
    assert obs.shape == (4, 1)


def test_postselection_output_layout_bit_packed():
    sampler = _make(DET_OBS_CIRCUIT, seed=0)
    mask = np.array([True, False])
    result = sampler.sample(4, postselection_mask=mask, bit_packed=True)
    assert result.dtype == np.uint8
    assert result.shape == (4, 1)


# ────────────────────────── surface-code integration ─────────────────────────


def test_postselection_surface_code_fully_direct_unchanged():
    """Surface-code detectors are all direct; mask=0 must not change samples."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
    )
    c = Circuit.from_stim_program(circ)
    assert (
        c.compile_detector_sampler()._direct_detector_mask.all()
    ), "expected all detectors direct for this circuit"

    mask = np.zeros(c.num_detectors, dtype=np.bool_)
    a = c.compile_detector_sampler(seed=0).sample(100, batch_size=16)
    b = c.compile_detector_sampler(seed=0).sample(
        100, batch_size=16, postselection_mask=mask
    )
    assert np.array_equal(a, b)


def test_postselection_surface_code_caller_filter():
    """After postselection, caller filters survivors via mask; they have no fired detectors."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
    )
    c = Circuit.from_stim_program(circ)
    num_det = c.num_detectors
    mask = np.zeros(num_det, dtype=np.bool_)
    mask[0] = True

    samples = c.compile_detector_sampler(seed=0).sample(
        200, batch_size=32, postselection_mask=mask
    )
    survivors = _keep(samples[:, :num_det], mask)
    assert survivors.any()
    assert not np.any(samples[survivors] & mask)
