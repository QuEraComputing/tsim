import numpy as np
import pytest
import stim

import tsim.sampler as sampler_module
from tsim.circuit import Circuit

MIXED_DIRECT_CIRCUIT = """
X_ERROR(0.1) 0
M 0
DETECTOR rec[-1]
DETECTOR rec[-1] rec[-1]
"""

FULLY_DIRECT_CIRCUIT = """
X_ERROR(0.5) 0
M 0
DETECTOR rec[-1]
"""

ALWAYS_DISCARD_CIRCUIT = """
X_ERROR(1) 0
M 0
DETECTOR rec[-1]
DETECTOR rec[-1] rec[-1]
"""


def _mixed_sampler(seed: int = 0):
    return Circuit(MIXED_DIRECT_CIRCUIT).compile_detector_sampler(seed=seed)


def _survivor_mask(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return True for rows that survive postselection on detector columns."""
    return ~np.any(samples & mask, axis=1)


def test_postselection_mask_invalid_shape_raises():
    sampler = _mixed_sampler()
    with pytest.raises(ValueError, match="postselection_mask must have shape"):
        sampler.sample(1, postselection_mask=np.array([True, False, True]))


def test_postselection_preserves_row_count():
    sampler = _mixed_sampler(seed=1)
    mask = np.array([True, False])
    samples = sampler.sample(20, postselection_mask=mask, batch_size=4)
    assert samples.shape == (20, 2)


def test_postselection_discarded_rows_have_false_non_direct_columns():
    sampler = _mixed_sampler(seed=2)
    mask = np.array([True, False])
    samples = sampler.sample(100, postselection_mask=mask, batch_size=8)

    # Detector 0 is direct; detector 1 is a compiled component.
    discarded = samples[:, 0] & mask[0]
    assert np.any(discarded)
    assert np.all(~samples[discarded, 1])


def test_postselection_survivors_match_unpostselected_sampling():
    seed = 7
    mask = np.array([True, False])
    sampler = _mixed_sampler(seed=seed)
    with_post = sampler.sample(200, postselection_mask=mask, batch_size=16)

    sampler2 = _mixed_sampler(seed=seed)
    without_post = sampler2.sample(200, batch_size=16)

    survivors = _survivor_mask(with_post, mask)
    assert np.array_equal(with_post[survivors], without_post[survivors])


def test_postselection_none_matches_default():
    seed = 3
    sampler = Circuit(MIXED_DIRECT_CIRCUIT).compile_detector_sampler(seed=seed)
    default = sampler.sample(10, batch_size=4)

    sampler2 = Circuit(MIXED_DIRECT_CIRCUIT).compile_detector_sampler(seed=seed)
    explicit_none = sampler2.sample(10, batch_size=4, postselection_mask=None)

    assert np.array_equal(default, explicit_none)


def test_postselection_fully_direct_matches_unpostselected_sampling():
    seed = 11
    mask = np.array([True])
    sampler = Circuit(FULLY_DIRECT_CIRCUIT).compile_detector_sampler(seed=seed)
    samples = sampler.sample(500, postselection_mask=mask)

    sampler2 = Circuit(FULLY_DIRECT_CIRCUIT).compile_detector_sampler(seed=seed)
    reference = sampler2.sample(500)

    assert np.array_equal(samples, reference)


def test_postselection_skips_jax_for_direct_discards(monkeypatch):
    sampler = Circuit(ALWAYS_DISCARD_CIRCUIT).compile_detector_sampler(seed=0)
    mask = np.array([True, False])
    calls: list[int] = []

    original = sampler_module.sample_program

    def counting_sample_program(program, f_params, key):
        calls.append(int(f_params.shape[0]))
        return original(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", counting_sample_program)

    samples = sampler.sample(10, postselection_mask=mask, batch_size=4)
    assert np.all(samples[:, 0])
    assert np.all(~samples[:, 1])
    assert calls == []


def test_postselection_with_detector_reference_sample():
    seed = 0
    mask = np.array([True, False])
    kwargs = {
        "use_detector_reference_sample": True,
        "batch_size": 4,
    }

    with_post = _mixed_sampler(seed=seed).sample(40, postselection_mask=mask, **kwargs)
    survivors = _survivor_mask(with_post, mask)
    assert survivors.any()
    assert not np.any(with_post[survivors] & mask)

    discarded = ~survivors
    assert discarded.any()
    assert np.all(with_post[discarded, 0])
    assert np.all(~with_post[discarded, 1])


def test_postselection_with_detector_reference_fully_direct():
    mask = np.array([True])
    circuit = Circuit(FULLY_DIRECT_CIRCUIT)
    samples = circuit.compile_detector_sampler(seed=2).sample(
        50,
        postselection_mask=mask,
        use_detector_reference_sample=True,
    )
    survivors = _survivor_mask(samples, mask)
    assert survivors.any()
    assert not np.any(samples[survivors] & mask)


def test_postselection_with_observable_reference_sample():
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    mask = np.array([True, False])
    kwargs = {
        "separate_observables": True,
        "use_observable_reference_sample": True,
        "postselection_mask": mask,
        "batch_size": 4,
    }

    dets1, obs1 = c.compile_detector_sampler(seed=3).sample(20, **kwargs)
    dets2, obs2 = c.compile_detector_sampler(seed=3).sample(20, **kwargs)
    assert np.array_equal(dets1, dets2)
    assert np.array_equal(obs1, obs2)


def test_postselection_zero_shots():
    sampler = _mixed_sampler()
    mask = np.array([True, False])
    result = sampler.sample(0, postselection_mask=mask)
    assert result.shape == (0, 2)

    dets, obs = sampler.sample(0, postselection_mask=mask, separate_observables=True)
    assert dets.shape == (0, 2)
    assert obs.shape == (0, 0)


def test_postselection_negative_shots_raises():
    sampler = _mixed_sampler()
    with pytest.raises(ValueError, match="shots must be non-negative"):
        sampler.sample(-1, postselection_mask=np.array([True, False]))


def test_postselection_invalid_batch_size_raises():
    sampler = _mixed_sampler()
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        sampler.sample(1, batch_size=0, postselection_mask=np.array([True, False]))


def test_postselection_respects_output_layout_flags():
    c = Circuit("""
        R 0 1 2
        X 2
        M 0 1 2
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """)
    sampler = c.compile_detector_sampler(seed=0)
    mask = np.array([True, False])

    appended = sampler.sample(2, append_observables=True, postselection_mask=mask)
    assert appended.shape == (2, 3)

    prepended = sampler.sample(2, prepend_observables=True, postselection_mask=mask)
    assert prepended.shape == (2, 3)

    dets, obs = sampler.sample(2, separate_observables=True, postselection_mask=mask)
    assert dets.shape == (2, 2)
    assert obs.shape == (2, 1)


def test_postselection_non_direct_mask_does_not_skip_jax(monkeypatch):
    """Postselection on a non-direct detector still runs JAX for every shot."""
    sampler = _mixed_sampler(seed=9)
    mask = np.array([False, True])
    jax_rows: list[int] = []

    original = sampler_module.sample_program

    def counting_sample_program(program, f_params, key):
        jax_rows.append(int(f_params.shape[0]))
        return original(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", counting_sample_program)

    sampler.sample(16, postselection_mask=mask, batch_size=8)
    assert sum(jax_rows) == 16


def test_postselection_batch_padding(monkeypatch):
    """Leftover survivors are padded to a full JAX batch and padding is discarded."""
    sampler = _mixed_sampler(seed=4)
    mask = np.array([True, False])
    seen_batch_sizes: list[int] = []

    original = sampler_module.sample_program

    def counting_sample_program(program, f_params, key):
        seen_batch_sizes.append(int(f_params.shape[0]))
        return original(program, f_params, key)

    monkeypatch.setattr(sampler_module, "sample_program", counting_sample_program)

    sampler.sample(10, postselection_mask=mask, batch_size=4)
    assert seen_batch_sizes
    assert all(batch_size == 4 for batch_size in seen_batch_sizes)


def test_postselection_surface_code_fully_direct():
    """Typical QEC circuits are fully direct; postselection must not change samples."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
    )
    c = Circuit.from_stim_program(circ)
    mask = np.zeros(c.num_detectors, dtype=np.bool_)
    mask[0] = True

    sampler = c.compile_detector_sampler(seed=0)
    with_mask = sampler.sample(100, postselection_mask=mask, batch_size=16)
    without_mask = c.compile_detector_sampler(seed=0).sample(100, batch_size=16)
    assert np.array_equal(with_mask, without_mask)


def test_postselection_caller_can_filter_survivors():
    sampler = _mixed_sampler(seed=6)
    mask = np.array([True, False])
    samples = sampler.sample(100, postselection_mask=mask, batch_size=8)

    survivors = _survivor_mask(samples, mask)
    assert survivors.any()
    assert not np.any(samples[survivors] & mask)
    assert np.any(samples[~survivors] & mask)
