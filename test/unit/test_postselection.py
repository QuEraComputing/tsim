"""Unit tests for CompiledDetectorSampler.sample postselection_mask feature."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from test.helpers.statevector import set_sharing
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import stim

import tsim.sampler as sampler_module
import tsim.statevector.shared as shared
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

# Detector 0 is constant 1 (no noise); detector 1 is a random compiled component.
CONSTANT_AND_COMPONENT_CIRCUIT = """
R 0 1
X 0
H 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
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


def _direct_outputs(sampler, f_params: np.ndarray) -> np.ndarray:
    """Return the direct outputs of ``f_params`` in their columns, False elsewhere."""
    program = sampler._program
    columns = program.output_order[: len(program.direct_f_indices)]
    out = np.zeros((len(f_params), program.num_outputs), dtype=np.bool_)
    out[:, columns] = program.direct_flips
    has_f = program.direct_f_indices >= 0
    out[:, columns[has_f]] ^= f_params[:, program.direct_f_indices[has_f]] != 0
    return out


def _keep(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean row mask of shots not discarded by postselection."""
    return ~np.any(samples & mask, axis=1)


@contextmanager
def _record_noise(sampler) -> Iterator[Callable[[], np.ndarray]]:
    """Record the noise ``sampler`` draws; yield a function returning its f-params.

    Every batch or call draws the seeds of its noise blocks in shot order, and a
    block's seed alone fixes its faults, so the f-params are drawn again from the
    recorded seeds.
    """
    channel_sampler = sampler._channel_sampler
    drawn: list[tuple[int, np.ndarray]] = []
    block_seeds, sample = channel_sampler.block_seeds, channel_sampler.sample

    def record_seeds(n: int) -> np.ndarray:
        seeds = block_seeds(n)
        drawn.append((n, seeds))
        return seeds

    with patch.object(channel_sampler, "block_seeds", side_effect=record_seeds):
        yield lambda: np.concatenate([sample(n, seeds) for n, seeds in drawn])


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


def test_postselection_returns_only_survivors():
    """Return at most ``shots`` rows, none of which fires a masked detector."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=0)
    mask = np.array([True, False])
    assert sampler.sample(0, postselection_mask=mask).shape == (0, 2)
    samples = sampler.sample(17, batch_size=4, postselection_mask=mask)
    assert 0 < len(samples) < 17
    assert samples.shape[1] == 2
    assert not np.any(samples & mask)


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


@pytest.mark.parametrize("strategy", ["cat5", "active_statevector"])
@pytest.mark.parametrize("bit_packed", [False, True])
def test_postselection_all_discarded_returns_no_rows(strategy, bit_packed):
    sampler = Circuit(ALWAYS_DISCARD_OBS_CIRCUIT).compile_detector_sampler(
        strategy=strategy, seed=0
    )
    mask = np.array([True, False])
    dets, obs = sampler.sample(
        20,
        batch_size=4,
        postselection_mask=mask,
        separate_observables=True,
        bit_packed=bit_packed,
    )
    assert dets.shape == (0, 1 if bit_packed else 2)
    assert obs.shape == (0, 1)


@pytest.mark.parametrize("strategy", ["cat5", "active_statevector"])
def test_postselection_acceptance_rate(strategy):
    sampler = Circuit(MIXED_DIRECT_CIRCUIT).compile_detector_sampler(
        strategy=strategy, seed=2
    )
    shots = 4_000
    samples = sampler.sample(
        shots, batch_size=512, postselection_mask=np.array([True, False])
    )
    assert abs(len(samples) / shots - 0.5) < 5 * np.sqrt(0.25 / shots)
    assert not samples[:, 0].any()


# Detector 0 is postselected; direct detector 2 and observable 0 are not, so a
# survivor sampled from another shot's f-params shows up in their columns. The X
# gates make the reference bits 1, and detectors 0 and 1 share an f-bit.
DIRECT_CIRCUIT = """
X 0 2
X_ERROR(0.5) 0 2
R 1
H 1
T 1
H 1
M 0 1 2
DETECTOR rec[-3]
DETECTOR rec[-2] rec[-3]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.mark.parametrize("shots,batch_size", [(37, 4), (37, 1), (5, 8), (64, 7)])
@pytest.mark.parametrize("reference", [False, True])
def test_survivors_keep_their_own_direct_outputs(shots, batch_size, reference):
    """Direct outputs come from each shot's own f-params, flips and reference."""
    sampler = _make(DIRECT_CIRCUIT, seed=11)
    mask = np.array([True, False, False])
    with _record_noise(sampler) as drawn:
        samples = sampler.sample(
            shots,
            batch_size=batch_size,
            postselection_mask=mask,
            append_observables=True,
            use_detector_reference_sample=reference,
        )
    expected = _direct_outputs(sampler, drawn()[:shots])
    if reference:
        expected[:, :3] ^= sampler._reference_sample()[:3]
    expected = expected[~expected[:, 0]]
    assert np.array_equal(samples[:, [0, 2, 3]], expected[:, [0, 2, 3]])


def test_postselection_direct_cols_always_equal_numpy():
    """Direct detector columns match NumPy computation for every row."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=3)
    mask = np.array([True, False])
    with _record_noise(sampler) as drawn:
        samples = sampler.sample(8, batch_size=4, postselection_mask=mask)

    expected_direct = _direct_outputs(sampler, drawn()[:8])
    expected_direct = expected_direct[~expected_direct[:, 0]]
    direct_det = sampler._direct_detector_mask
    assert np.array_equal(
        samples[:, : sampler._num_detectors] & direct_det,
        expected_direct[:, : sampler._num_detectors] & direct_det,
    )


# ────────────────────────── JAX-skip behaviour ───────────────────────────────


def _record_rows(monkeypatch, store: list[int]) -> None:
    """Append the shots that reach the components to ``store``, per call.

    With JAX those are the rows of every ``sample_program`` call, and with shot
    words the live shots of every ``run_lanes`` call.
    """
    sample_program, run_lanes = sampler_module.sample_program, shared.run_lanes

    def spy_program(program, f_params, *args, **kwargs):
        store.append(f_params.shape[0])
        return sample_program(program, f_params, *args, **kwargs)

    def spy_lanes(work, words, alive, *args):
        store.append(int(alive.sum()))
        return run_lanes(work, words, alive, *args)

    monkeypatch.setattr(sampler_module, "sample_program", spy_program)
    monkeypatch.setattr(shared, "run_lanes", spy_lanes)


def test_components_never_run_for_all_direct_discards(monkeypatch):
    """When every shot is discarded by a direct detector, no component runs."""
    sampler = _make(ALWAYS_DISCARD_CIRCUIT, seed=0)
    mask = np.array([True, False])
    calls: list[int] = []

    _record_rows(monkeypatch, calls)
    sampler.sample(10, batch_size=4, postselection_mask=mask)
    assert calls == []


@pytest.mark.parametrize("use_reference", [False, True])
@pytest.mark.parametrize("strategy", ["cat5", "active_statevector"])
def test_postselection_on_a_constant_detector(monkeypatch, use_reference, strategy):
    sampler = Circuit(CONSTANT_AND_COMPONENT_CIRCUIT).compile_detector_sampler(
        strategy=strategy, seed=0
    )
    assert sampler._direct_detector_mask.tolist() == [True, False]
    mask = np.array([True, False])
    calls: list[int] = []

    _record_rows(monkeypatch, calls)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        samples = sampler.sample(
            16,
            batch_size=4,
            postselection_mask=mask,
            use_detector_reference_sample=use_reference,
        )
    always_read_1 = [w for w in caught if "always read 1" in str(w.message)]
    if use_reference:
        assert always_read_1 == []
        assert sum(calls) >= 16
        assert samples.shape == (16, 2)
        assert not samples[:, 0].any()
        assert samples[:, 1].any()
    else:
        assert len(always_read_1) == 1
        assert "[0]" in str(always_read_1[0].message)
        assert calls == []
        assert samples.shape == (0, 2)


@pytest.mark.parametrize("constant,use_reference", [(0, False), (0, True), (1, True)])
def test_nonrejecting_constant_postselection_uses_normal_batches(
    monkeypatch, constant, use_reference
):
    circuit = CONSTANT_AND_COMPONENT_CIRCUIT
    if not constant:
        circuit = circuit.replace("X 0", "")
    # NOTE: only the stabilizer-rank strategy buffers postselection survivors.
    sampler = Circuit(circuit).compile_detector_sampler(strategy="cat5", seed=0)

    def fail(*args, **kwargs):
        raise AssertionError("A nonrejecting constant must not buffer survivors")

    monkeypatch.setattr(sampler, "_sample_batches_with_postselection", fail)
    mask = np.array([True, False])
    samples = sampler.sample(
        19,
        batch_size=5,
        postselection_mask=mask,
        use_detector_reference_sample=use_reference,
    )
    assert not samples[:, 0].any()
    assert mask.tolist() == [True, False]


@pytest.mark.parametrize("strategy", ["cat5", "active_statevector"])
@pytest.mark.parametrize("use_reference", [False, True])
def test_constant_outputs_with_component_postselection(strategy, use_reference):
    circuit = Circuit("""
        X 0
        H 1
        T 1
        H 1
        X_ERROR(0.2) 2
        M 0 1 2
        DETECTOR rec[-1]
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-3]
        """)
    sampler = circuit.compile_detector_sampler(strategy=strategy, seed=0)
    mask = np.array([True, True, False])
    detectors, observables = sampler.sample(
        128,
        batch_size=17,
        postselection_mask=mask,
        separate_observables=True,
        use_observable_reference_sample=use_reference,
    )
    assert 0 < len(detectors) < 128
    assert len(observables) == len(detectors)
    assert not detectors[:, :2].any()
    assert detectors[:, 2].all()
    assert (observables[:, 0] == (not use_reference)).all()


def test_components_run_on_fewer_shots_than_sampled():
    """The components see fewer shots than sampled when some are direct-discarded."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=0)
    mask = np.array([True, False])
    rows: list[int] = []

    with pytest.MonkeyPatch.context() as monkeypatch:
        _record_rows(monkeypatch, rows)
        samples = sampler.sample(32, batch_size=8, postselection_mask=mask)

    assert sum(rows) < 32
    assert sum(rows) >= len(samples)


def test_postselection_jax_batch_size_fixed(monkeypatch):
    """Every JAX call uses the same batch_size (no recompilation)."""
    sampler = Circuit(MIXED_DIRECT_CIRCUIT).compile_detector_sampler(
        strategy="cat5", seed=4
    )
    mask = np.array([True, False])
    seen: list[int] = []

    _record_rows(monkeypatch, seen)
    sampler.sample(10, batch_size=4, postselection_mask=mask)

    assert seen, "expected at least one JAX call for surviving shots"
    assert all(b == 4 for b in seen), f"non-uniform batch sizes: {seen}"


def test_postselection_non_direct_mask_runs_components_for_all(monkeypatch):
    """Mask on non-direct detector only → the components run for every shot."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=9)
    mask = np.array([False, True])  # det1 is a component
    rows: list[int] = []

    _record_rows(monkeypatch, rows)
    sampler.sample(16, batch_size=8, postselection_mask=mask)
    assert sum(rows) == 16


def test_postselection_mixed_mask_skips_jax_only_on_direct_discard(monkeypatch):
    """Mixed mask: JAX skipped only when a masked direct detector fires."""
    sampler = _make(MIXED_DIRECT_CIRCUIT, seed=1)
    mask = np.array([True, True])
    rows: list[int] = []

    _record_rows(monkeypatch, rows)
    samples = sampler.sample(32, batch_size=8, postselection_mask=mask)
    assert sum(rows) < 32
    assert sum(rows) >= len(samples)


def test_postselection_detector_reference_xor_before_discard_check(monkeypatch):
    """Discard check uses XOR'd detectors; ref fire cancels raw fire on direct det."""
    # NOTE: detector 0 fires in 90% of shots, and its reference bit is 1.
    circuit = """
    X 0
    X_ERROR(0.1) 0
    R 1
    H 1
    M 0 1
    DETECTOR rec[-2]
    DETECTOR rec[-1] rec[-2]
    """
    mask = np.array([True, False])
    jax_without: list[int] = []
    jax_with: list[int] = []
    s1 = _make(circuit, seed=0)
    with monkeypatch.context() as patch_rows:
        _record_rows(patch_rows, jax_without)
        s1.sample(32, batch_size=8, postselection_mask=mask)

    s2 = _make(circuit, seed=0)
    _record_rows(monkeypatch, jax_with)
    s2.sample(
        32,
        batch_size=8,
        postselection_mask=mask,
        use_detector_reference_sample=True,
    )

    # NOTE: the reference sample itself is one more row through sample_program.
    assert sum(jax_with) - 1 > sum(jax_without)


@pytest.mark.parametrize("strategy", ["cat5", "active_statevector"])
def test_postselection_observable_reference_flips_every_sampled_row(
    strategy, monkeypatch
):
    """The observable reference flips the observable of every returned row."""
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
    mask = np.array([True, False])

    def observables(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sampler = Circuit(circuit).compile_detector_sampler(strategy=strategy, seed=2)
        monkeypatch.setattr(sampler, "_reference_sample", lambda: reference)
        return sampler.sample(
            128,
            batch_size=8,
            postselection_mask=mask,
            separate_observables=True,
            use_observable_reference_sample=True,
        )

    dets, raw = observables(np.zeros(3, dtype=np.bool_))
    _, flipped = observables(np.array([False, False, True]))
    assert 0 < len(dets) < 128
    assert not dets[:, 0].any()
    np.testing.assert_array_equal(flipped, ~raw)


# ────────────────────────── fully-direct fast path ───────────────────────────


def test_postselection_fully_direct_no_jax(monkeypatch):
    """Fully-direct circuits never call sample_program."""
    sampler = _make(FULLY_DIRECT_CIRCUIT, seed=0)
    mask = np.array([True])
    spy: list[int] = []
    _record_rows(monkeypatch, spy)
    result = sampler.sample(10, postselection_mask=mask)
    assert len(result) <= 10
    assert not result.any()
    assert spy == []


def test_postselection_fully_direct_matches_default():
    """Fully-direct + all-False mask → identical to default sampling."""
    a = _make(FULLY_DIRECT_CIRCUIT, seed=11).sample(50)
    b = _make(FULLY_DIRECT_CIRCUIT, seed=11).sample(
        50, postselection_mask=np.zeros(1, dtype=np.bool_)
    )
    assert np.array_equal(a, b)


def test_postselection_fully_direct_detector_reference():
    """With a detector reference, fully-direct survivors are the unmasked samples."""
    mask = np.array([True])
    with_mask = _make(FULLY_DIRECT_CIRCUIT, seed=0).sample(
        12, postselection_mask=mask, use_detector_reference_sample=True
    )
    without = _make(FULLY_DIRECT_CIRCUIT, seed=0).sample(
        12, use_detector_reference_sample=True
    )
    assert np.array_equal(with_mask, without[~without[:, 0]])


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


def test_postselection_detector_reference_survivors():
    """The masked detectors of every returned row read 0 after the reference XOR."""
    mask = np.array([True, False])
    kwargs = {"batch_size": 8, "use_detector_reference_sample": True}
    samples = _make(MIXED_DIRECT_CIRCUIT, seed=3).sample(
        64, postselection_mask=mask, **kwargs
    )
    assert 0 < len(samples) < 64
    assert not np.any(samples & mask)


@pytest.mark.parametrize("strategy", ["cat5", "active_statevector"])
def test_postselection_reference_does_not_advance_channel_rng(strategy):
    """The reference sample must not draw from the channel sampler's RNG."""
    sampler = Circuit(MIXED_DIRECT_CIRCUIT).compile_detector_sampler(
        strategy=strategy, seed=0
    )
    channel_sampler = sampler._channel_sampler
    calls: list[str] = []
    with (
        patch.object(
            channel_sampler, "block_seeds", side_effect=lambda n: calls.append("seeds")
        ),
        patch.object(
            channel_sampler, "sample", side_effect=lambda *a: calls.append("sample")
        ),
    ):
        sampler._reference_sample()

    assert calls == [], f"the reference sample drew noise: {calls}"


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


def test_postselection_observable_reference_with_all_discarded():
    mask = np.array([True, False])
    dets, obs = _make(ALWAYS_DISCARD_OBS_CIRCUIT, seed=1).sample(
        16,
        batch_size=8,
        postselection_mask=mask,
        separate_observables=True,
        use_observable_reference_sample=True,
    )
    assert dets.shape == (0, 2)
    assert obs.shape == (0, 1)


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


def test_postselection_surface_code_survivors():
    """Postselection returns the shots in which no masked detector fires."""
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

    unmasked = c.compile_detector_sampler(seed=0).sample(200, batch_size=32)
    samples = c.compile_detector_sampler(seed=0).sample(
        200, batch_size=32, postselection_mask=mask
    )
    assert 0 < len(samples) < 200
    assert not np.any(samples & mask)
    np.testing.assert_array_equal(samples, unmasked[_keep(unmasked, mask)])


# Detector 0 and observable 0 are direct, with no statevector component.
DIRECT_ONLY_OBS_CIRCUIT = """
X_ERROR(0.5) 0 1
M 0 1
DETECTOR rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.mark.parametrize("strategy", ["active_statevector", "cat5"])
@pytest.mark.parametrize("bit_packed", [False, True])
def test_direct_only_postselection_returns_survivors(strategy, bit_packed):
    """Without components the survivors keep their own observables."""
    sampler = Circuit(DIRECT_ONLY_OBS_CIRCUIT).compile_detector_sampler(
        strategy=strategy, seed=4
    )
    assert not sampler._program.components
    samples = sampler.sample(
        2_000,
        append_observables=True,
        bit_packed=bit_packed,
        postselection_mask=np.array([True]),
    )
    if bit_packed:
        samples = np.unpackbits(samples, axis=1, bitorder="little")[:, :2] == 1
    assert 0 < len(samples) < 2_000
    assert not samples[:, 0].any()
    assert samples[:, 1].any() and not samples[:, 1].all()


# Detector 0 is direct; detector 1, detector 2 and observable 0 are statevector
# components, which run in that order.
TWO_COMPONENTS_CIRCUIT = """
R 0 1 2 3 4
X_ERROR(0.2) 4
H 0 2
T 0 2
H 0 2
T 0 2
CNOT 0 1
CNOT 2 3
X_ERROR(0.1) 1 3
T 1 3
H 0 1 2 3
M 4 0 1 2 3
DETECTOR rec[-5]
DETECTOR rec[-4] rec[-3]
DETECTOR rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-4]
"""


@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("masked", [1, 2])
def test_a_component_discard_skips_the_later_components(monkeypatch, share, masked):
    """The components after a pruning one skip its shots.

    The survivors are the shots of the unmasked call that the mask keeps.
    """
    circuit = Circuit(TWO_COMPONENTS_CIRCUIT)
    shots = 5_000
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[masked] = True

    def sample(postselection_mask, batch_size=None):
        sampler = set_sharing(
            circuit.compile_detector_sampler(strategy="active_statevector", seed=8),
            share,
        )
        components = sampler._program.components
        assert [c.output_indices for c in components] == [(1,), (2,), (3,)]
        return sampler.sample(
            shots,
            batch_size=batch_size,
            append_observables=True,
            postselection_mask=postselection_mask,
        )

    free = sample(None)
    live: list[int] = []
    with monkeypatch.context() as patch_rows:
        _record_rows(patch_rows, live)
        kept = sample(mask)
    survivors = len(kept)
    assert 0 < survivors < shots
    # NOTE: the components before the pruning one see every shot, those after
    # it the survivors alone.
    assert sum(live) == masked * shots + (3 - masked) * survivors
    assert np.array_equal(sample(mask, batch_size=7), kept)
    assert np.array_equal(kept, free[~free[:, masked]])


# Detectors 0 and 1 are stabilizer-rank component outputs; detector 2 is direct.
COMPONENT_DETECTORS_CIRCUIT = """
R 0 1 2
H 0 1
T 0 1
CNOT 0 1
T 1
H 0 1
X_ERROR(0.2) 0 1 2
M 0 1 2
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-2]
"""


@pytest.mark.parametrize("use_reference", [False, True])
@pytest.mark.parametrize("batch_size", [7, 64])
@pytest.mark.parametrize("masked", [[0], [0, 1], [0, 2]])
def test_stabilizer_rank_survivors_are_the_unmasked_samples_kept(
    use_reference, batch_size, masked
):
    circuit = Circuit(COMPONENT_DETECTORS_CIRCUIT)
    mask = np.zeros(circuit.num_detectors, dtype=np.bool_)
    mask[masked] = True

    def sample(postselection_mask):
        sampler = circuit.compile_detector_sampler(strategy="cat5", seed=6)
        assert sampler._direct_detector_mask.tolist() == [False, False, True]
        return sampler.sample(
            300,
            batch_size=batch_size,
            append_observables=True,
            use_detector_reference_sample=use_reference,
            postselection_mask=postselection_mask,
        )

    unmasked = sample(None)
    survivors = sample(mask)
    if 2 not in masked:
        np.testing.assert_array_equal(survivors, unmasked[_keep(unmasked[:, :3], mask)])
    assert 0 < len(survivors) < 300
    assert not np.any(survivors[:, :3] & mask)


@pytest.mark.parametrize("masked", [[0], [0, 2]])
def test_stabilizer_rank_postselection_compiles_once(masked):
    """Survivor counts vary between calls, but the compiled shapes do not."""
    circuit = Circuit(COMPONENT_DETECTORS_CIRCUIT)
    sampler = circuit.compile_detector_sampler(strategy="cat5", seed=0)
    mask = np.zeros(circuit.num_detectors, dtype=np.bool_)
    mask[masked] = True
    compact: Any = sampler_module._compact
    counts = {len(sampler.sample(500, batch_size=128, postselection_mask=mask))}
    compiled = compact._cache_size()
    for _ in range(5):
        counts.add(len(sampler.sample(500, batch_size=128, postselection_mask=mask)))
    assert len(counts) > 1
    assert compact._cache_size() == compiled
