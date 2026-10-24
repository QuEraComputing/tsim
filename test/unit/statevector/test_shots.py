from test.helpers.statevector import (
    CIRCUITS,
    coherent_surface_code,
    force_legacy,
    force_trees,
)
from test.integration.test_trajectory_sharing import GHZ_WITH_DIRECT_DETECTOR
from test.unit.statevector.test_tables import _assert_same_rows, _flagged_chain

import numba
import numpy as np
import pytest

import tsim.sampler as sampler_module
import tsim.utils.kernels as kernels
from tsim.circuit import Circuit
from tsim.noise.channels import _BLOCK_SHOTS
from tsim.statevector.program import StatevectorComponent
from tsim.statevector.shared import packed_symbols
from tsim.statevector.shots import sample_shots
from tsim.statevector.tables import _shot_keys as table_shot_keys
from tsim.statevector.trees import _shot_keys as tree_shot_keys

# NOTE: a T-gate GHZ component, a direct detector and a direct observable.
DIRECT = """
R 0 1 2 3
H 0
T 0
CX 0 1
CX 0 2
X_ERROR(0.2) 3
M 3
DETECTOR rec[-1]
X_ERROR(0.1) 3
M 3
OBSERVABLE_INCLUDE(1) rec[-1]
M 0 1 2
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-3]
"""


def _backend(sampler):
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    return backend


@pytest.mark.parametrize("text", [DIRECT, _flagged_chain(8), CIRCUITS["t_ghz"]])
@pytest.mark.parametrize("masked", [False, True])
def test_shot_states_match_the_f_matrix(text, masked):
    """Direct outputs and every component's words are those of the f-bits."""
    sampler = Circuit(text).compile_detector_sampler(seed=4)
    backend = _backend(sampler)
    program = sampler._program
    mask = np.ones(sampler.circuit.num_detectors, bool) if masked else None
    fused = backend.fused(mask, None)
    assert fused is not None
    layout, plans, _ = fused
    shots = 3 * _BLOCK_SHOTS + 17
    channels = sampler._channel_sampler
    seeds = channels.block_seeds(shots)
    f = channels.sample(shots, seeds)
    noise = channels.fire_tables
    out = np.empty((shots, program.num_outputs), dtype=bool)
    state = np.empty((shots, layout.words), dtype=np.uint64)
    discarded = sample_shots(
        0,
        out,
        state,
        seeds,
        _BLOCK_SHOTS,
        noise.p_fire,
        noise.outcome_start,
        noise.cdf,
        layout.signature_start,
        layout.signature_word,
        layout.signature_bits,
        layout.flips,
        layout.screens,
        layout.direct_bit,
        layout.kept,
    )
    _, direct_flips = backend.layouts(mask, None)
    columns = program.output_order[: len(program.direct_f_indices)]
    has_f = program.direct_f_indices >= 0
    direct = np.broadcast_to(direct_flips, (shots, len(columns))).copy()
    direct[:, has_f] ^= f[:, program.direct_f_indices[has_f]] != 0
    is_detector = columns < sampler.circuit.num_detectors
    expected_discard = np.zeros(shots, dtype=bool)
    if mask is not None:
        expected_discard = (
            direct & is_detector & mask[np.minimum(columns, len(mask) - 1)]
        ).any(axis=1)
    np.testing.assert_array_equal(discarded, expected_discard)
    kept = ~discarded[:, None] | is_detector[None, :]
    np.testing.assert_array_equal(out[:, columns], direct & kept)
    for component, plan, words in zip(
        program.components, plans, layout.components, strict=True
    ):
        assert isinstance(component, StatevectorComponent)
        assert plan is not None and words is not None
        selection = component.tape.f_selection
        symbols = state[:, words.symbols : words.symbols + words.symbol_words]
        symbols = symbols.copy().view(np.uint32)[:, : component.num_words]
        expected = packed_symbols(
            0, f, selection, component.num_words, kernels.PARALLEL_WORK, 1
        )
        np.testing.assert_array_equal(symbols, expected)
        engine = plan.tables if plan.tables is not None else plan.trees
        assert engine is not None
        shot_keys = table_shot_keys if plan.tables is not None else tree_shot_keys
        keys, patterns = shot_keys(
            0,
            f,
            selection,
            engine.key_columns,
            engine.output_columns,
            kernels.PARALLEL_WORK,
            1,
        )
        got_keys = state[:, words.key : words.key + words.key_words]
        np.testing.assert_array_equal(got_keys, keys.reshape(shots, -1))
        got_patterns = state[:, words.pattern : words.pattern + words.pattern_words]
        np.testing.assert_array_equal(got_patterns, patterns)


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize(
    ("text", "masked"),
    [
        (DIRECT, False),
        (DIRECT, True),
        (_flagged_chain(8), True),
        (_flagged_chain(8), False),
    ],
)
def test_fused_samples_match_the_unfused_path(text, masked, reference, monkeypatch):
    circuit = Circuit(text)
    mask = np.ones(circuit.num_detectors, bool) if masked else None
    options = {
        "postselection_mask": mask,
        "append_observables": True,
        "use_detector_reference_sample": reference,
    }
    shots = 1 << 15
    fused = circuit.compile_detector_sampler(seed=0)
    assert _backend(fused).fused(None, None) is not None
    got = fused.sample(shots, **options)
    with monkeypatch.context() as patch:
        force_legacy(patch)
        legacy = circuit.compile_detector_sampler(seed=1)
        expected = legacy.sample(shots, **options)
    _assert_same_rows(got, expected)


def test_discarded_shots_keep_only_their_direct_detectors():
    sampler = Circuit(DIRECT).compile_detector_sampler(seed=2)
    mask = np.zeros(sampler.circuit.num_detectors, bool)
    mask[0] = True
    samples = sampler.sample(20_000, postselection_mask=mask, append_observables=True)
    discarded = samples[:, 0]
    assert 0.1 < discarded.mean() < 0.3
    # NOTE: the direct observable (column 3) and the component's outputs read False.
    assert not samples[discarded][:, 1:].any()
    assert samples[~discarded][:, 3].any() and samples[~discarded][:, 2].any()


def test_fused_samples_do_not_depend_on_the_threads():
    circuit = Circuit(_flagged_chain(8))
    mask = np.ones(circuit.num_detectors, bool)
    runs = []
    threads = numba.get_num_threads()
    try:
        for count in (1, threads):
            numba.set_num_threads(count)
            sampler = circuit.compile_detector_sampler(seed=7)
            runs.append(sampler.sample(50_000, postselection_mask=mask))
    finally:
        numba.set_num_threads(threads)
    np.testing.assert_array_equal(runs[0], runs[1])


# NOTE: a T-gate observable and a direct detector that reads 1 without noise.
FLIPPED_DIRECT = """
R 0 1 2
H 0
T 0
X_ERROR(0.1) 0
H 0
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
X 1
X_ERROR(0.2) 1
M 1
DETECTOR rec[-1]
X_ERROR(0.2) 2
M 2
DETECTOR rec[-1]
"""


def test_a_component_detector_mask_does_not_leak_into_an_unmasked_call():
    sampler = Circuit(GHZ_WITH_DIRECT_DETECTOR).compile_detector_sampler(seed=0)
    sampler.sample(50_000, postselection_mask=np.array([True, False, False]))
    samples = sampler.sample(50_000)
    assert samples[samples[:, 0], 1].mean() > 0.4


def test_a_flipped_direct_detector_mask_does_not_leak_into_an_unmasked_call():
    sampler = Circuit(FLIPPED_DIRECT).compile_detector_sampler(seed=0)
    sampler.sample(50_000, postselection_mask=np.array([True, False]))
    samples = sampler.sample(50_000, append_observables=True)
    assert samples[:, -1].mean() > 0.1


def test_a_flipped_direct_detector_mask_applies_after_an_unmasked_call():
    sampler = Circuit(FLIPPED_DIRECT).compile_detector_sampler(seed=0)
    sampler.sample(50_000, append_observables=True)
    samples = sampler.sample(
        50_000, postselection_mask=np.array([True, False]), append_observables=True
    )
    assert samples[:, -1].mean() < 0.06


def _fused_and_legacy(monkeypatch, text, sample, trees=False):
    """Return samples of the fused and the unfused path, from different seeds.

    Both use one reference sample, which is a draw when some outputs are random.
    """
    reference = Circuit(text).compile_detector_sampler(seed=2)
    bits = reference._compute_reference_sample()
    with monkeypatch.context() as patch:
        if trees:
            force_trees(patch)
        fused = Circuit(text).compile_detector_sampler(seed=0)
        fused._compute_reference_sample = lambda: bits
        got = sample(fused)
        assert _backend(fused).fused(None, None) is not None
    with monkeypatch.context() as patch:
        if trees:
            force_trees(patch)
        force_legacy(patch)
        legacy = Circuit(text).compile_detector_sampler(seed=1)
        legacy._compute_reference_sample = lambda: bits
        expected = sample(legacy)
    return got, expected


@pytest.mark.parametrize("trees", [False, True])
@pytest.mark.parametrize("name", ["two_components", "midcircuit_feedforward", "t_ghz"])
def test_fused_samples_of_several_components_match(name, trees, monkeypatch):
    def sample(sampler):
        mask = np.ones(sampler.circuit.num_detectors, bool)
        return sampler.sample(1 << 15, postselection_mask=mask, append_observables=True)

    got, expected = _fused_and_legacy(monkeypatch, CIRCUITS[name], sample, trees)
    _assert_same_rows(got, expected)


def test_fused_samples_with_wide_keys_and_symbols_match(monkeypatch):
    """Over 64 f-bits: multiword symbols and keys, on trees."""
    text = coherent_surface_code(3, 3)

    def sample(sampler):
        mask = np.ones(sampler.circuit.num_detectors, bool)
        return sampler.sample(1 << 14, postselection_mask=mask, append_observables=True)

    got, expected = _fused_and_legacy(monkeypatch, text, sample, trees=True)
    for columns in ([0, 1, 2], [-3, -2, -1], [5, 9, -1]):
        _assert_same_rows(got[:, columns], expected[:, columns])


@pytest.mark.parametrize(
    "options",
    [
        {"postselection_mask": np.array([True, True, False])},
        {"postselection_mask": np.array([False, True, True])},
        {
            "postselection_mask": np.array([True, False, True]),
            "use_observable_reference_sample": True,
        },
        {
            "postselection_mask": np.array([False, False, True]),
            "use_detector_reference_sample": True,
            "batch_size": 1_000,
        },
    ],
)
def test_fused_samples_with_component_masks_and_references_match(options, monkeypatch):
    def sample(sampler):
        return sampler.sample(1 << 15, append_observables=True, **options)

    got, expected = _fused_and_legacy(monkeypatch, GHZ_WITH_DIRECT_DETECTOR, sample)
    _assert_same_rows(got, expected)


@pytest.mark.parametrize("reference", [False, True])
def test_fused_samples_with_a_flipped_direct_detector_match(reference, monkeypatch):
    def sample(sampler):
        return sampler.sample(
            1 << 15,
            postselection_mask=np.array([True, True]),
            use_detector_reference_sample=reference,
            append_observables=True,
        )

    got, expected = _fused_and_legacy(monkeypatch, FLIPPED_DIRECT, sample)
    _assert_same_rows(got, expected)


def test_fused_measurement_samples_with_small_batches_match(monkeypatch):
    circuit = Circuit(DIRECT)
    fused = circuit.compile_sampler(seed=0)
    assert _backend(fused).fused(None, None) is not None
    got = fused.sample(1 << 15, batch_size=1_000)
    with monkeypatch.context() as patch:
        force_legacy(patch)
        expected = circuit.compile_sampler(seed=1).sample(1 << 15, batch_size=1_000)
    _assert_same_rows(got, expected)
