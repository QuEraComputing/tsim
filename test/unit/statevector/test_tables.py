import dataclasses
import itertools
import pickle
from test.helpers.statevector import (
    CIRCUITS,
    assert_distributions_close,
    assert_matches_distribution,
    coherent_surface_code,
    f_patterns,
    output_indices,
    plan_distribution,
    quantum_components,
)

import numpy as np
import pytest

import tsim.sampler as sampler_module
import tsim.statevector.tables as tables_module
import tsim.utils.kernels as kernels
from tsim.circuit import Circuit
from tsim.statevector.program import StatevectorComponent, lower, plan_component
from tsim.statevector.shared import _unpack, sample_shared_component, sharing_layout
from tsim.statevector.tables import OutcomeTables
from tsim.utils.rng import splitmix64

NAMES = sorted(CIRCUITS)


def _compiled(name):
    for sample_detectors in (True, False):
        components, layout = quantum_components(
            CIRCUITS[name], sample_detectors=sample_detectors
        )
        for component in components:
            indices = output_indices(component)
            planned, component_layout = plan_component(component, indices, layout)
            yield planned, lower(planned, component_layout)


def _f_params(compiled, patterns, shots_each):
    selection = list(compiled.tape.f_selection)
    f_params = np.zeros(
        (shots_each * len(patterns), max(selection, default=-1) + 1), dtype=np.uint8
    )
    for i, f in enumerate(patterns):
        if selection:
            f_params[i * shots_each : (i + 1) * shots_each, selection] = f
    return f_params


def _entry_probabilities(store, table):
    """Return the probability of each entry of a table under the alias draw."""
    lo, hi = store.start[table], store.start[table + 1]
    probability = np.zeros(hi - lo)
    for bucket in range(lo, hi):
        keep = float(store.threshold[bucket]) / 2.0**64
        probability[bucket - lo] += keep / (hi - lo)
        probability[store.alias[bucket] - lo] += (1 - keep) / (hi - lo)
    return probability


def _table_distribution(tables, compiled, f):
    """Return the exact output distribution of ``f``'s table and its coins."""
    tables.sample(_f_params(compiled, [f], 1), 0, tables.layout)
    store, tape, words = tables._store, compiled.tape, compiled.num_words
    f_bits = np.zeros(32 * words, dtype=bool)
    f_bits[: len(f)] = f
    key_bits = _unpack(tables.layout.key_rows, words)
    key = sum(
        1 << k for k, row in enumerate(key_bits) if np.count_nonzero(row & f_bits) & 1
    )
    (table,) = np.flatnonzero(store.keys == key)
    coins = np.flatnonzero(_unpack(tables.layout.shot_dormant, words))
    output_bits = _unpack(tape.output_row, words)
    num_outputs = compiled.num_outputs
    result = {}
    lo = store.start[table]
    for offset, p in enumerate(_entry_probabilities(store, table)):
        pattern = store.patterns[lo + offset]
        branch = np.array(
            [
                (pattern[o >> 6] >> np.uint64(o & 63)) & np.uint64(1)
                for o in range(num_outputs)
            ],
            dtype=bool,
        )
        for values in itertools.product((False, True), repeat=len(coins)):
            symbols = f_bits.copy()
            symbols[coins] = values
            outputs = (
                (output_bits & symbols).sum(axis=1) % 2 ^ branch ^ tape.output_const
            )
            key_ = tuple(int(bit) for bit in outputs)
            result[key_] = result.get(key_, 0.0) + p / 2 ** len(coins)
    return result


@pytest.mark.parametrize("name", NAMES)
def test_each_table_holds_its_keys_exact_distribution(name):
    for planned, compiled in _compiled(name):
        tables = OutcomeTables(compiled)
        for f in f_patterns(planned.num_f, limit=8):
            assert_distributions_close(
                _table_distribution(tables, compiled, f),
                plan_distribution(planned, f),
                atol=1e-12,
            )


@pytest.mark.parametrize("name", NAMES)
def test_samples_follow_the_plan_distribution(name):
    shots_each = 10_000
    for planned, compiled in _compiled(name):
        tables = OutcomeTables(compiled)
        patterns = f_patterns(planned.num_f, limit=3)
        samples, deviation, stats = tables.sample(
            _f_params(compiled, patterns, shots_each), 7, tables.layout
        )
        assert deviation < 1e-9
        assert stats["built"] == stats["tables"] <= len(patterns)
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
        rows, _, _ = OutcomeTables(compiled).sample(
            _f_params(compiled, patterns, 4_000), 9, layout
        )
        # NOTE: postselection tests the outputs with their reference bits.
        fired = rows & postselected
        discarding = np.min(
            np.where(fired, layout.ready[None, :], compiled.num_actions), axis=1
        )
        late = layout.ready[None, :] > discarding[:, None]
        assert not np.any(rows & late), "an output set after the discarding one is 1"
        # NOTE: the outputs up to the discarding one keep their values.
        unflipped, _, _ = OutcomeTables(compiled).sample(
            _f_params(compiled, patterns, 4_000), 9, sharing_layout(compiled)
        )
        np.testing.assert_array_equal((rows ^ flipped)[~late], unflipped[~late])


@pytest.mark.parametrize("name", NAMES)
def test_samples_do_not_depend_on_which_batch_built_a_table(name):
    for planned, compiled in _compiled(name):
        patterns = f_patterns(planned.num_f, limit=6)
        f_params = _f_params(compiled, patterns, 500)
        np.random.default_rng(3).shuffle(f_params)
        fresh = OutcomeTables(compiled)
        warmed = OutcomeTables(compiled)
        warmed.sample(_f_params(compiled, patterns[::-1], 1), 1, warmed.layout)
        first, _, built = fresh.sample(f_params, 11, fresh.layout)
        again, _, rebuilt = fresh.sample(f_params, 11, fresh.layout)
        other, _, _ = warmed.sample(f_params, 11, warmed.layout)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(first, other)
        assert built["built"] == fresh.num_tables and rebuilt["built"] == 0


@pytest.mark.parametrize("name", NAMES)
def test_parallel_kernels_give_the_serial_samples(name, monkeypatch):
    for planned, compiled in _compiled(name):
        f_params = _f_params(compiled, f_patterns(planned.num_f, 3), 1_000)
        samples = []
        for threshold in (np.iinfo(np.int64).max, 0):
            with monkeypatch.context() as patch:
                patch.setattr(kernels, "PARALLEL_WORK", threshold)
                tables = OutcomeTables(compiled)
                rows, _, _ = tables.sample(f_params, 19, tables.layout)
            samples.append(rows)
        np.testing.assert_array_equal(samples[1], samples[0])


def test_a_rare_entry_is_drawn_at_its_probability():
    shots, p = 1 << 22, 1e-5
    start = np.array([0, 2])
    threshold, alias = tables_module._alias(np.array([1 - p, p]), start)
    draws, coins, _ = tables_module._streams(np.uint64(5))
    samples = tables_module._draw(
        0,
        np.zeros((shots, 1), dtype=np.uint64),
        np.zeros(shots, dtype=np.int64),
        start,
        threshold,
        alias,
        np.array([[0], [1]], dtype=np.uint64),
        np.zeros((0, 1), dtype=np.uint64),
        np.zeros(1, dtype=np.uint64),
        np.zeros(1, dtype=np.uint64),
        np.zeros(1, dtype=np.int64),
        draws,
        coins,
        kernels.PARALLEL_WORK,
        1,
    )
    hits = int(samples.sum())
    assert abs(hits - shots * p) <= 5 * np.sqrt(shots * p) + 3


def test_alias_tables_keep_every_probability():
    rng = np.random.default_rng(1)
    probability = np.concatenate([rng.random(7), [1e-18, 0.5], rng.random(3) * 1e-9])
    start = np.array([0, 7, 9, 12])
    threshold, alias = tables_module._alias(probability, start)
    store = tables_module._Store(
        keys=np.zeros(3, np.uint64),
        slot_keys=np.zeros(8, np.uint64),
        slot_tables=np.zeros(8, np.int64),
        start=start,
        threshold=threshold,
        alias=alias,
        patterns=np.zeros((12, 1), np.uint64),
    )
    for table in range(3):
        lo, hi = start[table], start[table + 1]
        expected = probability[lo:hi] / probability[lo:hi].sum()
        got = _entry_probabilities(store, table)
        # NOTE: thresholds resolve 2^-64 of a bucket of 1/n.
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=2.0**-64)


def test_lookups_find_every_key_and_miss_the_others():
    rng = np.random.default_rng(0)
    keys = np.unique(rng.integers(0, 2**63, 300, dtype=np.uint64))
    keys = np.concatenate([[np.uint64(0)], keys])
    slot_keys, slot_tables = tables_module._hashed(keys)
    queries = np.concatenate([keys, keys[::-1], keys + np.uint64(1)])
    tables, misses = tables_module._lookup(queries, slot_keys, slot_tables)
    present = np.isin(queries, keys)
    np.testing.assert_array_equal(
        tables[present], np.searchsorted(keys, queries[present])
    )
    assert np.all(tables[~present] == -1) and misses == np.count_nonzero(~present)


def _tabled(text):
    backend = Circuit(text).compile_detector_sampler(seed=0)._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    return [tables is not None for tables in backend.tables()]


def test_small_components_whose_keys_repeat_use_tables():
    assert _tabled(CIRCUITS["t_ghz"]) == [True]
    assert all(_tabled(CIRCUITS["two_components"]))


def test_components_whose_branches_overflow_memory_use_lanes(monkeypatch):
    monkeypatch.setattr(tables_module, "_BUILD_BYTES", 2**8)
    assert _tabled(CIRCUITS["t_ghz"]) == [False]


def test_tables_need_keys_that_repeat():
    """The same coherent surface code uses tables at low noise, lanes at high."""
    assert _tabled(coherent_surface_code(3, 2, noise=1e-3)) == [True]
    assert _tabled(coherent_surface_code(3, 2, noise=0.2)) == [False]


@pytest.mark.parametrize("name", NAMES)
def test_shots_whose_key_has_no_table_take_lanes(name):
    shots_each = 6_000
    for planned, compiled in _compiled(name):
        tables = OutcomeTables(compiled)
        patterns = f_patterns(planned.num_f, limit=3)
        f_params = _f_params(compiled, patterns, shots_each)
        samples, _, stats = tables.sample(
            f_params, 3, tables.layout, np.zeros(1, dtype=np.uint64)
        )
        keyed = sum(
            any(np.count_nonzero(row[: len(f)] & np.array(f, bool)) & 1 for row in key)
            for f in patterns
            for key in [_unpack(tables.layout.key_rows, compiled.num_words)]
        )
        assert stats["tables"] == 1
        assert stats["lane shots"] == keyed * shots_each
        for i, f in enumerate(patterns):
            rows = samples[i * shots_each : (i + 1) * shots_each]
            assert_matches_distribution(rows, plan_distribution(planned, f))


@pytest.mark.parametrize("name", ["t_ghz", "midcircuit_feedforward", "two_components"])
def test_the_fault_keys_cover_at_least_their_estimate(name):
    shots = 1 << 15
    sampler = Circuit(CIRCUITS[name]).compile_detector_sampler(seed=1)
    f_params = sampler._channel_sampler.sample(shots)
    noise = sampler._channel_sampler.fire_tables
    for component in sampler._program.components:
        assert isinstance(component, StatevectorComponent)
        layout = sharing_layout(component)
        keys, covered = tables_module._fault_keys(component, layout, noise)
        tables = OutcomeTables(component)
        _, _, stats = tables.sample(f_params, 0, tables.layout, keys)
        rate = 1 - stats["lane shots"] / shots
        assert rate >= covered - 5 * np.sqrt(covered * (1 - covered) / shots)


def _flagged_chain(rounds, noise=0.2, unflagged=0.0, flipped=()):
    """Return a circuit whose faults flip a T gate's sign and a direct detector.

    With ``unflagged``, each round also flips the sign alone, as the else branch
    of its fault. The flags of ``flipped`` rounds read 1 without noise.
    """
    lines = ["R " + " ".join(map(str, range(rounds + 1)))]
    for r in range(rounds):
        lines += ["H 0", "T 0", f"E({noise}) Z0 X{r + 1}"]
        if unflagged:
            lines.append(f"ELSE_CORRELATED_ERROR({unflagged}) Z0")
    lines += [f"X {r + 1}" for r in flipped]
    lines += ["H 0", "M 0"] + [f"M {r + 1}" for r in range(rounds)]
    lines += [f"DETECTOR rec[-{r + 1}]" for r in range(rounds)]
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[-{rounds + 1}]")
    return "\n".join(lines)


def _picked(backend, mask=None):
    """Return the one component's tables and keys under ``mask``."""
    (picked,) = backend.tables(mask)
    assert picked is not None
    return picked


def test_a_mask_whose_detectors_catch_the_faults_gives_tables():
    sampler = Circuit(_flagged_chain(16)).compile_detector_sampler(seed=0)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    mask = np.ones(sampler.circuit.num_detectors, dtype=bool)
    assert backend.tables() == (None,)
    tables, keys = _picked(backend, mask)
    np.testing.assert_array_equal(keys, [0])
    sampler.sample(1_000, postselection_mask=mask)
    sampler.sample(1_000, postselection_mask=mask)
    assert _picked(backend, mask) == (tables, keys) and tables.num_tables == 1
    assert backend.tables() == (None,)


def test_masks_share_the_tables_of_a_component():
    sampler = Circuit(_flagged_chain(16)).compile_detector_sampler(seed=0)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    full = np.ones(sampler.circuit.num_detectors, dtype=bool)
    most = full.copy()
    # NOTE: the last detector flags the fault of the first round.
    most[-1] = False
    tables, _ = _picked(backend, full)
    shared, keys = _picked(backend, most)
    assert shared is tables and keys is not None and len(keys) == 2
    sampler.sample(1_000, postselection_mask=full)
    sampler.sample(4_000, postselection_mask=most)
    assert tables.num_tables == 2


@pytest.mark.parametrize("every", [1, 2, 3])
def test_the_passing_fault_keys_cover_their_estimate(every):
    """Among the shots that the screen passes, the keys cover their estimate.

    Every round's sign also flips without its flag, so the estimate depends on
    the chance of the caught faults as well; it is exact for this circuit.
    """
    circuit = _flagged_chain(12, noise=0.1, unflagged=0.05)
    sampler = Circuit(circuit).compile_detector_sampler(seed=1)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    mask = np.zeros(sampler.circuit.num_detectors, dtype=bool)
    mask[::every] = True
    screened = backend.screened(mask, None)
    assert len(screened) == np.count_nonzero(mask)
    f_params = sampler._channel_sampler.sample(1 << 16)
    f_params = f_params[~f_params[:, screened].any(axis=1)]
    (component,) = sampler._program.components
    assert isinstance(component, StatevectorComponent)
    noise = sampler._channel_sampler.fire_tables
    keys, covered = tables_module._fault_keys(
        component, sharing_layout(component), noise, screened
    )
    tables = OutcomeTables(component)
    _, _, stats = tables.sample(f_params, 0, tables.layout, keys)
    shots = len(f_params)
    rate = 1 - stats["lane shots"] / shots
    assert abs(rate - covered) <= 5 * np.sqrt(covered * (1 - covered) / shots)


@pytest.mark.parametrize("reference", [False, True])
def test_detectors_that_read_one_without_noise_screen_only_with_a_reference(
    reference,
):
    circuit = Circuit(_flagged_chain(4, flipped=(1,)))
    sampler = circuit.compile_detector_sampler(seed=0)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    mask = np.ones(circuit.num_detectors, dtype=bool)
    flip = sampler._compute_reference_sample() if reference else None
    screened = backend.screened(mask, flip)
    assert len(screened) == (4 if reference else 3)


def _chain(rounds, noisy_rounds):
    """Return a one-qubit circuit whose every round measures a random outcome."""
    lines = ["R 0"]
    for round_ in range(rounds):
        lines.append("R_X(0.3) 0")
        if round_ < noisy_rounds:
            lines.append("DEPOLARIZE1(0.001) 0")
        lines += ["R_X(0.3) 0", "M 0"]
    return "\n".join(lines)


@pytest.mark.parametrize(("rounds", "noisy_rounds"), [(22, 12), (21, 0)])
def test_deep_narrow_trees_use_lanes(rounds, noisy_rounds):
    sampler = Circuit(_chain(rounds, noisy_rounds)).compile_sampler(seed=0)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    assert backend.tables() == (None,)


def test_a_rare_outcome_keeps_its_probability_on_either_side():
    (component,) = (
        Circuit("R 0\nH 0\nT 0\nH 0\nM 0").compile_sampler()._program.components
    )
    assert isinstance(component, StatevectorComponent)
    rotation = int(np.flatnonzero(component.tape.sin)[0])
    # NOTE: amplitude 1e-9 on either outcome of the measurement.
    for cos_rotation, sin_rotation in ((1.0, 1e-9), (1e-9, 1.0)):
        cos, sin = component.tape.cos.copy(), component.tape.sin.copy()
        cos[rotation], sin[rotation] = cos_rotation, sin_rotation
        tilted = dataclasses.replace(
            component, tape=component.tape._replace(cos=cos, sin=sin)
        )
        tables = OutcomeTables(tilted)
        tables.sample(np.zeros((1, 0), dtype=np.uint8), 0, tables.layout)
        probability = np.sort(_entry_probabilities(tables._store, 0))
        assert len(probability) == 2
        # NOTE: alias thresholds resolve 2^-64 of a bucket of 1/2.
        np.testing.assert_allclose(probability[0], 1e-18, rtol=0, atol=2.0**-64)


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
    for keys in (None, np.ones(1, dtype=np.uint64)):
        tables = OutcomeTables(broken)
        _, deviation, _ = tables.sample(f_params, 1, tables.layout, keys)
        assert np.isnan(deviation)


def test_the_streams_of_a_batch_do_not_overlap():
    for seed in range(4):
        draws, coins, lanes = tables_module._streams(np.uint64(seed))
        # NOTE: the lanes' executor starts its stream at its seed's next output.
        _, executor = splitmix64(lanes)
        starts = {int(draws), int(coins), int(lanes), int(executor)}
        assert len(starts) == 4


def test_samplers_with_tables_pickle_with_their_tables():
    sampler = Circuit(CIRCUITS["t_interference"]).compile_detector_sampler(seed=0)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    tables, _ = _picked(backend)
    sampler.sample(1_000)
    copy = pickle.loads(pickle.dumps(sampler))  # noqa: S301
    copied, _ = _picked(copy._backend)
    assert copied.num_tables == tables.num_tables > 0
    assert copy.sample(1_000).shape == (1_000, sampler.circuit.num_detectors)


def test_keys_wider_than_64_bits_have_no_tables():
    sampler = Circuit(coherent_surface_code(3, 3)).compile_detector_sampler(seed=0)
    (component,) = sampler._program.components
    assert isinstance(component, StatevectorComponent)
    assert len(sharing_layout(component).key_rows) > 64
    with pytest.raises(ValueError, match="64 bits"):
        OutcomeTables(component)


def _row_counts(samples):
    rows = np.packbits(samples, axis=1)
    keys, counts = np.unique(rows, axis=0, return_counts=True)
    return {key.tobytes(): count for key, count in zip(keys, counts, strict=True)}


def _assert_same_rows(got, expected):
    got, expected = _row_counts(got), _row_counts(expected)
    for key in set(got) | set(expected):
        a, b = got.get(key, 0), expected.get(key, 0)
        assert abs(a - b) <= 5 * np.sqrt(a + b) + 3, (a, b)


def test_more_than_64_outputs_and_coins_match_lanes():
    """Outputs and dormant coins span two words; tables agree with lanes."""
    coins = 70
    lines = ["R " + " ".join(map(str, range(coins + 1))), "H 0", "T 0"]
    lines += [f"CX 0 {q}" for q in range(1, coins + 1)]
    lines += ["X_ERROR(0.1) 0", "MX " + " ".join(map(str, range(1, coins + 1)))]
    lines += ["H 0", "M 0"]
    sampler = Circuit("\n".join(lines)).compile_sampler(seed=0)
    (component,) = sampler._program.components
    assert isinstance(component, StatevectorComponent)
    tables = OutcomeTables(component)
    assert component.num_outputs == coins + 1
    assert tables._coin_columns.shape == (coins, 2)
    f_params = sampler._channel_sampler.sample(1 << 15)
    rng = np.random.default_rng(0)
    postselected = rng.random(component.num_outputs) < 0.05
    flipped = rng.random(component.num_outputs) < 0.5
    layout = sharing_layout(component, postselected, flipped)
    got, _, _ = tables.sample(f_params, 1, layout)
    expected, _, _ = sample_shared_component(component, layout, f_params, 2)
    # NOTE: the coins make most rows unique, so compare outputs from both words
    # in triples.
    for columns in ([0, 1, 65], [2, 69, 70], [30, 64, 70]):
        _assert_same_rows(got[:, columns], expected[:, columns])


def test_postselected_samples_from_tables_match_lanes(monkeypatch):
    circuit = Circuit(_flagged_chain(8))
    mask = np.zeros(circuit.num_detectors, dtype=bool)
    mask[::2] = True
    shots = 1 << 15
    sampler = circuit.compile_detector_sampler(seed=0)
    backend = sampler._backend
    assert isinstance(backend, sampler_module._StatevectorBackend)
    assert backend.tables(mask) != (None,)
    got = sampler.sample(shots, postselection_mask=mask, append_observables=True)
    monkeypatch.setattr(sampler_module, "table_keys", lambda *args: None)
    lanes = circuit.compile_detector_sampler(seed=1)
    expected = lanes.sample(shots, postselection_mask=mask, append_observables=True)
    _assert_same_rows(got, expected)
