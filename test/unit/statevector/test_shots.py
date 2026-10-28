from test.helpers.statevector import CIRCUITS, shot_sampler

import numpy as np
import pytest

import tsim.statevector.shared as shared
import tsim.statevector.shots as shots_module
from tsim.circuit import Circuit
from tsim.noise.channels import _BLOCK_SHOTS
from tsim.statevector.shots import bool_outputs, packed_outputs

# NOTE: flipped direct detectors and observables, unflipped ones, and two T
# components, so that the outputs are reordered.
DIRECT_FLIPS = """
R 0 1 2 3 4 5
H 0
T 0
H 0
X_ERROR(0.3) 1
X 2
X_ERROR(0.2) 2
M 2
DETECTOR rec[-1]
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
X 3
X_ERROR(0.1) 3
M 3
OBSERVABLE_INCLUDE(0) rec[-1]
X_ERROR(0.4) 4
M 4
OBSERVABLE_INCLUDE(1) rec[-1]
H 5
T 5
H 5
M 5
OBSERVABLE_INCLUDE(2) rec[-1]
"""


def _bits(words):
    return np.unpackbits(words.view(np.uint8), axis=1, bitorder="little").astype(bool)


@pytest.mark.parametrize("rows", [1, 300, 9_000])
@pytest.mark.parametrize("name", [*sorted(CIRCUITS), "direct_flips"])
def test_noise_writes_each_outputs_f_part_and_each_key(name, rows):
    text = DIRECT_FLIPS if name == "direct_flips" else CIRCUITS[name]
    sampler = Circuit(text).compile_detector_sampler(seed=1)
    program, layout = sampler._program, shot_sampler(sampler).layout
    if name == "direct_flips":
        assert program.direct_flips.any()
        assert program.output_reindex is not None
    channels = sampler._channel_sampler
    seeds = channels.block_seeds(rows)
    f = channels.sample(rows, seeds).astype(np.uint8)
    words = np.empty((rows, layout.num_words), dtype=np.uint64)
    channels.sample_words(words, layout.noise, seeds)
    bits = _bits(words)

    expected = np.zeros((rows, program.num_outputs), dtype=bool)
    for position, column in enumerate(program.direct_f_indices):
        if column >= 0:
            expected[:, program.output_order[position]] = f[:, column] != 0
    for component, key, key_word in zip(
        layout.components, layout.keys, layout.key_starts, strict=True
    ):
        tape = component.tape
        f_bits = f[:, list(tape.f_selection)]
        output_bits = tape.output_row[:, : len(tape.f_selection)].astype(np.uint8)
        f_parts = (f_bits @ output_bits.T) % 2 == 1
        expected[:, list(component.output_indices)] = f_parts
        keys = (f_bits @ key.rows.T.astype(np.uint8)) % 2 == 1
        np.testing.assert_array_equal(
            bits[:, 64 * key_word : 64 * key_word + len(key.rows)], keys
        )
    np.testing.assert_array_equal(bits[:, : program.num_outputs], expected)
    # NOTE: the bits past the outputs in their words stay zero.
    assert not bits[:, program.num_outputs : 64 * layout.output_words].any()


def test_unpacked_and_packed_outputs_match_packbits():
    rng = np.random.default_rng(4)
    num_outputs = 75
    bits = rng.random((50, 128)) < 0.5
    bits[:, num_outputs:] = False
    words = np.ascontiguousarray(shared.pack_bits(bits, 2))
    columns = np.array([70, 3, 3, 74, 0])
    np.testing.assert_array_equal(bool_outputs(words, columns), bits[:, columns])
    for count in (0, 1, 8, 13, 64, 75):
        expected = np.packbits(bits[:, :count], axis=1, bitorder="little")
        np.testing.assert_array_equal(
            packed_outputs(words, count, num_outputs), expected
        )
    # NOTE: a leading run of outputs ends in a masked copy, so the words keep theirs.
    np.testing.assert_array_equal(_bits(words)[:, :128], bits)


# NOTE: two T components: a detector on one, an observable on the other.
TWO_T_COMPONENTS = """
R 0 1
H 0 1
T 0 1
H 0 1
M 0 1
DETECTOR rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def test_a_reference_sample_leaves_the_samples_of_a_seed_alone():
    def sample(reference):
        sampler = Circuit(TWO_T_COMPONENTS).compile_detector_sampler(seed=3)
        return sampler.sample(
            2000, append_observables=True, use_detector_reference_sample=reference
        )

    np.testing.assert_array_equal(sample(False)[:, 1], sample(True)[:, 1])


@pytest.mark.parametrize(
    "per_shot, budget, shots, expected",
    [
        # NOTE: one lane fits but not ten, so the one-shot chunks run one at a time.
        (3 * 2**30, 4 * 2**30, 10, (1, 1)),
        # NOTE: the call fits in one chunk, which then runs alone.
        (50 * 2**20, 2 * 2**30, 8, (256, 1)),
        (50 * 2**20, 8 * 2**30, 64, (256, 1)),
        # NOTE: otherwise a chunk per thread, as small as the memory needs.
        (50 * 2**20, 2 * 2**30, 64, (4, 10)),
        (3 * 2**20, 4 * 2**30, 262_144, (136, 10)),
        (1000, 2**40, 262_144, (26_215, 10)),
        (1000, 2**40, 1000, (256, 4)),
    ],
)
def test_the_chunks_that_run_at_once_fit_in_memory(
    per_shot, budget, shots, expected, monkeypatch
):
    sampler = shot_sampler(Circuit(TWO_T_COMPONENTS).compile_detector_sampler(seed=1))
    monkeypatch.setattr(shots_module, "memory_budget", lambda: budget)
    monkeypatch.setattr(sampler, "bytes_per_shot", lambda: per_shot)
    assert sampler._schedule(shots, None, 10) == expected


def test_the_memory_estimate_holds_every_lane_symbol():
    text = "R 0 1\nH 0 1\n" + "T 0\nT 1\nMPP X0*X1\n" * 300
    sampler = shot_sampler(Circuit(text).compile_sampler(seed=1))
    program = max(sampler.plan(None).programs, key=lambda p: p.lane_words)
    assert program.lane_words >= 4
    amplitudes = 3 * 16 * 2**program.width
    assert sampler.bytes_per_shot() >= amplitudes + 4 * 8 * program.lane_words


@pytest.mark.parametrize("postselect", [False, True])
@pytest.mark.parametrize("bit_packed", [False, True])
def test_calls_whose_words_do_not_fit_run_in_parts_of_the_same_samples(
    bit_packed, postselect, monkeypatch
):
    shots = 3 * _BLOCK_SHOTS + 5

    def sample():
        sampler = Circuit(DIRECT_FLIPS).compile_detector_sampler(seed=9)
        mask = np.ones(sampler._num_detectors, bool) if postselect else None
        return sampler, lambda: sampler.sample(
            shots,
            bit_packed=bit_packed,
            separate_observables=True,
            postselection_mask=mask,
        )

    _, whole = sample()
    expected = whole()
    sampler, parted = sample()
    executor = shot_sampler(sampler)
    budget = 2 * 8 * executor.layout.num_words * _BLOCK_SHOTS
    monkeypatch.setattr(shots_module, "memory_budget", lambda: budget)
    starts = []
    sample_part = executor._sample_part

    def spy(start, *args):
        starts.append(start)
        return sample_part(start, *args)

    monkeypatch.setattr(executor, "_sample_part", spy)
    for got, want in zip(parted(), expected, strict=True):
        np.testing.assert_array_equal(got, want)
    assert starts == [0, _BLOCK_SHOTS, 2 * _BLOCK_SHOTS, 3 * _BLOCK_SHOTS]


def _brickwork(width: int, layers: int) -> str:
    rng = np.random.default_rng(0)
    lines = ["R " + " ".join(map(str, range(width)))]
    for layer in range(layers):
        for q in range(width):
            a, b, c = rng.random(3)
            lines.append(f"U3({a:.3f}, {b:.3f}, {c:.3f}) {q}")
        lines += [f"CNOT {q} {q + 1}" for q in range(layer % 2, width - 1, 2)]
    lines.append("M " + " ".join(map(str, range(width))))
    lines += [f"DETECTOR rec[-{q + 1}]" for q in range(width)]
    return "\n".join(lines)


def test_a_wide_components_small_call_fills_every_thread():
    sampler = shot_sampler(Circuit(_brickwork(12, 15)).compile_detector_sampler(seed=1))
    assert max(c.width for c in sampler.layout.components) == 12
    shots = shots_module.MIN_CHUNK_SHOTS + 1
    assert sampler._schedule(shots, None, 10) == (26, 10)
    assert sampler._schedule(10, None, 10) == (16, 1)
    # NOTE: a narrow component's chunks keep their least size.
    narrow = shot_sampler(Circuit(TWO_T_COMPONENTS).compile_detector_sampler(seed=1))
    assert narrow._schedule(shots, None, 10) == (256, 2)


@pytest.mark.parametrize("mask", [False, True])
def test_an_empty_call_leaves_the_next_samples_alone(mask):
    def sampler():
        return Circuit(TWO_T_COMPONENTS).compile_detector_sampler(seed=5)

    def sample(s, shots):
        m = np.ones(s._num_detectors, bool) if mask else None
        return s.sample(shots, append_observables=True, postselection_mask=m)

    fresh, used = sampler(), sampler()
    assert sample(used, 0).shape == (0, 2)
    np.testing.assert_array_equal(sample(used, 3000), sample(fresh, 3000))


# NOTE: two components whose noise gives them key words.
KEYED_COMPONENTS = "R 0 1\n" + "".join(
    f"H {q}\nT {q}\nX_ERROR(0.1) {q}\nT {q}\nH {q}\nM {q}\nDETECTOR rec[-1]\n"
    for q in (0, 1)
)


def test_packed_outputs_do_not_keep_the_component_keys_alive():
    sampler = Circuit(KEYED_COMPONENTS).compile_detector_sampler(seed=1)
    layout = shot_sampler(sampler).layout
    assert layout.num_words > layout.output_words
    packed = sampler.sample(1000, bit_packed=True)
    expected = np.packbits(sampler.sample(1000), axis=1, bitorder="little")
    assert packed.base is None or packed.base.nbytes == packed.nbytes
    assert packed.shape == expected.shape
