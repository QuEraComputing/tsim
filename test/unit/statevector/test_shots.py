from test.helpers.statevector import CIRCUITS, shot_sampler

import numpy as np
import pytest

import tsim.statevector.shared as shared
from tsim.circuit import Circuit
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
        output_bits = shared.unpack_symbols(tape.output_row, component.num_words)
        f_parts = (f_bits @ output_bits[:, : len(tape.f_selection)].T) % 2 == 1
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
