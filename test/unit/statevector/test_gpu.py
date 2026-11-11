"""The host side of the GPU backends: noise tables, row packing and the tape generator."""

import re
from test.helpers.statevector import CIRCUITS

import numpy as np
import pytest

from tsim.circuit import Circuit
from tsim.noise.channels import ChannelSampler
from tsim.statevector.gpu import noise, rows, tape
from tsim.statevector.shots import ShotSampler

# NOTE: 200 detectors (four words), a fault that flips all four words (past the
# two that a record holds), two-qubit depolarizing noise (15 outcomes, alias
# tables) and an uneven single-qubit Pauli channel.
NOISY = "\n".join(
    [
        "R " + " ".join(map(str, range(200))),
        "E(0.05) X0 X70 X140 X199",
        "DEPOLARIZE2(0.03) 3 4 130 131",
        "PAULI_CHANNEL_1(0.02, 0.07, 0.01) 5 66",
        "H 5 66",
        "X_ERROR(0.01) " + " ".join(map(str, range(10, 200, 3))),
        "M " + " ".join(map(str, range(200))),
        *(f"DETECTOR rec[{i - 200}]" for i in range(200)),
        "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]",
    ]
)


def _shots(text: str) -> tuple[ShotSampler, ChannelSampler]:
    sampler = Circuit(text).compile_detector_sampler(seed=1)
    assert isinstance(sampler._shots, ShotSampler)
    return sampler._shots, sampler._channel_sampler


def _model_words(tables: noise.NoiseTables, shots: int, seed: int) -> np.ndarray:
    """Draw shot words from the tables as a noise kernel reads them.

    Each segment's trials fire independently with probability ``1 - e^log_keep``;
    a fault draws a slot of its channel and a 24-bit fraction, takes the slot's
    alias past its threshold and XORs the outcome's flips into its row.
    """
    rng = np.random.default_rng(seed)
    S, groups = tables.block_shots, tables.groups
    first, per, begin, end, log_keep = groups.segments
    out = np.zeros((shots, tables.num_words), dtype=np.uint64)
    for base in range(0, shots, S):
        ns = min(S, shots - base)
        for k in range(len(first)):
            p = -np.expm1(np.float64(log_keep[k]))
            trials = np.arange(begin[k], end[k])
            fired = trials[rng.random(len(trials)) < p]
            for trial in fired:
                channel, row = divmod(int(trial), S)
                if row >= ns:
                    continue
                n = int(per[k])
                slot = int(rng.integers(n))
                fraction = int(rng.integers(1 << 24))
                at = int(first[k]) + channel * n + slot
                record = groups.records[at]
                if fraction >= int(record["alias"]) & 0x1FFFFFF:
                    record = groups.records[at - slot + (int(record["alias"]) >> 25)]
                extra = int(record["extra"])
                words = [int(record["words"]) & 0xFFFF, int(record["words"]) >> 16]
                masks = [record["m0"], record["m1"]]
                for j in range(extra & 255):
                    if j < 2:
                        word, mask = words[j], masks[j]
                    else:
                        flip = tables.overflow[(extra >> 8) + j - 2]
                        word, mask = int(flip["word"]), flip["mask"]
                    out[base + row, word] ^= np.uint64(mask)
    return out


def _bits(words: np.ndarray) -> np.ndarray:
    return np.unpackbits(words.view(np.uint8), axis=1, bitorder="little")


@pytest.mark.parametrize("shared_words", [True, False])
def test_noise_tables_draw_the_channel_samplers_distribution(shared_words):
    host, channels = _shots(NOISY)
    layout = host.layout
    tables = noise.noise_tables(
        channels, layout.noise, layout.num_words, 14 * 1024, shared_words
    )
    assert (tables.overflow["word"][:-1] < layout.num_words).all()
    assert len(tables.overflow) > 1
    shots = 20_000
    got = _bits(_model_words(tables, shots, seed=2))
    expected = np.zeros((shots, layout.num_words), dtype=np.uint64)
    channels.sample_words(expected, layout.noise)
    expected = _bits(expected)
    p = (got.mean(0) + expected.mean(0)) / 2
    sigma = np.sqrt(np.maximum(p * (1 - p), 1 / shots) * 2 / shots)
    assert (np.abs(got.mean(0) - expected.mean(0)) <= 5 * sigma).all()
    # NOTE: the correlated fault flips detectors 0, 70, 140 and 199 together.
    joint = [0, 70, 140, 199]
    for bits in (got, expected):
        together = bits[:, joint].all(axis=1).mean()
        assert abs(together - 0.05 * (1 - 0.01) ** 2) < 0.01


def test_noise_tables_size_blocks_for_their_memory():
    host, channels = _shots(NOISY)
    layout = host.layout
    shared = noise.noise_tables(channels, layout.noise, layout.num_words, 14 * 1024)
    assert shared.shared_words
    assert 8 * layout.num_words * shared.block_shots <= 14 * 1024 - 8 * 256
    assert 8 * shared.groups.max_events <= (
        14 * 1024 - 8 * layout.num_words * shared.block_shots
    )
    wide = noise.noise_tables(
        channels, layout.noise, layout.num_words, 2048, shared_words=False
    )
    assert wide.block_shots == noise.MAX_BLOCK_SHOTS
    assert wide.groups.max_events <= 2048 // 8
    with pytest.raises(ValueError, match="do not fit"):
        noise.noise_tables(channels, layout.noise, layout.num_words, 2048)
    with pytest.raises(ValueError, match="at least one shot word"):
        noise.noise_tables(channels, layout.noise, 0, 14 * 1024)
    # NOTE: without the words, the memory still has to hold 256 events.
    with pytest.raises(ValueError, match="fewer than 256 events"):
        noise.noise_tables(
            channels, layout.noise, layout.num_words, 1024, shared_words=False
        )


def test_noise_tables_reject_an_outcome_whose_flip_count_overflows():
    # NOTE: a record counts its flips in 8 bits: one fault that flips a detector
    # in each of 256 words, then one that flips 255.
    def tables(num_words):
        targets = " ".join(f"X{64 * w}" for w in range(num_words))
        text = "\n".join(
            [
                f"E(0.05) {targets}",
                "M " + " ".join(str(64 * w) for w in range(num_words)),
                *(
                    f"DETECTOR rec[{w - num_words}]\nSHIFT_COORDS(1)"
                    + "\nDETECTOR" * 63
                    for w in range(num_words)
                ),
            ]
        )
        host, channels = _shots(text)
        layout = host.layout
        assert np.diff(layout.noise.start).max() == num_words
        return noise.noise_tables(
            channels, layout.noise, layout.num_words, 14 * 1024, shared_words=False
        )

    with pytest.raises(ValueError, match="up to 255 flipped words"):
        tables(256)
    fits = tables(255)
    assert int(fits.records["extra"][0]) & 255 == 255
    assert len(fits.overflow) == 253 + 1


def test_pass_rate_and_deferred_channels():
    host, channels = _shots(NOISY)
    layout = host.layout
    read = np.zeros(layout.num_words, dtype=np.uint64)
    # NOTE: detectors 0 and 5: the correlated fault and the uneven Pauli channel.
    read[0] = np.uint64(1 | 1 << 5)
    deferred = noise.deferred_channels(channels, layout.noise, read)
    hits = (read[layout.noise.word] & layout.noise.mask) != 0
    start = channels._tables.outcome_start
    for c in range(len(start) - 1):
        flips = layout.noise.start[start[c]], layout.noise.start[start[c + 1]]
        assert deferred[c] == (not hits[flips[0] : flips[1]].any())
    assert 0 < deferred.sum() < len(deferred)
    shots = 200_000
    words = np.zeros((shots, layout.num_words), dtype=np.uint64)
    channels.sample_words(words, layout.noise)
    passed = ((words & read) == 0).all(axis=1).mean()
    rate = noise.pass_rate(channels, layout.noise, read)
    assert abs(passed - rate) < 5 * np.sqrt(rate * (1 - rate) / shots)
    tables = noise.noise_tables(channels, layout.noise, layout.num_words, 14 * 1024)
    first, later = tables.subset(~deferred), tables.subset(deferred)
    assert np.isclose(first.faults + later.faults, tables.groups.faults)


def test_select_reads_the_columns_of_rows():
    rng = np.random.default_rng(3)
    num_outputs, count = 150, 40
    bits = rng.integers(0, 2, (64, num_outputs)).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    for columns in (
        np.arange(count),
        np.arange(num_outputs),
        np.sort(rng.choice(num_outputs, count, replace=False)),
    ):
        got = rows.select(packed, columns, num_outputs, bit_packed=False)
        np.testing.assert_array_equal(got, bits[:, columns].astype(bool))
        got = rows.select(packed, columns, num_outputs, bit_packed=True)
        np.testing.assert_array_equal(
            got, np.packbits(bits[:, columns], axis=1, bitorder="little")
        )


TEST_DIALECT = tape.Dialect(
    inline="inline",
    noinline="noinline",
    private="thread ",
    deviation="device atomic_uint*",
    constant="constant",
    float2="float2",
    rsqrt="rsqrt",
    fmax="fmax",
    u64="UL",
    shuffle2="shuffle({value}, {lanes})",
    shuffle1="shuffle({value}, {lanes})",
    lane_mask=False,
    goto=False,
    amplitude_chunk=32,
    pair_projections=True,
)


def _programs(text, mask=True):
    host, _ = _shots(text)
    num_detectors = Circuit(text).num_detectors
    plan = host.plan(np.ones(num_detectors, bool) if mask and num_detectors else None)
    return list(plan.programs)


def _wide_unrolled():
    rng = np.random.default_rng(0)
    lines = ["X_ERROR(0.2) " + " ".join(map(str, range(11)))]
    for _ in range(7):
        perm = rng.permutation(11)
        for a, b in zip(perm[::2], perm[1::2], strict=False):
            for q in (a, b):
                t = rng.uniform(-1, 1, 3)
                lines.append(f"U3({t[0]},{t[1]},{t[2]}) {q}")
            lines.append(f"CX {a} {b}")
    lines.append("M " + " ".join(map(str, range(11))))
    lines += [f"DETECTOR rec[{i - 11}]" for i in range(11)]
    return "\n".join(lines)


def test_tapes_break_out_of_a_block_without_goto():
    programs = _programs(CIRCUITS["t_ghz"])
    assert any(p.pruned.any() or p.prune_first.any() for p in programs)
    for i, program in enumerate(programs):
        source = tape.tape_function(program, i, 0, {}, TEST_DIALECT)
        assert "goto" not in source and "done:" not in source
        assert source.count("    do {") == source.count("} while (false);") == 1
        if program.pruned.any():
            assert "break; }" in source
        goto = tape.tape_function(
            program, i, 0, {}, tape.Dialect(**{**TEST_DIALECT.__dict__, "goto": True})
        )
        assert "do {" not in goto and "done:" in goto


def test_unrolled_tapes_split_their_amplitudes_into_chunks():
    (program,) = _programs(_wide_unrolled(), mask=False)
    assert program.width == 11 and not tape.loops(program, 5, 0)
    source = tape.tape_function(program, 0, 5, {}, TEST_DIALECT)
    assert re.search(r"\ba\[", source) is None
    assert "float2 a0[32], a1[32];" in source and "a1[31]" in source
    whole = tape.tape_function(
        program,
        0,
        5,
        {},
        tape.Dialect(**{**TEST_DIALECT.__dict__, "amplitude_chunk": None}),
    )
    assert "float2 a[64];" in whole


def test_projections_read_pairs_or_every_partner_first():
    (program,) = _programs(_wide_unrolled(), mask=False)
    unpaired = tape.Dialect(**{**TEST_DIALECT.__dict__, "pair_projections": False})
    paired = tape.tape_function(program, 0, 5, {}, TEST_DIALECT)
    whole = tape.tape_function(program, 0, 5, {}, unpaired)
    # NOTE: a non-diagonal projection loads its partners as q<loc>: all of them
    # before it updates any, or a pair in a block of its own.
    assert re.search(r"\n      float2 q63 = ", whole)
    assert not re.search(r"\n      float2 q\d+ = ", paired)
    assert re.search(r"\{ float2 q\d+ = .*, q\d+ = ", paired)


@pytest.mark.parametrize(("p", "qubits"), [(2e-8, 192), (1e-9, 3200)])
def test_tiny_probability_segments_take_the_kernels_fine_path(p, qubits):
    # NOTE: a 24-bit skip reaches only some trials of a segment of noise below
    # p ~ 2^-24; segments of fewer than 2^-12 expected faults draw from 48 bits.
    text = "\n".join(
        [f"X_ERROR({p!r}) " + " ".join(map(str, range(qubits)))]
        + ["M " + " ".join(map(str, range(qubits)))]
        + [f"DETECTOR rec[{i - qubits}]" for i in range(qubits)]
    )
    host, channels = _shots(text)
    layout = host.layout
    tables = noise.noise_tables(channels, layout.noise, layout.num_words, 46 * 1024)
    _, _, begin, end, log_keep = tables.groups.segments
    tiny = -log_keep < noise.TINY_P
    assert tiny.any()
    expected = (end - begin).astype(np.float32) * -log_keep
    assert (expected[tiny] < 2.0**-12).all()
    assert (end - begin).sum() == tables.block_shots * qubits
