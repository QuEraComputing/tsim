"""The host side of the GPU backends: noise tables, row packing and the tape generator."""

import ctypes
import re
import shutil
import subprocess
from test.helpers.statevector import CIRCUITS

import numpy as np
import pytest

from tsim.circuit import Circuit
from tsim.noise.channels import ChannelSampler
from tsim.statevector import shared
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
    assert noise.words_fit(layout.num_words, 14 * 1024)
    assert not noise.words_fit(layout.num_words, 2048)
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


def test_split_defers_the_channels_that_postselection_cannot_see():
    host, channels = _shots(NOISY)
    layout = host.layout
    tables = noise.noise_tables(channels, layout.noise, layout.num_words, 14 * 1024)
    # NOTE: postselect detector 0 only: every channel but those that flip it waits.
    mask = np.zeros(200, bool)
    mask[0] = True
    plan = host.plan(mask)
    psm = np.zeros(layout.output_words, np.uint64)
    psm[0] = 1
    found = noise.split(tables, channels, layout, plan, psm)
    assert found is not None
    first, later = found
    assert 0 < first.faults < later.faults
    assert np.isclose(first.faults + later.faults, tables.groups.faults)
    # NOTE: reading every output leaves nothing to defer.
    every = np.full(layout.output_words, ~np.uint64(0))
    assert noise.split(tables, channels, layout, plan, every) is None
    assert noise.screen_pass_rate(channels, layout, host.plan(None)) == 1.0


def test_select_reads_the_columns_of_rows():
    rng = np.random.default_rng(3)
    num_outputs, count = 150, 40
    bits = rng.integers(0, 2, (64, num_outputs)).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    for columns in (
        np.arange(count),
        np.arange(count - 3),
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


GXX = shutil.which("g++")

# NOTE: a tape in C++ for one thread per shot (tb = 0): no shuffles. ``draw_one``
# follows the CPU executor's ``shared._cut`` in double.
CPP_DIALECT = tape.Dialect(
    inline="static inline",
    noinline="__attribute__((noinline)) static",
    private="",
    deviation="unsigned*",
    constant="static const",
    float2="float2",
    rsqrt="rsqrt",
    fmax="std::fmax",
    u64="ULL",
    shuffle2="{value}",
    shuffle1="{value}",
    lane_mask=False,
    goto=False,
    amplitude_chunk=32,
    pair_projections=True,
)

CPP_PRELUDE = r"""
#include <cmath>
#include <cstring>
typedef unsigned long long u64;
struct float2 {
    float x, y;
    float2() = default;
    float2(float x_, float y_) : x(x_), y(y_) {}
};
#define GOLDEN 0x9E3779B97F4A7C15ULL
static inline u64 mix64(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline u64 draw(u64 seed, u64 shot, u64 slots, u64 slot) {
    return mix64(seed + (shot * slots + slot) * GOLDEN + GOLDEN);
}
static inline int par(u64 v) { return __builtin_popcountll(v) & 1; }
static inline u64 threshold(double p) {
    double scaled = p * 18446744073709551616.0;
    return scaled < 1.8446744073709550e19 ? (u64)scaled : 0xFFFFFFFFFFFFF800ULL;
}
static inline int draw_one(u64 r, float w0, float w1, float) {
    double total = (double)w0 + (double)w1;
    if (w1 <= w0) return r < threshold(w1 / total);
    return r < 0ULL - threshold(w0 / total);
}
static inline float rsqrt(float x) { return 1.f / std::sqrt(x); }
static inline float2 cmul(float2 p, float2 q) {
    return float2(p.x * q.x - p.y * q.y, p.x * q.y + p.y * q.x);
}
static inline float2 cmac(float2 a, float c, float s, float2 q) {
    return float2(std::fmaf(c, a.x, s * q.x), std::fmaf(c, a.y, s * q.y));
}
static inline float2 cmac_i(float2 a, float c, float s, float2 q) {
    return float2(std::fmaf(c, a.x, -s * q.y), std::fmaf(c, a.y, s * q.x));
}
static inline void deviate(unsigned* deviation, float total) {
    float d = std::fabs(total - 1.f);
    unsigned bits;
    std::memcpy(&bits, &d, 4);
    if (!(d <= 1e-3f) && bits > *deviation) *deviation = bits;
}
"""


def _compile_tapes(sources: list[str], directory) -> list:
    """Compile tape functions ``component_0`` into ``run_<k>(words, shots, nwords, seed, grave, ev, nev, deviation)``."""
    parts = [CPP_PRELUDE]
    for k, source in enumerate(sources):
        parts.append(
            f"namespace v{k} {{\n{source}\n"
            f'extern "C" void run_{k}(u64* words, long shots, long nwords, u64 seed,'
            " int* grave, float* ev, long nev, unsigned* deviation) {\n"
            "    for (long s = 0; s < shots; ++s)\n"
            "        grave[s] = component_0(words + s * nwords, (u64)s, seed, 0,"
            " nev ? ev + s * nev : nullptr, deviation);\n}\n}"
        )
    cpp, lib = directory / "tapes.cpp", directory / "tapes.so"
    cpp.write_text("\n".join(parts))
    # NOTE: -O0 compiles the unrolled tapes several times faster; ``cmac`` and
    # ``cmac_i`` fuse explicitly, so the arithmetic does not depend on it.
    assert GXX is not None
    flags = ["-std=c++17", "-O0", "-ffp-contract=off", "-w", "-shared", "-fPIC"]
    subprocess.run([GXX, *flags, "-o", str(lib), str(cpp)], check=True)  # noqa: S603
    dll = ctypes.CDLL(str(lib))
    functions = []
    for k in range(len(sources)):
        f = getattr(dll, f"run_{k}")
        f.restype = None
        functions.append(f)
    return functions


def _run_tape(f, words, seed, nev):
    words = words.copy()
    shots, nwords = words.shape
    grave = np.empty(shots, np.int32)
    ev = np.zeros((shots, max(nev, 1)), np.float32)
    deviation = np.zeros(1, np.uint32)
    f(
        words.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_long(shots),
        ctypes.c_long(nwords),
        ctypes.c_uint64(seed),
        grave.ctypes.data_as(ctypes.c_void_p),
        ev.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_long(nev),
        deviation.ctypes.data_as(ctypes.c_void_p),
    )
    assert tape.deviation(deviation[0]) == 0.0
    return words, grave, ev[:, :nev]


def _looping_circuit(n=5, layers=5, seed=0):
    """A width-``n`` component with every kind of action that a looping tape runs.

    T gates (diagonal rotations), U3 (others), X, Z and Pauli-product mid-circuit
    measurements with feedforward, noise, postselected detectors, and two
    observables.
    """
    rng = np.random.default_rng(seed)
    lines = ["R " + " ".join(map(str, range(n))), "H " + " ".join(map(str, range(n)))]
    for layer in range(layers):
        for q in range(n):
            t = rng.uniform(-1, 1, 3)
            lines.append(f"U3({t[0]:.4f},{t[1]:.4f},{t[2]:.4f}) {q}")
        lines.append("T " + " ".join(map(str, range(0, n, 2))))
        perm = rng.permutation(n)
        lines.append("CX " + " ".join(map(str, perm[: n & ~1])))
        lines.append(f"DEPOLARIZE1(0.02) {rng.integers(n)}")
        q = int(rng.integers(n))
        if layer % 3 == 0:
            lines += [f"MX {q}", f"CZ rec[-1] {(q + 1) % n}"]
        elif layer % 3 == 1:
            lines += [f"MPP X{q}*Z{(q + 2) % n}*Y{(q + 4) % n}"]
        else:
            lines += [f"M {q}", f"CX rec[-1] {(q + 3) % n}"]
    lines.append("M " + " ".join(map(str, range(n))))
    lines += [
        f"DETECTOR rec[-{n + 1}]",
        "DETECTOR rec[-1] rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-3]",
        "OBSERVABLE_INCLUDE(1) rec[-4] rec[-5]",
    ]
    return "\n".join(lines)


@pytest.mark.skipif(GXX is None, reason="needs g++")
@pytest.mark.parametrize(
    "text",
    [_looping_circuit(), CIRCUITS["midcircuit_feedforward"], CIRCUITS["t_ghz"]],
    ids=["looping", "feedforward", "t_ghz"],
)
def test_looping_and_unrolled_tapes_run_as_the_cpu_executor(
    text, tmp_path, monkeypatch
):
    host, _ = _shots(text)
    layout = host.layout
    num_detectors = Circuit(text).num_detectors
    plan = host.plan(np.ones(num_detectors, bool))
    (component,) = layout.components
    (program,) = plan.programs
    observables = {
        o: index - num_detectors
        for o, index in enumerate(component.output_indices)
        if index >= num_detectors
    }
    assert observables
    # NOTE: unrolled in each dialect variant, and looping.
    variants = [
        CPP_DIALECT,
        tape.Dialect(**{**CPP_DIALECT.__dict__, "goto": True}),
        tape.Dialect(**{**CPP_DIALECT.__dict__, "amplitude_chunk": None}),
        tape.Dialect(**{**CPP_DIALECT.__dict__, "pair_projections": False}),
    ]
    assert not tape.loops(program, 0, 0)
    sources = [tape.tape_function(program, 0, 0, observables, d) for d in variants]
    monkeypatch.setattr(tape, "UNROLL_BUDGET", 0)
    assert tape.loops(program, 0, 0)
    looping = tape.tape_function(program, 0, 0, observables, CPP_DIALECT)
    assert "#pragma unroll 1" in looping
    *unrolled, looped = _compile_tapes([*sources, looping], tmp_path)

    shots, seed = 2000, 0x1234_5678_9ABC_DEF0
    rng = np.random.default_rng(7)
    words = rng.integers(0, 2**63, (shots, layout.num_words), dtype=np.uint64)
    words <<= np.uint64(1)
    words |= rng.integers(0, 2, words.shape, dtype=np.uint64)
    expected = words.copy()
    alive = np.ones(shots, np.bool_)
    shared.run_lanes(
        shots << program.width,
        expected,
        alive,
        0,
        np.uint64(seed),
        True,
        program,
        False,
        1 << 60,
        1,
    )
    assert 0 < alive.sum() < shots
    got, grave, ev = _run_tape(looped, words, seed, len(observables))
    np.testing.assert_array_equal(grave == np.iinfo(np.int32).max, alive)
    np.testing.assert_array_equal(got[alive], expected[alive])
    assert np.abs(ev).max() > 0
    for f in unrolled:
        other, other_grave, other_ev = _run_tape(f, words, seed, len(observables))
        np.testing.assert_array_equal(other_grave, grave)
        np.testing.assert_array_equal(other, got)
        np.testing.assert_allclose(other_ev, ev, atol=1e-5)


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


def test_tape_kernel_fills_in_every_component():
    text = CIRCUITS["two_components"]
    host, _ = _shots(text)
    layout = host.layout
    num_detectors = Circuit(text).num_detectors
    mask = np.ones(num_detectors, bool)
    plan = host.plan(mask)
    tb = tape.threads_per_shot(layout)
    kernel = tape.tape_kernel(layout, plan, mask, num_detectors, tb, TEST_DIALECT)
    count = len(plan.programs)
    assert count == len(layout.components) > 1
    assert kernel.calls.count("grave = min(grave, component_") == count
    assert kernel.seeds.count("const u64 seed_") == count
    assert all(f"component_{i}(" in kernel.functions for i in range(count))
    assert kernel.hits != "0" and kernel.psm_words.any()
    unmasked = tape.tape_kernel(
        layout, host.plan(None), None, num_detectors, tb, TEST_DIALECT
    )
    assert unmasked.hits == "0" and not unmasked.psm_words.any()
    assert tape.supports(layout)


def test_deviation_reports_only_what_float32_rounding_does_not_explain():
    def bits(value):
        return np.float32(value).view(np.uint32)

    assert tape.deviation(bits(1e-4)) == 0.0
    assert tape.deviation(bits(0.01)) == pytest.approx(0.01)
    assert np.isnan(tape.deviation(np.uint32(0x7FC00000)))
