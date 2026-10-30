"""Noise sampling into shot words on the GPU.

A block owns ``S`` consecutive shots and keeps their words in shared memory,
word major. The channels with equal fire probability and equal numbers of
outcomes form a group, and within a block the group's channels make
``S * len(group)`` Bernoulli trials, channel major (trial ``i`` is channel
``group[i / S]`` on shot ``i % S``). The host cuts the groups into *segments* of
about ``FAULTS_PER_SEGMENT`` expected faults. In phase 1 the threads walk the
segments with geometric skips and record each fault as an *event* in shared
memory: its row, the record of the alias slot it draws and the draw's
fraction; phase 2 applies the events: it keeps the slot's outcome or takes its
alias and XORs the outcome's flips into the words. Recording first keeps the
warps converged in phase 2, and many of its table reads in flight.

Each launch lays out its channels' outcomes in group order, so a group's
channel ``c`` has its records at ``first + c * outcomes`` and a fault reads one
aligned 32-byte record (and its alias's, in the same channel):

    alias:  bits 0-24 keep threshold (probability * 2^24), bits 25-31 alias slot
    words:  the first two flipped words (16 bits each)
    extra:  bits 0-7 number of flips, bits 8-31 start of the rest in the overflow
    masks:  the first two flip masks

The flips past an outcome's first two are 16-byte ``(word, mask)`` pairs, in
order.

A block then screens its shots (constants, postselected direct detectors),
writes their rows (coalesced; only the survivors' unless every row is asked
for), appends the surviving shots to a queue for the tape kernel and marks
them in a bitmap of the batch's shots. The samples follow ``ChannelSampler``'s distribution, not its
streams: every block draws from its own counter-based stream.

Under postselection the channels can be drawn in two passes. Those with an
outcome that flips a bit the screen, the tapes or the postselection read go
first (``noise``); the others cannot change whether a shot is kept, so
``noise_kept`` draws them after the tapes, on the kept shots only. Channels are
independent and their flips commute, so the kept shots' distribution is
unchanged.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np

from tsim.noise.channels import ChannelSampler, WordTables

FAULTS_PER_SEGMENT = 4.0
# NOTE: rows take 8 bits of an event, and the block's words and events share
# 46 KB of shared memory (the events' share is dynamic, set per launch).
MAX_BLOCK_SHOTS = 256
SHARED_BYTES = 46 * 1024

_RECORD = np.dtype(
    [
        ("alias", "<u4"),
        ("words", "<u4"),
        ("extra", "<u4"),
        ("pad", "<u4"),
        ("m0", "<u8"),
        ("m1", "<u8"),
    ],
    align=True,
)
_FLIP = np.dtype([("word", "<u4"), ("pad", "<u4"), ("mask", "<u8")], align=True)

SOURCE = r"""
struct __align__(32) Record {{ unsigned alias; unsigned words; unsigned extra; unsigned pad; u64 m0; u64 m1; }};
struct __align__(16) Flip {{ unsigned word; unsigned pad; u64 mask; }};

// The fault drawn at record ``rec`` (bits 0-23 of ``draw`` its fraction, bits
// 24-31 its slot in the channel): the slot's outcome or its alias's.
__device__ __forceinline__ void apply_fault(u64* sw, int row, unsigned rec, unsigned draw,
    const Record* __restrict__ records, const Flip* __restrict__ overflow)
{{
    Record r = records[rec];
    if ((draw & 0xFFFFFF) >= (r.alias & 0x1FFFFFF)) r = records[rec - (draw >> 24) + (r.alias >> 25)];
    int flips = r.extra & 255;
    if (flips > 0) atomicXor(&sw[(r.words & 0xFFFF) * {S} + row], r.m0);
    if (flips > 1) atomicXor(&sw[(r.words >> 16) * {S} + row], r.m1);
    for (int j = 2; j < flips; ++j) {{
        Flip f = overflow[(r.extra >> 8) + j - 2];
        atomicXor(&sw[f.word * {S} + row], f.mask);
    }}
}}

// The returned outputs of the block's ``ns`` rows ``sw[w * S + s]``, bit-packed
// into {NB} bytes per row, contiguously at ``dst``. Output word ``k`` of a row
// is the pieces ``pieces[start[k] : start[k + 1]]``, each ``(word, bit | (length
// - 1) << 6 | position << 12)``: ``length`` bits from ``bit`` of a shot word.
// Consecutive threads pack one word of consecutive rows (one piece, distinct
// banks) into ``stage``, from which the block's rows go out 16 bytes at a time.
__device__ __forceinline__ void pack_rows(const u64* sw, int ns, const uint2* __restrict__ pieces,
    const unsigned* __restrict__ start, unsigned char* stage, unsigned char* __restrict__ dst)
{{
    for (int i = threadIdx.x; i < {NK} * {S}; i += blockDim.x) {{
        int k = i / {S}, s = i - k * {S};
        if (s >= ns) continue;
        u64 v = 0;
        for (unsigned p = start[k]; p < start[k + 1]; ++p) {{
            uint2 q = pieces[p];
            u64 x = sw[q.x * {S} + s] >> (q.y & 63);
            int length = ((q.y >> 6) & 63) + 1;
            if (length < 64) x &= (1ULL << length) - 1;
            v |= x << (q.y >> 12);
        }}
        unsigned char* at = stage + s * {NB} + 8 * k;
        #pragma unroll
        for (int j = 0; j < 8; ++j) if (8 * k + j < {NB}) at[j] = (unsigned char)(v >> (8 * j));
    }}
    __syncthreads();
    int total = ns * {NB};
    for (int i = threadIdx.x; i < total / 16; i += blockDim.x) ((uint4*)dst)[i] = ((const uint4*)stage)[i];
    for (int i = total / 16 * 16 + threadIdx.x; i < total; i += blockDim.x) dst[i] = stage[i];
}}

// Phases 1 and 2: draw the segments' faults on the block's ``ns`` shots and XOR them into ``sw``.
__device__ __forceinline__ void add_faults(u64* sw, uint2* events, int* num_events, int max_events,
    int ns, u64 stream, int num_segments,
    const int* __restrict__ seg_first, const int* __restrict__ seg_outcomes,
    const int* __restrict__ seg_begin, const int* __restrict__ seg_end,
    const float* __restrict__ seg_log_keep, const Record* __restrict__ records,
    const Flip* __restrict__ overflow)
{{
    for (int sg = threadIdx.x; sg < num_segments; sg += blockDim.x) {{
        float lk = seg_log_keep[sg];
        int first = seg_first[sg], outcomes = seg_outcomes[sg], end = seg_end[sg];
        int idx = seg_begin[sg] - 1;
        u64 state = mix64(stream ^ ((u64)sg * GOLDEN));
        while (true) {{
            state += GOLDEN;
            u64 r = mix64(state);
            float skip = lk < -3.0e38f ? 0.f : floorf(__logf(unit(r)) / lk);
            if (skip >= (float)(end - 1 - idx)) break;
            idx += 1 + (int)skip;
            int ch = idx / {S}, row = idx - ch * {S};
            if (row >= ns) continue;  // the last block's missing shots
            // NOTE: the skip reads the high 24 bits of r, the outcome its low 32:
            // a slot from the high 24 of those, the fraction of the slot below.
            unsigned long long x = (unsigned long long)((unsigned)r >> 8) * (unsigned)outcomes;
            unsigned slot = (unsigned)(x >> 24);
            unsigned rec = (unsigned)(first + ch * outcomes) + slot;
            unsigned draw = (unsigned)(x & 0xFFFFFF) | (slot << 24);
            int e = atomicAdd(num_events, 1);
            if (e < max_events) events[e] = make_uint2((unsigned)row | (rec << 8), draw);
            else apply_fault(sw, row, rec, draw, records, overflow);
        }}
    }}
    __syncthreads();
    int ne = min(*num_events, max_events);
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {{
        uint2 v = events[e];
        apply_fault(sw, v.x & 255, v.x >> 8, v.y, records, overflow);
    }}
    __syncthreads();
}}

extern "C" __global__ void __launch_bounds__(256) noise(
    u64* __restrict__ out, long long shots, u64 seed, int num_segments, int max_events,
    const int* __restrict__ seg_first, const int* __restrict__ seg_outcomes,
    const int* __restrict__ seg_begin, const int* __restrict__ seg_end,
    const float* __restrict__ seg_log_keep, const Record* __restrict__ records,
    const Flip* __restrict__ overflow,
    const u64* __restrict__ constant, const u64* __restrict__ screen, const u64* __restrict__ kept,
    unsigned* __restrict__ queue, unsigned* __restrict__ count, unsigned* __restrict__ survivors,
    int every_row)
{{
    __shared__ u64 sw[{NW} * {S}];
    extern __shared__ uint2 events[];
    __shared__ int num_events;
    __shared__ bool passed[{S}];
    long long base = (long long)blockIdx.x * {S};
    int ns = (int)min((long long){S}, shots - base);
    for (int i = threadIdx.x; i < {NW} * {S}; i += blockDim.x) sw[i] = 0;
    if (threadIdx.x == 0) num_events = 0;
    __syncthreads();
    add_faults(sw, events, &num_events, max_events, ns, seed ^ ((u64)blockIdx.x * 0xD1B54A32D192ED03ULL),
               num_segments, seg_first, seg_outcomes, seg_begin, seg_end, seg_log_keep, records,
               overflow);
    for (int s0 = 0; s0 < {S}; s0 += blockDim.x) {{
        int s = s0 + threadIdx.x;
        bool alive = false;
        if (s < ns) {{
            u64 hit = 0;
            #pragma unroll
            for (int i = 0; i < {OW}; ++i) {{
                u64 v = sw[i * {S} + s] ^ constant[i];
                hit |= v & screen[i];
                sw[i * {S} + s] = v;
            }}
            if (hit) {{
                #pragma unroll
                for (int i = 0; i < {OW}; ++i) sw[i * {S} + s] &= kept[i];
            }}
            alive = !hit;
        }}
        if (s < {S}) passed[s] = alive;
        // NOTE: one atomic per warp: the leader reserves the survivors' slots.
        unsigned ballot = __ballot_sync(0xffffffffu, alive);
        long long word = (base + s0 + (threadIdx.x & ~31)) >> 5;
        if (survivors != nullptr && (threadIdx.x & 31) == 0 && s0 + (threadIdx.x & ~31) < ns)
            survivors[word] = ballot;
        if (ballot) {{
            int lane = threadIdx.x & 31, leader = __ffs(ballot) - 1;
            unsigned b = 0;
            if (lane == leader) b = atomicAdd(count, __popc(ballot));
            b = __shfl_sync(0xffffffffu, b, leader);
            if (alive) queue[b + __popc(ballot & ((1u << lane) - 1))] = (unsigned)(base + s);
        }}
    }}
    __syncthreads();
    // NOTE: without ``every_row`` only the shots that pass the screen are read again.
    u64* rows = out + base * {NW};
    for (int i = threadIdx.x; i < ns * {NW}; i += blockDim.x) {{
        int s = i / {NW}, word = i - s * {NW};
        if (every_row || passed[s]) rows[i] = sw[word * {S} + s];
    }}
}}

// The deferred channels' faults on the ``n`` kept shots ``index``, then their
// returned outputs bit-packed (``pack_rows``). The deferred channels flip output
// words only, so a block loads just those.
extern "C" __global__ void __launch_bounds__(256) noise_kept(
    const u64* __restrict__ out, long long n, const unsigned* __restrict__ index, u64 seed,
    int num_segments, int max_events,
    const int* __restrict__ seg_first, const int* __restrict__ seg_outcomes,
    const int* __restrict__ seg_begin, const int* __restrict__ seg_end,
    const float* __restrict__ seg_log_keep, const Record* __restrict__ records,
    const Flip* __restrict__ overflow,
    unsigned char* __restrict__ dst, const uint2* __restrict__ pieces,
    const unsigned* __restrict__ piece_start)
{{
    __shared__ u64 sw[{OW} * {S}];
    extern __shared__ __align__(16) uint2 events[];
    __shared__ int num_events;
    long long base = (long long)blockIdx.x * {S};
    int ns = (int)min((long long){S}, n - base);
    for (int i = threadIdx.x; i < ns * {OW}; i += blockDim.x) {{
        int s = i / {OW}, word = i - s * {OW};
        sw[word * {S} + s] = out[(long long)index[base + s] * {NW} + word];
    }}
    if (threadIdx.x == 0) num_events = 0;
    __syncthreads();
    add_faults(sw, events, &num_events, max_events, ns, seed ^ ((u64)blockIdx.x * 0xD1B54A32D192ED03ULL),
               num_segments, seg_first, seg_outcomes, seg_begin, seg_end, seg_log_keep, records,
               overflow);
    // NOTE: the events are applied, so their buffer stages the packed rows.
    pack_rows(sw, ns, pieces, piece_start, (unsigned char*)events, dst + base * {NB});
}}
"""


@dataclass(frozen=True)
class NoiseGroups:
    """A set of channels as one launch draws them: their groups, cut into segments.

    Attributes:
        max_events: Events a block records before it applies faults at once.
        segments: ``(first, outcomes, begin, end, log_keep)`` of each segment:
            its group's first record and outcomes per channel, its trial range
            and log(1 - p).
        records: The channels' outcome records, in group order.
        faults: Expected faults per shot.

    """

    max_events: int
    segments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    records: np.ndarray
    faults: float


@dataclass(frozen=True)
class NoiseTables:
    """The host tables of the noise kernels.

    Attributes:
        block_shots: Shots per block (``S``).
        groups: Every channel's groups.
        records: One 32-byte record per fault outcome, in channel order.
        overflow: The ``(word, mask)`` flips past an outcome's first two.
        p_fire: Each channel's fire probability.
        outcome_start: Each channel's first outcome (and the end).
        num_words: Words per shot.

    """

    block_shots: int
    groups: NoiseGroups
    records: np.ndarray
    overflow: np.ndarray
    p_fire: np.ndarray
    outcome_start: np.ndarray
    num_words: int

    def source(self, output_words: int, packed_bytes: int) -> str:
        """Return the CUDA source of the ``noise`` and ``noise_kept`` kernels.

        ``noise_kept`` packs rows of ``packed_bytes`` bytes.
        """
        return SOURCE.format(
            NW=self.num_words,
            OW=output_words,
            S=self.block_shots,
            NB=max(packed_bytes, 1),
            NK=max(-(-packed_bytes // 8), 1),
        )

    def subset(self, select: np.ndarray) -> NoiseGroups:
        """Return the groups of the channels ``select`` (a boolean mask)."""
        return _groups(self, select)


def _groups(tables: NoiseTables, select: np.ndarray) -> NoiseGroups:
    p_fire, outcome_start, S = tables.p_fire, tables.outcome_start, tables.block_shots
    outcomes = np.diff(outcome_start)
    faults = float(p_fire[select].sum())
    max_events = min(
        max(256, int(2 * faults * S + 256)),
        (SHARED_BYTES - 8 * tables.num_words * S) // 8,
    )
    # NOTE: channels of equal fire probability and outcomes form a group, so a
    # group's channel c has its records at first + c * outcomes.
    chosen = np.flatnonzero(select)
    order = chosen[np.lexsort((outcomes[chosen], p_fire[chosen]))]
    p_sorted, n_sorted = p_fire[order], outcomes[order]
    change = (np.diff(p_sorted, prepend=-1.0) != 0) | (
        np.diff(n_sorted, prepend=-1) != 0
    )
    starts = np.flatnonzero(change)
    group_start = np.append(starts, len(order))
    record_start = np.concatenate([[0], np.cumsum(n_sorted)])
    first, per, begin, end, log_keep = [], [], [], [], []
    for g in range(len(starts)):
        lo, hi = int(group_start[g]), int(group_start[g + 1])
        p = float(p_sorted[lo])
        lk = -np.inf if p >= 1.0 else float(np.log1p(-p))
        trials = S * (hi - lo)
        pieces = max(1, round(trials * min(p, 1.0) / FAULTS_PER_SEGMENT))
        cuts = np.linspace(0, trials, pieces + 1).astype(np.int64)
        for a, b in itertools.pairwise(cuts):
            if b > a:
                first.append(int(record_start[lo]))
                per.append(int(n_sorted[lo]))
                begin.append(a)
                end.append(b)
                log_keep.append(lk)
    if max(end, default=0) >= 1 << 31:
        raise ValueError("too many noise channels for the CUDA noise sampler")
    # NOTE: the segments with the most expected faults first, so the threads that
    # take a second segment take short ones.
    expected = (np.array(end) - np.array(begin)) * -np.expm1(np.array(log_keep))
    by_work = np.argsort(-expected, kind="stable")
    # NOTE: each chosen channel's outcomes, in group order.
    within = np.arange(int(record_start[-1])) - np.repeat(record_start[:-1], n_sorted)
    records = tables.records[np.repeat(outcome_start[order], n_sorted) + within]
    return NoiseGroups(
        max_events=max_events,
        segments=(
            np.asarray(first, np.int32)[by_work],
            np.asarray(per, np.int32)[by_work],
            np.asarray(begin, np.int32)[by_work],
            np.asarray(end, np.int32)[by_work],
            np.asarray(log_keep, np.float32)[by_work],
        ),
        records=records,
        faults=faults,
    )


def _outcome_hits(flips: WordTables, read: np.ndarray) -> np.ndarray:
    """Return which outcomes flip a bit of ``read`` (as shot words)."""
    hit = (read[flips.word] & flips.mask) != 0
    # NOTE: hits per outcome as differences of running counts.
    per_flip = np.concatenate([[0], np.cumsum(hit)])
    return per_flip[flips.start[1:]] - per_flip[flips.start[:-1]] > 0


def deferred_channels(
    channels: ChannelSampler, flips: WordTables, read: np.ndarray
) -> np.ndarray:
    """Return which channels flip none of the bits ``read`` (as shot words)."""
    hits = np.concatenate([[0], np.cumsum(_outcome_hits(flips, read))])
    start = channels._tables.outcome_start
    return hits[start[1:]] == hits[start[:-1]]


def pass_rate(channels: ChannelSampler, flips: WordTables, read: np.ndarray) -> float:
    """Return the share of shots whose faults flip no bit of ``read``, to first order.

    Each channel's chance to flip one is its fire probability times its hitting
    outcomes' probabilities (from its alias table); flips that cancel are not
    counted.
    """
    fire = channels._tables
    n = np.diff(fire.outcome_start)
    per = np.repeat(n, n).astype(np.float64)
    q = fire.alias_prob / per
    np.add.at(q, fire.alias_index, (1 - fire.alias_prob) / per)
    hits = np.concatenate([[0.0], np.cumsum(q * _outcome_hits(flips, read))])
    start = fire.outcome_start
    p = np.clip(fire.p_fire * (hits[start[1:]] - hits[start[:-1]]), 0.0, 1.0)
    return float(np.exp(np.log1p(-np.minimum(p, 1 - 1e-12)).sum()))


def noise_tables(
    channels: ChannelSampler, flips: WordTables, num_words: int
) -> NoiseTables:
    """Return the noise kernels' tables for rows of ``num_words`` shot words."""
    fire = channels._tables
    if len(fire.p_fire) >= 1 << 24:
        raise ValueError("the CUDA noise sampler supports fewer than 2^24 channels")
    if num_words >= 1 << 16:
        raise ValueError("the CUDA noise sampler supports fewer than 2^16 shot words")
    block_shots = max(
        32, min(MAX_BLOCK_SHOTS, (SHARED_BYTES // 2 // (8 * num_words)) // 32 * 32)
    )
    if 8 * num_words * block_shots > SHARED_BYTES - 8 * 256:
        raise ValueError(
            f"{num_words} shot words do not fit the noise kernel's shared memory"
        )
    outcome_start = fire.outcome_start
    if len(fire.alias_prob) >= 1 << 24:
        raise ValueError("the CUDA noise sampler supports fewer than 2^24 outcomes")
    records = np.zeros(len(fire.alias_prob), dtype=_RECORD)
    overflow: list[tuple[int, int]] = []
    for c in range(len(fire.p_fire)):
        lo, hi = int(outcome_start[c]), int(outcome_start[c + 1])
        if hi - lo > 127:
            raise ValueError(
                "the CUDA noise sampler supports channels of up to 127 outcomes"
            )
        for k in range(lo, hi):
            keep = (
                1 << 24
                if hi - lo == 1
                else min(round(float(fire.alias_prob[k]) * (1 << 24)), 1 << 24)
            )
            records[k]["alias"] = keep | ((int(fire.alias_index[k]) - lo) << 25)
            j0, j1 = int(flips.start[k]), int(flips.start[k + 1])
            words = flips.word[j0:j1].tolist()
            masks = flips.mask[j0:j1].tolist()
            records[k]["words"] = (words[0] if words else 0) | (
                (words[1] if len(words) > 1 else 0) << 16
            )
            records[k]["m0"] = masks[0] if masks else 0
            records[k]["m1"] = masks[1] if len(masks) > 1 else 0
            records[k]["extra"] = len(words) | (len(overflow) << 8)
            overflow.extend(zip(words[2:], masks[2:], strict=True))
    if len(overflow) >= 1 << 24:
        raise ValueError("too many noise flips for the CUDA noise sampler")
    pairs = np.zeros(len(overflow) + 1, dtype=_FLIP)
    if overflow:
        pairs["word"][:-1] = [w for w, _ in overflow]
        pairs["mask"][:-1] = [m for _, m in overflow]
    tables = NoiseTables(
        block_shots=block_shots,
        groups=NoiseGroups(0, (np.empty(0, np.int32),) * 5, records[:0], 0.0),
        records=records,
        overflow=pairs,
        p_fire=fire.p_fire,
        outcome_start=outcome_start,
        num_words=num_words,
    )
    every = np.ones(len(fire.p_fire), dtype=bool)
    return replace(tables, groups=_groups(tables, every))
