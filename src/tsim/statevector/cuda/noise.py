"""Noise sampling into shot words on the GPU.

A block owns ``S`` consecutive shots and keeps their words in shared memory,
word major. The channels with equal fire probability form a group, and within
a block the group's channels make ``S * len(group)`` Bernoulli trials, channel
major (trial ``i`` is channel ``group[i / S]`` on shot ``i % S``). The host cuts
the groups into *segments* of about ``FAULTS_PER_SEGMENT`` expected faults. In
phase 1 the threads walk the segments with geometric skips and record each
fault as an *event* (its row, its channel and 32 random bits) in shared memory;
phase 2 applies the events: it picks each fault's outcome with the channel's
alias table and XORs its flips into the words. Recording first keeps the warps
converged in phase 2, and many of its table reads in flight.

Each fault outcome is one aligned 32-byte record, so a fault reads about two
sectors of the tables: its channel's ``(first outcome, outcomes)`` pair and a
record (and its alias's record):

    alias:  bits 0-24 keep threshold (probability * 2^24), bits 25-31 alias slot
    words:  the first two flipped words (16 bits each)
    extra:  bits 0-7 number of flips, bits 8-31 start of the rest in the overflow
    masks:  the first two flip masks

A block then screens its shots (constants, postselected direct detectors),
writes their rows (coalesced) and appends the surviving shots to a queue for
the tape kernel. The samples follow ``ChannelSampler``'s distribution, not its
streams: every block draws from its own counter-based stream.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from tsim.noise.channels import ChannelSampler, WordTables

FAULTS_PER_SEGMENT = 4.0
# NOTE: rows take 8 bits of an event, and the block's words and events share
# 46 KB of static shared memory.
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

SOURCE = r"""
struct __align__(32) Record {{ unsigned alias; unsigned words; unsigned extra; unsigned pad; u64 m0; u64 m1; }};

__device__ __forceinline__ void apply_fault(u64* sw, int row, int2 channel, unsigned bits,
    const Record* __restrict__ records, const int* __restrict__ overflow_word,
    const u64* __restrict__ overflow_mask)
{{
    unsigned long long x = (unsigned long long)(bits >> 8) * (unsigned)channel.y;
    int slot = (int)(x >> 24);
    unsigned fraction = (unsigned)(x & 0xFFFFFF);
    Record r = records[channel.x + slot];
    if (fraction >= (r.alias & 0x1FFFFFF)) r = records[channel.x + (r.alias >> 25)];
    int flips = r.extra & 255;
    if (flips > 0) atomicXor(&sw[(r.words & 0xFFFF) * {S} + row], r.m0);
    if (flips > 1) atomicXor(&sw[(r.words >> 16) * {S} + row], r.m1);
    for (int j = 2; j < flips; ++j) {{
        int o = (r.extra >> 8) + j - 2;
        atomicXor(&sw[overflow_word[o] * {S} + row], overflow_mask[o]);
    }}
}}

extern "C" __global__ void __launch_bounds__(256) noise(
    u64* __restrict__ out, long long shots, u64 seed, int num_segments,
    const int* __restrict__ seg_first, const int* __restrict__ seg_begin,
    const int* __restrict__ seg_end, const float* __restrict__ seg_log_keep,
    const int2* __restrict__ channels, const Record* __restrict__ records,
    const int* __restrict__ overflow_word, const u64* __restrict__ overflow_mask,
    const u64* __restrict__ constant, const u64* __restrict__ screen, const u64* __restrict__ kept,
    unsigned* __restrict__ queue, unsigned* __restrict__ count)
{{
    __shared__ u64 sw[{NW} * {S}];
    __shared__ uint2 events[{E}];
    __shared__ int num_events;
    long long base = (long long)blockIdx.x * {S};
    int ns = (int)min((long long){S}, shots - base);
    for (int i = threadIdx.x; i < {NW} * {S}; i += blockDim.x) sw[i] = 0;
    if (threadIdx.x == 0) num_events = 0;
    __syncthreads();
    for (int sg = threadIdx.x; sg < num_segments; sg += blockDim.x) {{
        float lk = seg_log_keep[sg];
        int first = seg_first[sg], end = seg_end[sg];
        int idx = seg_begin[sg] - 1;
        u64 state = mix64(seed ^ ((u64)blockIdx.x * 0xD1B54A32D192ED03ULL) ^ ((u64)sg * GOLDEN));
        while (true) {{
            state += GOLDEN;
            u64 r = mix64(state);
            float skip = lk < -3.0e38f ? 0.f : floorf(__logf(unit(r)) / lk);
            if (skip >= (float)(end - 1 - idx)) break;
            idx += 1 + (int)skip;
            int ch = idx / {S}, row = idx - ch * {S};
            if (row >= ns) continue;  // the last block's missing shots
            // NOTE: the skip reads the high 24 bits of r, the outcome its low 32.
            unsigned bits = (unsigned)r;
            int e = atomicAdd(&num_events, 1);
            if (e < {E}) events[e] = make_uint2((unsigned)row | ((unsigned)(first + ch) << 8), bits);
            else apply_fault(sw, row, channels[first + ch], bits, records, overflow_word, overflow_mask);
        }}
    }}
    __syncthreads();
    int ne = min(num_events, {E});
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {{
        uint2 v = events[e];
        apply_fault(sw, v.x & 255, channels[v.x >> 8], v.y, records, overflow_word, overflow_mask);
    }}
    __syncthreads();
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
        // NOTE: one atomic per warp: the leader reserves the survivors' slots.
        unsigned ballot = __ballot_sync(0xffffffffu, alive);
        if (ballot) {{
            int lane = threadIdx.x & 31, leader = __ffs(ballot) - 1;
            unsigned b = 0;
            if (lane == leader) b = atomicAdd(count, __popc(ballot));
            b = __shfl_sync(0xffffffffu, b, leader);
            if (alive) queue[b + __popc(ballot & ((1u << lane) - 1))] = (unsigned)(base + s);
        }}
    }}
    __syncthreads();
    u64* rows = out + base * {NW};
    for (int i = threadIdx.x; i < ns * {NW}; i += blockDim.x) {{
        int s = i / {NW}, word = i - s * {NW};
        rows[i] = sw[word * {S} + s];
    }}
}}
"""


@dataclass(frozen=True)
class NoiseTables:
    """The host tables of the noise kernel.

    Attributes:
        block_shots: Shots per block (``S``).
        max_events: Events a block records before it applies faults at once.
        segments: ``(first, begin, end, log_keep)`` of each segment: its group's
            first channel (in group order), its trial range and log(1 - p).
        channels: Each channel's ``(first outcome, outcomes)``, in group order.
        records: One 32-byte record per fault outcome.
        overflow_word: Words of the flips past an outcome's first two.
        overflow_mask: Masks of the flips past an outcome's first two.

    """

    block_shots: int
    max_events: int
    segments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    channels: np.ndarray
    records: np.ndarray
    overflow_word: np.ndarray
    overflow_mask: np.ndarray

    def source(self, num_words: int, output_words: int) -> str:
        """Return the CUDA source of the ``noise`` kernel."""
        return SOURCE.format(
            NW=num_words, OW=output_words, S=self.block_shots, E=self.max_events
        )


def noise_tables(
    channels: ChannelSampler, flips: WordTables, num_words: int
) -> NoiseTables:
    """Return the noise kernel's tables for rows of ``num_words`` shot words."""
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
    faults = float(fire.p_fire.sum()) * block_shots
    max_events = min(
        max(256, int(2 * faults + 256)),
        (SHARED_BYTES - 8 * num_words * block_shots) // 8,
    )

    # NOTE: channels of equal fire probability form a group.
    order = np.argsort(fire.p_fire, kind="stable")
    p_sorted = fire.p_fire[order]
    starts = np.flatnonzero(np.diff(p_sorted, prepend=-1.0) != 0)
    group_start = np.append(starts, len(order))
    first, begin, end, log_keep = [], [], [], []
    for g in range(len(starts)):
        lo, hi = int(group_start[g]), int(group_start[g + 1])
        p = float(p_sorted[lo])
        lk = -np.inf if p >= 1.0 else float(np.log1p(-p))
        trials = block_shots * (hi - lo)
        pieces = max(1, round(trials * min(p, 1.0) / FAULTS_PER_SEGMENT))
        cuts = np.linspace(0, trials, pieces + 1).astype(np.int64)
        for a, b in itertools.pairwise(cuts):
            if b > a:
                first.append(lo)
                begin.append(a)
                end.append(b)
                log_keep.append(lk)
    if max(end, default=0) >= 1 << 31:
        raise ValueError("too many noise channels for the CUDA noise sampler")
    # NOTE: the segments with the most expected faults first, so the threads that
    # take a second segment take short ones.
    expected = (np.array(end) - np.array(begin)) * -np.expm1(np.array(log_keep))
    by_work = np.argsort(-expected, kind="stable")

    outcome_start = fire.outcome_start
    records = np.zeros(len(fire.alias_prob), dtype=_RECORD)
    overflow_word: list[int] = []
    overflow_mask: list[int] = []
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
            records[k]["extra"] = len(words) | (len(overflow_word) << 8)
            overflow_word.extend(words[2:])
            overflow_mask.extend(masks[2:])
    if len(overflow_word) >= 1 << 24:
        raise ValueError("too many noise flips for the CUDA noise sampler")
    pairs = np.stack([outcome_start[order], np.diff(outcome_start)[order]], axis=1)
    return NoiseTables(
        block_shots=block_shots,
        max_events=max_events,
        segments=(
            np.asarray(first, np.int32)[by_work],
            np.asarray(begin, np.int32)[by_work],
            np.asarray(end, np.int32)[by_work],
            np.asarray(log_keep, np.float32)[by_work],
        ),
        channels=pairs.astype(np.int32),
        records=records,
        overflow_word=np.asarray([*overflow_word, 0], np.int32),
        overflow_mask=np.asarray([*overflow_mask, 0], np.uint64),
    )
