"""Noise sampling into shot words on a CUDA GPU.

The kernels read the tables of ``tsim.statevector.gpu.noise`` (see there): a
block owns ``S`` consecutive shots and keeps their words in shared memory,
word major, walks segments of the channel groups with geometric skips,
records the faults as events in shared memory and then XORs their outcomes'
flips into the shots' words. Recording first keeps the warps converged while
the faults are applied, and many of their table reads in flight. Rows too wide
for shared memory keep their words in device memory, where the faults are XORed
in with device atomics.

A block then screens its shots (constants, postselected direct detectors),
writes their rows (coalesced; only the survivors' unless every row is asked
for), appends the surviving shots to a queue for the tape kernel and marks
them in a bitmap of the batch's shots. Under postselection, ``noise_kept``
draws the noise that cannot change whether a shot is kept
(``gpu.noise.deferred_channels``) on the kept shots only, after the tapes.
"""

from __future__ import annotations

from tsim.noise.channels import ChannelSampler, WordTables
from tsim.statevector.gpu.noise import (
    NoiseTables,
    noise_tables as _noise_tables,
    words_fit,
)

THREADS = 256
"""Threads per block of the noise kernels."""

SHARED_BYTES = 46 * 1024
"""Shared memory of a block's words and events (the events' share is set per launch)."""

_NOISE_SIGNATURE = r"""
extern "C" __global__ void __launch_bounds__({NT}) noise(
    u64* __restrict__ out, long long shots, u64 seed, int num_segments, int max_events,
    const int* __restrict__ seg_first, const int* __restrict__ seg_outcomes,
    const int* __restrict__ seg_begin, const int* __restrict__ seg_end,
    const float* __restrict__ seg_log_keep, const Record* __restrict__ records,
    const Flip* __restrict__ overflow,
    const u64* __restrict__ constant, const u64* __restrict__ screen,
    unsigned* __restrict__ queue, unsigned* __restrict__ count, unsigned* __restrict__ survivors)
"""

# NOTE: the noise kernel with the block's words in shared memory, word major.
NOISE_SHARED = _NOISE_SIGNATURE + r"""{{
    __shared__ u64 sw[{NW} * {S}];
    extern __shared__ uint2 events[];
    __shared__ int num_events;
    __shared__ bool passed[{S}];
    long long base = (long long)blockIdx.x * {S};
    int ns = (int)min((long long){S}, shots - base);
    for (int i = threadIdx.x; i < {NW} * {S}; i += blockDim.x) sw[i] = 0;
    if (threadIdx.x == 0) num_events = 0;
    __syncthreads();
    add_faults<{S}, 1>(sw, events, &num_events, max_events, ns, block_stream(seed), num_segments,
                       seg_first, seg_outcomes, seg_begin, seg_end, seg_log_keep, records, overflow);
    screen_shots<{S}, 1>(sw, ns, base, constant, screen, passed, queue, count, survivors);
    __syncthreads();
    // NOTE: only the shots that pass the screen are read again.
    u64* rows = out + base * {NW};
    for (int i = threadIdx.x; i < ns * {NW}; i += blockDim.x) {{
        int s = i / {NW}, word = i - s * {NW};
        if (passed[s]) rows[i] = sw[word * {S} + s];
    }}
}}
"""

# NOTE: the noise kernel for rows too wide for shared memory: the faults are XORed
# into the rows in device memory, and the events have all of the shared memory.
NOISE_DEVICE = _NOISE_SIGNATURE + r"""{{
    extern __shared__ uint2 events[];
    __shared__ int num_events;
    __shared__ bool passed[{S}];
    long long base = (long long)blockIdx.x * {S};
    int ns = (int)min((long long){S}, shots - base);
    u64* rows = out + base * {NW};
    for (int i = threadIdx.x; i < ns * {NW}; i += blockDim.x) rows[i] = 0;
    if (threadIdx.x == 0) num_events = 0;
    __syncthreads();
    add_faults<1, {NW}>(rows, events, &num_events, max_events, ns, block_stream(seed), num_segments,
                        seg_first, seg_outcomes, seg_begin, seg_end, seg_log_keep, records, overflow);
    screen_shots<1, {NW}>(rows, ns, base, constant, screen, passed, queue, count, survivors);
}}
"""

NOISE_KEPT = r"""
// The deferred channels' faults on the ``n`` kept shots ``index``, then their
// returned outputs bit-packed (``pack_rows``). The deferred channels flip output
// words only, so a block loads just those.
extern "C" __global__ void __launch_bounds__({NT}) noise_kept(
    const u64* __restrict__ out, long long n, const unsigned* __restrict__ index, u64 seed,
    int num_segments, int max_events,
    const int* __restrict__ seg_first, const int* __restrict__ seg_outcomes,
    const int* __restrict__ seg_begin, const int* __restrict__ seg_end,
    const float* __restrict__ seg_log_keep, const Record* __restrict__ records,
    const Flip* __restrict__ overflow,
    unsigned char* __restrict__ dst)
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
    add_faults<{S}, 1>(sw, events, &num_events, max_events, ns, block_stream(seed), num_segments,
                       seg_first, seg_outcomes, seg_begin, seg_end, seg_log_keep, records, overflow);
    // NOTE: the events are applied, so their buffer stages the packed rows.
    pack_rows<{S}>(sw, ns, (unsigned char*)events, dst + base * {NB});
}}
"""

SOURCE = r"""
struct __align__(32) Record {{ unsigned alias; unsigned words; unsigned extra; unsigned keep; u64 m0; u64 m1; }};
struct __align__(16) Flip {{ unsigned word; unsigned pad; u64 mask; }};

// The fault drawn at record ``rec`` (bits 0-23 of ``draw`` the high 24 bits of
// its 40-bit fraction, bits 24-31 its slot in the channel): the slot's outcome
// if the fraction is below the keep threshold (gpu/noise.py), else its alias's.
// The fraction's low 16 bits matter only when its high bits tie with the
// threshold's, and then come from a hash of the fault. Word ``w`` of row ``s``
// is ``sw[w * WS + s * SS]``.
template <int WS, int SS>
__device__ __forceinline__ void apply_fault(u64* sw, int row, unsigned rec, unsigned draw,
    u64 stream, const Record* __restrict__ records, const Flip* __restrict__ overflow)
{{
    Record r = records[rec];
    unsigned high = draw & 0xFFFFFF, keep = r.alias & 0x1FFFFFF;
    if (high > keep || (high == keep &&
            (unsigned)(mix64(stream ^ ((u64)(rec << 8 | row) * GOLDEN)) & 0xFFFF) >= r.keep))
        r = records[rec - (draw >> 24) + (r.alias >> 25)];
    int flips = r.extra & 255;
    if (flips > 0) atomicXor(&sw[(r.words & 0xFFFF) * WS + row * SS], r.m0);
    if (flips > 1) atomicXor(&sw[(r.words >> 16) * WS + row * SS], r.m1);
    for (int j = 2; j < flips; ++j) {{
        Flip f = overflow[(r.extra >> 8) + j - 2];
        atomicXor(&sw[f.word * WS + row * SS], f.mask);
    }}
}}

// The outputs of a block's ``ns`` rows ``sw[w * S + s]``, bit-packed into
// {NB} bytes per row (the bits past them cleared), contiguously at ``dst``.
// Consecutive threads pack one word of consecutive rows (distinct banks) into
// ``stage``, from which the block's rows go out 16 bytes at a time.
template <int S>
__device__ __forceinline__ void pack_rows(const u64* sw, int ns, unsigned char* stage,
    unsigned char* __restrict__ dst)
{{
    for (int i = threadIdx.x; i < {NK} * S; i += blockDim.x) {{
        int k = i / S, s = i - k * S;
        if (s >= ns) continue;
        u64 v = sw[k * S + s];
        if (k == {NK} - 1) v &= {LAST_WORD};
        unsigned char* at = stage + s * {NB} + 8 * k;
        #pragma unroll
        for (int j = 0; j < 8; ++j) if (8 * k + j < {NB}) at[j] = (unsigned char)(v >> (8 * j));
    }}
    __syncthreads();
    int total = ns * {NB};
    for (int i = threadIdx.x; i < total / 16; i += blockDim.x) ((uint4*)dst)[i] = ((const uint4*)stage)[i];
    for (int i = total / 16 * 16 + threadIdx.x; i < total; i += blockDim.x) dst[i] = stage[i];
}}

// The stream of this block. NOTE: hashed, so that a segment's stream
// ``mix64(stream ^ sg * GOLDEN)`` cannot equal another block's: with ``seed ^ bid * C``
// alone, two (block, segment) pairs with equal XORs would share their draws.
__device__ __forceinline__ u64 block_stream(u64 seed) {{
    return mix64(seed ^ ((u64)blockIdx.x * 0xD1B54A32D192ED03ULL));
}}

// One segment's faults. With ``FINE``, for segments of fewer than 2^-12 expected
// faults, a draw within 2^-12 of u = 1 takes 24 more bits, which ``unit`` alone
// cannot resolve: it would fire such a segment at least 2^-24 of the time.
template <int WS, int SS, bool FINE>
__device__ __forceinline__ void add_segment_faults(u64* sw, uint2* events, int* num_events,
    int max_events, int ns, u64 stream, u64 state, float lk, int first, int outcomes, int idx,
    int end, const Record* __restrict__ records, const Flip* __restrict__ overflow)
{{
    while (true) {{
        state += GOLDEN;
        u64 r = mix64(state);
        float skip;
        if (!FINE) skip = lk < -3.0e38f ? 0.f : floorf(__logf(unit(r)) / lk);
        else if ((~r >> 40) >= 4096) skip = floorf(__logf(unit(r)) / lk);
        else {{
            // NOTE: 1 - u = (j + w) 2^-24, with w from a fresh draw.
            float w = ((mix64(state ^ 0x5851F42D4C957F2DULL) >> 40) + 1) * 0x1p-24f;
            skip = floorf(log1pf(-((float)(~r >> 40) + w) * 0x1p-24f) / lk);
        }}
        if (skip >= (float)(end - 1 - idx)) break;
        idx += 1 + (int)skip;
        int ch = idx / {S}, row = idx - ch * {S};
        if (row >= ns) continue;  // the last block's missing shots
        // NOTE: the skip reads the high 24 bits of r, the outcome its low 40: a
        // slot, and the fraction of the slot below.
        u64 x = (r & 0xFFFFFFFFFFULL) * (unsigned)outcomes;
        unsigned slot = (unsigned)(x >> 40);
        unsigned rec = (unsigned)(first + ch * outcomes) + slot;
        unsigned draw = (unsigned)((x >> 16) & 0xFFFFFF) | (slot << 24);
        int e = atomicAdd(num_events, 1);
        if (e < max_events) events[e] = make_uint2((unsigned)row | (rec << 8), draw);
        else apply_fault<WS, SS>(sw, row, rec, draw, stream, records, overflow);
    }}
}}

template <int WS, int SS>
__device__ __noinline__ void add_fine_segment_faults(u64* sw, uint2* events, int* num_events,
    int max_events, int ns, u64 stream, u64 state, float lk, int first, int outcomes, int idx,
    int end, const Record* __restrict__ records, const Flip* __restrict__ overflow)
{{
    add_segment_faults<WS, SS, true>(sw, events, num_events, max_events, ns, stream, state, lk,
                                     first, outcomes, idx, end, records, overflow);
}}

// Phases 1 and 2: draw the segments' faults on the block's ``ns`` shots and XOR
// them into ``sw`` (laid out as in ``apply_fault``).
template <int WS, int SS>
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
        if ((float)(end - 1 - idx) * -lk < 0x1p-12f)
            add_fine_segment_faults<WS, SS>(sw, events, num_events, max_events, ns, stream, state,
                                            lk, first, outcomes, idx, end, records, overflow);
        else
            add_segment_faults<WS, SS, false>(sw, events, num_events, max_events, ns, stream,
                                              state, lk, first, outcomes, idx, end, records,
                                              overflow);
    }}
    __syncthreads();
    int ne = min(*num_events, max_events);
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {{
        uint2 v = events[e];
        apply_fault<WS, SS>(sw, v.x & 255, v.x >> 8, v.y, stream, records, overflow);
    }}
    __syncthreads();
}}

// Screens the block's ``ns`` shots (words laid out as in ``apply_fault``): XORs
// the constants into their output words, marks the survivors in ``passed`` and
// in the bitmap ``survivors`` (if given), and appends them to ``queue`` after
// the ``count`` queued so far.
template <int WS, int SS>
__device__ __forceinline__ void screen_shots(u64* sw, int ns, long long base,
    const u64* __restrict__ constant, const u64* __restrict__ screen, bool* passed,
    unsigned* __restrict__ queue, unsigned* __restrict__ count, unsigned* __restrict__ survivors)
{{
    for (int s0 = 0; s0 < {S}; s0 += blockDim.x) {{
        int s = s0 + threadIdx.x;
        bool alive = false;
        if (s < ns) {{
            u64 hit = 0;
            #pragma unroll
            for (int i = 0; i < {OW}; ++i) {{
                u64 v = sw[i * WS + s * SS] ^ constant[i];
                hit |= v & screen[i];
                sw[i * WS + s * SS] = v;
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
}}
{NOISE}{KEPT}"""


def noise_tables(
    channels: ChannelSampler, flips: WordTables, num_words: int
) -> NoiseTables:
    """Return the noise kernels' tables, for CUDA's shared memory.

    Rows too wide for ``SHARED_BYTES`` keep their words in device memory
    (``shared_words`` False), where the noise kernel XORs the faults in.

    Raises:
        ValueError: If the noise exceeds the kernels' limits.

    """
    return _noise_tables(
        channels,
        flips,
        num_words,
        shared_bytes=SHARED_BYTES,
        shared_words=words_fit(num_words, SHARED_BYTES),
    )


def source(tables: NoiseTables, output_words: int, num_outputs: int) -> str:
    """Return the CUDA source of the ``noise`` and ``noise_kept`` kernels, and ``pack_rows``.

    ``noise_kept`` packs rows of ``num_outputs`` outputs. Without
    ``tables.shared_words`` there is no ``noise_kept``: the rows of a block do
    not fit in shared memory.
    """
    packed_bytes = -(-num_outputs // 8)
    if tables.shared_words:
        noise, kept = NOISE_SHARED, NOISE_KEPT
    else:
        noise, kept = NOISE_DEVICE, ""
    return SOURCE.replace("{NOISE}{KEPT}", noise + kept).format(
        NT=THREADS,
        NW=tables.num_words,
        OW=output_words,
        S=tables.block_shots,
        NB=max(packed_bytes, 1),
        NK=max(-(-packed_bytes // 8), 1),
        LAST_WORD=f"0x{(1 << (num_outputs % 64 or 64)) - 1:016X}ULL",
    )
