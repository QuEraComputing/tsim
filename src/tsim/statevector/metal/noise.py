"""Noise sampling into shot words on a Metal GPU.

The kernels read the tables of ``tsim.statevector.gpu.noise`` (see there): a
threadgroup owns ``S`` consecutive shots, walks segments of the channel groups
with geometric skips, records the faults as events in threadgroup memory and
then XORs their outcomes' flips into the shots' words. Metal has no 64-bit
atomics, so a flip XORs the two halves of a word with 32-bit atomics, and a
threadgroup has 32 KB of threadgroup memory. Rows too wide for it keep their
words in device memory, where the faults are XORed in with device atomics.

A block then screens its shots (constants, postselected direct detectors),
writes the survivors' rows, appends them to a queue for the tape kernel and
marks them in a bitmap of the batch's shots. Under postselection,
``noise_kept`` draws the noise that cannot change whether a shot is kept
(``gpu.noise.deferred_channels``) on the kept shots only, after the tapes.
"""

from __future__ import annotations

from tsim.noise.channels import ChannelSampler, WordTables
from tsim.statevector.gpu.noise import (
    NoiseTables,
    noise_tables as _noise_tables,
    words_fit,
)

SHARED_BYTES = 14 * 1024
"""Threadgroup memory of a block's words and events.

A threadgroup may have 32 KB, but the kernel is latency bound: with 14 KB, a
GPU core keeps more blocks resident, and the noise of end-to-end cultivation
d=5 takes 7.3 instead of 11.5 ns per shot on an M4 Pro (most other circuits
gain 10-25%, distillation loses up to 15%).
"""

_SEGMENT_PARAMS = """\
    device const int* seg_first [[buffer(2)]], device const int* seg_outcomes [[buffer(3)]],
    device const int* seg_begin [[buffer(4)]], device const int* seg_end [[buffer(5)]],
    device const float* seg_log_keep [[buffer(6)]], device const Record* records [[buffer(7)]],
    device const Flip* overflow [[buffer(8)]],
"""

# NOTE: the signature of both ``noise`` kernels: with the block's words in
# threadgroup memory or in device memory.
_NOISE_SIGNATURE = (
    """\
kernel void noise(
    device u64* out [[buffer(0)]], constant NoiseArgs& args [[buffer(1)]],
"""
    + _SEGMENT_PARAMS
    + """\
    device const u64* constants [[buffer(9)]], device const u64* screen [[buffer(10)]],
    device uint* queue [[buffer(11)]], device atomic_uint* count [[buffer(12)]],
    device uint* survivors [[buffer(13)]],
    threadgroup uint2* events [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]], uint bid [[threadgroup_position_in_grid]],
    uint nt [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]])
"""
)

NOISE_SHARED = _NOISE_SIGNATURE + r"""{{
    threadgroup u64 sw[{NW} * {S}];
    threadgroup atomic_int num_events;
    threadgroup bool passed[{S}];
    long base = (long)bid * {S};
    int ns = (int)min((long){S}, args.shots - base);
    for (uint i = tid; i < {NW} * {S}; i += nt) sw[i] = 0;
    if (tid == 0) atomic_store_explicit(&num_events, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Segments seg = {{seg_first, seg_outcomes, seg_begin, seg_end, seg_log_keep}};
    add_faults<threadgroup u64*, {S}, 1, false>(sw, events, &num_events, args.max_events, ns, block_stream(args.seed, bid),
               args.num_segments, seg, records, overflow, tid, nt);
    // NOTE: the screen reads the words in parallel, a thread per word and shot (a
    // shot's threads only ever clear its flag); the constants go into the rows.
    for (uint s = tid; s < {S}; s += nt) passed[s] = (int)s < ns;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {OW} * {S}; i += nt) {{
        uint word = i / {S}, s = i - word * {S};
        if ((sw[i] ^ constants[word]) & screen[word]) passed[s] = false;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s0 = 0; s0 < {S}; s0 += nt) {{
        int s = s0 + tid;
        bool alive = s < {S} && passed[s];
        enqueue(alive, s, s0 + (tid & ~31u), base, ns, lane, args.has_survivors, queue, count, survivors);
    }}
    device u64* rows = out + base * {NW};
    for (int i = tid; i < ns * {NW}; i += nt) {{
        int s = i / {NW}, word = i - s * {NW};
        if (passed[s]) rows[i] = sw[word * {S} + s] ^ (word < {OW} ? constants[word] : 0);
    }}
}}

"""

NOISE_DEVICE = _NOISE_SIGNATURE + r"""{{
    threadgroup atomic_int num_events;
    threadgroup bool passed[{S}];
    long base = (long)bid * {S};
    int ns = (int)min((long){S}, args.shots - base);
    device u64* rows = out + base * {NW};
    for (int i = tid; i < ns * {NW}; i += nt) rows[i] = 0;
    if (tid == 0) atomic_store_explicit(&num_events, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    Segments seg = {{seg_first, seg_outcomes, seg_begin, seg_end, seg_log_keep}};
    add_faults<device u64*, 1, {NW}, true>(rows, events, &num_events, args.max_events, ns,
               block_stream(args.seed, bid), args.num_segments, seg, records,
               overflow, tid, nt);
    // NOTE: as in the kernel above, the screen reads the words in parallel.
    for (uint s = tid; s < {S}; s += nt) passed[s] = (int)s < ns;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < ns * {OW}; i += nt) {{
        int s = i / {OW}, word = i - s * {OW};
        device u64* at = rows + (long)s * {NW} + word;
        u64 v = *at ^ constants[word];
        *at = v;
        if (v & screen[word]) passed[s] = false;
    }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (int s0 = 0; s0 < {S}; s0 += nt) {{
        int s = s0 + tid;
        bool alive = s < {S} && passed[s];
        enqueue(alive, s, s0 + (tid & ~31u), base, ns, lane, args.has_survivors, queue, count, survivors);
    }}
}}

"""

NOISE_KEPT = (
    r"""
struct KeptArgs {{ u64 seed; int num_segments; int max_events; }};

// The deferred channels' faults on the kept shots ``index`` (``*n`` of them),
// then their returned outputs bit-packed (``pack_rows``).
kernel void noise_kept(
    device const u64* out [[buffer(0)]], constant KeptArgs& args [[buffer(1)]],
"""
    + _SEGMENT_PARAMS
    + r"""    device const uint* n [[buffer(9)]], device const uint* index [[buffer(10)]],
    device uchar* dst [[buffer(11)]], device const uint* cursor [[buffer(12)]],
    threadgroup uint2* events [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]], uint bid [[threadgroup_position_in_grid]],
    uint nt [[threads_per_threadgroup]])
{{
    threadgroup u64 sw[{OW} * {S}];
    threadgroup atomic_int num_events;
    long base = (long)bid * {S};
    if (base >= (long)*n) return;
    // NOTE: rows past the call's capacity (cursor[1]) are not written.
    int ns = (int)min(min((long){S}, (long)*n - base), (long)cursor[1] - (long)cursor[0] - base);
    if (ns <= 0) return;
    for (int i = tid; i < ns * {OW}; i += nt) {{
        int s = i / {OW}, word = i - s * {OW};
        sw[word * {S} + s] = out[(long)index[base + s] * {NW} + word];
    }}
    if (tid == 0) atomic_store_explicit(&num_events, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Segments seg = {{seg_first, seg_outcomes, seg_begin, seg_end, seg_log_keep}};
    add_faults<threadgroup u64*, {S}, 1, false>(sw, events, &num_events, args.max_events, ns,
               block_stream(args.seed, bid), args.num_segments, seg, records,
               overflow, tid, nt);
    pack_rows(sw, ns, (threadgroup uchar*)events, dst, ((long)*cursor + base) * {NB}, tid, nt);
}}
"""
)

SOURCE = r"""
struct Record {{ uint alias; uint words; uint extra; uint keep; u64 m0; u64 m1; }};
struct Flip {{ uint word; uint pad; u64 mask; }};

inline void xor64(threadgroup u64* sw, int index, u64 mask) {{
    threadgroup atomic_uint* halves = (threadgroup atomic_uint*)(sw + index);
    uint lo = uint(mask), hi = uint(mask >> 32);
    if (lo) atomic_fetch_xor_explicit(halves, lo, memory_order_relaxed);
    if (hi) atomic_fetch_xor_explicit(halves + 1, hi, memory_order_relaxed);
}}

inline void xor64(device u64* sw, int index, u64 mask) {{
    device atomic_uint* halves = (device atomic_uint*)(sw + index);
    uint lo = uint(mask), hi = uint(mask >> 32);
    if (lo) atomic_fetch_xor_explicit(halves, lo, memory_order_relaxed);
    if (hi) atomic_fetch_xor_explicit(halves + 1, hi, memory_order_relaxed);
}}

// The fault drawn at record ``rec`` (bits 0-23 of ``draw`` the high 24 bits of
// its 40-bit fraction, bits 24-31 its slot in the channel): the slot's outcome
// if the fraction is below the keep threshold (gpu/noise.py), else its alias's.
// The fraction's low 16 bits matter only when its high bits tie with the
// threshold's, and then come from a hash of the fault. Word ``w`` of row ``s``
// is ``sw[w * WS + s * SS]``.
template <typename W, int WS, int SS>
inline void apply_fault(W sw, int row, uint rec, uint draw, u64 stream,
    device const Record* records, device const Flip* overflow)
{{
    Record r = records[rec];
    uint high = draw & 0xFFFFFF, keep = r.alias & 0x1FFFFFF;
    if (high > keep || (high == keep &&
            (uint)(mix64(stream ^ ((u64)(rec << 8 | row) * GOLDEN)) & 0xFFFF) >= r.keep))
        r = records[rec - (draw >> 24) + (r.alias >> 25)];
    int flips = r.extra & 255;
    if (flips > 0) xor64(sw, (r.words & 0xFFFF) * WS + row * SS, r.m0);
    if (flips > 1) xor64(sw, (r.words >> 16) * WS + row * SS, r.m1);
    for (int j = 2; j < flips; ++j) {{
        Flip f = overflow[(r.extra >> 8) + j - 2];
        xor64(sw, f.word * WS + row * SS, f.mask);
    }}
}}

// The outputs of the block's ``ns`` rows ``sw[w * S + s]``, bit-packed into
// {NB} bytes per row (the bits past them cleared), contiguously at ``dst``.
inline void pack_rows(threadgroup const u64* sw, int ns, threadgroup uchar* stage,
    device uchar* dst, long at, uint tid, uint nt)
{{
    for (uint i = tid; i < {NK} * {S}; i += nt) {{
        int k = i / {S}, s = i - k * {S};
        if (s >= ns) continue;
        u64 v = sw[k * {S} + s];
        if (k == {NK} - 1) v &= {LAST_WORD};
        threadgroup uchar* at = stage + s * {NB} + 8 * k;
        for (int j = 0; j < 8; ++j) if (8 * k + j < {NB}) at[j] = uchar(v >> (8 * j));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // NOTE: the rows go to byte ``at`` of the (aligned) buffer ``dst``: bytes up
    // to a 16-byte boundary, then 16 at a time, then the rest.
    int total = ns * {NB};
    int head = min(total, (int)((16 - (at & 15)) & 15));
    device uchar* d = dst + at;
    for (int i = tid; i < head; i += nt) d[i] = stage[i];
    int body = (total - head) / 16;
    device uint4* aligned = (device uint4*)(d + head);
    if ((head & 3) == 0) {{
        threadgroup const uint* from = (threadgroup const uint*)(stage + head);
        for (int i = tid; i < body; i += nt)
            aligned[i] = uint4(from[4 * i], from[4 * i + 1], from[4 * i + 2], from[4 * i + 3]);
    }} else {{
        for (int i = tid; i < body; i += nt) {{
            threadgroup const uchar* b = stage + head + 16 * i;
            uint4 v;
            for (int k = 0; k < 4; ++k)
                v[k] = uint(b[4 * k]) | uint(b[4 * k + 1]) << 8 | uint(b[4 * k + 2]) << 16 | uint(b[4 * k + 3]) << 24;
            aligned[i] = v;
        }}
    }}
    for (int i = head + body * 16 + tid; i < total; i += nt) d[i] = stage[i];
}}

// The stream of block ``bid``. NOTE: hashed, so that a segment's stream
// ``mix64(stream ^ sg * GOLDEN)`` cannot equal another block's: with ``seed ^ bid * C``
// alone, two (block, segment) pairs with equal XORs would share their draws.
inline u64 block_stream(u64 seed, uint bid) {{
    return mix64(seed ^ ((u64)bid * 0xD1B54A32D192ED03UL));
}}

struct Segments {{
    device const int* first; device const int* outcomes; device const int* begin;
    device const int* end; device const float* log_keep;
}};

// One segment's faults. With ``FINE``, for segments of fewer than 2^-12 expected
// faults, a draw within 2^-12 of u = 1 takes 24 more bits, which ``unit`` alone
// cannot resolve: it would fire such a segment at least 2^-24 of the time.
template <typename W, int WS, int SS, bool FINE>
inline void add_segment_faults(W sw, threadgroup uint2* events, threadgroup atomic_int* num_events,
    int max_events, int ns, u64 stream, u64 state, float lk, int first, int outcomes, int idx,
    int end, device const Record* records, device const Flip* overflow)
{{
    while (true) {{
        state += GOLDEN;
        u64 r = mix64(state);
        float skip;
        if (!FINE) skip = lk < -3.0e38f ? 0.f : floor(fast::log(unit(r)) / lk);
        else if ((~r >> 40) >= 4096) skip = floor(fast::log(unit(r)) / lk);
        else {{
            // NOTE: 1 - u = y = (j + w) 2^-24 <= 2^-12, with w from a fresh draw;
            // Metal has no log1p, and the series to y^3 is exact to ~1e-11 here.
            float w = float((mix64(state ^ 0x5851F42D4C957F2DUL) >> 40) + 1) * 0x1p-24f;
            float y = (float(~r >> 40) + w) * 0x1p-24f;
            skip = floor(-y * (1.f + y * (0.5f + y * (1.f / 3.f))) / lk);
        }}
        if (skip >= (float)(end - 1 - idx)) break;
        idx += 1 + (int)skip;
        int ch = idx / {S}, row = idx - ch * {S};
        if (row >= ns) continue;  // the last block's missing shots
        // NOTE: the skip reads the high 24 bits of r, the outcome its low 40: a
        // slot, and the fraction of the slot below.
        u64 x = (r & 0xFFFFFFFFFFUL) * (uint)outcomes;
        uint slot = (uint)(x >> 40);
        uint rec = (uint)(first + ch * outcomes) + slot;
        uint draw = (uint)((x >> 16) & 0xFFFFFF) | (slot << 24);
        int e = atomic_fetch_add_explicit(num_events, 1, memory_order_relaxed);
        if (e < max_events) events[e] = uint2((uint)row | (rec << 8), draw);
        else apply_fault<W, WS, SS>(sw, row, rec, draw, stream, records, overflow);
    }}
}}

template <typename W, int WS, int SS>
__attribute__((noinline)) void add_fine_segment_faults(W sw, threadgroup uint2* events,
    threadgroup atomic_int* num_events, int max_events, int ns, u64 stream, u64 state, float lk,
    int first, int outcomes, int idx, int end, device const Record* records,
    device const Flip* overflow)
{{
    add_segment_faults<W, WS, SS, true>(sw, events, num_events, max_events, ns, stream, state, lk,
                                        first, outcomes, idx, end, records, overflow);
}}

// Phases 1 and 2: draw the segments' faults on the block's ``ns`` shots and XOR
// them into ``sw`` (laid out as in ``apply_fault``; in device memory if ``DEV``).
template <typename W, int WS, int SS, bool DEV>
inline void add_faults(W sw, threadgroup uint2* events, threadgroup atomic_int* num_events,
    int max_events, int ns, u64 stream, int num_segments, Segments seg,
    device const Record* records, device const Flip* overflow, uint tid, uint nt)
{{
    for (int sg = tid; sg < num_segments; sg += nt) {{
        float lk = seg.log_keep[sg];
        int first = seg.first[sg], outcomes = seg.outcomes[sg], end = seg.end[sg];
        int idx = seg.begin[sg] - 1;
        u64 state = mix64(stream ^ ((u64)sg * GOLDEN));
        if ((float)(end - 1 - idx) * -lk < 0x1p-12f)
            add_fine_segment_faults<W, WS, SS>(sw, events, num_events, max_events, ns, stream,
                                               state, lk, first, outcomes, idx, end, records,
                                               overflow);
        else
            add_segment_faults<W, WS, SS, false>(sw, events, num_events, max_events, ns, stream,
                                                 state, lk, first, outcomes, idx, end, records,
                                                 overflow);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int ne = min(atomic_load_explicit(num_events, memory_order_relaxed), max_events);
    for (int e = tid; e < ne; e += nt) {{
        uint2 v = events[e];
        apply_fault<W, WS, SS>(sw, v.x & 255, v.x >> 8, v.y, stream, records, overflow);
    }}
    if (DEV) threadgroup_barrier(mem_flags::mem_device);
    else threadgroup_barrier(mem_flags::mem_threadgroup);
}}

struct NoiseArgs {{ long shots; u64 seed; int num_segments; int max_events; int has_survivors; }};

// Marks a SIMD group's surviving shots (of the block's ``ns``, from ``first``)
// in the bitmap ``survivors`` if ``has_survivors``, and appends them to
// ``queue`` after the ``count`` queued so far.
inline void enqueue(bool alive, int s, int first, long base, int ns, uint lane,
    int has_survivors, device uint* queue, device atomic_uint* count, device uint* survivors)
{{
    uint ballot = uint(static_cast<simd_vote::vote_t>(simd_ballot(alive)));
    if (has_survivors && lane == 0 && first < ns) survivors[(base + first) >> 5] = ballot;
    if (ballot) {{
        uint leader = ctz(ballot), b = 0;
        if (lane == leader) b = atomic_fetch_add_explicit(count, popcount(ballot), memory_order_relaxed);
        b = simd_broadcast(b, (ushort)leader);
        if (alive) queue[b + popcount(ballot & ((1u << lane) - 1))] = (uint)(base + s);
    }}
}}

{NOISE}{KEPT}"""


def noise_tables(
    channels: ChannelSampler, flips: WordTables, num_words: int
) -> NoiseTables:
    """Return the noise kernels' tables, for Metal's threadgroup memory.

    Rows too wide for ``SHARED_BYTES`` keep their words in device memory
    (``shared_words`` False), where the noise kernel XORs the faults in.
    """
    return _noise_tables(
        channels,
        flips,
        num_words,
        shared_bytes=SHARED_BYTES,
        shared_words=words_fit(num_words, SHARED_BYTES),
    )


def source(tables: NoiseTables, output_words: int, num_outputs: int) -> str:
    """Return the MSL source of the ``noise`` and ``noise_kept`` kernels.

    ``noise_kept`` packs rows of ``num_outputs`` outputs. Without
    ``tables.shared_words`` there is no ``noise_kept``: the rows of a block do
    not fit in threadgroup memory.
    """
    packed_bytes = -(-num_outputs // 8)
    if tables.shared_words:
        noise, kept = NOISE_SHARED, NOISE_KEPT
    else:
        noise, kept = NOISE_DEVICE, ""
    return SOURCE.replace("{NOISE}{KEPT}", noise + kept).format(
        NW=tables.num_words,
        OW=output_words,
        S=tables.block_shots,
        NB=max(packed_bytes, 1),
        NK=max(-(-packed_bytes // 8), 1),
        LAST_WORD=f"0x{(1 << (num_outputs % 64 or 64)) - 1:016X}UL",
    )
