"""The Metal sampler's kernels other than the noise ones, and their source.

``tape`` runs the components' tapes on the queued shots; ``count_bits`` and
``list_bits`` list the accepted shots from the bitmap; ``gather_rows``,
``unpack_rows``, ``select_rows`` and ``gather_float`` copy out rows, booleans,
chosen outputs and expectations;
``reset`` and ``advance`` keep a call's counters on the GPU.
"""

from __future__ import annotations

from tsim.statevector.gpu.noise import NoiseTables
from tsim.statevector.gpu.tape import TapeKernel
from tsim.statevector.metal.codegen import PRELUDE
from tsim.statevector.metal.noise import source as noise_source
from tsim.statevector.shots import ShotLayout

SCAN_THREADS = 1024
"""Threads per threadgroup of ``count_bits`` and ``list_bits``: one per bitmap word."""

TEMPLATE = r"""
struct TapeArgs {{ long first_shot; u64 seed; int flags; }};

// flags: 1 clears rejected shots from ``survivors`` and counts the accepted ones,
// 2 writes the expectations.
kernel void tape(
    device u64* out [[buffer(0)]], constant TapeArgs& args [[buffer(1)]],
    device const uint* queue [[buffer(2)]], device const uint* count [[buffer(3)]],
    device atomic_uint* survivors [[buffer(4)]], device atomic_uint* acc_count [[buffer(5)]],
    device float* expectations [[buffer(6)]], device atomic_uint* deviation [[buffer(7)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{{
    int t = lane & {T1};
    uint q = gid >> {tb};
    if (q >= count[0]) return;
    const u64 seed = args.seed;
{seeds}
    uint row_index = queue[q];
    u64 shot = (u64)args.first_shot + row_index;
    device u64* row = out + (long)row_index * {NW};
    u64 w[{NW}];
    for (int i = 0; i < {NW}; ++i) w[i] = row[i];
    float ev[{NO}];
    int grave = 0x7fffffff;
{calls}
    if (t == 0) {{
{stores}
        if (args.flags & 2)
            for (int i = 0; i < {NO}; ++i) expectations[(long)row_index * {NO} + i] = ev[i];
    }}
    if (args.flags & 1) {{
        u64 hit = {hits};
        if (t == 0 && hit)
            atomic_fetch_and_explicit(&survivors[row_index >> 5], ~(1u << (row_index & 31)), memory_order_relaxed);
        uint ballot = uint(static_cast<simd_vote::vote_t>(simd_ballot(t == 0 && hit == 0)));
        if (ballot && simd_is_first()) atomic_fetch_add_explicit(acc_count, popcount(ballot), memory_order_relaxed);
    }}
}}

// The counts of a batch: its screened survivors and its accepted shots.
kernel void reset(device uint* counts [[buffer(0)]]) {{ counts[0] = 0; counts[1] = 0; }}

// Moves the cursor past a batch's ``*n`` rows.
kernel void advance(device uint* cursor [[buffer(0)]], device const uint* n [[buffer(1)]]) {{
    cursor[0] += *n;
}}

inline uint block_scan(uint v, thread uint& total, threadgroup uint* warps, uint lane, uint warp, uint nwarps) {{
    uint x = simd_prefix_inclusive_sum(v);
    if (lane == 31) warps[warp] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {{
        uint y = lane < nwarps ? warps[lane] : 0;
        warps[lane] = simd_prefix_inclusive_sum(y);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = (warp ? warps[warp - 1] : 0) + x - v;
    total = warps[nwarps - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return before;
}}

// The shots of the set bits of ``bits`` (``words`` words), in order: ``count_bits``
// counts each block's {SCAN} words, ``list_bits`` lists a block's shots after
// those of the blocks before it.
kernel void count_bits(
    device const uint* bits [[buffer(0)]], constant long& words [[buffer(1)]],
    device uint* totals [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]], uint bid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]],
    uint nwarps [[simdgroups_per_threadgroup]])
{{
    threadgroup uint warps[32];
    long i = (long)bid * {SCAN} + tid;
    uint total;
    block_scan(i < words ? popcount(bits[i]) : 0, total, warps, lane, warp, nwarps);
    if (tid == 0) totals[bid] = total;
}}

kernel void list_bits(
    device const uint* bits [[buffer(0)]], constant long& words [[buffer(1)]],
    device const uint* totals [[buffer(2)]], device uint* index [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]], uint bid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]],
    uint nwarps [[simdgroups_per_threadgroup]])
{{
    threadgroup uint warps[32];
    uint earlier = 0, total;
    for (uint b = tid; b < bid; b += {SCAN}) earlier += totals[b];
    block_scan(earlier, earlier, warps, lane, warp, nwarps);
    long i = (long)bid * {SCAN} + tid;
    uint w = i < words ? bits[i] : 0;
    uint at = earlier + block_scan(popcount(w), total, warps, lane, warp, nwarps);
    for (; w; w &= w - 1) index[at++] = (uint)(32 * i + ctz(w));
}}

// The rows ``index[j]`` (or every row, without ``has_index``) for ``j < *n``,
// densely after the ``*cursor`` rows of the call's earlier batches: each row's
// first {NB} bytes, which hold its outputs bit-packed (the bits past them
// cleared). A thread writes 4 aligned bytes at a time.
kernel void gather_rows(
    device const uchar* rows [[buffer(0)]], device const uint* n [[buffer(1)]],
    device const uint* index [[buffer(2)]], constant int& has_index [[buffer(3)]],
    device uchar* dst [[buffer(4)]], device const uint* cursor [[buffer(5)]],
    uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]])
{{
    long count = min((long)*n, (long)cursor[1] - (long)cursor[0]);
    if (count <= 0) return;
    long at = (long)cursor[0] * {NB}, end = at + count * {NB};
    for (long p = (at & ~3L) + 4L * gid; p < end; p += 4L * grid) {{
        // NOTE: a batch's rows span less than 4 GB (``BATCH_BYTES``).
        uint k = (uint)(max(p, at) - at);
        uint r = k / {NB}, b = k - r * {NB};
        uint sh = 8 * (b & 3);
        if (p >= at && p + 4 <= end && b + 4 <= {NB} && (sh == 0 || (b & ~3u) + 8 <= {RB})) {{
            // NOTE: the 4 bytes lie in one row: two aligned loads at most.
            long source = has_index ? (long)index[r] : r;
            device const uint* w = (device const uint*)(rows + source * {RB}) + (b >> 2);
            uint v = sh ? (w[0] >> sh) | (w[1] << (32 - sh)) : w[0];
            if (b + 4 == {NB}) v &= ({LAST}u << 24) | 0xFFFFFFu;
            *(device uint*)(dst + p) = v;
            continue;
        }}
        uint v = 0;
        for (int j = (int)(max(p, at) - p); j < 4 && p + j < end; ++j) {{
            long source = has_index ? (long)index[r] : r;
            uint x = rows[source * {RB} + b];
            if (b == {NB} - 1) x &= {LAST};
            v |= x << (8 * j);
            if (++b == {NB}) {{
                b = 0;
                ++r;
            }}
        }}
        for (int j = 0; j < 4; ++j)
            if (p + j >= at && p + j < end) dst[p + j] = uchar(v >> (8 * j));
    }}
}}

// The first ``n`` packed rows of ``{NB}`` bytes, one byte per output (``{NOUT}``
// per row), as booleans: a thread unpacks 4 outputs, which share a packed byte.
kernel void unpack_rows(
    device const uchar* packed [[buffer(0)]], device uchar* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]])
{{
    const uint groups = ({NOUT} + 3) / 4;
    for (uint i = gid; i < n * groups; i += grid) {{
        uint r = i / groups, g = i - r * groups;
        uint b = packed[(ulong)r * {NB} + (g >> 1)] >> (4 * (g & 1));
        device uchar* out = dst + (ulong)r * {NOUT} + 4 * g;
        for (uint j = 0; j < 4 && 4 * g + j < {NOUT}; ++j) out[j] = uchar((b >> j) & 1);
    }}
}}

// Outputs ``columns[k]`` (``k < count``) of the first ``n`` packed rows of ``{NB}``
// bytes, into rows of ``width`` bytes (``args = (n, count, width, bools)``): one
// byte per output with ``bools``, else bit-packed with zero bits past them. A
// thread writes one byte; the host keeps ``n * width`` at most ``MAX_COPY_ITEMS``.
kernel void select_rows(
    device const uchar* packed [[buffer(0)]], device uchar* dst [[buffer(1)]],
    device const uint* columns [[buffer(2)]], constant uint4& args [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]])
{{
    const uint width = args.z, per = args.w ? 1 : 8;
    for (uint i = gid; i < args.x * width; i += grid) {{
        uint r = i / width, k = (i - r * width) * per;
        device const uchar* row = packed + (ulong)r * {NB};
        uint v = 0;
        for (uint b = 0; b < per && k + b < args.y; ++b) {{
            uint c = columns[k + b];
            v |= ((row[c >> 3] >> (c & 7)) & 1u) << b;
        }}
        dst[i] = uchar(v);
    }}
}}

// Observable expectations of rows ``index[j]`` (or of row ``j``, without
// ``has_index``) for ``j < *n``, densely, after the ``*cursor`` rows of the
// call's earlier batches.
kernel void gather_float(
    device const float* values [[buffer(0)]], device const uint* n [[buffer(1)]],
    constant int& width [[buffer(2)]], device const uint* index [[buffer(3)]],
    device float* dst [[buffer(4)]], device const uint* cursor [[buffer(5)]],
    constant int& has_index [[buffer(6)]],
    uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]])
{{
    long total = min((long)*n, (long)cursor[1] - (long)cursor[0]) * width;
    dst += (long)*cursor * width;
    for (long k = gid; k < total; k += grid) {{
        long j = k / width; int c = (int)(k - j * width);
        long r = has_index ? (long)index[j] : j;
        dst[k] = values[r * width + c];
    }}
}}
"""


def source(
    layout: ShotLayout,
    tables: NoiseTables,
    parts: TapeKernel,
    tb: int,
    num_observables: int,
) -> str:
    """Return the MSL source of every kernel, for tapes of ``2^tb`` threads a shot."""
    nb = -(-layout.num_outputs // 8)
    return (
        PRELUDE
        + noise_source(tables, layout.output_words, layout.num_outputs)
        + parts.functions
        + TEMPLATE.format(
            T1=(1 << tb) - 1,
            tb=tb,
            NW=layout.num_words,
            SCAN=SCAN_THREADS,
            NO=num_observables,
            NB=max(nb, 1),
            RB=8 * layout.num_words,
            NOUT=max(layout.num_outputs, 1),
            LAST=hex((1 << (layout.num_outputs % 8 or 8)) - 1),
            calls=parts.calls,
            seeds=parts.seeds,
            stores=parts.stores,
            hits=parts.hits,
        )
    )


def names(shared_words: bool) -> list[str]:
    """Return the kernels of a source; ``noise_kept`` needs ``shared_words``."""
    found = ["noise", "tape", "count_bits", "list_bits", "gather_rows"]
    found += ["gather_float", "reset", "advance", "unpack_rows", "select_rows"]
    return [*found, "noise_kept"] if shared_words else found
