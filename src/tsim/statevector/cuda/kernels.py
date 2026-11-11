"""The CUDA sampler's kernels other than the noise ones, and their source.

``tape`` runs the components' tapes on the queued shots; ``count_bits`` and
``list_bits`` list the accepted shots from the bitmap; ``gather_columns`` and
``gather_float`` gather rows and expectations for the copy to the host;
``publish`` stores a batch's counts into pinned host memory.
"""

from __future__ import annotations

from tsim.statevector.cuda.codegen import PRELUDE
from tsim.statevector.cuda.noise import source as noise_source
from tsim.statevector.gpu.noise import NoiseTables
from tsim.statevector.gpu.tape import TapeKernel
from tsim.statevector.shots import ShotLayout

TAPE_THREADS = 128
"""Threads per block of the tape kernel."""

COPY_THREADS = 256
"""Threads per block of the kernels that gather rows and expectations."""

SCAN_THREADS = 1024
"""Threads per block of ``count_bits`` and ``list_bits``: one per bitmap word."""

TEMPLATE = r"""
extern "C" __global__ void __launch_bounds__({block}) tape(
    u64* __restrict__ out, long long first_shot, u64 seed,
    const unsigned* __restrict__ queue, const unsigned* __restrict__ count,
    unsigned* __restrict__ survivors, unsigned* __restrict__ acc_count,
    float* __restrict__ expectations, unsigned* __restrict__ deviation)
{{
    int t = threadIdx.x & {T1};
    unsigned gmask = {gmask};
{seeds}
    long long groups = ((long long)gridDim.x * blockDim.x) >> {tb};
    long long n = *count;
    // NOTE: whole warps leave the loop together, so the shuffles keep their groups.
    long long rounds = (n + groups - 1) / groups;
    long long g0 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> {tb};
    for (long long r = 0; r < rounds; ++r) {{
        long long q = g0 + r * groups;
        bool valid = q < n;
        u64 row_index = queue[valid ? q : n - 1];
        u64 shot = first_shot + row_index;
        u64* row = out + row_index * {NW};
        // NOTE: the words that no component reads are dead loads, dropped.
        u64 w[{NW}];
        #pragma unroll
        for (int i = 0; i < {NW}; ++i) w[i] = row[i];
        float ev[{NO}];
        int grave = 0x7fffffff;
{calls}
        if (valid && t == 0) {{
{stores}
            if (expectations != nullptr) {{
                #pragma unroll
                for (int i = 0; i < {NO}; ++i) expectations[row_index * {NO} + i] = ev[i];
            }}
        }}
        if (survivors != nullptr) {{
            // NOTE: the postselected direct outputs passed the screen already;
            // the shots that fail now leave the screen's bitmap.
            u64 hit = {hits};
            if (valid && t == 0 && hit) atomicAnd(&survivors[row_index >> 5], ~(1u << (row_index & 31)));
            unsigned ballot = __ballot_sync(0xffffffffu, valid && t == 0 && hit == 0);
            if (ballot && (threadIdx.x & 31) == __ffs(ballot) - 1) atomicAdd(acc_count, __popc(ballot));
        }}
    }}
}}

// The shots of the ``words`` words of ``bits`` set, in order, as indices into
// ``index``: ``count_bits`` counts each block's {SCAN} words, ``list_bits`` lists
// a block's shots after those of the blocks before it.
__device__ __forceinline__ unsigned block_scan(unsigned v, unsigned* total) {{
    __shared__ unsigned warps[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {{ unsigned y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }}
    if (lane == 31) warps[warp] = x;
    __syncthreads();
    if (warp == 0) {{
        unsigned w = lane < (blockDim.x >> 5) ? warps[lane] : 0;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {{ unsigned y = __shfl_up_sync(0xffffffffu, w, o); if (lane >= o) w += y; }}
        warps[lane] = w;
    }}
    __syncthreads();
    unsigned before = (warp ? warps[warp - 1] : 0) + x - v;
    *total = warps[31];
    __syncthreads();
    return before;
}}

extern "C" __global__ void __launch_bounds__({SCAN}) count_bits(
    const unsigned* __restrict__ bits, long long words, unsigned* __restrict__ totals) {{
    long long i = (long long)blockIdx.x * {SCAN} + threadIdx.x;
    unsigned total;
    block_scan(i < words ? __popc(bits[i]) : 0, &total);
    if (threadIdx.x == 0) totals[blockIdx.x] = total;
}}

extern "C" __global__ void __launch_bounds__({SCAN}) list_bits(
    const unsigned* __restrict__ bits, long long words, const unsigned* __restrict__ totals,
    unsigned* __restrict__ index) {{
    unsigned earlier = 0, total;
    for (int b = threadIdx.x; b < blockIdx.x; b += blockDim.x) earlier += totals[b];
    block_scan(earlier, &earlier);
    long long i = (long long)blockIdx.x * {SCAN} + threadIdx.x;
    unsigned w = i < words ? bits[i] : 0;
    unsigned at = earlier + block_scan(__popc(w), &total);
    for (; w; w &= w - 1) index[at++] = (unsigned)(32 * i + __ffs(w) - 1);
}}

// The outputs of rows ``index[j]`` (or every row), bit-packed densely
// (``pack_rows``): a block packs {S} rows' output words.
extern "C" __global__ void __launch_bounds__({COPY}) gather_columns(
    const u64* __restrict__ rows, long long n, const unsigned* __restrict__ index,
    unsigned char* __restrict__ dst) {{
    __shared__ u64 sw[{OW} * {S}];
    extern __shared__ __align__(16) unsigned char stage[];
    long long base = (long long)blockIdx.x * {S};
    int ns = (int)min((long long){S}, n - base);
    for (int i = threadIdx.x; i < ns * {OW}; i += blockDim.x) {{
        int s = i / {OW}, w = i - s * {OW};
        long long r = index != nullptr ? (long long)index[base + s] : base + s;
        sw[w * {S} + s] = rows[r * {NW} + w];
    }}
    __syncthreads();
    pack_rows(sw, ns, stage, dst + base * {NB});
}}

// The batch's counts, stored into pinned host memory: a read that does not
// queue behind the copies to the host.
extern "C" __global__ void publish(const unsigned* __restrict__ counts, volatile unsigned* host) {{
    if (threadIdx.x < 3) host[threadIdx.x] = counts[threadIdx.x];
    __threadfence_system();
}}

// Observable expectations of rows ``index[j]``, densely.
extern "C" __global__ void gather_float(const float* __restrict__ values, long long n, int width,
                                        const unsigned* __restrict__ index, float* __restrict__ dst) {{
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n * width;
         k += (long long)gridDim.x * blockDim.x) {{
        long long j = k / width; int c = (int)(k - j * width);
        dst[k] = values[(long long)index[j] * width + c];
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
    """Return the CUDA source of every kernel, for tapes of ``2^tb`` threads a shot."""
    T = 1 << tb
    # NOTE: a shot's lanes within its warp.
    gmask = (
        "0xffffffffu"
        if T == 32
        else f"((1u << {T}) - 1) << (threadIdx.x & 31 & ~{T - 1})"
    )
    return (
        PRELUDE
        + noise_source(tables, layout.output_words, layout.num_outputs)
        + parts.functions
        + TEMPLATE.format(
            block=TAPE_THREADS,
            SCAN=SCAN_THREADS,
            COPY=COPY_THREADS,
            T1=T - 1,
            tb=tb,
            gmask=gmask,
            NW=layout.num_words,
            OW=layout.output_words,
            NO=num_observables,
            S=tables.block_shots,
            # NOTE: the rows keep every output, the postselected ones too.
            NB=max(-(-layout.num_outputs // 8), 1),
            calls=parts.calls,
            seeds=parts.seeds,
            stores=parts.stores,
            hits=parts.hits,
        )
    )
