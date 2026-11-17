"""CUDA source of a component's tape (see ``tsim.statevector.gpu.tape``).

Partner amplitudes on other threads come from ``__shfl_xor_sync`` among the
shot's group of lanes (``gmask``), and a pruned shot jumps to the end. Near
the boundary ``draw_one`` decides an outcome in double precision.
"""

from __future__ import annotations

from tsim.statevector.gpu import tape

CUDA = tape.Dialect(
    inline="__device__ __forceinline__",
    noinline="__device__ __noinline__",
    private="",
    deviation="unsigned*",
    constant="__device__ const",
    float2="make_float2",
    rsqrt="rsqrtf",
    fmax="fmaxf",
    u64="ULL",
    shuffle2="shfl2({value}, {lanes}, gmask)",
    shuffle1="__shfl_xor_sync(gmask, {value}, {lanes})",
    lane_mask=True,
    goto=True,
)

PRELUDE = r"""
typedef unsigned long long u64;
#define GOLDEN 0x9E3779B97F4A7C15ULL

__device__ __forceinline__ u64 mix64(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// The CPU executor's draw: splitmix64 at index shot * slots + slot of stream seed.
__device__ __forceinline__ u64 draw(u64 seed, u64 shot, u64 slots, u64 slot) {
    return mix64(seed + (shot * slots + slot) * GOLDEN + GOLDEN);
}
__device__ __forceinline__ int par(u64 v) { return __popcll(v) & 1; }
// draw_one in double precision, from the rarer outcome's share, which float32
// keeps to full relative precision: a rare 0 takes the top of the range, as on
// the CPU (``shared._cut``). Not inlined, as it is rarely taken.
__device__ __noinline__ int draw_one_exact(u64 r, float w0, float w1, float total) {
    int rare0 = w0 < w1;
    u64 below = (u64)((double)(rare0 ? w0 : w1) / (double)total * 18446744073709551616.0);
    return ((rare0 ? ~r : r) < below) ^ rare0;
}
// Whether the 64-bit draw r gives outcome 1, of weight w1 against w0. float32
// decides it unless r / 2^64 lies within 2^-21 of the share w1 / total, which
// covers both roundings; there draw_one_exact does.
__device__ __forceinline__ int draw_one(u64 r, float w0, float w1, float total) {
    float p = w1 / total, x = __ull2float_rz(r) * 0x1p-64f;
    if (x < p * (1.f - 0x1p-21f)) return 1;
    if (x > p * (1.f + 0x1p-21f)) return 0;
    return draw_one_exact(r, w0, w1, total);
}
// A uniform in (0, 1] from the high 24 bits, so that logf never sees 0.
__device__ __forceinline__ float unit(u64 r) { return ((r >> 40) + 1) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float2 cmul(float2 p, float2 q) {
    return make_float2(p.x * q.x - p.y * q.y, p.x * q.y + p.y * q.x);
}
// c a + s q and c a + i s q
__device__ __forceinline__ float2 cmac(float2 a, float c, float s, float2 q) {
    return make_float2(fmaf(c, a.x, s * q.x), fmaf(c, a.y, s * q.y));
}
__device__ __forceinline__ float2 cmac_i(float2 a, float c, float s, float2 q) {
    return make_float2(fmaf(c, a.x, -s * q.y), fmaf(c, a.y, s * q.x));
}
__device__ __forceinline__ float2 shfl2(float2 v, int lane_xor, unsigned mask) {
    return make_float2(__shfl_xor_sync(mask, v.x, lane_xor), __shfl_xor_sync(mask, v.y, lane_xor));
}
// The largest normalization deviation, as the bits of a non-negative float.
__device__ __forceinline__ void deviate(unsigned* deviation, float total) {
    float d = fabsf(total - 1.f);
    if (!(d <= 1e-3f)) atomicMax(deviation, d == d ? __float_as_uint(d) : 0x7f800000u);
}
"""
