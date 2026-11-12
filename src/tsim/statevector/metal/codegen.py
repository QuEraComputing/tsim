"""Metal source of a component's tape (see ``tsim.statevector.gpu.tape``).

The Metal Shading Language has SIMD-group shuffles for warp shuffles, no
``goto`` (a pruned shot breaks out of a ``do``-block) and no double precision:
near the boundary ``draw_one`` refines the rarer outcome's share as a
float-float ``q + e / total`` instead. Unrolled tapes keep their amplitudes in
arrays of 32, which Metal's compiler keeps in registers, and project partner
amplitudes in pairs, which keeps fewer values live.
"""

from __future__ import annotations

from tsim.statevector.gpu import tape

METAL = tape.Dialect(
    inline="inline",
    noinline="__attribute__((noinline))",
    private="thread ",
    deviation="device atomic_uint*",
    constant="constant",
    float2="float2",
    rsqrt="rsqrt",
    fmax="fmax",
    u64="UL",
    shuffle2="simd_shuffle_xor({value}, (ushort){lanes})",
    shuffle1="simd_shuffle_xor({value}, (ushort){lanes})",
    lane_mask=False,
    goto=False,
    # NOTE: Metal's compiler moves a thread array of 64 complex amplitudes (a
    # width-11 component on 32 threads) to the stack, which costs the tape 1.8x.
    amplitude_chunk=32,
    pair_projections=True,
)

PRELUDE = r"""
#include <metal_stdlib>
using namespace metal;
typedef ulong u64;
#define GOLDEN 0x9E3779B97F4A7C15UL

inline u64 mix64(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}
// The CPU executor's draw: splitmix64 at index shot * slots + slot of stream seed.
inline u64 draw(u64 seed, u64 shot, u64 slots, u64 slot) {
    return mix64(seed + (shot * slots + slot) * GOLDEN + GOLDEN);
}
inline int par(u64 v) { return popcount(v) & 1; }
// draw_one against the rarer outcome's share, which float32 keeps to full
// relative precision: a rare 0 takes the top of the range, as on the CPU
// (``shared._cut``). The share is refined to about 46 bits as q + e / total
// (Metal has no double): q correctly rounded, e its exact residual. Not
// inlined, as it is rarely taken.
__attribute__((noinline)) int draw_one_exact(u64 r, float w0, float w1, float total) {
    int rare0 = w0 < w1;
    float w = rare0 ? w0 : w1;
    float q = precise::divide(w, total);
    float e = fma(-q, total, w);
    u64 below = u64(q * 0x1p64f) + u64(long(precise::divide(e, total) * 0x1p64f));
    return ((rare0 ? ~r : r) < below) ^ rare0;
}
// Whether the 64-bit draw r gives outcome 1, of weight w1 against w0. float32
// decides it unless r / 2^64 lies within 2^-21 of the share w1 / total, which
// covers both roundings and fast division's error; there draw_one_exact does.
inline int draw_one(u64 r, float w0, float w1, float total) {
    float p = w1 / total, x = float(r) * 0x1p-64f;
    if (x < p * (1.f - 0x1p-21f)) return 1;
    if (x > p * (1.f + 0x1p-21f)) return 0;
    return draw_one_exact(r, w0, w1, total);
}
// A uniform in (0, 1] from the high 24 bits, so that log never sees 0.
inline float unit(u64 r) { return float((r >> 40) + 1) * (1.0f / 16777216.0f); }
inline float2 cmul(float2 p, float2 q) {
    return float2(p.x * q.x - p.y * q.y, p.x * q.y + p.y * q.x);
}
// c a + s q and c a + i s q
inline float2 cmac(float2 a, float c, float s, float2 q) {
    return float2(fma(c, a.x, s * q.x), fma(c, a.y, s * q.y));
}
inline float2 cmac_i(float2 a, float c, float s, float2 q) {
    return float2(fma(c, a.x, -s * q.y), fma(c, a.y, s * q.x));
}
// The largest normalization deviation, as the bits of a non-negative float.
inline void deviate(device atomic_uint* deviation, float total) {
    float d = fabs(total - 1.f);
    if (!(d <= 1e-3f))
        atomic_fetch_max_explicit(deviation, d == d ? as_type<uint>(d) : 0x7f800000u, memory_order_relaxed);
}
"""
