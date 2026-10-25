"""CUDA source of a component's tape: straight-line FP32 code over its amplitudes.

A shot's ``2^W`` amplitudes are spread over ``T = 2^tb`` threads: the lowest
``tb`` axes are thread axes, so amplitude ``g`` is local ``g >> tb`` of thread
``g & (T - 1)``, held in registers. A Pauli with X on a thread axis reads its
partner amplitudes with ``__shfl_xor_sync`` among the group. Every update is
written per amplitude ``g`` and its partner ``g ^ x``: a rotation is
``a_g ← cos a_g + κ s (-1)^{par((g^x)&z)} a_{g^x}``, with ``κ`` 1 or i, ``s`` its
signed strength. Masks, angles and amplitude indices are constants of the
generated code, so a tape compiles to a few thousand fused multiply-adds.

The tape function follows ``shared.run_lanes`` for one shot on its own lane: its
lane symbols are the key, the postselected f-parts and its branch bits; it draws
a measurement's outcome as the CPU executor does, by its seed, shot and action;
a postselected output that reads 1 ends it (its *grave* is that action); at the
end it adds its branch parts, coins and hidden outputs to the shot's words.

With ``expectations``, the tape also returns, for each observable output, the
expectation of ``(-1)^output`` given the shot's history before the measurement
that sets the observable's last branch bit ``b``:
``(-1)^(output ⊕ b) (w0 - w1) / (w0 + w1)``. A fair coin's is 0 and an output
without branch bits has ``±1``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tsim.statevector.program import MEASURE_DORMANT, ROTATE, ROTATE_DIAG
from tsim.statevector.shared import LaneProgram

TINY = 1e-30

CERTAIN = 1e-11
"""Outcome weight, relative to the total, below which an outcome is impossible.

float32 amplitudes leave a certain outcome's other branch a weight of ~1e-14 to
1e-12; genuine outcome probabilities of tsim's circuits (T gates, Pauli noise,
small-angle rotations) stay above ~1e-10.
"""

UNROLL_BUDGET = 12_000
"""Amplitude updates (local spans summed over the actions) up to which a tape unrolls.

A longer tape loops over its amplitudes instead: they then live in local memory
(cached in L1), but NVRTC compiles the tape in seconds instead of minutes.
"""

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


def hex64(value) -> str:
    """Return a CUDA unsigned 64-bit literal."""
    return f"0x{int(value):016X}ULL"


def f32(value: float) -> str:
    """Return a CUDA float literal of ``value`` rounded to float32."""
    return repr(float(np.float32(value))) + "f"


def _par(value: int) -> int:
    return bin(value).count("1") & 1


def row_parity(row: np.ndarray) -> str:
    """Return CUDA for the parity of ``sym & row``, or ``0``."""
    terms = [f"(sym[{w}] & {hex64(m)})" for w, m in enumerate(row) if int(m)]
    return f"par({' ^ '.join(terms)})" if terms else "0"


class _Tape:
    """Emits one component's actions for a thread group of ``2^tb`` threads."""

    def __init__(self, program: LaneProgram, tb: int, emit: Callable[[str], None]):
        self.p = program
        self.tb = tb
        self.T = 1 << tb
        self.emit = emit
        spans = [self.span(a) for a in range(len(program.kind))]
        # NOTE: one runtime index puts the whole amplitude array in local memory,
        # so a tape either unrolls every action or loops every wide one.
        self.loop = sum(spans) > UNROLL_BUDGET

    def shfl(self, expr: str, xt: int) -> str:
        return expr if xt == 0 else f"shfl2({expr}, {xt}, gmask)"

    def span(self, a: int) -> int:
        return 1 << max(int(self.p.extent[a]) - self.tb, 0)

    def masks(self, a: int) -> tuple[int, int, int, int]:
        x, z, T = int(self.p.x_mask[a]), int(self.p.z_mask[a]), self.T
        return x & (T - 1), x >> self.tb, z & (T - 1), z >> self.tb

    def rotation(self, a: int) -> None:
        p, emit = self.p, self.emit
        xt, xl, zt, zl = self.masks(a)
        emit(f"    f = {row_parity(p.sign_row[a])} ^ {int(p.sign_const[a])};")
        if int(p.kind[a]) == ROTATE_DIAG:
            g0, g1 = p.factor[a]
            tpar = f" ^ par((u64)(t & {zt}))" if zt else ""
            emit(f"    {{ int ff = f{tpar};")
            emit(
                f"      float2 h0 = ff ? make_float2({f32(g1.real)}, {f32(g1.imag)}) : make_float2({f32(g0.real)}, {f32(g0.imag)});"
            )
            emit(
                f"      float2 h1 = ff ? make_float2({f32(g0.real)}, {f32(g0.imag)}) : make_float2({f32(g1.real)}, {f32(g1.imag)});"
            )
            if self.loop:
                emit(
                    f"      #pragma unroll 1\n      for (int l = 0; l < {self.span(a)}; ++l) a[l] = cmul(a[l], par((u64)(l & {zl})) ? h1 : h0);"
                )
            else:
                for loc in range(self.span(a)):
                    emit(
                        f"      a[{loc}] = cmul(a[{loc}], {'h1' if _par(loc & zl) else 'h0'});"
                    )
            emit("    }")
            return
        coupling = p.coupling[a]
        imaginary = coupling.real == 0
        strength = coupling.imag if imaginary else coupling.real
        c = f32(p.cos[a])
        tpar = f" ^ par((u64)((t ^ {xt}) & {zt}))" if zt else ""
        emit(f"    s = (f{tpar}) ? {f32(-strength)} : {f32(strength)};")
        update = "cmac_i" if imaginary else "cmac"

        def sign(loc: int) -> str:
            return "-s" if _par((loc ^ xl) & zl) else "s"

        if self.loop:
            self._rotation_loop(a, c, update)
            return
        if xl:
            low = xl & -xl
            for loc in range(self.span(a)):
                if loc & low:
                    continue
                m = loc ^ xl
                emit(
                    f"    {{ float2 ql = {self.shfl(f'a[{m}]', xt)}, qm = {self.shfl(f'a[{loc}]', xt)};"
                )
                emit(
                    f"      a[{loc}] = {update}(a[{loc}], {c}, {sign(loc)}, ql); a[{m}] = {update}(a[{m}], {c}, {sign(m)}, qm); }}"
                )
        else:
            for loc in range(self.span(a)):
                emit(
                    f"    {{ float2 q = {self.shfl(f'a[{loc}]', xt)}; a[{loc}] = {update}(a[{loc}], {c}, {sign(loc)}, q); }}"
                )

    def weights(self, a: int) -> None:
        """Set ``w0``, ``w1``: the outcome weights of a splitting measurement."""
        p, emit = self.p, self.emit
        if int(p.kind[a]) == MEASURE_DORMANT:
            emit("    w0 = 0.5f; w1 = 0.5f;")
            return
        x = int(p.x_mask[a])
        xt, xl, zt, zl = self.masks(a)
        emit("    w0 = 0.f; w1 = 0.f;")
        if self.loop:
            self._weights_loop(a)
        elif x == 0:
            emit(f"    {{ int tp = {f'par((u64)(t & {zt}))' if zt else '0'};")
            for loc in range(self.span(a)):
                emit(
                    f"      {{ float m2 = a[{loc}].x * a[{loc}].x + a[{loc}].y * a[{loc}].y; if (tp ^ {_par(loc & zl)}) w1 += m2; else w0 += m2; }}"
                )
            emit("    }")
        else:
            ip = p.i_power[a]
            emit(f"    {{ int tp = {f'par((u64)((t ^ {xt}) & {zt}))' if zt else '0'};")
            emit(
                f"      float2 ipw = tp ? make_float2({f32(-ip.real)}, {f32(-ip.imag)}) : make_float2({f32(ip.real)}, {f32(ip.imag)});"
            )
            for loc in range(self.span(a)):
                m = loc ^ xl
                sg = "-1.f" if _par(m & zl) else "1.f"
                emit(
                    f"      {{ float2 b = cmul({self.shfl(f'a[{m}]', xt)}, ipw); b.x *= {sg}; b.y *= {sg};"
                )
                emit(
                    f"        float px = a[{loc}].x + b.x, py = a[{loc}].y + b.y, mx = a[{loc}].x - b.x, my = a[{loc}].y - b.y;"
                )
                emit(
                    "        w0 += 0.25f * (px * px + py * py); w1 += 0.25f * (mx * mx + my * my); }"
                )
            emit("    }")
        if self.tb:
            emit(
                f"    for (int o = 1; o < {self.T}; o <<= 1) {{ w0 += __shfl_xor_sync(gmask, w0, o); w1 += __shfl_xor_sync(gmask, w1, o); }}"
            )

    def project(self, a: int) -> None:
        """Project onto ``outcome`` (``weight`` set) and free the pivot."""
        p, emit, tb = self.p, self.emit, self.tb
        x = int(p.x_mask[a])
        xt, xl, zt, zl = self.masks(a)
        pivot = int(p.pivot[a])
        ip = p.i_power[a]
        emit(
            f"    scale = rsqrtf(fmaxf({'weight' if x == 0 else '2.f * weight'}, {TINY}f));"
        )
        emit(f"    {{ int lowhalf = {f'!((t >> {pivot}) & 1)' if pivot < tb else '1'};")
        lbit = (1 << (pivot - tb)) if pivot >= tb else 0
        span = self.span(a)
        if self.loop:
            self._project_loop(a, lbit)
        elif x == 0:
            bt = (1 << pivot) if pivot < tb else 0
            emit(f"      int tp = {f'par((u64)(t & {zt}))' if zt else '0'};")
            for loc in range(span):
                if lbit and (loc & lbit):
                    emit(f"      a[{loc}] = make_float2(0.f, 0.f);")
                    continue
                hi = loc | lbit
                emit(
                    f"      {{ float2 h = {self.shfl(f'a[{hi}]', bt)}; int keep = (tp ^ {_par(loc & zl)}) == outcome;"
                )
                emit(
                    f"        a[{loc}] = lowhalf ? (keep ? make_float2(a[{loc}].x * scale, a[{loc}].y * scale) : make_float2(h.x * scale, h.y * scale)) : make_float2(0.f, 0.f);"
                )
                if lbit:
                    emit(f"        a[{hi}] = make_float2(0.f, 0.f);")
                emit("      }")
        else:
            emit(f"      int tp = {f'par((u64)((t ^ {xt}) & {zt}))' if zt else '0'};")
            emit(
                f"      float2 ipw = (tp ^ outcome) ? make_float2({f32(-ip.real)}, {f32(-ip.imag)}) : make_float2({f32(ip.real)}, {f32(ip.imag)});"
            )
            # NOTE: every partner is read before any amplitude changes.
            for loc in range(span):
                emit(f"      float2 q{loc} = {self.shfl(f'a[{loc ^ xl}]', xt)};")
            for loc in range(span):
                if lbit and (loc & lbit):
                    emit(f"      a[{loc}] = make_float2(0.f, 0.f);")
                    continue
                sg = "-1.f" if _par((loc ^ xl) & zl) else "1.f"
                emit(
                    f"      {{ float2 b = cmul(q{loc}, ipw); a[{loc}] = lowhalf ? make_float2((a[{loc}].x + {sg} * b.x) * scale, (a[{loc}].y + {sg} * b.y) * scale) : make_float2(0.f, 0.f); }}"
                )
        emit("    }")

    # NOTE: the loops index ``a`` at runtime, so wide tapes keep their
    # amplitudes in local memory (cached in L1) and compile quickly.

    def _rotation_loop(self, a: int, c: str, update: str) -> None:
        emit = self.emit
        xt, xl, _, zl = self.masks(a)
        span = self.span(a)
        if xl:
            low = xl & -xl
            emit(f"    #pragma unroll 1\n    for (int u = 0; u < {span // 2}; ++u) {{")
            emit(
                f"      int l = ((u & ~{low - 1}) << 1) | (u & {low - 1}), m = l ^ {xl};"
            )
            emit(
                f"      float2 ql = {self.shfl('a[m]', xt)}, qm = {self.shfl('a[l]', xt)};"
            )
            emit(
                f"      float sl = par((u64)((l ^ {xl}) & {zl})) ? -s : s, sm = par((u64)((m ^ {xl}) & {zl})) ? -s : s;"
            )
            emit(
                f"      a[l] = {update}(a[l], {c}, sl, ql); a[m] = {update}(a[m], {c}, sm, qm);"
            )
            emit("    }")
        else:
            emit(f"    #pragma unroll 1\n    for (int l = 0; l < {span}; ++l) {{")
            emit(f"      float2 q = {self.shfl('a[l]', xt)};")
            emit(f"      a[l] = {update}(a[l], {c}, par((u64)(l & {zl})) ? -s : s, q);")
            emit("    }")

    def _weights_loop(self, a: int) -> None:
        emit = self.emit
        x = int(self.p.x_mask[a])
        xt, xl, zt, zl = self.masks(a)
        span = self.span(a)
        if x == 0:
            emit(f"    {{ int tp = {f'par((u64)(t & {zt}))' if zt else '0'};")
            emit(f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) {{")
            emit("        float m2 = a[l].x * a[l].x + a[l].y * a[l].y;")
            emit(f"        if (tp ^ par((u64)(l & {zl}))) w1 += m2; else w0 += m2;")
            emit("      }")
            emit("    }")
            return
        ip = self.p.i_power[a]
        emit(f"    {{ int tp = {f'par((u64)((t ^ {xt}) & {zt}))' if zt else '0'};")
        emit(
            f"      float2 ipw = tp ? make_float2({f32(-ip.real)}, {f32(-ip.imag)}) : make_float2({f32(ip.real)}, {f32(ip.imag)});"
        )
        emit(f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) {{")
        emit(f"        int m = l ^ {xl};")
        emit(f"        float2 b = cmul({self.shfl('a[m]', xt)}, ipw);")
        emit(f"        if (par((u64)(m & {zl}))) {{ b.x = -b.x; b.y = -b.y; }}")
        emit(
            "        float px = a[l].x + b.x, py = a[l].y + b.y, mx = a[l].x - b.x, my = a[l].y - b.y;"
        )
        emit(
            "        w0 += 0.25f * (px * px + py * py); w1 += 0.25f * (mx * mx + my * my);"
        )
        emit("      }")
        emit("    }")

    def _project_loop(self, a: int, lbit: int) -> None:
        emit, tb = self.emit, self.tb
        x = int(self.p.x_mask[a])
        xt, xl, zt, zl = self.masks(a)
        pivot = int(self.p.pivot[a])
        span = self.span(a)
        if x == 0:
            bt = (1 << pivot) if pivot < tb else 0
            emit(f"      int tp = {f'par((u64)(t & {zt}))' if zt else '0'};")
            emit(f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) {{")
            emit(f"        if ({lbit} & l) continue;")
            emit(
                f"        float2 h = {self.shfl(f'a[l | {lbit}]', bt)}; int keep = (tp ^ par((u64)(l & {zl}))) == outcome;"
            )
            emit(
                "        a[l] = lowhalf ? (keep ? make_float2(a[l].x * scale, a[l].y * scale) : make_float2(h.x * scale, h.y * scale)) : make_float2(0.f, 0.f);"
            )
            emit("      }")
            if lbit:
                emit(
                    f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) if (l & {lbit}) a[l] = make_float2(0.f, 0.f);"
                )
            return
        ip = self.p.i_power[a]
        emit(f"      int tp = {f'par((u64)((t ^ {xt}) & {zt}))' if zt else '0'};")
        emit(
            f"      float2 ipw = (tp ^ outcome) ? make_float2({f32(-ip.real)}, {f32(-ip.imag)}) : make_float2({f32(ip.real)}, {f32(ip.imag)});"
        )
        emit(f"      float2 q[{span}];")
        emit(
            f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) q[l] = {self.shfl(f'a[l ^ {xl}]', xt)};"
        )
        emit(f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) {{")
        emit(f"        if ({lbit} & l) {{ a[l] = make_float2(0.f, 0.f); continue; }}")
        emit(
            f"        float2 b = cmul(q[l], ipw); float sg = par((u64)((l ^ {xl}) & {zl})) ? -1.f : 1.f;"
        )
        emit(
            "        a[l] = lowhalf ? make_float2((a[l].x + sg * b.x) * scale, (a[l].y + sg * b.y) * scale) : make_float2(0.f, 0.f);"
        )
        emit("      }")

    def prune(self, tests: np.ndarray, at: int) -> None:
        conds = [row_parity(self.p.prune_row[o]) for o in np.flatnonzero(tests)]
        conds = [c for c in conds if c != "0"]
        if conds:
            self.emit(f"    if ({' | '.join(conds)}) {{ grave = {at}; goto done; }}")


def tape_function(
    program: LaneProgram, index: int, tb: int, observables: dict[int, int]
) -> str:
    """Return ``component_<index>``: one shot's run of a component's tape.

    Args:
        program: The component's lane program.
        index: The component's index (names the function).
        tb: Log2 of the threads per shot.
        observables: Slot of each of the component's observable outputs
            (output slot in the program -> column of the expectations).

    """
    lines: list[str] = []
    emit = lines.append
    tape = _Tape(program, tb, emit)
    p = program
    A = len(p.kind)
    n = 1 << max(p.width - tb, 0)
    num_coins = p.coin_pattern.shape[0]
    slots = A + (num_coins + 63) // 64
    ready = p.ready
    # NOTE: the measurements whose weights an observable's expectation reads.
    watched = {int(ready[o]) for o in observables if ready[o] >= 0}

    emit(
        f"__device__ __forceinline__ int component_{index}(u64* words, u64 shot, u64 seed, int t, unsigned gmask, float* ev, unsigned* deviation) {{"
    )
    emit(f"    u64 sym[{p.lane_words}];")
    for w in range(p.key_words):
        emit(f"    sym[{w}] = words[{p.key_start + w}];")
    for j, (pw, pm) in enumerate(zip(p.ps_word, p.ps_mask, strict=True)):
        emit(f"    sym[{p.key_words + j}] = words[{int(pw)}] & {hex64(pm)};")
    for w in range(p.key_words + len(p.ps_word), p.lane_words):
        emit(f"    sym[{w}] = 0;")
    emit(f"    float2 a[{n}];")
    emit(
        f"    #pragma unroll\n    for (int i = 0; i < {n}; ++i) a[i] = make_float2(0.f, 0.f);"
    )
    emit("    if (t == 0) a[0].x = 1.f;")
    emit("    int grave = 0x7fffffff;")
    emit("    float w0, w1, total, weight, scale, s; int outcome, f;")
    for a in sorted(watched):
        emit(f"    float ev_{a} = 0.f; int b_{a} = 0;")
    if p.prune_first.any():
        tape.prune(p.prune_first, -1)
    for a in range(A):
        kind = int(p.kind[a])
        emit(f"    // action {a}: kind {kind}")
        if kind in (ROTATE, ROTATE_DIAG):
            tape.rotation(a)
            continue
        if not p.splits[a]:
            continue
        tape.weights(a)
        emit("    total = w0 + w1;")
        if kind != MEASURE_DORMANT:
            emit("    if (t == 0) deviate(deviation, total);")
        # NOTE: float32 rounding leaves an outcome that is certain a weight of up to
        # ~1e-12 of the total; it counts as impossible (CERTAIN), and the others
        # are drawn against the full 64-bit draw, as the CPU executor draws them.
        emit(
            f"    if (w1 <= {CERTAIN}f * total) outcome = 0; else if (w0 <= {CERTAIN}f * total) outcome = 1;"
        )
        emit(
            f"    else outcome = draw(seed, shot, {slots}, {a}) < (u64)((double)w1 / (double)total * 18446744073709551616.0);"
        )
        emit("    weight = outcome ? w1 : w0;")
        if a in watched:
            emit(
                f"    ev_{a} = total > {TINY}f ? (w0 - w1) / total : 0.f; b_{a} = outcome;"
            )
        if kind != MEASURE_DORMANT:
            tape.project(a)
        bit = int(p.branch[a])
        if bit >= 0:
            emit(f"    if (outcome) sym[{bit >> 6}] |= 1ULL << {bit & 63};")
        if p.pruned[a]:
            tape.prune(p.prune_outputs[a], a)
    emit("done:")
    for o in range(len(p.output_bit)):
        w = int(p.out_words[p.output_word[o]])
        parity = row_parity(p.branch_row[o])
        if parity != "0":
            emit(f"    if ({parity}) words[{w}] ^= {hex64(1 << int(p.output_bit[o]))};")
    if num_coins:
        emit("    { u64 coins = 0;")
        for j in range(num_coins):
            if j % 64 == 0:
                emit(f"      coins = draw(seed, shot, {slots}, {A + j // 64});")
            flips = [
                (int(p.out_words[w]), int(m))
                for w, m in enumerate(p.coin_pattern[j])
                if int(m)
            ]
            if flips:
                body = " ".join(f"words[{w}] ^= {hex64(m)};" for w, m in flips)
                emit(f"      if ((coins >> {j % 64}) & 1) {{ {body} }}")
        emit("    }")
    for o in range(len(p.output_bit)):
        if ready[o] >= 0:
            w = int(p.out_words[p.output_word[o]])
            emit(
                f"    if (grave < {int(ready[o])}) words[{w}] &= ~{hex64(1 << int(p.output_bit[o]))};"
            )
    if observables:
        emit("    if (ev != nullptr) {")
        for o, column in observables.items():
            w = int(p.out_words[p.output_word[o]])
            bit = f"(int)((words[{w}] >> {int(p.output_bit[o])}) & 1)"
            r = int(ready[o])
            if r < 0:
                value = f"({bit} ? -1.f : 1.f)"
            elif int(p.kind[r]) == MEASURE_DORMANT or not p.splits[r]:
                value = "0.f"
            else:
                value = f"((({bit}) ^ b_{r}) ? -ev_{r} : ev_{r})"
            emit(f"        ev[{column}] = grave < {max(r, 0)} ? 0.f : {value};")
        emit("    }")
    emit("    return grave;")
    emit("}")
    return "\n".join(lines)
