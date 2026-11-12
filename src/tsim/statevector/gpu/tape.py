"""GPU source of a component's tape: straight-line FP32 code over its amplitudes.

A shot's ``2^W`` amplitudes are spread over ``T = 2^tb`` threads: the lowest
``tb`` axes are thread axes, so amplitude ``g`` is local ``g >> tb`` of thread
``g & (T - 1)``, held in registers. A Pauli with X on a thread axis reads its
partner amplitudes with a shuffle among the group. Every update is
written per amplitude ``g`` and its partner ``g ^ x``: a rotation is
``a_g ← cos a_g + κ s (-1)^{par((g^x)&z)} a_{g^x}``, with ``κ`` 1 or i, ``s`` its
signed strength. Masks, angles and amplitude indices are constants of the
generated code, so a tape compiles to a few thousand fused multiply-adds.
The same tape is written in each backend's language (``Dialect``); a backend's
prelude defines the helpers that it calls (``draw``, ``draw_one``, ``cmul``,
``cmac``, ``cmac_i``, ``par``, ``deviate``).

The tape function follows ``shared.run_lanes`` for one shot on its own lane: its
lane symbols are the key, the postselected f-parts and its branch bits; it draws
a measurement's outcome as the CPU executor does, by its seed, shot and action;
a postselected output that reads 1 ends it (its *grave* is that action); at the
end it adds its branch parts, coins and hidden outputs to the shot's words.
The actions before the first one that reads a lane symbol run alike for every
shot (``shared.common_prefix``): they run once, on the host, and every shot
starts from the amplitudes they leave.

With ``expectations``, the tape also returns, for each observable output, the
expectation of ``(-1)^output`` given the shot's history before the measurement
that sets the observable's last branch bit ``b``:
``(-1)^(output ⊕ b) (w0 - w1) / (w0 + w1)``. A fair coin's is 0 and an output
without branch bits has ``±1``. The expectation ignores postselection that
follows it, so when a postselected output of the component is tested at or
after that measurement, the tape returns the sampled ``(-1)^output`` instead:
the expectation's average over the accepted shots would be biased, the sample's
is not.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsim.statevector.program import MEASURE_DORMANT, ROTATE, ROTATE_DIAG
from tsim.statevector.shared import LaneProgram, common_prefix, pack_bits

if TYPE_CHECKING:
    from tsim.statevector.shots import ShotLayout, ShotPlan

MAX_WIDTH = 11
"""Widest component the GPU executors run: 32 threads of 64 amplitudes."""

AMPLITUDE_AXES = 5
"""Log2 of the amplitudes per thread from which a shot takes more threads."""

DEVIATION_TOLERANCE = 1e-3
"""Normalization deviation that float32 amplitudes tolerate without a warning."""

TINY = 1e-30

CERTAIN = 1e-11
"""Outcome weight, relative to the total, below which an outcome is impossible.

float32 amplitudes leave a certain outcome's other branch a weight of ~1e-14 to
1e-12, which the GPU cannot tell from a genuine outcome probability that small.
"""

RESOLVABLE = 1e-8
"""Smallest branch weight of a rotation (``cos^2`` or ``sin^2`` of its angle) that
a GPU executor runs: a smaller one could give an outcome a genuine probability
below ``CERTAIN``, which the GPU would never draw."""

_STRUCTURAL = 1e-20

UNROLL_BUDGET = 12_000
"""Amplitude updates (local spans summed over the actions) up to which a tape unrolls.

A longer tape loops over its amplitudes instead: they then live in local memory
(cached in L1), but it compiles in seconds instead of minutes.
"""

BLOCK_AXES = 4
"""Local axes that a looping tape's run of rotations may mix in one pass.

A looping tape applies consecutive rotations whose X masks share these axes
together, subcube by subcube, with the subcube's amplitudes in registers, so
it reads and writes the amplitudes once per run instead of once per rotation.
"""


@dataclass(frozen=True)
class Dialect:
    """How a GPU language spells the parts of a tape that differ between them.

    Attributes:
        inline: Qualifiers of the component function, inlined into its kernel.
        noinline: Qualifiers of a looping tape's rotation passes.
        private: Address space of pointers to a thread's own arrays.
        deviation: Type of the pointer to the largest normalization deviation.
        constant: Qualifiers of a program-scope constant array.
        float2: Constructor of a ``float2``.
        rsqrt: Reciprocal square root of a float.
        fmax: Larger of two floats.
        u64: Suffix of a 64-bit unsigned literal.
        shuffle2: A ``float2`` ``{value}`` from the lane ``{lanes}`` (an XOR) away.
        shuffle1: A ``float`` ``{value}`` from the lane ``{lanes}`` away.
        lane_mask: Whether the shuffles take the group's lane mask ``gmask``,
            which the tape functions then pass on.
        goto: Whether a pruned shot jumps to the end; otherwise it breaks out
            of a ``do``-block around the actions.
        amplitude_chunk: Amplitudes per array of an unrolled tape (a power of
            two, so the arrays are full), or None for one array: a compiler
            may move a long array to the stack.
        pair_projections: Whether a projection reads and updates each pair of
            partner amplitudes at once, instead of every partner first: fewer
            values are live.

    """

    inline: str
    noinline: str
    private: str
    deviation: str
    constant: str
    float2: str
    rsqrt: str
    fmax: str
    u64: str
    shuffle2: str
    shuffle1: str
    lane_mask: bool
    goto: bool
    amplitude_chunk: int | None = None
    pair_projections: bool = False

    def __post_init__(self) -> None:
        """Check that the chunk size is a positive power of two."""
        chunk = self.amplitude_chunk
        assert chunk is None or (chunk > 0 and not chunk & (chunk - 1))

    def hex64(self, value) -> str:
        """Return an unsigned 64-bit literal."""
        return f"0x{int(value):016X}{self.u64}"

    def row_parity(self, row: np.ndarray) -> str:
        """Return the parity of ``sym & row``, or ``0``."""
        terms = [f"(sym[{w}] & {self.hex64(m)})" for w, m in enumerate(row) if int(m)]
        return f"par({' ^ '.join(terms)})" if terms else "0"


def supports(layout: ShotLayout) -> bool:
    """Return whether a GPU executor runs ``layout``.

    It needs shot words, components of at most ``MAX_WIDTH`` qubits and rotations
    whose branches the GPU resolves (``rarest_branch`` at least ``RESOLVABLE``). A
    layout without shot words (no outputs and no noise) stays on the CPU.
    """
    return (
        layout.num_words > 0
        and all(c.width <= MAX_WIDTH for c in layout.components)
        and rarest_branch(layout) >= RESOLVABLE
    )


def rarest_branch(layout: ShotLayout) -> float:
    """Return the smallest nonzero branch weight (``cos^2``, ``sin^2``) of a rotation.

    It is 1 without rotations; a weight below ``_STRUCTURAL`` (of a Clifford
    angle, in float64) counts as zero.
    """
    rarest = 1.0
    for component in layout.components:
        t = component.tape
        rotations = (t.kind == ROTATE) | (t.kind == ROTATE_DIAG)
        weights = np.concatenate([t.cos[rotations] ** 2, t.sin[rotations] ** 2])
        weights = weights[weights >= _STRUCTURAL]
        if len(weights):
            rarest = min(rarest, float(weights.min()))
    return rarest


def threads_per_shot(layout: ShotLayout, amplitude_axes: int = AMPLITUDE_AXES) -> int:
    """Return log2 of a shot's threads: one per ``2^amplitude_axes`` amplitudes, at most 32."""
    width = max((c.width for c in layout.components), default=0)
    return min(5, max(0, width - amplitude_axes))


def deviation(bits) -> float:
    """Return the normalization deviation of a call (the bits of a float32), or 0.

    It is 0 within ``DEVIATION_TOLERANCE``; a NaN stays NaN, which must warn.
    """
    value = float(np.asarray(bits, np.uint32).reshape(1).view(np.float32)[0])
    return 0.0 if value <= DEVIATION_TOLERANCE else value


@dataclass(frozen=True)
class TapeKernel:
    """The parts of a backend's tape kernel that a plan fills in.

    Attributes:
        functions: Every component's tape function, ``component_<i>``.
        seeds: The components' streams ``seed_<i>``, from the call's ``seed``.
        calls: The calls of the tape functions, each lowering ``grave``.
        stores: The stores of the words that the tapes write, ``row[k] = w[k]``.
        hits: The postselected bits among those words, OR-ed (``0`` if none).
        psm_words: The postselection mask, as output words.

    """

    functions: str
    seeds: str
    calls: str
    stores: str
    hits: str
    psm_words: np.ndarray


def tape_kernel(
    layout: ShotLayout,
    plan: ShotPlan,
    mask: np.ndarray | None,
    num_detectors: int,
    tb: int,
    dialect: Dialect,
) -> TapeKernel:
    """Return the tape kernel's parts for ``plan`` of postselection ``mask``.

    The outputs from ``num_detectors`` on are observables, whose expectations
    the tapes return in that order.
    """
    functions = []
    for i, (program, component) in enumerate(
        zip(plan.programs, layout.components, strict=True)
    ):
        observables = {
            o: index - num_detectors
            for o, index in enumerate(component.output_indices)
            if index >= num_detectors
        }
        functions.append(tape_function(program, i, tb, observables, dialect))
    lanes = "t, gmask" if dialect.lane_mask else "t"
    count = len(plan.programs)
    bits = np.zeros(64 * layout.output_words, dtype=bool)
    if mask is not None:
        bits[: len(mask)] = mask
    psm_words = pack_bits(bits, layout.output_words)
    # NOTE: the tapes write only their outputs' words, and test the postselected
    # outputs among them.
    written = sorted({int(w) for p in plan.programs for w in p.out_words})
    hits = [
        f"(w[{w}] & {dialect.hex64(psm_words[w])})" for w in written if psm_words[w]
    ]
    return TapeKernel(
        functions="\n\n".join(functions),
        seeds="\n".join(
            f"    const u64 seed_{i} = mix64(seed + {i + 1}{dialect.u64} * GOLDEN);"
            for i in range(count)
        ),
        calls="\n".join(
            f"    grave = min(grave, component_{i}(w, shot, seed_{i}, {lanes}, ev, deviation));"
            for i in range(count)
        ),
        stores="\n".join(f"        row[{w}] = w[{w}];" for w in written),
        hits=" | ".join(hits) or "0",
        psm_words=psm_words,
    )


def f32(value: float) -> str:
    """Return a float literal of ``value`` rounded to float32."""
    return repr(float(np.float32(value))) + "f"


def _par(value: int) -> int:
    return bin(value).count("1") & 1


def loops(program: LaneProgram, tb: int, start: int) -> bool:
    """Return whether the tape from action ``start`` loops instead of unrolling.

    One runtime index puts the whole amplitude array in local memory, so a tape
    either unrolls every action or loops every wide one.
    """
    extent = program.extent[start:].astype(np.int64)
    return int((1 << np.maximum(extent - tb, 0)).sum()) > UNROLL_BUDGET


class _Tape:
    """Emits one component's actions for a thread group of ``2^tb`` threads."""

    def __init__(
        self,
        program: LaneProgram,
        tb: int,
        emit: Callable[[str], None],
        dialect: Dialect,
        name: str = "run",
        start: int = 0,
    ):
        self.d = dialect
        self.p = program
        self.tb = tb
        self.T = 1 << tb
        self.emit = emit
        self.name = name
        self.helpers: list[str] = []
        self.loop = loops(program, tb, start)

    def shfl(self, expr: str, xt: int) -> str:
        return expr if xt == 0 else self.d.shuffle2.format(value=expr, lanes=xt)

    def f2(self, x: str, y: str) -> str:
        return f"{self.d.float2}({x}, {y})"

    def span(self, a: int) -> int:
        return 1 << max(int(self.p.extent[a]) - self.tb, 0)

    def masks(self, a: int) -> tuple[int, int, int, int]:
        x, z, T = int(self.p.x_mask[a]), int(self.p.z_mask[a]), self.T
        return x & (T - 1), x >> self.tb, z & (T - 1), z >> self.tb

    def rotation(self, a: int) -> None:
        p, emit = self.p, self.emit
        xt, xl, zt, zl = self.masks(a)
        emit(f"    f = {self.d.row_parity(p.sign_row[a])} ^ {int(p.sign_const[a])};")
        if int(p.kind[a]) == ROTATE_DIAG:
            g0, g1 = p.factor[a]
            tpar = f" ^ par((u64)(t & {zt}))" if zt else ""
            emit(f"    {{ int ff = f{tpar};")
            emit(
                f"      float2 h0 = ff ? {self.d.float2}({f32(g1.real)}, {f32(g1.imag)}) : {self.d.float2}({f32(g0.real)}, {f32(g0.imag)});"
            )
            emit(
                f"      float2 h1 = ff ? {self.d.float2}({f32(g0.real)}, {f32(g0.imag)}) : {self.d.float2}({f32(g1.real)}, {f32(g1.imag)});"
            )
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
                f"      float2 ipw = tp ? {self.d.float2}({f32(-ip.real)}, {f32(-ip.imag)}) : {self.d.float2}({f32(ip.real)}, {f32(ip.imag)});"
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
                f"    for (int o = 1; o < {self.T}; o <<= 1) {{ w0 += {self.d.shuffle1.format(value='w0', lanes='o')}; w1 += {self.d.shuffle1.format(value='w1', lanes='o')}; }}"
            )

    def project(self, a: int) -> None:
        """Project onto ``outcome`` (``weight`` set) and free the pivot."""
        p, emit, tb = self.p, self.emit, self.tb
        x = int(p.x_mask[a])
        xt, xl, zt, zl = self.masks(a)
        pivot = int(p.pivot[a])
        ip = p.i_power[a]
        emit(
            f"    scale = {self.d.rsqrt}({self.d.fmax}({'weight' if x == 0 else '2.f * weight'}, {TINY}f));"
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
                    emit(f"      a[{loc}] = {self.d.float2}(0.f, 0.f);")
                    continue
                hi = loc | lbit
                emit(
                    f"      {{ float2 h = {self.shfl(f'a[{hi}]', bt)}; int keep = (tp ^ {_par(loc & zl)}) == outcome;"
                )
                emit(
                    f"        a[{loc}] = lowhalf ? (keep ? {self.d.float2}(a[{loc}].x * scale, a[{loc}].y * scale) : {self.d.float2}(h.x * scale, h.y * scale)) : {self.d.float2}(0.f, 0.f);"
                )
                if lbit:
                    emit(f"        a[{hi}] = {self.d.float2}(0.f, 0.f);")
                emit("      }")
        else:
            emit(f"      int tp = {f'par((u64)((t ^ {xt}) & {zt}))' if zt else '0'};")
            emit(
                f"      float2 ipw = (tp ^ outcome) ? {self.d.float2}({f32(-ip.real)}, {f32(-ip.imag)}) : {self.d.float2}({f32(ip.real)}, {f32(ip.imag)});"
            )
            if self.d.pair_projections:
                self._project_pairs(a, lbit)
                emit("    }")
                return
            # NOTE: every partner is read before any amplitude changes.
            for loc in range(span):
                emit(f"      float2 q{loc} = {self.shfl(f'a[{loc ^ xl}]', xt)};")
            for loc in range(span):
                if lbit and (loc & lbit):
                    emit(f"      a[{loc}] = {self.d.float2}(0.f, 0.f);")
                    continue
                sg = "-1.f" if _par((loc ^ xl) & zl) else "1.f"
                emit(
                    f"      {{ float2 b = cmul(q{loc}, ipw); a[{loc}] = lowhalf ? {self.d.float2}((a[{loc}].x + {sg} * b.x) * scale, (a[{loc}].y + {sg} * b.y) * scale) : {self.d.float2}(0.f, 0.f); }}"
                )
        emit("    }")

    def _project_pairs(self, a: int, lbit: int) -> None:
        """Project, unrolled, one pair of partner amplitudes ``l``, ``l ^ xl`` at a time.

        An amplitude's update reads only its partner, so each pair is read
        before either changes, and few values are live.
        """
        emit = self.emit
        xt, xl, _, zl = self.masks(a)
        for loc in range(self.span(a)):
            m = loc ^ xl
            if m < loc:
                continue
            pair = [loc] if m == loc else [loc, m]
            reads = ", ".join(f"q{j} = {self.shfl(f'a[{j ^ xl}]', xt)}" for j in pair)
            emit(f"      {{ float2 {reads};")
            for j in pair:
                if lbit and (j & lbit):
                    emit(f"        a[{j}] = {self.f2('0.f', '0.f')};")
                    continue
                sg = "-1.f" if _par((j ^ xl) & zl) else "1.f"
                x_new = f"(a[{j}].x + {sg} * b.x) * scale"
                y_new = f"(a[{j}].y + {sg} * b.y) * scale"
                emit(
                    f"        {{ float2 b = cmul(q{j}, ipw); a[{j}] = lowhalf ? {self.f2(x_new, y_new)} : {self.f2('0.f', '0.f')}; }}"
                )
            emit("      }")

    # NOTE: the loops index ``a`` at runtime, so wide tapes keep their
    # amplitudes in local memory (cached in L1) and compile quickly.

    def run(self, a: int) -> tuple[list[int], int]:
        """Return the rotations from ``a`` that one pass applies, and the next action.

        The run ends at a splitting measurement or where its X masks would mix
        more than ``BLOCK_AXES`` local axes; measurements that do not split
        change no amplitude.
        """
        p, rotations, axes, b = self.p, [], 0, a
        while b < len(p.kind):
            kind = int(p.kind[b])
            if kind in (ROTATE, ROTATE_DIAG):
                joined = axes | (int(p.x_mask[b]) >> self.tb)
                if rotations and bin(joined).count("1") > BLOCK_AXES:
                    break
                rotations.append(b)
                axes = joined
            elif p.splits[b]:
                break
            b += 1
        return rotations, b

    def rotation_run(self, rotations: list[int]) -> None:
        """Apply ``rotations`` in one pass over the subcubes of their X axes.

        The pass is a function of its own, so that the compiler takes a long tape's
        runs one by one; the caller passes each rotation's signed strength (or
        diagonal factors), which depend on the shot.
        """
        p, emit = self.p, self.emit
        axes = 0
        for a in rotations:
            axes |= int(p.x_mask[a]) >> self.tb
        bits = [i for i in range(axes.bit_length()) if axes >> i & 1]
        # NOTE: the subcube's amplitude j is local base + offset[j].
        offset = [
            sum(1 << b for i, b in enumerate(bits) if j >> i & 1)
            for j in range(1 << len(bits))
        ]
        index = {o: j for j, o in enumerate(offset)}
        span = max(self.span(a) for a in rotations)
        name = f"{self.name}_{rotations[0]}"
        mask_param = "unsigned gmask, " if self.d.lane_mask else ""
        mask_arg = "gmask, " if self.d.lane_mask else ""
        params, args = [], []
        for a in rotations:
            xt, xl, zt, zl = self.masks(a)
            f = f"({self.d.row_parity(p.sign_row[a])} ^ {int(p.sign_const[a])})"
            if int(p.kind[a]) == ROTATE_DIAG:
                g0, g1 = p.factor[a]
                tpar = f" ^ par((u64)(t & {zt}))" if zt else ""
                params += [f"float2 h0_{a}", f"float2 h1_{a}"]
                v0 = f"{self.d.float2}({f32(g0.real)}, {f32(g0.imag)})"
                v1 = f"{self.d.float2}({f32(g1.real)}, {f32(g1.imag)})"
                args += [f"({f}{tpar}) ? {v1} : {v0}", f"({f}{tpar}) ? {v0} : {v1}"]
            else:
                coupling = p.coupling[a]
                strength = coupling.imag if coupling.real == 0 else coupling.real
                tpar = f" ^ par((u64)((t ^ {xt}) & {zt}))" if zt else ""
                params.append(f"float s{a}")
                args.append(f"({f}{tpar}) ? {f32(-strength)} : {f32(strength)}")
        body = [
            f"{self.d.noinline} void {name}({self.d.private}float2* a, int t, {mask_param}{', '.join(params)}) {{",
            f"    #pragma unroll 1\n    for (int base = 0; base < {span}; base = ((base | {axes}) + 1) & ~{axes}) {{",
            f"        float2 r[{len(offset)}];",
        ]
        body += [f"        r[{j}] = a[base + {o}];" for j, o in enumerate(offset)]
        for a in rotations:
            xt, xl, zt, zl = self.masks(a)
            # NOTE: a local index's sign parity is its subcube's (base) and its offset's.
            outside = zl & ~axes
            pb = f"par((u64)(base & {outside}))" if outside else "0"
            if int(p.kind[a]) == ROTATE_DIAG:
                body.append(f"        {{ int pb = {pb};")
                body += [
                    f"          r[{j}] = cmul(r[{j}], (pb ^ {_par(o & zl)}) ? h1_{a} : h0_{a});"
                    for j, o in enumerate(offset)
                ]
                body.append("        }")
                continue
            coupling = p.coupling[a]
            update = "cmac_i" if coupling.real == 0 else "cmac"
            c = f32(p.cos[a])
            body.append(f"        {{ float sb = {pb} ? -s{a} : s{a};")
            sign = ["-sb" if _par((o ^ xl) & zl) else "sb" for o in offset]
            for j, o in enumerate(offset):
                m = o ^ xl
                if xl and m < o:
                    continue
                if xl:
                    k = index[m]
                    body.append(
                        f"          {{ float2 ql = {self.shfl(f'r[{k}]', xt)}, qm = {self.shfl(f'r[{j}]', xt)};"
                    )
                    body.append(
                        f"            r[{j}] = {update}(r[{j}], {c}, {sign[j]}, ql); r[{k}] = {update}(r[{k}], {c}, {sign[k]}, qm); }}"
                    )
                else:
                    body.append(
                        f"          {{ float2 q = {self.shfl(f'r[{j}]', xt)}; r[{j}] = {update}(r[{j}], {c}, {sign[j]}, q); }}"
                    )
            body.append("        }")
        body += [f"        a[base + {o}] = r[{j}];" for j, o in enumerate(offset)]
        body += ["    }", "}"]
        self.helpers.append("\n".join(body))
        emit(
            f"    // actions {rotations[0]}-{rotations[-1]}: {len(rotations)} rotations on local axes {axes:#x}"
        )
        emit(f"    {name}(a, t, {mask_arg}{', '.join(args)});")

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
            f"      float2 ipw = tp ? {self.d.float2}({f32(-ip.real)}, {f32(-ip.imag)}) : {self.d.float2}({f32(ip.real)}, {f32(ip.imag)});"
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
                f"        a[l] = lowhalf ? (keep ? {self.d.float2}(a[l].x * scale, a[l].y * scale) : {self.d.float2}(h.x * scale, h.y * scale)) : {self.d.float2}(0.f, 0.f);"
            )
            emit("      }")
            if lbit:
                emit(
                    f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) if (l & {lbit}) a[l] = {self.d.float2}(0.f, 0.f);"
                )
            return
        ip = self.p.i_power[a]
        emit(f"      int tp = {f'par((u64)((t ^ {xt}) & {zt}))' if zt else '0'};")
        emit(
            f"      float2 ipw = (tp ^ outcome) ? {self.d.float2}({f32(-ip.real)}, {f32(-ip.imag)}) : {self.d.float2}({f32(ip.real)}, {f32(ip.imag)});"
        )
        emit(f"      float2 q[{span}];")
        emit(
            f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) q[l] = {self.shfl(f'a[l ^ {xl}]', xt)};"
        )
        emit(f"      #pragma unroll 1\n      for (int l = 0; l < {span}; ++l) {{")
        emit(
            f"        if ({lbit} & l) {{ a[l] = {self.d.float2}(0.f, 0.f); continue; }}"
        )
        emit(
            f"        float2 b = cmul(q[l], ipw); float sg = par((u64)((l ^ {xl}) & {zl})) ? -1.f : 1.f;"
        )
        emit(
            f"        a[l] = lowhalf ? {self.d.float2}((a[l].x + sg * b.x) * scale, (a[l].y + sg * b.y) * scale) : {self.d.float2}(0.f, 0.f);"
        )
        emit("      }")

    def prune(self, tests: np.ndarray, at: int) -> None:
        conds = [self.d.row_parity(self.p.prune_row[o]) for o in np.flatnonzero(tests)]
        conds = [c for c in conds if c != "0"]
        if conds:
            leave = "goto done" if self.d.goto else "break"
            self.emit(f"    if ({' | '.join(conds)}) {{ grave = {at}; {leave}; }}")


def tape_function(
    program: LaneProgram,
    index: int,
    tb: int,
    observables: dict[int, int],
    dialect: Dialect,
) -> str:
    """Return ``component_<index>``: one shot's run of a component's tape.

    Args:
        program: The component's lane program.
        index: The component's index (names the function).
        tb: Log2 of the threads per shot.
        observables: Slot of each of the component's observable outputs
            (output slot in the program -> column of the expectations).
        dialect: The language to write it in.

    """
    lines: list[str] = []
    emit = lines.append
    p = program
    A = len(p.kind)
    n = 1 << max(p.width - tb, 0)
    start, amps = common_prefix(p)
    d = dialect
    tape = _Tape(p, tb, emit, d, f"run{index}", start)
    loaded = 0
    if start:
        # NOTE: the prefix leaves amplitudes outside its last action's span zero.
        loaded = tape.span(start - 1)
        values = ", ".join(
            f"{{{f32(v.real)}, {f32(v.imag)}}}" for v in amps[: loaded << tb]
        )
        tape.helpers.append(
            f"{d.constant} float2 start{index}[{loaded << tb}] = {{{values}}};"
        )
    num_coins = p.coin_pattern.shape[0]
    slots = A + (num_coins + 63) // 64
    ready = p.ready
    # NOTE: an expectation is exact only if no postselected output of the component
    # is tested at or after its measurement (other components and direct outputs
    # depend on it only through the f-bits, which it is conditioned on).
    exact = {o: ready[o] < 0 or not p.pruned[ready[o] :].any() for o in observables}
    # NOTE: the measurements whose weights an observable's expectation reads.
    watched = {int(ready[o]) for o in observables if ready[o] >= 0 and exact[o]}

    mask = "unsigned gmask, " if d.lane_mask else ""
    emit(
        f"{d.inline} int component_{index}({d.private}u64* words, u64 shot, u64 seed, int t, {mask}{d.private}float* ev, {d.deviation} deviation) {{"
    )
    emit(f"    u64 sym[{p.lane_words}];")
    for w in range(p.key_words):
        emit(f"    sym[{w}] = words[{p.key_start + w}];")
    for j, (pw, pm) in enumerate(zip(p.ps_word, p.ps_mask, strict=True)):
        emit(f"    sym[{p.key_words + j}] = words[{int(pw)}] & {d.hex64(pm)};")
    for w in range(p.key_words + len(p.ps_word), p.lane_words):
        emit(f"    sym[{w}] = 0;")
    chunk = d.amplitude_chunk
    split = chunk is not None and not tape.loop and n > chunk
    first = len(lines)
    if split:
        assert chunk is not None
        # NOTE: unrolled, every index is a constant: ``_split_amplitudes`` sends
        # ``a[i]`` to chunk ``i / chunk``.
        emit(f"    float2 {', '.join(f'a{c}[{chunk}]' for c in range(n // chunk))};")
        for i in range(loaded, n):
            emit(f"    a[{i}] = {d.float2}(0.f, 0.f);")
        for i in range(loaded if start else 0):
            emit(f"    a[{i}] = start{index}[{i << tb} | t];")
    else:
        emit(f"    float2 a[{n}];")
        emit(
            f"    #pragma unroll\n    for (int i = {loaded}; i < {n}; ++i) a[i] = {d.float2}(0.f, 0.f);"
        )
        if start:
            emit(
                f"    #pragma unroll\n    for (int i = 0; i < {loaded}; ++i) a[i] = start{index}[(i << {tb}) | t];"
            )
    if not start:
        emit("    if (t == 0) a[0].x = 1.f;")
    emit("    int grave = 0x7fffffff;")
    emit("    float w0, w1, total, weight, scale, s; int outcome, f;")
    for a in sorted(watched):
        emit(f"    float ev_{a} = 0.f; int b_{a} = 0;")
    if not d.goto:
        emit("    do {")
    if p.prune_first.any():
        tape.prune(p.prune_first, -1)
    a = start - 1
    while a + 1 < A:
        a += 1
        kind = int(p.kind[a])
        if kind in (ROTATE, ROTATE_DIAG) and tape.loop:
            rotations, after = tape.run(a)
            tape.rotation_run(rotations)
            a = after - 1
            continue
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
            f"    else outcome = draw_one(draw(seed, shot, {slots}, {a}), w0, w1, total);"
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
            emit(f"    if (outcome) sym[{bit >> 6}] |= 1{d.u64} << {bit & 63};")
        if p.pruned[a]:
            tape.prune(p.prune_outputs[a], a)
    emit("done:" if d.goto else "    } while (false);")
    for o in range(len(p.output_bit)):
        w = int(p.out_words[p.output_word[o]])
        parity = d.row_parity(p.branch_row[o])
        if parity != "0":
            emit(
                f"    if ({parity}) words[{w}] ^= {d.hex64(1 << int(p.output_bit[o]))};"
            )
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
                body = " ".join(f"words[{w}] ^= {d.hex64(m)};" for w, m in flips)
                emit(f"      if ((coins >> {j % 64}) & 1) {{ {body} }}")
        emit("    }")
    for o in range(len(p.output_bit)):
        if ready[o] >= 0:
            w = int(p.out_words[p.output_word[o]])
            emit(
                f"    if (grave < {int(ready[o])}) words[{w}] &= ~{d.hex64(1 << int(p.output_bit[o]))};"
            )
    if observables:
        emit("    if (ev != nullptr) {")
        for o, column in observables.items():
            w = int(p.out_words[p.output_word[o]])
            bit = f"(int)((words[{w}] >> {int(p.output_bit[o])}) & 1)"
            r = int(ready[o])
            if r >= 0 and not p.splits[r]:
                # NOTE: a fair coin that no postselected output reads.
                value = "0.f"
            elif r < 0 or not exact[o]:
                value = f"({bit} ? -1.f : 1.f)"
            elif int(p.kind[r]) == MEASURE_DORMANT:
                value = "0.f"
            else:
                value = f"((({bit}) ^ b_{r}) ? -ev_{r} : ev_{r})"
            emit(f"        ev[{column}] = grave < {max(r, 0)} ? 0.f : {value};")
        emit("    }")
    emit("    return grave;")
    emit("}")
    if split:
        assert chunk is not None
        lines[first:] = [_split_amplitudes(line, chunk) for line in lines[first:]]
    return "\n\n".join([*tape.helpers, "\n".join(lines)])


_AMPLITUDE = re.compile(r"\ba\[(\d+)\]")


def _split_amplitudes(line: str, chunk: int) -> str:
    """Return ``line`` with each ``a[i]`` in its chunk's array."""

    def place(match: re.Match) -> str:
        c, i = divmod(int(match.group(1)), chunk)
        return f"a{c}[{i}]"

    return _AMPLITUDE.sub(place, line)
