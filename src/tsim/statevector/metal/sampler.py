"""``MetalShotSampler``: ``ShotSampler`` with the noise and the tapes on a Metal GPU.

A batch of shots is one command buffer of these kernels, which run one after
another:

    noise  draws every shot's noise into its words and screens it; the screened
           survivors go to a queue
    tape   runs every component on the queued shots, a SIMD group of ``2^tb``
           threads per shot with its amplitudes in registers; with a
           postselection mask it clears the rejected shots from the screen's
           bitmap of survivors
    list   lists the accepted shots in order from the bitmap
    pack   gathers the accepted rows, bit-packed (or draws the deferred noise
           on them first, ``noise_kept``)

Every buffer is shared with the CPU (unified memory), so nothing is copied:
the packing kernels write each batch's rows into the array that the call
returns, after the rows of the batches before it (a cursor in the GPU's memory
counts them). The host never waits between batches: it commits each batch's
command buffer as it encodes it, the batches reuse one set of buffers (so
Metal runs them one after another), and the GPU resets their counters itself.

The kernels are compiled once per postselection mask (Metal caches compiled
shaders across processes). Amplitudes are complex64, so a sample follows the
CPU executor's distribution to within float32 rounding of the outcome
probabilities; the noise and the outcome draws come from other streams than
the CPU's, so a seed's samples differ from the CPU's, and they depend on the
batch size.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tsim.noise.channels import ChannelSampler
from tsim.statevector.gpu import FORKED, tape
from tsim.statevector.gpu.noise import NoiseGroups, screen_pass_rate, split
from tsim.statevector.gpu.rows import select as _select
from tsim.statevector.metal import runtime
from tsim.statevector.metal.codegen import METAL, PRELUDE
from tsim.statevector.metal.noise import noise_tables, source as noise_source
from tsim.statevector.shots import ShotPlan, ShotSampler

BATCH_SHOTS = 1 << 20
"""Shots of one batch (one command buffer)."""

BATCH_BYTES = 256 << 20
"""Bytes of shot words of one batch, which bound its shots below ``BATCH_SHOTS``."""

NOISE_THREADS = 256
"""Threads per threadgroup of the noise kernel."""

TAPE_THREADS = 128
"""Threads per threadgroup of the tape kernel."""

MAX_CALL_SHOTS = 1 << 31
"""Most shots of one GPU call; larger calls run in parts (the counters are 32-bit)."""

_KERNELS = r"""
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
// counts each block's 1024 words, ``list_bits`` lists a block's shots after
// those of the blocks before it.
kernel void count_bits(
    device const uint* bits [[buffer(0)]], constant long& words [[buffer(1)]],
    device uint* totals [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]], uint bid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]],
    uint nwarps [[simdgroups_per_threadgroup]])
{{
    threadgroup uint warps[32];
    long i = (long)bid * 1024 + tid;
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
    for (uint b = tid; b < bid; b += 1024) earlier += totals[b];
    block_scan(earlier, earlier, warps, lane, warp, nwarps);
    long i = (long)bid * 1024 + tid;
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


@dataclass
class _Launch:
    """The device tables of one noise launch: its segments and records."""

    segments: list[runtime.Buffer]
    records: runtime.Buffer
    num_segments: int
    max_events: int


@dataclass
class _Module:
    """A plan's compiled kernels and its device constants."""

    library: runtime.Library
    tb: int
    has_mask: bool
    screen: runtime.Buffer
    plan: ShotPlan
    psm_words: np.ndarray
    passing: float
    """Estimated share of shots that pass the screen."""
    splits: dict[bool, tuple[_Launch, _Launch] | None] = field(default_factory=dict)


@dataclass
class _Slot:
    """The buffers that a call's batches reuse, one batch after another.

    ``counts`` holds the batch's screened survivors, its accepted shots and the
    call's largest normalization deviation (as float bits), then the call's
    cursor: the rows that its batches have packed, and the rows that its
    output has room for.
    """

    words: runtime.Buffer
    queue: runtime.Buffer
    accepted: runtime.Buffer
    counts: runtime.Buffer
    totals: runtime.Buffer
    index: runtime.Buffer
    values: runtime.Buffer | None = None


class MetalShotSampler:
    """Samples shot words of a statevector program on a Metal GPU.

    It keeps ``ShotSampler``'s interface (``sample``, ``direct_bools``,
    ``reference``, ``plan``, ``layout``); ``sample`` returns only the shots that
    pass postselection. ``sample_accepted`` can also return each observable's
    expectation.
    """

    supports = staticmethod(tape.supports)

    def __init__(
        self,
        host: ShotSampler,
        channels: ChannelSampler,
        num_detectors: int,
        seed: np.random.SeedSequence,
    ) -> None:
        """Upload the noise tables of ``host``'s shot words.

        ``host`` draws the reference samples, on the CPU.

        Raises:
            ImportError: If PyObjC's Metal is not installed.
            RuntimeError: If there is no Apple GPU.

        """
        runtime.check_simd_width()
        assert self.supports(host.layout)
        self._args = (host, channels, num_detectors, seed)
        # NOTE: a call's batches share the sampler's buffers, so calls from several
        # threads run one at a time.
        self._lock = threading.RLock()
        self._host = host
        layout = host.layout
        self._num_detectors = num_detectors
        self._observables = np.arange(num_detectors, layout.num_outputs)
        self._rng = np.random.default_rng(seed)
        self._channels = channels
        self._noise = noise_tables(channels, layout.noise, layout.num_words)
        self._overflow = runtime.upload(self._noise.overflow.view(np.uint8))
        self._all = self._launch(self._noise.groups)
        self._modules: dict[bytes, _Module] = {}
        self._constants: dict[bytes, runtime.Buffer] = {}
        self._slots: dict[tuple[int, bool], _Slot] = {}
        self._shares: dict[tuple, float] = {}

    device = "metal"
    """Where this sampler samples (``SamplerStats.device``)."""

    def __reduce__(self):
        """Pickle the CPU sampler and the inputs; the GPU state is rebuilt on load."""
        return _restore, (*self._args, self._rng.bit_generator.state)

    @property
    def layout(self):
        """Return the shot words' layout."""
        return self._host.layout

    def plan(self, mask: np.ndarray | None) -> ShotPlan:
        """Return the plan for a postselection ``mask``."""
        return self._host.plan(mask)

    def bytes_per_shot(self) -> int:
        """Return the host memory of one shot's words."""
        return 8 * self.layout.num_words

    def reference(self) -> tuple[np.ndarray, float]:
        """Return a noiseless sample of every output (on the CPU executor)."""
        return self._host.reference()

    # -- compilation ----------------------------------------------------------

    def _module(self, mask: np.ndarray | None) -> _Module:
        key = b"" if mask is None else np.asarray(mask, bool).tobytes()
        if key in self._modules:
            return self._modules[key]
        plan = self.plan(mask)
        layout = self.layout
        tb = tape.threads_per_shot(layout)
        T = 1 << tb
        parts = tape.tape_kernel(layout, plan, mask, self._num_detectors, tb, METAL)
        # NOTE: the rows keep every output, the postselected ones too, as the CPU's do.
        nb = -(-layout.num_outputs // 8)
        source = (
            PRELUDE
            + noise_source(self._noise, layout.output_words, layout.num_outputs)
            + parts.functions
            + _KERNELS.format(
                T1=T - 1,
                tb=tb,
                NW=layout.num_words,
                NO=max(1, len(self._observables)),
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
        library = runtime.Library(source)
        names = ["noise", "tape", "count_bits", "list_bits", "gather_rows"]
        names += ["gather_float", "reset", "advance", "unpack_rows"]
        if self._noise.shared_words:
            names.append("noise_kept")
        for name in names:
            library.pipeline(name)
        compiled = _Module(
            library=library,
            tb=tb,
            has_mask=mask is not None,
            screen=runtime.upload(plan.screen),
            plan=plan,
            psm_words=parts.psm_words,
            passing=screen_pass_rate(self._channels, layout, plan),
        )
        self._modules[key] = compiled
        return compiled

    def _launch(self, groups: NoiseGroups) -> _Launch:
        """Upload a noise launch's segments and records."""
        return _Launch(
            segments=[runtime.upload(a) for a in groups.segments],
            records=runtime.upload(groups.records.view(np.uint8)),
            num_segments=len(groups.segments[0]),
            max_events=groups.max_events,
        )

    def _split(
        self, module: _Module, expectations: bool
    ) -> tuple[_Launch, _Launch] | None:
        """Return a postselected batch's noise passes (screened, deferred), or None.

        As ``gpu.noise.split``; with ``expectations`` the observables' bits are
        read too.
        """
        # NOTE: rows too wide for threadgroup memory draw all their noise at once.
        if not module.has_mask or not self._noise.shared_words:
            return None
        if expectations not in module.splits:
            found = split(
                self._noise,
                self._channels,
                self.layout,
                module.plan,
                module.psm_words,
                self._num_detectors if expectations else None,
            )
            module.splits[expectations] = (
                None
                if found is None
                else (self._launch(found[0]), self._launch(found[1]))
            )
        return module.splits[expectations]

    # -- execution --------------------------------------------------------------

    def _constant(self, flip: np.ndarray | None) -> runtime.Buffer:
        """Return the outputs' constants XOR the reference bits ``flip``."""
        key = b"" if flip is None else np.asarray(flip, bool).tobytes()
        if key not in self._constants:
            self._constants[key] = runtime.upload(self._host._constant(flip))
        return self._constants[key]

    def _batch_shots(self, shots: int, batch_size: int | None) -> int:
        """Return the shots of a call's batches."""
        # NOTE: at most 2^16 words a row, so a batch stays below 512 MB, and the
        # kernels' 32-bit offsets within it do not wrap.
        cap = max(1 << 10, BATCH_BYTES // (8 * self.layout.num_words))
        return max(1, min(batch_size or BATCH_SHOTS, BATCH_SHOTS, cap, shots))

    def _slot(self, shots: int, expectations: bool) -> _Slot:
        """Return the buffers for batches of up to ``shots`` shots."""
        key = (shots, expectations)
        slot = self._slots.get(key)
        if slot is None:
            nw, no = self.layout.num_words, max(1, len(self._observables))
            slot = _Slot(
                words=runtime.buffer(shots * nw, np.uint64),
                queue=runtime.buffer(shots, np.uint32),
                accepted=runtime.buffer(-(-shots // 32), np.uint32),
                counts=runtime.buffer(8, np.uint32),
                totals=runtime.buffer(-(-shots // 32 // 1024) + 1, np.uint32),
                index=runtime.buffer(shots, np.uint32),
            )
            if expectations:
                slot.values = runtime.buffer(shots * no, np.float32)
            self._slots = {k: v for k, v in self._slots.items() if k[1] != expectations}
            self._slots[key] = slot
        return slot

    def _encode_batch(
        self,
        encoder: runtime.Encoder,
        module: _Module,
        slot: _Slot,
        shots: int,
        first_shot: int,
        constant: runtime.Buffer,
        expectations: bool,
        launch: _Launch,
    ) -> None:
        """Encode a batch's noise and tape kernels.

        Only the shots that pass the screen get their words written.
        """
        library = module.library
        words = slot.words
        encoder.dispatch(library.pipeline("reset"), [slot.counts], 1, 1)
        seed = np.uint64(self._rng.integers(0, 2**64, dtype=np.uint64))
        noise_seed = np.uint64(self._rng.integers(0, 2**64, dtype=np.uint64))
        S = self._noise.block_shots
        args = np.zeros(1, dtype=_NOISE_ARGS)
        args[0] = (
            shots,
            noise_seed,
            launch.num_segments,
            launch.max_events,
            int(module.has_mask),
        )
        encoder.dispatch(
            library.pipeline("noise"),
            [
                words,
                args,
                *launch.segments,
                launch.records,
                self._overflow,
                constant,
                module.screen,
                slot.queue,
                (slot.counts, 0),
                slot.accepted,
            ],
            -(-shots // S),
            NOISE_THREADS,
            threadgroup_bytes=8 * max(launch.max_events, 1),
        )
        if self.layout.components:
            tape_args = np.zeros(1, dtype=_TAPE_ARGS)
            flags = int(module.has_mask) | (2 if expectations else 0)
            tape_args[0] = (first_shot, seed, flags, 0)
            values = slot.values if slot.values is not None else slot.counts
            threads = shots << module.tb
            encoder.dispatch(
                library.pipeline("tape"),
                [
                    words,
                    tape_args,
                    slot.queue,
                    (slot.counts, 0),
                    slot.accepted,
                    (slot.counts, 4),
                    values,
                    (slot.counts, 8),
                ],
                -(-threads // TAPE_THREADS),
                TAPE_THREADS,
            )

    def _encode_pack(
        self,
        encoder: runtime.Encoder,
        module: _Module,
        slot: _Slot,
        shots: int,
        split: tuple[_Launch, _Launch] | None,
        expectations: bool,
        rows: runtime.Buffer,
        values: runtime.Buffer | None,
        fixed: tuple[runtime.Buffer, int],
    ) -> None:
        """Encode the listing and packing of a batch's accepted rows into ``rows``.

        They follow the rows of the call's earlier batches (at the cursor), and
        their expectations go to ``values``; ``fixed`` holds the batch's shots,
        which a call without a mask returns.
        """
        library = module.library
        S = self._noise.block_shots
        nb = -(-self.layout.num_outputs // 8)
        no = len(self._observables)
        if module.has_mask:
            words = -(-shots // 32)
            blocks = -(-words // 1024)
            encoder.dispatch(
                library.pipeline("count_bits"),
                [slot.accepted, np.int64(words), slot.totals],
                blocks,
                1024,
            )
            encoder.dispatch(
                library.pipeline("list_bits"),
                [slot.accepted, np.int64(words), slot.totals, slot.index],
                blocks,
                1024,
            )
            count = (slot.counts, 4) if self.layout.components else (slot.counts, 0)
        else:
            count = fixed
        cursor = (slot.counts, 12)
        blocks = -(-shots // S)
        if nb and split is not None:
            launch = split[1]
            kept_args = np.zeros(1, dtype=_KEPT_ARGS)
            kept_args[0] = (
                np.uint64(self._rng.integers(0, 2**64, dtype=np.uint64)),
                launch.num_segments,
                launch.max_events,
            )
            encoder.dispatch(
                library.pipeline("noise_kept"),
                [
                    slot.words,
                    kept_args,
                    *launch.segments,
                    launch.records,
                    self._overflow,
                    count,
                    slot.index,
                    rows,
                    cursor,
                ],
                blocks,
                256,
                threadgroup_bytes=max(8 * launch.max_events, S * nb),
            )
        elif nb:
            threads = min(-(-shots * nb // 4) + 1, 1 << 20)
            encoder.dispatch(
                library.pipeline("gather_rows"),
                [
                    slot.words,
                    count,
                    slot.index,
                    np.int32(module.has_mask),
                    rows,
                    cursor,
                ],
                -(-threads // 256),
                256,
            )
        if expectations and no and self.layout.components:
            assert slot.values is not None and values is not None
            encoder.dispatch(
                library.pipeline("gather_float"),
                [
                    slot.values,
                    count,
                    np.int32(no),
                    slot.index,
                    values,
                    cursor,
                    np.int32(module.has_mask),
                ],
                256,
                256,
            )
        encoder.dispatch(library.pipeline("advance"), [cursor, count], 1, 1)

    def sample(
        self,
        shots: int,
        columns: list[np.ndarray],
        *,
        bit_packed: bool,
        batch_size: int | None = None,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], float]:
        """Return the output bits ``columns`` of ``shots`` shots and the largest norm deviation.

        As ``ShotSampler.sample``: with a ``mask`` only the shots that pass
        postselection are returned. ``batch_size`` bounds the shots of a batch.
        Out of GPU memory, and in a process forked after its parent used the
        GPU, the call warns and samples on the CPU.
        """

        def on_cpu():
            return self._host.sample(
                shots,
                columns,
                bit_packed=bit_packed,
                batch_size=batch_size,
                mask=mask,
                flip=flip,
            )

        if _forked():
            return on_cpu()
        try:
            rows, _, deviation = self.sample_accepted(
                shots,
                mask=mask,
                flip=flip,
                batch_size=batch_size,
                bools=not bit_packed,
            )
        except MemoryError as error:
            _warn_out_of_memory(error)
            return on_cpu()
        if not bit_packed:
            return [_columns(rows, c) for c in columns], deviation
        num_outputs = self.layout.num_outputs
        return [_select(rows, c, num_outputs, bit_packed) for c in columns], deviation

    def _capacity(self, key: tuple, module: _Module, shots: int) -> int:
        """Return the rows to make room for: the expected accepted shots and a margin.

        The share comes from the previous call of ``key``, or else the screen's
        estimate (which the tapes' postselection only lowers).
        """
        share = self._shares.get(key, module.passing if module.has_mask else 1.0)
        expected = shots * min(1.0, share)
        return min(shots, int(1.05 * expected + 8 * np.sqrt(expected) + 4096))

    def _run_accepted(
        self,
        module: _Module,
        slot: _Slot,
        shots: int,
        batch: int,
        constant: runtime.Buffer,
        expectations: bool,
        split: tuple[_Launch, _Launch] | None,
        rows: runtime.Buffer,
        values: runtime.Buffer | None,
        capacity: int,
    ) -> None:
        """Run a call's batches into ``rows`` (and ``values``) of ``capacity`` rows."""
        launch = split[0] if split else self._all
        starts = range(0, shots, batch)
        fixed = runtime.buffer(len(starts), np.uint32)
        slot.counts.array[:] = 0
        slot.counts.array[4] = capacity
        commands = []
        try:
            for k, start in enumerate(starts):
                n = min(batch, shots - start)
                fixed.array[k] = n
                encoder = runtime.Encoder()
                self._encode_batch(
                    encoder, module, slot, n, start, constant, expectations, launch
                )
                self._encode_pack(
                    encoder,
                    module,
                    slot,
                    n,
                    split,
                    expectations,
                    rows,
                    values,
                    (fixed, 4 * k),
                )
                commands.append(encoder.commit())
        except BaseException:
            self._settle(commands)
            raise
        self._wait_all(commands)

    def _wait_all(self, commands: list[Any]) -> None:
        """Wait for a call's command buffers; on any error, let them all finish first.

        Raises:
            MemoryError: If one ran out of GPU memory.
            RuntimeError: If one failed otherwise.

        """
        try:
            for command in commands:
                runtime.wait(command)
        except BaseException:
            self._settle(commands)
            raise

    def _settle(self, commands: list[Any]) -> None:
        """Drop the batch buffers and wait for ``commands``, through further interrupts.

        A call that fails or is interrupted leaves its batches on the GPU; they
        must finish before the next call reuses (or the pool hands out) a buffer
        that they write.
        """
        self._slots.clear()
        for command in commands:
            while True:
                try:
                    command.waitUntilCompleted()
                    break
                except KeyboardInterrupt:
                    continue

    def sample_accepted(
        self,
        shots: int,
        *,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
        expectations: bool = False,
        batch_size: int | None = None,
        bools: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """Sample ``shots`` shots; return only those that pass postselection.

        A shot passes when none of the ``mask``'s detectors is set (every shot
        passes without a mask). Rows keep their shots' order and hold every output;
        with ``bools``, one boolean per output, unpacked on the GPU.

        Returns:
            ``(outputs, expectations, deviation)``: the passing shots' outputs,
            bit-packed as ``np.packbits(..., bitorder="little")`` would (``uint8``,
            in shared memory), with ``expectations`` each observable's
            expectation of ``(-1)^observable`` given the shot's history before
            the measurement that settles it, or its sampled ``±1`` when a
            postselected output of its component is tested at or after that
            measurement (else None), and the largest normalization deviation
            beyond float32 rounding.

        Raises:
            MemoryError: If the GPU runs out of memory.

        """
        if shots > MAX_CALL_SHOTS:
            parts = [
                self.sample_accepted(
                    min(MAX_CALL_SHOTS, shots - start),
                    mask=mask,
                    flip=flip,
                    expectations=expectations,
                    batch_size=batch_size,
                    bools=bools,
                )
                for start in range(0, shots, MAX_CALL_SHOTS)
            ]
            found = [p[1] for p in parts if p[1] is not None]
            values = np.concatenate(found) if expectations else None
            return (
                np.concatenate([p[0] for p in parts]),
                values,
                max(p[2] for p in parts),
            )
        with self._lock:
            return self._sample_call(
                shots,
                mask=mask,
                flip=flip,
                expectations=expectations,
                batch_size=batch_size,
                bools=bools,
            )

    def _sample_call(
        self,
        shots: int,
        *,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
        expectations: bool = False,
        batch_size: int | None = None,
        bools: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """One call of ``sample_accepted``, of at most ``MAX_CALL_SHOTS`` shots."""
        module = self._module(mask)
        nb, no = -(-self.layout.num_outputs // 8), len(self._observables)
        if not shots:
            values = np.zeros((0, no), np.float32) if expectations else None
            if bools:
                return np.zeros((0, self.layout.num_outputs), bool), values, 0.0
            return np.zeros((0, nb), np.uint8), values, 0.0
        constant = self._constant(flip)
        split = self._split(module, expectations)
        batch = self._batch_shots(shots, batch_size)
        slot = self._slot(batch, expectations)
        key = (module.plan, expectations)
        state = self._rng.bit_generator.state
        capacity = self._capacity(key, module, shots)
        while True:
            # NOTE: a command buffer makes all of its buffers resident, so the
            # output has room for the expected rows only; a call that overflows
            # it runs again, on the same streams, with room for every shot.
            rows = runtime.buffer(capacity * nb, np.uint8, recycle=True)
            values = (
                runtime.buffer(capacity * no, np.float32, recycle=True)
                if expectations
                else None
            )
            self._run_accepted(
                module,
                slot,
                shots,
                batch,
                constant,
                expectations,
                split,
                rows,
                values,
                capacity,
            )
            total = int(slot.counts.array[3])
            if total <= capacity:
                break
            self._rng.bit_generator.state = state
            capacity = shots
        self._shares[key] = total / max(shots, 1)
        # NOTE: fast math records a NaN total as NaN (not inf), which must warn.
        deviation = tape.deviation(slot.counts.array[2])
        packed = rows.array[: total * nb].reshape(total, nb)
        expected = None
        if values is not None:
            expected = values.array[: total * no].reshape(total, no)
            if not self.layout.components:
                expected[:] = 0
            self._direct_expectations(packed, expected)
        if bools:
            return self._unpack(module, rows, total), expected, deviation
        return packed, expected, deviation

    def _unpack(self, module: _Module, rows: runtime.Buffer, total: int) -> np.ndarray:
        """Return the first ``total`` packed ``rows`` as booleans, unpacked on the GPU."""
        num_outputs = self.layout.num_outputs
        nb = -(-num_outputs // 8)
        groups = -(-num_outputs // 4)
        if total * groups >= 1 << 32:
            packed = rows.array[: total * nb].reshape(total, nb)
            return _select(packed, np.arange(num_outputs), num_outputs, False)
        out = runtime.buffer(total * num_outputs, np.uint8, recycle=True)
        if total and num_outputs:
            threads = min(total * groups, 1 << 20)
            encoder = runtime.Encoder()
            encoder.dispatch(
                module.library.pipeline("unpack_rows"),
                [rows, out, np.uint32(total)],
                -(-threads // 256),
                256,
            )
            self._wait_all([encoder.commit()])
        return out.array[: total * num_outputs].reshape(total, num_outputs).view(bool)

    def _direct_expectations(self, rows: np.ndarray, values: np.ndarray) -> None:
        """Set the expectations of observables outside the components: ``±1``."""
        layout = self.layout
        in_components = {i for c in layout.components for i in c.output_indices}
        for column, index in enumerate(self._observables):
            if int(index) in in_components:
                continue
            k = int(index)
            bit = (rows[:, k >> 3] >> np.uint8(k & 7)) & np.uint8(1)
            values[:, column] = 1.0 - 2.0 * bit.astype(np.float32)

    def direct_bools(self, shots: int, flip: np.ndarray | None = None) -> np.ndarray:
        """Return every output bit of a program without components, as booleans."""
        if _forked():
            return self._host.direct_bools(shots, flip)
        try:
            rows, _, _ = self.sample_accepted(shots, flip=flip, bools=True)
        except MemoryError as error:
            _warn_out_of_memory(error)
            return self._host.direct_bools(shots, flip)
        return rows


def _forked() -> bool:
    """Return whether this process was forked after Metal's setup, with a warning."""
    if not runtime.forked_after_init():
        return False
    warnings.warn(FORKED, stacklevel=5)
    return True


def _warn_out_of_memory(error: MemoryError) -> None:
    warnings.warn(
        f"{error}; sampling on the CPU. Fewer shots per call fit.", stacklevel=5
    )


def _restore(
    host, channels, num_detectors, seed, state
) -> MetalShotSampler | ShotSampler:
    """Rebuild a pickled ``MetalShotSampler``, or its CPU sampler where Metal is missing."""
    if not runtime.available():
        return host
    shots = MetalShotSampler(host, channels, num_detectors, seed)
    shots._rng.bit_generator.state = state
    return shots


def _columns(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Return the outputs ``columns`` of boolean rows of every output, C-contiguous."""
    count = len(columns)
    if count == rows.shape[1] and np.array_equal(columns, np.arange(count)):
        return rows
    return np.ascontiguousarray(rows[:, columns])


_NOISE_ARGS = np.dtype(
    [
        ("shots", "<i8"),
        ("seed", "<u8"),
        ("num_segments", "<i4"),
        ("max_events", "<i4"),
        ("has_survivors", "<i4"),
    ],
    align=True,
)
_TAPE_ARGS = np.dtype(
    [("first_shot", "<i8"), ("seed", "<u8"), ("flags", "<i4"), ("pad", "<i4")],
    align=True,
)
_KEPT_ARGS = np.dtype(
    [("seed", "<u8"), ("num_segments", "<i4"), ("max_events", "<i4")], align=True
)
