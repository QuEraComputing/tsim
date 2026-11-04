"""``CudaShotSampler``: ``ShotSampler`` with the noise and the tapes on a CUDA GPU.

A batch of shots runs as these kernels on one stream:

    noise  draws every shot's noise into its words and screens it (``noise``);
           the screened survivors go to a queue
    tape   runs every component on the queued shots, a group of ``2^tb`` threads
           per shot with its amplitudes in registers (``codegen``); with a
           postselection mask it clears the rejected shots from the screen's
           bitmap of survivors, and with
           expectations it writes each observable's expectation
    kept   under postselection, draws the noise that cannot change whether a
           shot is kept on the accepted shots only (``noise_kept``), when that
           noise is at least ``DEFER_SHARE`` of the faults
    list   lists the accepted shots in order from the bitmap (``count_bits``,
           ``list_bits``)
    select gathers the rows (or the accepted rows) for the copy to the host

``sample_accepted`` runs a call as chunks of shots on two streams, so that a
chunk's copy to the host overlaps the next chunk's kernels.

The kernels are compiled with NVRTC once per postselection mask (CuPy caches
them on disk). Amplitudes are complex64, so a sample follows the CPU executor's
distribution to within float32 rounding of the outcome probabilities; the noise
and the outcome draws come from other streams than the CPU's, so a seed's
samples differ from the CPU's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tsim.core.types import CompiledProgram
from tsim.noise.channels import ChannelSampler
from tsim.statevector.cuda import runtime
from tsim.statevector.cuda.codegen import PRELUDE, hex64, tape_function
from tsim.statevector.cuda.noise import (
    NoiseGroups,
    deferred_channels,
    noise_tables,
    pass_rate,
)
from tsim.statevector.program import STABILIZER_RANK_HINT
from tsim.statevector.shared import pack_bits
from tsim.statevector.shots import ShotPlan, ShotSampler, bool_outputs, read_outputs

MAX_WIDTH = 11
"""Widest component the CUDA executor runs: 32 threads of 64 amplitudes."""

BATCH_SHOTS = 1 << 22
"""Shots of one batch on the device."""

DEVIATION_TOLERANCE = 1e-3
"""Normalization deviation that float32 amplitudes tolerate without a warning."""

DEFER_SHARE = 0.25
"""Share of the expected faults that postselection cannot see, from which they
are drawn on the accepted shots only, after the tapes."""

CHUNK_BYTES = 6 << 20
"""Bytes of kept rows per chunk that a call's chunks aim at: each chunk's copy to
the host overlaps the next chunk's kernels on the other stream."""

MAX_CHUNKS = 8
"""Most chunks of a batch, so that a chunk's fixed costs stay small."""

_KERNELS = r"""
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
        float ev[{NE}];
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
// ``index``: ``count_bits`` counts each block's 1024 words, ``list_bits`` lists
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

extern "C" __global__ void __launch_bounds__(1024) count_bits(
    const unsigned* __restrict__ bits, long long words, unsigned* __restrict__ totals) {{
    long long i = (long long)blockIdx.x * 1024 + threadIdx.x;
    unsigned total;
    block_scan(i < words ? __popc(bits[i]) : 0, &total);
    if (threadIdx.x == 0) totals[blockIdx.x] = total;
}}

extern "C" __global__ void __launch_bounds__(1024) list_bits(
    const unsigned* __restrict__ bits, long long words, const unsigned* __restrict__ totals,
    unsigned* __restrict__ index) {{
    unsigned earlier = 0, total;
    for (int b = threadIdx.x; b < blockIdx.x; b += blockDim.x) earlier += totals[b];
    block_scan(earlier, &earlier);
    long long i = (long long)blockIdx.x * 1024 + threadIdx.x;
    unsigned w = i < words ? bits[i] : 0;
    unsigned at = earlier + block_scan(__popc(w), &total);
    for (; w; w &= w - 1) index[at++] = (unsigned)(32 * i + __ffs(w) - 1);
}}

// The returned outputs of rows ``index[j]`` (or every row), bit-packed densely
// (``pack_rows``): a block packs {S} rows' output words.
extern "C" __global__ void __launch_bounds__(256) gather_columns(
    const u64* __restrict__ rows, long long n, const unsigned* __restrict__ index,
    const uint2* __restrict__ pieces, const unsigned* __restrict__ start,
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
    pack_rows(sw, ns, pieces, start, stage, dst + base * {NB});
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


@dataclass
class _Launch:
    """The device tables of one noise launch: its segments and records."""

    args: list[Any]
    num_segments: int
    max_events: int


@dataclass
class _Module:
    """A plan's compiled kernels and its device constants."""

    noise: Any
    noise_kept: Any
    tape: Any
    gather_columns: Any
    gather_float: Any
    publish: Any
    count_bits: Any
    list_bits: Any
    screen: Any
    kept: Any
    tape_blocks: int
    has_mask: bool
    columns: np.ndarray
    """The outputs that ``sample_accepted`` returns: those outside the mask."""
    pieces: Any
    piece_start: Any
    plan: ShotPlan
    psm_words: np.ndarray
    passing: float
    """Estimated share of shots that pass the screen."""
    splits: dict[bool, tuple[_Launch, _Launch] | None]
    """Each ``expectations`` value's noise passes (screened, deferred), or None."""


def _pinned_counts(cp) -> tuple[np.ndarray, Any]:
    """Return three pinned host counters and a device array over the same memory."""
    import cupyx

    host = cupyx.empty_pinned((3,), dtype=np.uint32)
    # NOTE: with unified addressing a pinned host pointer is also a device pointer.
    memory = cp.cuda.UnownedMemory(host.ctypes.data, host.nbytes, host)
    device = cp.ndarray((3,), np.uint32, cp.cuda.MemoryPointer(memory, 0))
    return host, device


def _pieces(columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return how ``pack_rows`` packs the outputs ``columns``: ``(pieces, start)``.

    Output word ``k`` holds ``columns[64 k : 64 k + 64]``; its pieces are
    ``pieces[start[k] : start[k + 1]]``, each a run of them that is consecutive
    in one shot word, as ``(word, bit | (length - 1) << 6 | position << 12)``.
    """
    pieces, start = [], [0]
    for k in range(0, len(columns), 64):
        word = columns[k : k + 64].tolist()
        i = 0
        while i < len(word):
            j = i
            while (
                j + 1 < len(word)
                and word[j + 1] == word[j] + 1
                and word[j + 1] >> 6 == word[i] >> 6
            ):
                j += 1
            c = word[i]
            pieces.append((c >> 6, (c & 63) | (j - i) << 6 | i << 12))
            i = j + 1
        start.append(len(pieces))
    return np.asarray(pieces or [(0, 0)], np.uint32), np.asarray(start, np.uint32)


class _HostRows:
    """A call's passing rows (and expectations) in pinned host memory, chunk by chunk."""

    def __init__(self, nb: int, no: int | None) -> None:
        self.nb, self.no = nb, no
        self.capacity, self.total = 0, 0
        self.rows: np.ndarray | None = None
        self.values: np.ndarray | None = None

    def allocate(self, passed: int, of: int, shots: int) -> None:
        """Reserve the rows of a call of ``shots``, ``passed`` of its first ``of`` passing."""
        rest = shots - of
        share = passed / of
        expected = rest * share
        self._resize(
            min(shots, passed + int(1.05 * expected + 6 * np.sqrt(expected) + 256))
        )

    def grow(self, capacity: int) -> None:
        """Make room for ``capacity`` rows, keeping the copied ones (no copy may be pending)."""
        rows, values, total = self.rows, self.values, self.total
        self._resize(capacity)
        assert self.rows is not None and rows is not None
        self.rows[:total] = rows[:total]
        if values is not None and self.values is not None:
            self.values[:total] = values[:total]

    def _resize(self, capacity: int) -> None:
        import cupyx

        self.capacity = capacity
        self.rows = cupyx.empty_pinned((capacity, self.nb), dtype=np.uint8)
        if self.no is not None:
            self.values = cupyx.empty_pinned((capacity, self.no), dtype=np.float32)

    def take(self, count: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the next ``count`` rows (and expectations) to fill."""
        assert self.rows is not None and self.total + count <= self.capacity
        lo, self.total = self.total, self.total + count
        values = None if self.values is None else self.values[lo : self.total]
        return self.rows[lo : self.total], values

    def result(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the filled rows and expectations."""
        if self.rows is None:
            self._resize(0)
        assert self.rows is not None
        values = None if self.values is None else self.values[: self.total]
        return self.rows[: self.total], values


class CudaShotSampler:
    """Samples shot words of a statevector program on a CUDA GPU.

    It keeps ``ShotSampler``'s interface (``sample``, ``direct_bools``,
    ``reference``, ``plan``, ``layout``) and adds ``sample_accepted``, which
    copies only the shots that pass postselection and can return each
    observable's expectation.
    """

    def __init__(
        self,
        program: CompiledProgram,
        channels: ChannelSampler,
        num_detectors: int,
        seed: np.random.SeedSequence,
    ) -> None:
        """Lay out the program's shot words and upload the noise tables.

        Raises:
            ImportError: If CuPy is not installed.
            ValueError: If a component is wider than ``MAX_WIDTH``.

        """
        self._cp = runtime.cupy()
        cp = self._cp
        host_seed, seed = seed.spawn(2)
        self._host = ShotSampler(program, channels, num_detectors, host_seed)
        layout = self._host.layout
        for component in layout.components:
            if component.width > MAX_WIDTH:
                raise ValueError(
                    f"device='cuda' runs components of up to {MAX_WIDTH} active qubits; "
                    f"the component with outputs {list(component.output_indices)} has "
                    f"{component.width}. Use device='cpu'. {STABILIZER_RANK_HINT}"
                )
        self._num_detectors = num_detectors
        self._observables = np.arange(num_detectors, layout.num_outputs)
        self._rng = np.random.default_rng(seed)
        self._channels = channels
        self._noise = noise_tables(channels, layout.noise, layout.num_words)
        n = self._noise
        self._record_args = [cp.asarray(n.overflow.view(np.uint8))]
        self._all = self._launch(n.groups)
        self._modules: dict[bytes, _Module] = {}
        self._buffers: dict[str, Any] = {}
        self._views: dict[tuple[str, tuple[int, ...]], Any] = {}
        self._constants: dict[bytes, Any] = {}
        # NOTE: each buffer slot has a kernel stream and a copy stream, so that a
        # slot's next kernels need not wait for its copy to the host.
        # NOTE: a slot's noise kernel runs on a stream of the least priority, so the
        # other slot's tape and packing, which the next copy waits for, go first.
        self._streams = [
            cp.cuda.Stream(non_blocking=True, priority=-1) for _ in range(2)
        ]
        self._noise_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
        self._copies = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
        self._published = [_pinned_counts(cp) for _ in range(2)]

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

    def _threads_per_shot(self) -> int:
        """Return log2 of the threads per shot: one per 32 amplitudes, at most 32."""
        width = max((c.width for c in self.layout.components), default=0)
        return min(5, max(0, width - 5))

    def _module(self, mask: np.ndarray | None) -> _Module:
        key = b"" if mask is None else np.asarray(mask, bool).tobytes()
        if key in self._modules:
            return self._modules[key]
        cp = self._cp
        plan = self.plan(mask)
        layout = self.layout
        tb = self._threads_per_shot()
        T = 1 << tb
        block = 128
        observables: list[dict[int, int]] = []
        for component in layout.components:
            slots = {
                o: int(np.searchsorted(self._observables, index))
                for o, index in enumerate(component.output_indices)
                if index >= self._num_detectors
            }
            observables.append(slots)
        functions = "\n\n".join(
            tape_function(program, i, tb, observables[i])
            for i, program in enumerate(plan.programs)
        )
        calls = "\n".join(
            f"        grave = min(grave, component_{i}(w, shot, seed_{i}, t, gmask, ev, deviation));"
            for i in range(len(plan.programs))
        )
        gmask = (
            "0xffffffffu"
            if T == 32
            else f"((1u << {T}) - 1) << (threadIdx.x & 31 & ~{T - 1})"
        )
        bits = np.zeros(64 * layout.output_words, dtype=bool)
        if mask is not None:
            bits[: len(mask)] = mask
        psm_words = pack_bits(bits, layout.output_words)
        columns = np.flatnonzero(~bits[: layout.num_outputs]).astype(np.int32)
        # NOTE: the tapes write only their outputs' words, and test the
        # postselected outputs among them.
        written = sorted({int(w) for p in plan.programs for w in p.out_words})
        stores = "\n".join(f"            row[{w}] = w[{w}];" for w in written)
        hits = " | ".join(
            f"(w[{w}] & {hex64(psm_words[w])})" for w in written if psm_words[w]
        )
        source = (
            PRELUDE
            + self._noise.source(layout.output_words, -(-len(columns) // 8))
            + functions
            + _KERNELS.format(
                block=block,
                T1=T - 1,
                tb=tb,
                gmask=gmask,
                NW=layout.num_words,
                OW=layout.output_words,
                NO=max(1, len(self._observables)),
                NE=max(1, len(self._observables)),
                S=self._noise.block_shots,
                NB=max(-(-len(columns) // 8), 1),
                calls=calls,
                # NOTE: each component's stream, from the call's seed.
                seeds="\n".join(
                    f"    const u64 seed_{i} = mix64(seed + {i + 1} * GOLDEN);"
                    for i in range(len(plan.programs))
                ),
                stores=stores,
                hits=hits or "0",
            )
        )
        module = self._compile(source)
        tape = module.get_function("tape")
        blocks_per_sm = 4
        if hasattr(cp.cuda.runtime, "occupancyMaxActiveBlocksPerMultiprocessor"):
            blocks_per_sm = max(
                1,
                cp.cuda.runtime.occupancyMaxActiveBlocksPerMultiprocessor(
                    tape.ptr, block, 0
                ),
            )
        pieces, piece_start = _pieces(columns)
        compiled = _Module(
            noise=module.get_function("noise"),
            noise_kept=module.get_function("noise_kept"),
            tape=tape,
            gather_columns=module.get_function("gather_columns"),
            gather_float=module.get_function("gather_float"),
            publish=module.get_function("publish"),
            count_bits=module.get_function("count_bits"),
            list_bits=module.get_function("list_bits"),
            screen=cp.asarray(plan.screen),
            kept=cp.asarray(plan.kept),
            tape_blocks=runtime.multiprocessors() * blocks_per_sm,
            has_mask=mask is not None,
            columns=columns,
            pieces=cp.asarray(pieces),
            piece_start=cp.asarray(piece_start),
            plan=plan,
            psm_words=psm_words,
            passing=self._passing(plan),
            splits={},
        )
        self._modules[key] = compiled
        return compiled

    def _passing(self, plan: ShotPlan) -> float:
        """Return the estimated share of shots that pass the screen of ``plan``."""
        layout = self.layout
        screen = np.zeros(layout.num_words, dtype=np.uint64)
        screen[: layout.output_words] = plan.screen
        return pass_rate(self._channels, layout.noise, screen) if screen.any() else 1.0

    def _launch(self, groups: NoiseGroups) -> _Launch:
        """Upload a noise launch's segments and records."""
        cp = self._cp
        return _Launch(
            args=[
                *(cp.asarray(a) for a in groups.segments),
                cp.asarray(groups.records.view(np.uint8)),
            ],
            num_segments=len(groups.segments[0]),
            max_events=groups.max_events,
        )

    def _split(
        self, module: _Module, expectations: bool
    ) -> tuple[_Launch, _Launch] | None:
        """Return a postselected batch's noise passes (screened, deferred), or None.

        The deferred channels flip none of the bits that the screen, the tapes or
        the postselection read: the postselected outputs, the keys, and with
        ``expectations`` the components' observables. Each key word is read
        whole, so the deferred channels flip output words only.
        """
        if not module.has_mask:
            return None
        if expectations in module.splits:
            return module.splits[expectations]
        layout = self.layout
        read = np.zeros(layout.num_words, dtype=np.uint64)
        read[: layout.output_words] = module.psm_words | module.plan.screen
        for program in module.plan.programs:
            start = program.key_start
            read[start : start + program.key_words] = ~np.uint64(0)
            np.bitwise_or.at(read, program.ps_word, program.ps_mask.astype(np.uint64))
        if expectations:
            bits = np.zeros(64 * layout.output_words, dtype=bool)
            for component in layout.components:
                for index in component.output_indices:
                    if index >= self._num_detectors:
                        bits[index] = True
            read[: layout.output_words] |= pack_bits(bits, layout.output_words)
        deferred = deferred_channels(self._channels, layout.noise, read)
        later = self._noise.subset(deferred)
        split = None
        if later.faults > 0 and later.faults >= DEFER_SHARE * self._noise.groups.faults:
            split = (self._launch(self._noise.subset(~deferred)), self._launch(later))
        module.splits[expectations] = split
        return split

    def _compile(self, source: str):
        """Compile with NVRTC; ptxas -O3 fails on some shuffled tapes, so lower levels follow."""
        cp = self._cp
        for level in ("3", "2", "1"):
            options = (
                runtime.arch(),
                "-std=c++17",
                "-diag-suppress=177",
                f"-Xptxas=-O{level}",
            )
            try:
                module = cp.RawModule(code=source, options=options)
                module.get_function("tape")
                return module
            except cp.cuda.compiler.CompileException:
                if level == "1":
                    raise
        raise AssertionError("unreachable")

    # -- execution --------------------------------------------------------------

    def _buffer(self, name: str, shape: tuple[int, ...], dtype) -> Any:
        """Return a device buffer of at least ``shape``, reused across calls."""
        view = self._views.get((name, shape))
        if view is not None and view.dtype == np.dtype(dtype):
            return view
        size = int(np.prod(shape))
        held = self._buffers.get(name)
        if held is None or held.size < size or held.dtype != np.dtype(dtype):
            held = self._cp.empty(size, dtype=dtype)
            self._buffers[name] = held
            # NOTE: views of the replaced buffer must not be handed out again.
            self._views = {k: v for k, v in self._views.items() if k[0] != name}
        view = held[:size].reshape(shape)
        self._views[(name, shape)] = view
        return view

    def _constant(self, flip: np.ndarray | None) -> Any:
        """Return the outputs' constants XOR the reference bits ``flip``, on the device."""
        key = b"" if flip is None else np.asarray(flip, bool).tobytes()
        if key not in self._constants:
            self._constants[key] = self._cp.asarray(self._host._constant(flip))
        return self._constants[key]

    def _run_batch(
        self,
        module: _Module,
        shots: int,
        first_shot: int,
        constant: Any,
        expectations: bool,
        launch: _Launch,
        every_row: bool,
        slot: int = 0,
        noise_stream: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Launch one batch on the current stream: ``(words, accepted, counts, values)``.

        ``accepted`` is the bitmap of the accepted shots (with a mask).

        ``launch`` holds the channels that the noise kernel draws; without
        ``every_row`` only the rows of the shots that pass the screen are written.
        ``counts`` holds the screened survivors, the accepted shots and the norm
        deviation's bits. ``slot`` names the device buffers; ``noise_stream``, if
        given, runs the noise kernel, after the current stream's work.
        """
        cp = self._cp
        layout = self.layout
        nw, no = layout.num_words, max(1, len(self._observables))
        words = self._buffer(f"words{slot}", (shots, nw), np.uint64)
        queue = self._buffer(f"queue{slot}", (shots,), np.uint32)
        accepted = self._buffer(f"accepted{slot}", (-(-shots // 32),), np.uint32)
        counts = self._buffer(f"counts{slot}", (3,), np.uint32)
        values = (
            self._buffer(f"values{slot}", (shots, no), np.float32)
            if expectations
            else None
        )
        current = cp.cuda.get_current_stream()
        if noise_stream is not None:
            noise_stream.wait_event(current.record())
            noise_stream.use()
        counts.fill(0)
        seed = np.uint64(self._rng.integers(0, 2**64, dtype=np.uint64))
        noise_seed = np.uint64(self._rng.integers(0, 2**64, dtype=np.uint64))
        S = self._noise.block_shots
        module.noise(
            ((shots + S - 1) // S,),
            (256,),
            (
                words,
                np.int64(shots),
                noise_seed,
                np.int32(launch.num_segments),
                np.int32(launch.max_events),
                *launch.args,
                *self._record_args,
                constant,
                module.screen,
                module.kept,
                queue,
                counts[0:1],
                accepted if module.has_mask else np.uint64(0),
                np.int32(every_row),
            ),
            shared_mem=8 * launch.max_events,
        )
        if noise_stream is not None:
            current.wait_event(noise_stream.record())
            current.use()
        if layout.components:
            module.tape(
                (module.tape_blocks,),
                (128,),
                (
                    words,
                    np.int64(first_shot),
                    seed,
                    queue,
                    counts[0:1],
                    accepted if module.has_mask else np.uint64(0),
                    counts[1:2],
                    values if values is not None else np.uint64(0),
                    counts[2:3],
                ),
            )
        elif module.has_mask:
            # NOTE: without components the screened survivors are the accepted shots.
            counts[1:2] = counts[0:1]
        return words, accepted, counts, values

    def _pack(
        self,
        module: _Module,
        split: tuple[_Launch, _Launch] | None,
        words,
        count: int,
        index,
    ) -> Any:
        """Return the ``count`` rows ``index`` (or all) on the device, as copied to the host.

        Their outputs ``module.columns``, bit-packed; with ``split`` the deferred
        noise kernel adds the deferred channels' faults first.
        """
        S = self._noise.block_shots
        m = len(module.columns)
        nb = -(-m // 8)
        # NOTE: a new array per chunk, so that refilling it never waits for a copy.
        dst = self._cp.empty((max(count, 1), nb), dtype=np.uint8)
        if not count or not nb:
            return dst
        blocks = (count + S - 1) // S
        if split is not None:
            launch = split[1]
            seed = np.uint64(self._rng.integers(0, 2**64, dtype=np.uint64))
            module.noise_kept(
                (blocks,),
                (256,),
                (
                    words,
                    np.int64(count),
                    index,
                    seed,
                    np.int32(launch.num_segments),
                    np.int32(launch.max_events),
                    *launch.args,
                    *self._record_args,
                    dst,
                    module.pieces,
                    module.piece_start,
                ),
                shared_mem=max(8 * launch.max_events, S * nb),
            )
        else:
            module.gather_columns(
                (blocks,),
                (256,),
                (
                    words,
                    np.int64(count),
                    index if index is not None else np.uint64(0),
                    module.pieces,
                    module.piece_start,
                    dst,
                ),
                shared_mem=S * nb,
            )
        return dst

    def _list(self, module: _Module, bits, shots: int, count: int, slot: int) -> Any:
        """Return the ``count`` shots of the bitmap ``bits`` of ``shots`` shots, in order."""
        words = -(-shots // 32)
        blocks = -(-words // 1024)
        totals = self._buffer(f"totals{slot}", (blocks,), np.uint32)
        index = self._buffer(f"index{slot}", (max(count, 1),), np.uint32)
        module.count_bits((blocks,), (1024,), (bits, np.int64(words), totals))
        module.list_bits((blocks,), (1024,), (bits, np.int64(words), totals, index))
        return index[:count]

    def _deviation(self, counts: np.ndarray) -> float:
        """Return a batch's normalization deviation beyond float32 rounding, else 0."""
        bits = np.asarray(counts[2:3], dtype=np.uint32)
        deviation = float(bits.view(np.float32)[0])
        return deviation if deviation > DEVIATION_TOLERANCE else 0.0

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

        As ``ShotSampler.sample``; ``batch_size`` bounds the shots of a device batch.
        """
        words, deviation = self._sample_words(shots, batch_size, mask, flip)
        rejected = self.plan(mask).rejected
        if rejected.any():
            words = words[~(words[:, : len(rejected)] & rejected).any(axis=1)]
        num_outputs = self.layout.num_outputs
        arrays = [
            read_outputs(words, c, num_outputs, bit_packed=bit_packed) for c in columns
        ]
        return arrays, deviation

    def _sample_words(
        self,
        shots: int,
        batch_size: int | None,
        mask: np.ndarray | None,
        flip: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        """Return the shot words of ``shots`` shots and the largest norm deviation."""
        module = self._module(mask)
        layout = self.layout
        out = np.empty((shots, layout.num_words), dtype=np.uint64)
        constant = self._constant(flip)
        batch = min(batch_size or BATCH_SHOTS, BATCH_SHOTS)
        deviation = 0.0
        for start in range(0, shots, batch):
            n = min(batch, shots - start)
            words, _, counts, _ = self._run_batch(
                module, n, start, constant, False, self._all, True
            )
            words.get(out=out[start : start + n])
            deviation = max(deviation, self._deviation(counts.get()))
        return out, deviation

    def _synchronize(self) -> None:
        """Wait for every kernel and copy of ``sample_accepted``."""
        for stream in (*self._streams, *self._copies):
            stream.synchronize()

    def _chunk_shots(self, shots: int, nb: int, passing: float) -> int:
        """Return a chunk's shots: about ``CHUNK_BYTES`` of rows of ``nb`` bytes.

        ``passing`` is the share of shots kept, and a batch of ``shots`` shots
        splits into at most ``MAX_CHUNKS`` chunks.
        """
        chunks = min(MAX_CHUNKS, max(1, math.ceil(nb * passing * shots / CHUNK_BYTES)))
        return -(-shots // chunks)

    def sample_accepted(
        self,
        shots: int,
        *,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
        expectations: bool = False,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """Sample ``shots`` shots; return only those that pass postselection.

        A shot passes when none of the ``mask``'s detectors is set (every shot
        passes without a mask). Rows keep their shots' order and hold only the
        outputs outside the mask, whose bits are known for a passing shot.

        The shots run in chunks on two streams: each chunk's rows go to the host
        while the next chunk's kernels run. The chunks hold about ``CHUNK_BYTES``
        of rows, by the screen's estimated share of passing shots, so a seed's
        samples do not depend on earlier calls.

        Returns:
            ``(outputs, expectations, deviation)``: the passing shots' outputs
            outside the mask, bit-packed as ``np.packbits(..., bitorder="little")`` would (``uint8``,
            in pinned host memory), with ``expectations`` each observable's
            expectation of ``(-1)^observable`` given the shot's history before
            the measurement that settles it, or its sampled ``±1`` when a
            postselected output of its component is tested at or after that
            measurement (else None), and the largest normalization deviation
            beyond float32 rounding.

        """
        cp = self._cp
        module = self._module(mask)
        layout = self.layout
        nb, no = -(-len(module.columns) // 8), len(self._observables)
        constant = self._constant(flip)
        split = self._split(module, expectations)
        launch = split[0] if split else self._all
        batch = min(batch_size or BATCH_SHOTS, BATCH_SHOTS)
        out = _HostRows(nb, no if expectations else None)
        state = {"start": 0, "slot": 0}
        size = self._chunk_shots(min(batch, shots), nb, module.passing)
        deviation = 0.0
        pending: list[tuple[int, int, int, tuple[Any, Any, Any, Any], Any]] = []
        # NOTE: the device rows in flight to the host, kept until the copies land,
        # and the streams that copy them.
        held: list[Any] = []
        used: set[Any] = set()

        def launch_next() -> None:
            first, slot = state["start"], state["slot"]
            n = min(size, shots - first)
            device = self._published[slot][1]
            ready = None
            with self._streams[slot]:
                launched = self._run_batch(
                    module,
                    n,
                    first,
                    constant,
                    expectations,
                    launch,
                    False,
                    slot,
                    self._noise_streams[slot],
                )
                # NOTE: no copy is in flight when the first chunk finishes, so it
                # reads its counts with a copy of its own.
                if first:
                    module.publish((1,), (32,), (launched[2], device))
                    ready = self._streams[slot].record()
            pending.append((slot, first, n, launched, ready))
            state["start"], state["slot"] = first + n, 1 - slot

        def finish() -> float:
            slot, first, n, (words, accepted, counts, values), ready = pending.pop(0)
            if ready is None:
                host_counts = counts.get(stream=self._streams[slot])
            else:
                ready.synchronize()
                host_counts = self._published[slot][0].copy()
            with self._streams[slot]:
                count = n if mask is None else int(host_counts[1])
                if first == 0:
                    out.allocate(count, n, shots)
                elif out.total + count > out.capacity:
                    # NOTE: a rare excess of passing shots: the copies in flight land first.
                    self._synchronize()
                    out.grow(out.total + count + (shots - first - n))
                rows_host, values_host = out.take(count)
                index = (
                    None
                    if mask is None
                    else self._list(module, accepted, n, count, slot)
                )
                stream = self._streams[slot]
                # NOTE: a call of one chunk has nothing to overlap its copy with.
                copy = stream if first + n == shots and not held else self._copies[slot]
                used.add(copy)
                if count and nb:
                    dst = self._pack(module, split, words, count, index)
                    if copy is not stream:
                        copy.wait_event(stream.record())
                    dst[:count].get(stream=copy, out=rows_host, blocking=False)
                    held.append(dst)
                if values_host is not None and count:
                    if no and layout.components:
                        if index is None:
                            index = cp.arange(n, dtype=cp.uint32)
                        dst = cp.empty((count, no), dtype=np.float32)
                        module.gather_float(
                            (runtime.multiprocessors() * 8,),
                            (256,),
                            (values, np.int64(count), np.int32(no), index, dst),
                        )
                        if copy is not stream:
                            copy.wait_event(stream.record())
                        dst.get(stream=copy, out=values_host, blocking=False)
                        held.append(dst)
                    else:
                        values_host[:] = 0
            return self._deviation(host_counts)

        # NOTE: each chunk is launched before the previous one is finished, so its
        # kernels overlap that copy.
        launch_next()
        while state["start"] < shots:
            launch_next()
            if len(pending) == 2:
                deviation = max(deviation, finish())
        while pending:
            deviation = max(deviation, finish())
        for stream in used:
            stream.synchronize()
        rows, values = out.result()
        if values is None:
            return rows, None, deviation
        self._direct_expectations(rows, values, module.columns)
        return rows, values, deviation

    def _direct_expectations(
        self, rows: np.ndarray, values: np.ndarray, columns: np.ndarray
    ) -> None:
        """Set the expectations of observables outside the components: ``±1``.

        ``columns`` are the outputs that ``rows`` hold.
        """
        layout = self.layout
        in_components = {i for c in layout.components for i in c.output_indices}
        for column, index in enumerate(self._observables):
            if int(index) in in_components:
                continue
            k = int(np.searchsorted(columns, index))
            bit = (rows[:, k >> 3] >> np.uint8(k & 7)) & np.uint8(1)
            values[:, column] = 1.0 - 2.0 * bit.astype(np.float32)

    def direct_bools(self, shots: int, flip: np.ndarray | None = None) -> np.ndarray:
        """Return every output bit of a program without components, as booleans."""
        words, _ = self._sample_words(shots, None, None, flip)
        return bool_outputs(words, np.arange(self.layout.num_outputs))
