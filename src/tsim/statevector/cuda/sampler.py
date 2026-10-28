"""``CudaShotSampler``: ``ShotSampler`` with the noise and the tapes on a CUDA GPU.

A batch of shots runs as three kernels on one stream:

    noise  draws every shot's noise into its words and screens it (``noise``);
           the screened survivors go to a queue
    tape   runs every component on the queued shots, a group of ``2^tb`` threads
           per shot with its amplitudes in registers (``codegen``); with a
           postselection mask it queues the accepted shots, and with
           expectations it writes each observable's expectation
    select gathers the rows (or the accepted rows) for the copy to the host

The kernels are compiled with NVRTC once per postselection mask (CuPy caches
them on disk). Amplitudes are complex64, so a sample follows the CPU executor's
distribution to within float32 rounding of the outcome probabilities; the noise
and the outcome draws come from other streams than the CPU's, so a seed's
samples differ from the CPU's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tsim.core.types import CompiledProgram
from tsim.noise.channels import ChannelSampler
from tsim.statevector.cuda import runtime
from tsim.statevector.cuda.codegen import PRELUDE, tape_function
from tsim.statevector.cuda.noise import noise_tables
from tsim.statevector.program import STABILIZER_RANK_HINT
from tsim.statevector.shared import pack_bits
from tsim.statevector.shots import ShotPlan, ShotSampler, bool_outputs, read_outputs

MAX_WIDTH = 11
"""Widest component the CUDA executor runs: 32 threads of 64 amplitudes."""

BATCH_SHOTS = 1 << 22
"""Shots of one batch on the device."""

DEVIATION_TOLERANCE = 1e-3
"""Normalization deviation that float32 amplitudes tolerate without a warning."""

_KERNELS = r"""
extern "C" __global__ void __launch_bounds__({block}) tape(
    u64* __restrict__ out, long long first_shot, const u64* __restrict__ seeds,
    const unsigned* __restrict__ queue, const unsigned* __restrict__ count,
    const u64* __restrict__ psm, unsigned* __restrict__ acc_queue, unsigned* __restrict__ acc_count,
    float* __restrict__ expectations, unsigned* __restrict__ deviation)
{{
    int t = threadIdx.x & {T1};
    unsigned gmask = {gmask};
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
        u64 w[{NW}];
        #pragma unroll
        for (int i = 0; i < {NW}; ++i) w[i] = row[i];
        float ev[{NE}];
        int grave = 0x7fffffff;
{calls}
        if (valid && t == 0) {{
            #pragma unroll
            for (int i = 0; i < {OW}; ++i) row[i] = w[i];
            if (expectations != nullptr) {{
                #pragma unroll
                for (int i = 0; i < {NO}; ++i) expectations[row_index * {NO} + i] = ev[i];
            }}
        }}
        if (acc_queue != nullptr) {{
            u64 hit = 0;
            #pragma unroll
            for (int i = 0; i < {OW}; ++i) hit |= w[i] & psm[i];
            bool accepted = valid && t == 0 && hit == 0;
            unsigned ballot = __ballot_sync(0xffffffffu, accepted);
            if (ballot) {{
                int lane = threadIdx.x & 31, leader = __ffs(ballot) - 1;
                unsigned b = 0;
                if (lane == leader) b = atomicAdd(acc_count, __popc(ballot));
                b = __shfl_sync(0xffffffffu, b, leader);
                if (accepted) acc_queue[b + __popc(ballot & ((1u << lane) - 1))] = (unsigned)row_index;
            }}
        }}
    }}
}}

// Rows ``index[j]`` (or every row) of their first ``ow`` words, densely.
extern "C" __global__ void gather(const u64* __restrict__ rows, long long n, int nw, int ow,
                                  const unsigned* __restrict__ index, u64* __restrict__ dst) {{
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n * ow;
         k += (long long)gridDim.x * blockDim.x) {{
        long long j = k / ow; int w = (int)(k - j * ow);
        long long i = index != nullptr ? (long long)index[j] : j;
        dst[k] = rows[i * nw + w];
    }}
}}

// The leading ``nb`` bytes of rows ``index[j]`` (or every row), densely: the
// bit-packed outputs, as the words hold them in little-endian order.
extern "C" __global__ void gather_bytes(const u64* __restrict__ rows, long long n, int nw, int nb,
                                        const unsigned* __restrict__ index, unsigned char* __restrict__ dst) {{
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n * nb;
         k += (long long)gridDim.x * blockDim.x) {{
        long long j = k / nb; int b = (int)(k - j * nb);
        long long i = index != nullptr ? (long long)index[j] : j;
        dst[k] = (unsigned char)(rows[i * nw + (b >> 3)] >> (8 * (b & 7)));
    }}
}}

// The outputs ``cols`` of rows ``index[j]``, bit-packed densely into ``nb`` bytes per
// row; ``run[b]`` marks the bytes whose eight outputs are consecutive.
extern "C" __global__ void gather_columns(const u64* __restrict__ rows, long long n, int nw, int nb,
                                          const unsigned* __restrict__ index,
                                          const int* __restrict__ cols, int m,
                                          const unsigned char* __restrict__ run,
                                          unsigned char* __restrict__ dst) {{
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n * nb;
         k += (long long)gridDim.x * blockDim.x) {{
        long long j = k / nb; int b = (int)(k - j * nb);
        const u64* row = rows + (index != nullptr ? (long long)index[j] : j) * nw;
        unsigned v = 0;
        if (run[b]) {{
            int c = cols[8 * b], w = c >> 6, s = c & 63;
            u64 bits = row[w] >> s;
            if (s > 56) bits |= row[w + 1] << (64 - s);
            v = (unsigned)(bits & 0xffu);
        }} else {{
            for (int t = 0; t < 8 && 8 * b + t < m; ++t) {{
                int c = cols[8 * b + t];
                v |= (unsigned)((row[c >> 6] >> (c & 63)) & 1u) << t;
            }}
        }}
        dst[k] = (unsigned char)v;
    }}
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
class _Module:
    """A plan's compiled kernels and its device constants."""

    noise: Any
    tape: Any
    gather: Any
    gather_bytes: Any
    gather_columns: Any
    gather_float: Any
    screen: Any
    kept: Any
    psm: Any
    tape_blocks: int
    has_mask: bool
    columns: np.ndarray
    """The outputs that ``sample_accepted`` returns: those outside the mask."""
    columns_d: Any
    runs_d: Any


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
        seed: int,
    ) -> None:
        """Lay out the program's shot words and upload the noise tables.

        Raises:
            ImportError: If CuPy is not installed.
            ValueError: If a component is wider than ``MAX_WIDTH``.

        """
        self._cp = runtime.cupy()
        cp = self._cp
        self._host = ShotSampler(program, channels, num_detectors, seed, share=True)
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
        self._rng = np.random.default_rng([seed, 2])
        self._noise = noise_tables(channels, layout.noise, layout.num_words)
        n = self._noise
        self._noise_args = [
            *(cp.asarray(a) for a in n.segments),
            cp.asarray(n.channels),
            cp.asarray(n.records.view(np.uint8)),
            cp.asarray(n.overflow_word),
            cp.asarray(n.overflow_mask),
        ]
        self._modules: dict[bytes, _Module] = {}
        self._buffers: dict[str, Any] = {}

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
        """Return log2 of the threads per shot: 64 amplitudes per thread at most."""
        width = max((c.width for c in self.layout.components), default=0)
        return 0 if width <= 6 else min(5, width - 5)

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
            f"        grave = min(grave, component_{i}(w, shot, seeds[{i}], t, gmask, ev, deviation));"
            for i in range(len(plan.programs))
        )
        gmask = (
            "0xffffffffu"
            if T == 32
            else f"((1u << {T}) - 1) << (threadIdx.x & 31 & ~{T - 1})"
        )
        source = (
            PRELUDE
            + self._noise.source(layout.num_words, layout.output_words)
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
                calls=calls,
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
        bits = np.zeros(64 * layout.output_words, dtype=bool)
        if mask is not None:
            bits[: len(mask)] = mask
        columns = np.flatnonzero(~bits[: layout.num_outputs]).astype(np.int32)
        # NOTE: a byte of consecutive outputs is a shift of the row's words.
        octets = np.pad(columns, (0, -len(columns) % 8), constant_values=-1)
        octets = octets.reshape(-1, 8)
        runs = (octets[:, 7] >= 0) & (octets[:, 7] - octets[:, 0] == 7)
        compiled = _Module(
            noise=module.get_function("noise"),
            tape=tape,
            gather=module.get_function("gather"),
            gather_bytes=module.get_function("gather_bytes"),
            gather_columns=module.get_function("gather_columns"),
            gather_float=module.get_function("gather_float"),
            screen=cp.asarray(plan.screen),
            kept=cp.asarray(plan.kept),
            psm=cp.asarray(pack_bits(bits, layout.output_words)),
            tape_blocks=runtime.multiprocessors() * blocks_per_sm,
            has_mask=mask is not None,
            columns=columns,
            columns_d=cp.asarray(columns if len(columns) else [0], dtype=np.int32),
            runs_d=cp.asarray(runs if len(runs) else [0], dtype=np.uint8),
        )
        self._modules[key] = compiled
        return compiled

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
        cp = self._cp
        size = int(np.prod(shape))
        held = self._buffers.get(name)
        if held is None or held.size < size or held.dtype != np.dtype(dtype):
            held = cp.empty(size, dtype=dtype)
            self._buffers[name] = held
        return held[:size].reshape(shape)

    def _run_batch(
        self,
        module: _Module,
        shots: int,
        first_shot: int,
        constant: np.ndarray,
        expectations: bool,
    ) -> tuple[Any, Any, Any, Any]:
        """Sample one batch on the device: ``(words, accepted, count, values)``."""
        cp = self._cp
        layout = self.layout
        nw, no = layout.num_words, max(1, len(self._observables))
        words = self._buffer("words", (shots, nw), np.uint64)
        queue = self._buffer("queue", (shots,), np.uint32)
        accepted = self._buffer("accepted", (shots,), np.uint32)
        counts = self._buffer("counts", (3,), np.uint32)
        values = (
            self._buffer("values", (shots, no), np.float32) if expectations else None
        )
        counts.fill(0)
        seeds = cp.asarray(
            self._rng.integers(
                0, 2**64, size=max(1, len(layout.components)), dtype=np.uint64
            )
        )
        noise_seed = np.uint64(self._rng.integers(0, 2**64, dtype=np.uint64))
        constant_d = cp.asarray(constant)
        S = self._noise.block_shots
        module.noise(
            ((shots + S - 1) // S,),
            (256,),
            (
                words,
                np.int64(shots),
                noise_seed,
                np.int32(len(self._noise.segments[0])),
                *self._noise_args,
                constant_d,
                module.screen,
                module.kept,
                queue,
                counts[0:1],
            ),
        )
        if layout.components:
            module.tape(
                (module.tape_blocks,),
                (128,),
                (
                    words,
                    np.int64(first_shot),
                    seeds,
                    queue,
                    counts[0:1],
                    module.psm,
                    accepted if module.has_mask else np.uint64(0),
                    counts[1:2],
                    values if values is not None else np.uint64(0),
                    counts[2:3],
                ),
            )
        elif module.has_mask:
            # NOTE: without components the screened survivors are the accepted shots.
            accepted, counts[1:2] = queue, counts[0:1]
        return words, accepted, counts, values

    def _to_host(self, device, rows: int, width: int, dtype) -> np.ndarray:
        """Copy ``device[:rows]`` to a new pinned host array (CuPy's pinned pool)."""
        import cupyx

        host = cupyx.empty_pinned((rows, width), dtype=dtype)
        if rows:
            device[:rows].get(out=host)
        return host

    def _gather_bytes(
        self, module: _Module, words, count: int, index, nb: int
    ) -> np.ndarray:
        """Return the leading ``nb`` bytes of ``count`` rows (``index``, or all)."""
        dst = self._buffer("dst_bytes", (max(count, 1), nb), np.uint8)
        module.gather_bytes(
            (runtime.multiprocessors() * 8,),
            (256,),
            (
                words,
                np.int64(count),
                np.int32(words.shape[1]),
                np.int32(nb),
                index if index is not None else np.uint64(0),
                dst,
            ),
        )
        return self._to_host(dst, count, nb, np.uint8)

    def _gather_columns(self, module: _Module, words, count: int, index) -> np.ndarray:
        """Return the outputs ``module.columns`` of ``count`` rows ``index``, bit-packed."""
        m = len(module.columns)
        if m == self.layout.num_outputs:
            return self._gather_bytes(module, words, count, index, -(-m // 8))
        nb = -(-m // 8)
        if not nb:
            return np.empty((count, 0), np.uint8)
        dst = self._buffer("dst_bytes", (max(count, 1), nb), np.uint8)
        if count:
            module.gather_columns(
                (runtime.multiprocessors() * 8,),
                (256,),
                (
                    words,
                    np.int64(count),
                    np.int32(words.shape[1]),
                    np.int32(nb),
                    index,
                    module.columns_d,
                    np.int32(m),
                    module.runs_d,
                    dst,
                ),
            )
        return self._to_host(dst, count, nb, np.uint8)

    def _gather_values(
        self, module: _Module, values, count: int, index, width: int
    ) -> np.ndarray:
        """Return the expectations of ``count`` rows ``index``."""
        dst = self._buffer("dst_float", (max(count, 1), width), np.float32)
        module.gather_float(
            (runtime.multiprocessors() * 8,),
            (256,),
            (values, np.int64(count), np.int32(width), index, dst),
        )
        return self._to_host(dst, count, width, np.float32)

    def _deviation(self, counts) -> float:
        """Return the batch's normalization deviation beyond float32 rounding, else 0."""
        bits = np.asarray(counts[2:3].get(), dtype=np.uint32)
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
        constant = self._host._constant(flip)
        batch = min(batch_size or BATCH_SHOTS, BATCH_SHOTS)
        deviation = 0.0
        for start in range(0, shots, batch):
            n = min(batch, shots - start)
            words, _, counts, _ = self._run_batch(module, n, start, constant, False)
            words.get(out=out[start : start + n])
            deviation = max(deviation, self._deviation(counts))
        return out, deviation

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
        constant = self._host._constant(flip)
        batch = min(batch_size or BATCH_SHOTS, BATCH_SHOTS)
        rows, values, deviation = [], [], 0.0
        for start in range(0, shots, batch):
            n = min(batch, shots - start)
            words, accepted, counts, ev = self._run_batch(
                module, n, start, constant, expectations
            )
            if mask is None:
                count, index = n, None
            else:
                count = int(counts[1:2].get()[0])
                # NOTE: the queue's order is the warps'; rows keep their shots'.
                index = cp.sort(accepted[:count])
            rows.append(self._gather_columns(module, words, count, index))
            if expectations:
                if no and layout.components:
                    if index is None:
                        index = cp.arange(n, dtype=cp.uint32)
                    values.append(self._gather_values(module, ev, count, index, no))
                else:
                    values.append(np.zeros((count, no), dtype=np.float32))
            deviation = max(deviation, self._deviation(counts))
        out_rows = (
            rows[0]
            if len(rows) == 1
            else np.concatenate(rows) if rows else np.empty((0, nb), np.uint8)
        )
        if not expectations:
            return out_rows, None, deviation
        out_values = (
            values[0]
            if len(values) == 1
            else np.concatenate(values) if values else np.empty((0, no), np.float32)
        )
        self._direct_expectations(out_rows, out_values, module.columns)
        return out_rows, out_values, deviation

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
