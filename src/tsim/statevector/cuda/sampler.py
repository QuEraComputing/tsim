"""``CudaShotSampler``: ``ShotSampler`` with the noise and the tapes on a CUDA GPU.

A batch of shots runs as these kernels on one stream:

    noise  draws every shot's noise into its words and screens it (``noise``);
           the screened survivors go to a queue
    tape   runs every component on the queued shots, a group of ``2^tb`` threads
           per shot with its amplitudes in registers (``codegen``, ``gpu.tape``); with a
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
import threading
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from tsim.noise.channels import ChannelSampler
from tsim.statevector.cuda import kernels, runtime
from tsim.statevector.cuda.codegen import CUDA
from tsim.statevector.cuda.kernels import COPY_THREADS, SCAN_THREADS, TAPE_THREADS
from tsim.statevector.cuda.noise import THREADS as NOISE_THREADS, noise_tables
from tsim.statevector.gpu import FORKED, tape
from tsim.statevector.gpu.noise import NoiseGroups, screen_pass_rate, split
from tsim.statevector.gpu.rows import select
from tsim.statevector.shots import ShotPlan, ShotSampler

# NOTE: the words of a batch's ``counts``.
_SURVIVORS, _ACCEPTED, _DEVIATION = range(3)

KERNEL_SHARED_BYTES = 48 * 1024
"""Shared memory that a kernel may use without opting in, static and dynamic."""

BATCH_SHOTS = 1 << 22
"""Shots of one batch on the device."""

MIN_BATCH_SHOTS = 1 << 14
"""Smallest batch that a call out of device memory retries with, before the CPU."""

CHUNK_BYTES = 6 << 20
"""Bytes of kept rows per chunk that a call's chunks aim at: each chunk's copy to
the host overlaps the next chunk's kernels on the other stream."""

MAX_CHUNKS = 8
"""Most chunks of a batch, so that a chunk's fixed costs stay small."""


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
    tape_blocks: int
    has_mask: bool
    plan: ShotPlan
    psm_words: np.ndarray
    passing: float
    """Estimated share of shots that pass the screen."""
    splits: dict[bool, tuple[_Launch, _Launch] | None]
    """Each ``expectations`` value's noise passes (screened, deferred), or None."""


def _pinned(shape: tuple[int, ...], dtype) -> np.ndarray:
    """Return an uninitialized array in pinned host memory."""
    import cupyx

    # NOTE: CuPy's stub types the dtype as a float type.
    return cupyx.empty_pinned(shape, dtype=dtype)  # pyright: ignore[reportArgumentType]


def _pinned_counts(cp) -> tuple[np.ndarray, Any]:
    """Return three pinned host counters and a device array over the same memory."""
    host = _pinned((3,), np.uint32)
    # NOTE: with unified addressing a pinned host pointer is also a device pointer.
    memory = cp.cuda.UnownedMemory(host.ctypes.data, host.nbytes, host)
    device = cp.ndarray((3,), np.uint32, cp.cuda.MemoryPointer(memory, 0))
    return host, device


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
        self.capacity = capacity
        self.rows = _pinned((capacity, self.nb), np.uint8)
        if self.no is not None:
            self.values = _pinned((capacity, self.no), np.float32)

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
    ``reference``, ``plan``, ``layout``); ``sample`` copies only the shots that
    pass postselection off the GPU. ``sample_accepted`` can also return each
    observable's expectation.
    """

    device = "cuda"

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
            ImportError: If CuPy is not installed.

        """
        self._cp = runtime.cupy()
        cp = self._cp
        assert self.supports(host.layout)
        self._args = (host, channels, num_detectors, seed)
        self._lock = threading.RLock()
        self._host = host
        layout = host.layout
        self._num_detectors = num_detectors
        self._observables = np.arange(num_detectors, layout.num_outputs)
        self._rng = np.random.default_rng(seed)
        self._channels = channels
        self._noise = noise_tables(channels, layout.noise, layout.num_words)
        n = self._noise
        # NOTE: gather_columns and noise_kept stage a block's output words and its
        # packed rows in shared memory.
        staged = 8 * layout.output_words * n.block_shots
        staged += n.block_shots * -(-layout.num_outputs // 8)
        if staged > KERNEL_SHARED_BYTES:
            raise ValueError(
                f"{layout.num_outputs} outputs do not fit the GPU kernels' shared memory"
            )
        self._record_args = [self._upload(n.overflow.view(np.uint8))]
        self._all = self._launch(n.groups)
        self._modules: dict[bytes | None, _Module] = {}
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
        # NOTE: lowered when a call runs out of device memory.
        self._max_batch = BATCH_SHOTS

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
        # NOTE: None, not b"", so that an empty mask is not taken for no mask.
        key = None if mask is None else np.asarray(mask, bool).tobytes()
        if key in self._modules:
            return self._modules[key]
        cp = self._cp
        plan = self.plan(mask)
        layout = self.layout
        tb = tape.threads_per_shot(layout)
        parts = tape.tape_kernel(layout, plan, mask, self._num_detectors, tb, CUDA)
        source = kernels.source(
            layout, self._noise, parts, tb, max(1, len(self._observables))
        )
        module = self._compile(source)
        tape_fn = module.get_function("tape")
        blocks_per_sm = 4
        if hasattr(cp.cuda.runtime, "occupancyMaxActiveBlocksPerMultiprocessor"):
            blocks_per_sm = max(
                1,
                cp.cuda.runtime.occupancyMaxActiveBlocksPerMultiprocessor(
                    tape_fn.ptr, TAPE_THREADS, 0
                ),
            )
        compiled = _Module(
            noise=module.get_function("noise"),
            noise_kept=module.get_function("noise_kept"),
            tape=tape_fn,
            gather_columns=module.get_function("gather_columns"),
            gather_float=module.get_function("gather_float"),
            publish=module.get_function("publish"),
            count_bits=module.get_function("count_bits"),
            list_bits=module.get_function("list_bits"),
            screen=self._upload(plan.screen),
            tape_blocks=runtime.multiprocessors() * blocks_per_sm,
            has_mask=mask is not None,
            plan=plan,
            psm_words=parts.psm_words,
            passing=screen_pass_rate(self._channels, layout, plan),
            splits={},
        )
        self._modules[key] = compiled
        return compiled

    def _upload(self, array: np.ndarray) -> Any:
        """Return ``array`` on the device, copied before any later kernel runs.

        ``cp.asarray`` copies on the current stream, usually the legacy default
        stream, which the sampler's non-blocking streams do not wait for: a kernel
        could read the array before the copy lands. Uploads are one-time, cached.
        """
        device = self._cp.asarray(array)
        self._cp.cuda.get_current_stream().synchronize()
        return device

    def _launch(self, groups: NoiseGroups) -> _Launch:
        """Upload a noise launch's segments and records."""
        return _Launch(
            args=[
                *(self._upload(a) for a in groups.segments),
                self._upload(groups.records.view(np.uint8)),
            ],
            num_segments=len(groups.segments[0]),
            max_events=groups.max_events,
        )

    def _split(
        self, module: _Module, expectations: bool
    ) -> tuple[_Launch, _Launch] | None:
        """Return a postselected batch's noise passes (screened, deferred), or None.

        As ``gpu.noise.split``; with ``expectations`` the observables' bits are
        read too. Each key word is read whole, so the deferred channels flip
        output words only.
        """
        if not module.has_mask:
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
            self._constants[key] = self._upload(self._host._constant(flip))
        return self._constants[key]

    def _run_batch(
        self,
        module: _Module,
        shots: int,
        first_shot: int,
        constant: Any,
        expectations: bool,
        launch: _Launch,
        slot: int = 0,
        noise_stream: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Launch one batch on the current stream: ``(words, accepted, counts, values)``.

        ``accepted`` is the bitmap of the accepted shots (with a mask).

        ``launch`` holds the channels that the noise kernel draws; only the rows
        of the shots that pass the screen are written.
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
            (NOISE_THREADS,),
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
                queue,
                counts[_SURVIVORS : _SURVIVORS + 1],
                accepted if module.has_mask else np.uint64(0),
            ),
            shared_mem=8 * launch.max_events,
        )
        if noise_stream is not None:
            current.wait_event(noise_stream.record())
            current.use()
        if layout.components:
            module.tape(
                (module.tape_blocks,),
                (TAPE_THREADS,),
                (
                    words,
                    np.int64(first_shot),
                    seed,
                    queue,
                    counts[_SURVIVORS : _SURVIVORS + 1],
                    accepted if module.has_mask else np.uint64(0),
                    counts[_ACCEPTED : _ACCEPTED + 1],
                    values if values is not None else np.uint64(0),
                    counts[_DEVIATION : _DEVIATION + 1],
                ),
            )
        elif module.has_mask:
            # NOTE: without components the screened survivors are the accepted shots.
            counts[_ACCEPTED : _ACCEPTED + 1] = counts[_SURVIVORS : _SURVIVORS + 1]
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

        Their outputs, bit-packed; with ``split`` the deferred noise kernel adds
        the deferred channels' faults first.
        """
        S = self._noise.block_shots
        nb = -(-self.layout.num_outputs // 8)
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
                (NOISE_THREADS,),
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
                ),
                shared_mem=max(8 * launch.max_events, S * nb),
            )
        else:
            module.gather_columns(
                (blocks,),
                (COPY_THREADS,),
                (
                    words,
                    np.int64(count),
                    index if index is not None else np.uint64(0),
                    dst,
                ),
                shared_mem=S * nb,
            )
        return dst

    def _list(self, module: _Module, bits, shots: int, count: int, slot: int) -> Any:
        """Return the ``count`` shots of the bitmap ``bits`` of ``shots`` shots, in order."""
        words = -(-shots // 32)
        blocks = -(-words // SCAN_THREADS)
        totals = self._buffer(f"totals{slot}", (blocks,), np.uint32)
        index = self._buffer(f"index{slot}", (max(count, 1),), np.uint32)
        grid, threads = (blocks,), (SCAN_THREADS,)
        module.count_bits(grid, threads, (bits, np.int64(words), totals))
        module.list_bits(grid, threads, (bits, np.int64(words), totals, index))
        return index[:count]

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
        postselection are returned. ``batch_size`` bounds the shots of a device
        batch.
        """
        accepted = None
        if not _forked():
            with self._lock:
                accepted = self._sample_or_none(shots, mask, flip, batch_size)
        if accepted is None:
            return self._host.sample(
                shots,
                columns,
                bit_packed=bit_packed,
                batch_size=batch_size,
                mask=mask,
                flip=flip,
            )
        rows, deviation = accepted
        num_outputs = self.layout.num_outputs
        return [select(rows, c, num_outputs, bit_packed) for c in columns], deviation

    def _sample_or_none(
        self,
        shots: int,
        mask: np.ndarray | None,
        flip: np.ndarray | None,
        batch_size: int | None,
    ) -> tuple[np.ndarray, float] | None:
        """Return ``sample_accepted``'s rows and deviation, or None if the GPU is full.

        Out of device memory, the call is retried in batches half as large, down
        to ``MIN_BATCH_SHOTS``; past that it warns and returns None, for the CPU.
        Later calls start from the batch that fit.
        """
        batch = min(batch_size or BATCH_SHOTS, self._max_batch)
        while True:
            try:
                rows, _, deviation = self.sample_accepted(
                    shots, mask=mask, flip=flip, batch_size=batch
                )
                return rows, deviation
            except Exception as error:
                if not runtime.out_of_memory(error):
                    raise
                self._release()
                self._max_batch = max(batch // 2, MIN_BATCH_SHOTS)
                if batch <= MIN_BATCH_SHOTS:
                    warnings.warn(
                        "The GPU ran out of memory, even for batches of "
                        f"{MIN_BATCH_SHOTS} shots; sampling on the CPU.",
                        stacklevel=5,
                    )
                    return None
                batch = self._max_batch

    def _release(self) -> None:
        """Wait for the device and free the buffers that calls reuse."""
        cp = self._cp
        cp.cuda.Device().synchronize()
        self._buffers.clear()
        self._views.clear()
        cp.get_default_memory_pool().free_all_blocks()

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
        passes without a mask). Rows keep their shots' order and hold every output.

        The shots run in chunks on two streams: each chunk's rows go to the host
        while the next chunk's kernels run. The chunks hold about ``CHUNK_BYTES``
        of rows, by the screen's estimated share of passing shots, so a seed's
        samples do not depend on earlier calls.

        Returns:
            ``(outputs, expectations, deviation)``: the passing shots' outputs,
            bit-packed as ``np.packbits(..., bitorder="little")`` would (``uint8``,
            in pinned host memory), with ``expectations`` each observable's
            expectation of ``(-1)^observable`` given the shot's history before
            the measurement that settles it, or its sampled ``±1`` when a
            postselected output of its component is tested at or after that
            measurement (else None), and the largest normalization deviation
            beyond float32 rounding.

        """
        with self._lock:
            return self._sample_accepted(
                shots,
                mask=mask,
                flip=flip,
                expectations=expectations,
                batch_size=batch_size,
            )

    def _sample_accepted(
        self,
        shots: int,
        *,
        mask: np.ndarray | None,
        flip: np.ndarray | None,
        expectations: bool,
        batch_size: int | None,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        cp = self._cp
        module = self._module(mask)
        layout = self.layout
        nb, no = -(-layout.num_outputs // 8), len(self._observables)
        if not shots:
            values = np.zeros((0, no), np.float32) if expectations else None
            return np.zeros((0, nb), np.uint8), values, 0.0
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
                count = n if mask is None else int(host_counts[_ACCEPTED])
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
                            held.append(index)
                        dst = cp.empty((count, no), dtype=np.float32)
                        module.gather_float(
                            (runtime.multiprocessors() * 8,),
                            (COPY_THREADS,),
                            (values, np.int64(count), np.int32(no), index, dst),
                        )
                        if copy is not stream:
                            copy.wait_event(stream.record())
                        dst.get(stream=copy, out=values_host, blocking=False)
                        held.append(dst)
                    else:
                        values_host[:] = 0
            return tape.deviation(host_counts[_DEVIATION])

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
        self._direct_expectations(rows, values)
        return rows, values, deviation

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
        accepted = None
        if not _forked():
            with self._lock:
                accepted = self._sample_or_none(shots, None, flip, None)
        if accepted is None:
            return self._host.direct_bools(shots, flip)
        rows, _ = accepted
        num_outputs = self.layout.num_outputs
        return select(rows, np.arange(num_outputs), num_outputs, False)


def _forked() -> bool:
    """Return whether this process was forked after its parent used CUDA, with a warning."""
    if not runtime.forked_after_init():
        return False
    warnings.warn(FORKED, stacklevel=5)
    return True


def _restore(
    host: ShotSampler,
    channels: ChannelSampler,
    num_detectors: int,
    seed: np.random.SeedSequence,
    state: dict,
) -> CudaShotSampler | ShotSampler:
    """Rebuild a pickled ``CudaShotSampler``, or its CPU sampler where CUDA is missing."""
    if not runtime.available():
        return host
    shots = CudaShotSampler(host, channels, num_detectors, seed)
    shots._rng.bit_generator.state = state
    return shots
