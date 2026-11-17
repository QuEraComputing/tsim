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
from dataclasses import dataclass
from typing import Any

import numpy as np

from tsim.noise.channels import ChannelSampler
from tsim.statevector.gpu import FORKED, tape
from tsim.statevector.gpu.noise import NoiseGroups, screen_pass_rate, split
from tsim.statevector.gpu.rows import select as _host_select
from tsim.statevector.metal import kernels, runtime
from tsim.statevector.metal.codegen import METAL
from tsim.statevector.metal.kernels import SCAN_THREADS
from tsim.statevector.metal.noise import noise_tables
from tsim.statevector.shots import ShotPlan, ShotSampler

BATCH_SHOTS = 1 << 20
"""Shots of one batch (one command buffer)."""

BATCH_BYTES = 256 << 20
"""Bytes of shot words of one batch, which bound its shots below ``BATCH_SHOTS``."""

NOISE_THREADS = 256
"""Threads per threadgroup of the noise kernels."""

COPY_THREADS = 256
"""Threads per threadgroup of the kernels that copy rows."""

MAX_COPY_THREADS = 1 << 20
"""Most threads of a copy kernel's dispatch; each thread loops over its share."""

MAX_COPY_ITEMS = (1 << 32) - MAX_COPY_THREADS
"""Most items (bytes) of a copy kernel's dispatch, which runs a 32-bit loop.

NOTE: a thread steps by the dispatch's threads, so an index must stay below 2^32
by that much; past it, it would wrap and the loop would not end.
"""

# NOTE: the words of ``_Slot.counts``.
_SURVIVORS, _ACCEPTED, _DEVIATION, _CURSOR, _CAPACITY = range(5)

TAPE_THREADS = 128
"""Threads per threadgroup of the tape kernel."""

WIDE_TAPE = 8
"""Width from which a shot's threads hold at most 16 amplitudes each, not 32."""

MAX_CALL_SHOTS = 1 << 31
"""Most shots of one GPU call; larger calls run in parts (the counters are 32-bit)."""


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
    split: tuple[_Launch, _Launch] | None
    """A postselected batch's noise passes (screened, deferred), or None."""


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


@dataclass
class _Call:
    """What a call's batches share: its kernels, buffers, noise passes and output."""

    module: _Module
    slot: _Slot
    constant: runtime.Buffer
    split: tuple[_Launch, _Launch] | None
    rows: runtime.Buffer
    """The accepted rows, bit-packed."""


class MetalShotSampler:
    """Samples shot words of a statevector program on a Metal GPU.

    It keeps ``ShotSampler``'s interface (``sample``, ``direct_bools``,
    ``reference``, ``plan``, ``layout``); ``sample`` returns only the shots that
    pass postselection.
    """

    supports = staticmethod(tape.supports)

    def __init__(
        self,
        host: ShotSampler,
        channels: ChannelSampler,
        num_detectors: int,
        seed: np.random.SeedSequence,
        *,
        required: bool = False,
    ) -> None:
        """Upload the noise tables of ``host``'s shot words.

        ``host`` draws the reference samples, on the CPU, and the calls that the
        GPU cannot run, unless ``required`` (``device="metal"``): then those raise.

        Raises:
            ImportError: If PyObjC's Metal is not installed.
            RuntimeError: If there is no Apple GPU.

        """
        runtime.check_device()
        assert self.supports(host.layout)
        self._args = (host, channels, num_detectors, seed, required)
        self._required = required
        # NOTE: a call's batches share the sampler's buffers, so calls from several
        # threads run one at a time.
        self._lock = threading.RLock()
        self._host = host
        layout = host.layout
        self._rng = np.random.default_rng(seed)
        self._channels = channels
        self._noise = noise_tables(channels, layout.noise, layout.num_words)
        self._overflow = runtime.upload(self._noise.overflow.view(np.uint8))
        self._all = self._launch(self._noise.groups)
        self._modules: dict[bytes, _Module] = {}
        self._constants: dict[bytes, runtime.Buffer] = {}
        self._slots: dict[int, _Slot] = {}
        self._shares: dict[ShotPlan, float] = {}
        self._cpu_only = False

    @property
    def device(self) -> str:
        """Return where this sampler samples (``SamplerStats.device``).

        It is ``"cpu"`` after a kernel failed to compile, and in a process forked
        after its parent used the GPU, where every call samples on the CPU.
        """
        if self._cpu_only or runtime.forked_after_init():
            return "cpu"
        return "metal"

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
        tb = _threads_per_shot(layout)
        parts = tape.tape_kernel(layout, plan, mask, tb, METAL)
        source = kernels.source(layout, self._noise, parts, tb)
        library = runtime.Library(source)
        for name in kernels.names(self._noise.shared_words):
            library.pipeline(name)
        compiled = _Module(
            library=library,
            tb=tb,
            has_mask=mask is not None,
            screen=runtime.upload(plan.screen),
            plan=plan,
            psm_words=parts.psm_words,
            passing=screen_pass_rate(self._channels, layout, plan),
            split=self._split(plan, parts.psm_words, mask is not None),
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
        self, plan: ShotPlan, psm_words: np.ndarray, has_mask: bool
    ) -> tuple[_Launch, _Launch] | None:
        """Return a postselected batch's noise passes (screened, deferred), or None.

        As ``gpu.noise.split``.
        """
        # NOTE: rows too wide for threadgroup memory draw all their noise at once.
        if not has_mask or not self._noise.shared_words:
            return None
        found = split(self._noise, self._channels, self.layout, plan, psm_words)
        if found is None:
            return None
        return self._launch(found[0]), self._launch(found[1])

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

    def _slot(self, shots: int) -> _Slot:
        """Return the buffers for batches of up to ``shots`` shots."""
        slot = self._slots.get(shots)
        if slot is None:
            nw = self.layout.num_words
            slot = _Slot(
                words=runtime.buffer(shots * nw, np.uint64),
                queue=runtime.buffer(shots, np.uint32),
                accepted=runtime.buffer(-(-shots // 32), np.uint32),
                counts=runtime.buffer(8, np.uint32),
                totals=runtime.buffer(-(-shots // 32 // SCAN_THREADS) + 1, np.uint32),
                index=runtime.buffer(shots, np.uint32),
            )
            self._slots = {shots: slot}
        return slot

    def _encode_batch(
        self, encoder: runtime.Encoder, call: _Call, shots: int, first_shot: int
    ) -> None:
        """Encode a batch's noise and tape kernels.

        Only the shots that pass the screen get their words written.
        """
        module, slot, constant = call.module, call.slot, call.constant
        launch = call.split[0] if call.split else self._all
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
                _count(slot, _SURVIVORS),
                slot.accepted,
            ],
            -(-shots // S),
            NOISE_THREADS,
            threadgroup_bytes=8 * max(launch.max_events, 1),
        )
        if self.layout.components:
            tape_args = np.zeros(1, dtype=_TAPE_ARGS)
            tape_args[0] = (first_shot, seed, int(module.has_mask), 0)
            threads = shots << module.tb
            encoder.dispatch(
                library.pipeline("tape"),
                [
                    words,
                    tape_args,
                    slot.queue,
                    _count(slot, _SURVIVORS),
                    slot.accepted,
                    _count(slot, _ACCEPTED),
                    _count(slot, _DEVIATION),
                ],
                -(-threads // TAPE_THREADS),
                TAPE_THREADS,
            )

    def _encode_pack(
        self,
        encoder: runtime.Encoder,
        call: _Call,
        shots: int,
        fixed: tuple[runtime.Buffer, int],
    ) -> None:
        """Encode the listing and packing of a batch's accepted rows into ``call.rows``.

        They follow the rows of the call's earlier batches (at the cursor);
        ``fixed`` holds the batch's shots, which a call without a mask returns.
        """
        module, slot, split = call.module, call.slot, call.split
        rows = call.rows
        library = module.library
        S = self._noise.block_shots
        nb = -(-self.layout.num_outputs // 8)
        if module.has_mask:
            words = -(-shots // 32)
            blocks = -(-words // SCAN_THREADS)
            encoder.dispatch(
                library.pipeline("count_bits"),
                [slot.accepted, np.int64(words), slot.totals],
                blocks,
                SCAN_THREADS,
            )
            encoder.dispatch(
                library.pipeline("list_bits"),
                [slot.accepted, np.int64(words), slot.totals, slot.index],
                blocks,
                SCAN_THREADS,
            )
            kept = _ACCEPTED if self.layout.components else _SURVIVORS
            count = _count(slot, kept)
        else:
            count = fixed
        cursor = _count(slot, _CURSOR)
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
                NOISE_THREADS,
                threadgroup_bytes=max(8 * launch.max_events, S * nb),
            )
        elif nb:
            threads = min(-(-shots * nb // 4) + 1, MAX_COPY_THREADS)
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
                -(-threads // COPY_THREADS),
                COPY_THREADS,
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
        Out of GPU memory, when a kernel does not compile, and in a process
        forked after its parent used the GPU, the call samples on the CPU, or
        raises if the sampler is ``required`` on the GPU (``_or_cpu``).
        """
        return self._or_cpu(
            lambda: self._accepted(
                shots,
                columns,
                mask=mask,
                flip=flip,
                batch_size=batch_size,
                bools=not bit_packed,
            ),
            lambda: self._host.sample(
                shots,
                columns,
                bit_packed=bit_packed,
                batch_size=batch_size,
                mask=mask,
                flip=flip,
            ),
        )

    def _or_cpu(self, on_gpu, on_cpu):
        """Return ``on_gpu()``, or ``on_cpu()`` where the GPU fails.

        It fails in a process forked after its parent used the GPU, out of GPU
        memory, and when a kernel does not compile; after the last the sampler
        samples on the CPU. A ``required`` sampler raises instead.

        Raises:
            RuntimeError: If ``required``, in a process forked after its parent
                used the GPU.
            MemoryError: If ``required``, out of GPU memory.
            runtime.CompileError: If ``required``, when a kernel does not compile.

        """
        if self._cpu_only:
            return on_cpu()
        if runtime.forked_after_init():
            if self._required:
                raise RuntimeError(f"Cannot sample on Metal: {FORKED}.")
            return on_cpu()
        try:
            return on_gpu()
        except MemoryError:
            if self._required:
                raise
            return on_cpu()
        except runtime.CompileError:
            if self._required:
                raise
            self._cpu_only = True
            return on_cpu()

    def _capacity(self, key: ShotPlan, module: _Module, shots: int) -> int:
        """Return the rows to make room for: the expected accepted shots and a margin.

        The share comes from the previous call of ``key``, or else the screen's
        estimate (which the tapes' postselection only lowers).
        """
        share = self._shares.get(key, module.passing if module.has_mask else 1.0)
        expected = shots * min(1.0, share)
        return min(shots, int(1.05 * expected + 8 * np.sqrt(expected) + 4096))

    def _run_accepted(self, call: _Call, shots: int, batch: int, capacity: int) -> None:
        """Run a call's batches into ``call.rows`` of ``capacity`` rows."""
        slot = call.slot
        starts = range(0, shots, batch)
        fixed = runtime.buffer(len(starts), np.uint32)
        slot.counts.array[:] = 0
        slot.counts.array[_CAPACITY] = capacity
        commands = []
        try:
            for k, start in enumerate(starts):
                n = min(batch, shots - start)
                fixed.array[k] = n
                encoder = runtime.Encoder()
                self._encode_batch(encoder, call, n, start)
                self._encode_pack(encoder, call, n, (fixed, 4 * k))
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

    def _accepted(
        self,
        shots: int,
        columns: list[np.ndarray],
        *,
        mask: np.ndarray | None = None,
        flip: np.ndarray | None = None,
        batch_size: int | None = None,
        bools: bool = False,
    ) -> tuple[list[np.ndarray], float]:
        """Sample ``shots`` shots; return only those that pass postselection.

        A shot passes when none of the ``mask``'s detectors is set (every shot
        passes without a mask). Rows keep their shots' order. Each entry of
        ``columns`` is an array of its outputs, in its order, selected on the
        GPU: bit-packed as ``np.packbits(..., bitorder="little")`` would
        (``uint8``, in shared memory), or with ``bools`` one boolean per output.

        Returns:
            ``(outputs, deviation)``: the passing shots' arrays and the largest
            normalization deviation beyond float32 rounding.

        Raises:
            MemoryError: If the GPU runs out of memory.

        """
        if shots > MAX_CALL_SHOTS:
            parts = [
                self._accepted(
                    min(MAX_CALL_SHOTS, shots - start),
                    columns,
                    mask=mask,
                    flip=flip,
                    batch_size=batch_size,
                    bools=bools,
                )
                for start in range(0, shots, MAX_CALL_SHOTS)
            ]
            outputs = [
                np.concatenate(cut) for cut in zip(*(p[0] for p in parts), strict=True)
            ]
            return outputs, max(p[1] for p in parts)
        with self._lock:
            return self._sample_call(
                shots, columns, mask=mask, flip=flip, batch_size=batch_size, bools=bools
            )

    def _sample_call(
        self,
        shots: int,
        columns: list[np.ndarray],
        *,
        mask: np.ndarray | None,
        flip: np.ndarray | None,
        batch_size: int | None,
        bools: bool,
    ) -> tuple[list[np.ndarray], float]:
        """One call of ``_accepted``, of at most ``MAX_CALL_SHOTS`` shots."""
        module = self._module(mask)
        nb = -(-self.layout.num_outputs // 8)
        if not shots:
            if bools:
                return [np.zeros((0, len(c)), bool) for c in columns], 0.0
            return [np.zeros((0, -(-len(c) // 8)), np.uint8) for c in columns], 0.0
        constant = self._constant(flip)
        batch = self._batch_shots(shots, batch_size)
        slot = self._slot(batch)
        key = module.plan
        state = self._rng.bit_generator.state
        capacity = self._capacity(key, module, shots)
        while True:
            # NOTE: a command buffer makes all of its buffers resident, so the
            # output has room for the expected rows only; a call that overflows
            # it runs again, on the same streams, with room for every shot.
            rows = runtime.buffer(capacity * nb, np.uint8, recycle=True)
            call = _Call(module, slot, constant, module.split, rows)
            self._run_accepted(call, shots, batch, capacity)
            total = int(slot.counts.array[_CURSOR])
            if total <= capacity:
                break
            self._rng.bit_generator.state = state
            capacity = shots
        self._shares[key] = total / max(shots, 1)
        # NOTE: fast math records a NaN total as NaN (not inf), which must warn.
        deviation = tape.deviation(slot.counts.array[_DEVIATION])
        return self._select(module, rows, total, columns, bools), deviation

    def _unpack(self, module: _Module, rows: runtime.Buffer, total: int) -> np.ndarray:
        """Return the first ``total`` packed ``rows`` as booleans, unpacked on the GPU."""
        num_outputs = self.layout.num_outputs
        nb = -(-num_outputs // 8)
        groups = -(-num_outputs // 4)
        if total * groups > MAX_COPY_ITEMS:
            packed = rows.array[: total * nb].reshape(total, nb)
            return _host_select(packed, np.arange(num_outputs), num_outputs, False)
        out = runtime.buffer(total * num_outputs, np.uint8, recycle=True)
        if total and num_outputs:
            threads = min(total * groups, MAX_COPY_THREADS)
            encoder = runtime.Encoder()
            encoder.dispatch(
                module.library.pipeline("unpack_rows"),
                [rows, out, np.uint32(total)],
                -(-threads // COPY_THREADS),
                COPY_THREADS,
            )
            self._wait_all([encoder.commit()])
        return out.array[: total * num_outputs].reshape(total, num_outputs).view(bool)

    def _select(
        self,
        module: _Module,
        rows: runtime.Buffer,
        total: int,
        columns: list[np.ndarray],
        bools: bool,
    ) -> list[np.ndarray]:
        """Return the outputs ``columns`` of the first ``total`` ``rows``, an array each.

        Bit-packed, or one boolean per output with ``bools``. All outputs in order
        are the rows (or ``_unpack``), leading whole bytes a view of them, and
        other columns are selected on the GPU.
        """
        num_outputs = self.layout.num_outputs
        nb = -(-num_outputs // 8)
        packed = rows.array[: total * nb].reshape(total, nb)
        selected: list[np.ndarray] = []
        encoder = None
        for chosen in columns:
            count = len(chosen)
            leading = np.array_equal(chosen, np.arange(count))
            if leading and count == num_outputs:
                selected.append(self._unpack(module, rows, total) if bools else packed)
                continue
            if leading and not bools and count % 8 == 0:
                selected.append(packed[:, : count // 8])
                continue
            width = count if bools else -(-count // 8)
            if total * width > MAX_COPY_ITEMS:
                found = _host_select(packed, chosen, num_outputs, not bools)
                selected.append(np.ascontiguousarray(found))
                continue
            out = runtime.buffer(total * width, np.uint8, recycle=True)
            if total * width:
                threads = min(total * width, MAX_COPY_THREADS)
                arguments = np.array([total, count, width, int(bools)], np.uint32)
                encoder = encoder or runtime.Encoder()
                encoder.dispatch(
                    module.library.pipeline("select_rows"),
                    [rows, out, runtime.upload(chosen.astype(np.uint32)), arguments],
                    -(-threads // COPY_THREADS),
                    COPY_THREADS,
                )
            array = out.array[: total * width].reshape(total, width)
            selected.append(array.view(bool) if bools else array)
        if encoder is not None:
            self._wait_all([encoder.commit()])
        return selected

    def direct_bools(
        self,
        shots: int,
        flip: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return every output bit of a program without components, as booleans."""
        every = [np.arange(self.layout.num_outputs)]
        return self._or_cpu(
            lambda: self._accepted(
                shots, every, flip=flip, batch_size=batch_size, bools=True
            )[0][0],
            lambda: self._host.direct_bools(shots, flip, batch_size),
        )


def _threads_per_shot(layout) -> int:
    """Return log2 of a shot's threads, as ``tape.threads_per_shot`` from ``WIDE_TAPE`` on."""
    # NOTE: fewer registers per thread pay for the shuffles once tapes are wide: on
    # an M4 Pro, 32 threads of 16 amplitudes run 9 qubits 1.4x and 8 qubits 1.1-1.2x
    # faster than 16 threads of 32; at 7 qubits they tie, at 6 four threads of 16
    # are 10% slower than two of 32.
    width = max((c.width for c in layout.components), default=0)
    if width >= WIDE_TAPE:
        return tape.threads_per_shot(layout, tape.AMPLITUDE_AXES - 1)
    return tape.threads_per_shot(layout)


def _count(slot: _Slot, word: int) -> tuple[runtime.Buffer, int]:
    """Return word ``word`` of ``slot.counts``, as a kernel argument."""
    return slot.counts, 4 * word


def _restore(
    host, channels, num_detectors, seed, required, state
) -> MetalShotSampler | ShotSampler:
    """Rebuild a pickled ``MetalShotSampler``, or its CPU sampler where Metal cannot run.

    As at compile time, a sampler ``required`` on the GPU raises where Metal cannot run.
    """
    from tsim.statevector import metal

    shots = metal.shot_sampler(host, channels, num_detectors, seed, required=required)
    if isinstance(shots, MetalShotSampler):
        shots._rng.bit_generator.state = state
    return shots


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
