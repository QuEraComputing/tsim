"""Shot words: each shot's outputs and component keys as packed bits, without f-bits.

A statevector program needs a shot's noise only as parities of its f-bits: each
direct output is one f-bit (or a constant), each component output has an f-part,
and each component reads its key (``shared.KeyBasis``). So each f-variable gets a
column of bits in the *shot words*, and the noise sampler XORs a fault's columns
straight into its shot's words: no f matrix is built. The outputs come first, in
their original order, so the first ``num_outputs`` bits of a shot's words become
its sample once the constants are XOR-ed in and the components have added their
branch parts (``shared``).

``ShotSampler`` samples a call's noise, then runs its components on a chunk of
shots per thread. The noise blocks and every per-shot draw are seeded by the call
and the shot's index, so the samples depend on neither the chunks nor the threads.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from dataclasses import dataclass

import numba
import numpy as np
import psutil

import tsim.statevector.shared as shared
from tsim.core.types import CompiledProgram
from tsim.noise.channels import _BLOCK_SHOTS, ChannelSampler, WordTables
from tsim.statevector.conditioned import component as conditioned
from tsim.statevector.program import STABILIZER_RANK_HINT, StatevectorComponent
from tsim.statevector.shared import KeyBasis, LaneProgram, key_basis, lane_program
from tsim.utils.kernels import PARALLEL_WORK, jit, kernel, run_tasks

# NOTE: beyond 2^16 shots a chunk costs memory without sharing more, and below
# 2^8 its fixed costs dominate.
MAX_CHUNK_SHOTS = 2**16
MIN_CHUNK_SHOTS = 2**8
# NOTE: the most scratch memory a sampler keeps between calls.
MAX_SCRATCH_BYTES = 2**24


def memory_budget() -> int:
    """Return the memory sampling may use: half of the free memory."""
    return psutil.virtual_memory().available // 2


@dataclass(frozen=True, eq=False)
class ShotLayout:
    """Where a program's outputs and component keys live in the shot words.

    Attributes:
        num_outputs: Number of outputs: bits ``0 … num_outputs-1``, in order.
        output_words: Number of words that hold the outputs.
        num_words: Number of words per shot.
        constant: Each output's constant (direct flips, component constants).
        direct: Whether each output is a direct output.
        direct_f: Each direct output's f-variable, -1 for a constant or another
            output.
        components: The statevector components.
        keys: Each component's key basis.
        key_starts: First shot word of each component's key.
        noise: The words each fault outcome flips.

    """

    num_outputs: int
    output_words: int
    num_words: int
    constant: np.ndarray
    direct: np.ndarray
    direct_f: np.ndarray
    components: tuple[StatevectorComponent | conditioned.ConditionedComponent, ...]
    keys: tuple[KeyBasis, ...]
    key_starts: tuple[int, ...]
    noise: WordTables


def shot_layout(program: CompiledProgram, channels: ChannelSampler) -> ShotLayout:
    """Return the shot-word layout of a program whose components are statevector ones."""
    num_outputs = program.num_outputs
    output_words = -(-num_outputs // 64)
    constant = np.zeros(num_outputs, dtype=bool)
    direct = np.zeros(num_outputs, dtype=bool)
    outputs = program.output_order[: len(program.direct_f_indices)]
    constant[outputs] = program.direct_flips
    direct[outputs] = True
    direct_f = np.full(num_outputs, -1, dtype=np.int64)
    direct_f[outputs] = program.direct_f_indices
    # NOTE: the shot-word bits that each f-variable flips, as (f-variable, bit).
    sets = program.direct_f_indices >= 0
    f_index, bit = [program.direct_f_indices[sets]], [outputs[sets]]
    components, keys, key_starts = [], [], []
    word = output_words
    for component in program.components:
        assert isinstance(
            component, (StatevectorComponent, conditioned.ConditionedComponent)
        )
        tape = component.tape
        selection = np.asarray(tape.f_selection, dtype=np.int64)
        slot, f = np.nonzero(tape.output_row[:, : len(selection)])
        f_index.append(selection[f])
        bit.append(np.asarray(component.output_indices, dtype=np.int64)[slot])
        constant[list(component.output_indices)] = tape.output_const
        key = (
            component.key
            if isinstance(component, conditioned.ConditionedComponent)
            else key_basis(component)
        )
        row, f = np.nonzero(key.rows)
        f_index.append(selection[f])
        bit.append(64 * word + row)
        components.append(component)
        keys.append(key)
        key_starts.append(word)
        word += -(-len(key.rows) // 64)
    return ShotLayout(
        num_outputs=num_outputs,
        output_words=output_words,
        num_words=word,
        constant=constant,
        direct=direct,
        direct_f=direct_f,
        components=tuple(components),
        keys=tuple(keys),
        key_starts=tuple(key_starts),
        noise=channels.word_tables(np.concatenate(f_index), np.concatenate(bit)),
    )


@dataclass(frozen=True, eq=False)
class ShotPlan:
    """How to finish a layout's shots under a postselection mask.

    Attributes:
        screen: The postselected direct outputs, as words: a shot with one of
            them set is discarded before the components.
        programs: Each component's lane program.
        rejected: Whether any detector is postselected: if so, only the shots
            that pass the screen and every component are returned.

    """

    screen: np.ndarray
    programs: tuple[LaneProgram, ...]
    rejected: bool


def shot_plan(
    layout: ShotLayout, num_detectors: int, mask: np.ndarray | None
) -> ShotPlan:
    """Return the plan of ``layout`` for a postselection ``mask`` over the detectors."""
    num_outputs, words = layout.num_outputs, layout.output_words
    postselected = np.zeros(num_outputs, dtype=bool)
    if mask is not None:
        postselected[:num_detectors] = mask
    programs = tuple(
        component.program(key_start, postselected[list(component.output_indices)])
        if isinstance(component, conditioned.ConditionedComponent)
        else lane_program(
            component, key, key_start, postselected[list(component.output_indices)]
        )
        for component, key, key_start in zip(
            layout.components, layout.keys, layout.key_starts, strict=True
        )
    )
    return ShotPlan(
        screen=shared.pack_bits(postselected & layout.direct, words),
        programs=programs,
        rejected=bool(postselected.any()),
    )


@jit
def _screen(words, constant, screen, alive):
    """XOR the constants into the outputs; a shot with a ``screen`` bit is not alive."""
    for s in range(words.shape[0]):
        hit = np.uint64(0)
        for w in range(len(constant)):
            words[s, w] ^= constant[w]
            hit |= words[s, w] & screen[w]
        alive[s] = hit == 0


@jit
def _spread(bits, out, target):
    """Write the 64 ``bits`` to ``out[target … target+7]``, a byte of 0 or 1 each."""
    ones = np.uint64(0x0101_0101_0101_0101)
    for i in range(8):
        byte = bits >> np.uint64(8 * i) & np.uint64(0xFF)
        # NOTE: byte k of the product keeps bit k; adding 0x7F carries it to bit 7.
        x = byte * ones & np.uint64(0x8040_2010_0804_0201)
        out[target + i] = (x + np.uint64(0x7F7F_7F7F_7F7F_7F7F)) >> np.uint64(7) & ones


@kernel
def _unpack(words, runs, width, out):
    """Write each shot's runs of output bits to its row of ``width`` bytes in ``out``.

    Run ``(bit, length)`` holds bits ``bit … bit+length-1``, and a row holds its
    runs in order. The rows form one stream of bits, and each 64 bits of it fill
    8 words of ``out``: 64 rows end on a whole word, so blocks of 64 rows run in
    parallel. ``out`` holds whole blocks of 8 words.
    """
    num_rows, num_words = words.shape
    for block in numba.prange((num_rows + 63) >> 6):
        target = 8 * width * block
        bits, filled = np.uint64(0), 0
        for s in range(64 * block, min(64 * block + 64, num_rows)):
            for r in range(runs.shape[0]):
                bit, length = runs[r, 0], runs[r, 1]
                while length > 0:
                    take = min(length, 64 - filled)
                    w, shift = bit >> 6, np.uint64(bit & 63)
                    chunk = words[s, w] >> shift
                    if shift != 0 and w + 1 < num_words:
                        chunk |= words[s, w + 1] << (np.uint64(64) - shift)
                    chunk &= np.uint64(0xFFFF_FFFF_FFFF_FFFF) >> np.uint64(64 - take)
                    bits |= chunk << np.uint64(filled)
                    filled += take
                    bit += take
                    length -= take
                    if filled == 64:
                        _spread(bits, out, target)
                        target += 8
                        bits, filled = np.uint64(0), 0
        if filled:
            _spread(bits, out, target)


@jit
def _compact(rows, alive, width):
    """Move the first ``width`` words of each alive shot to the front, in order.

    Returns the number of alive shots.
    """
    target = 0
    for s in range(rows.shape[0]):
        if alive[s]:
            if target != s:
                for w in range(width):
                    rows[target, w] = rows[s, w]
            target += 1
    return target


@kernel
def _copy_kept(words, keep, size, kept, starts, out):
    """Copy the rows ``keep`` flags in each run of ``size`` rows to ``out``, in order.

    Run ``i`` keeps ``kept[i]`` rows, from row ``starts[i]`` of ``out`` on; runs
    copy in parallel.
    """
    width = out.shape[1]
    for i in numba.prange(starts.shape[0]):
        first = i * size
        last = min(first + size, words.shape[0]) - 1
        while last >= first and not keep[last]:
            last -= 1
        target = starts[i]
        if 2 * kept[i] < last + 1 - first:
            # NOTE: few kept rows: copy only them.
            for s in range(first, last + 1):
                if keep[s]:
                    for w in range(width):
                        out[target, w] = words[s, w]
                    target += 1
        elif width == 1:
            # NOTE: every row up to the run's last kept one is written to
            # ``target``, which only a kept row moves on, so the loop has no
            # branch and never writes past the run's rows of ``out``.
            for s in range(first, last + 1):
                out[target, 0] = words[s, 0]
                target += keep[s]
        else:
            for s in range(first, last + 1):
                for w in range(width):
                    out[target, w] = words[s, w]
                target += keep[s]


@jit
def _join(words, kept, size, width):
    """Move the ``kept[i]`` rows at the front of each run of ``size`` rows together.

    Only the first ``width`` words move. Returns the number of rows kept.
    """
    filled = 0
    for i in range(kept.shape[0]):
        source = i * size
        if source != filled:
            for r in range(kept[i]):
                for w in range(width):
                    words[filled + r, w] = words[source + r, w]
        filled += kept[i]
    return filled


def _shrink(words: np.ndarray, rows: int) -> np.ndarray:
    """Return the first ``rows`` rows of ``words``, an array no view shares.

    The memory past them is given back without a copy, so that a view of the
    survivors does not keep a whole call's rows alive.

    Warning: the resize skips numpy's reference check, so ``words`` must have no
    other view left, or that view would read freed memory.
    """
    words.resize((rows, words.shape[1]), refcheck=False)
    return words


def bool_outputs(words: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Return the output bits ``columns`` of every shot as booleans."""
    columns = np.asarray(columns, dtype=np.int64)
    starts = np.flatnonzero(np.diff(columns, prepend=-2) != 1)
    lengths = np.diff(np.append(starts, len(columns)))
    runs = np.stack([columns[starts], lengths], axis=1)
    size = len(words) * len(columns)
    out = np.empty(8 * -(-size // 64), dtype=np.uint64)
    _unpack(size, words, runs, len(columns), out)
    return out.view(np.bool_)[:size].reshape(len(words), len(columns))


def packed_outputs(words: np.ndarray, count: int, num_outputs: int) -> np.ndarray:
    """Return the first ``count`` of ``num_outputs`` output bits as ``np.packbits`` does.

    Bits are in little order. The bits past the outputs are zero, so with all of
    them, or whole bytes, of words that hold only outputs the result is a view of
    ``words``; else a copy, masked past ``count``. A view of words that also hold
    component keys would keep the keys alive.
    """
    num_bytes = -(-count // 8)
    packed = words.view(np.uint8)[:, :num_bytes]
    only_outputs = words.shape[1] == -(-num_outputs // 64)
    if only_outputs and (count % 8 == 0 or count == num_outputs):
        return packed
    packed = packed.copy()
    if count % 8:
        packed[:, -1] &= np.uint8((1 << (count % 8)) - 1)
    return packed


def read_outputs(
    words: np.ndarray, columns: np.ndarray, num_outputs: int, *, bit_packed: bool
) -> np.ndarray:
    """Return the output bits ``columns`` of shot words, as booleans or bit-packed.

    Leading outputs in order are packed straight from the words.
    """
    if bit_packed and np.array_equal(columns, np.arange(len(columns))):
        return packed_outputs(words, len(columns), num_outputs)
    bits = bool_outputs(words, columns)
    return np.packbits(bits, axis=1, bitorder="little") if bit_packed else bits


class ShotSampler:
    """Samples shot words of a program whose components are statevector ones."""

    def __init__(
        self,
        program: CompiledProgram,
        channels: ChannelSampler,
        num_detectors: int,
        seed: np.random.SeedSequence,
    ) -> None:
        """Lay out the program's shot words.

        Raises:
            NotImplementedError: On a big-endian host.
            ValueError: If one shot needs more than half of the free memory.

        """
        # NOTE: a little-endian word's bytes hold its bits in packbits' "little"
        # order, which the output readers rely on.
        if sys.byteorder != "little":
            raise NotImplementedError("shot words need a little-endian host")
        self._layout = shot_layout(program, channels)
        self._channels = channels
        self._num_detectors = num_detectors
        # NOTE: private switch for tests and benchmarks; samples do not depend on it.
        self._share = True
        components, reference = seed.spawn(2)
        self._seeds = np.random.default_rng(components)
        # NOTE: reference samples draw from their own stream, so that asking for
        # one leaves the samples of a seed alone.
        self._reference_seeds = np.random.default_rng(reference)
        self._plans: dict[bytes, ShotPlan] = {}
        self._widest = max((c.width for c in self._layout.components), default=0)
        self._conditioned = bool(self._layout.components) and all(
            isinstance(c, conditioned.ConditionedComponent)
            for c in self._layout.components
        )
        self._direct_tables: WordTables | None = None
        # NOTE: noise words of a program without components, reused by the calls
        # that screen them, since only their survivors are returned.
        self._scratch: np.ndarray | None = None
        self._scratch_lock = threading.Lock()
        if not self._layout.components:
            f = self._layout.direct_f
            sets = np.flatnonzero(f >= 0)
            self._direct_tables = channels.word_tables(f[sets], sets, bits=1)
        # NOTE: lane counts of the latest chunks, for tests and diagnostics.
        self.sharing_stats: deque[dict] = deque(maxlen=10_000)
        peak = self.bytes_per_shot()
        if self._layout.components and peak > memory_budget():
            widest = max(self._layout.components, key=lambda c: c.width)
            raise ValueError(
                f"strategy='active_statevector' needs {peak} bytes per shot for the "
                f"component with outputs {list(widest.output_indices)} (width "
                f"{widest.width}), more than half of the available memory. "
                f"{STABILIZER_RANK_HINT}"
            )

    @property
    def layout(self) -> ShotLayout:
        """Return the shot words' layout."""
        return self._layout

    def bytes_per_shot(self) -> int:
        """Return the peak memory of one shot.

        That is its words, its own lane at worst, and 32 bytes for its place in
        the executor's shot lists.
        """
        lane = max(
            (
                16 << component.width
                if isinstance(component, conditioned.ConditionedComponent)
                else shared.lane_bytes(component, key)
                for component, key in zip(
                    self._layout.components, self._layout.keys, strict=True
                )
            ),
            default=0,
        )
        return 8 * self._layout.num_words + lane + 32

    def plan(self, mask: np.ndarray | None) -> ShotPlan:
        """Return the plan for a postselection ``mask``."""
        key = b"" if mask is None else np.asarray(mask, bool).tobytes()
        if key not in self._plans:
            self._plans[key] = shot_plan(self._layout, self._num_detectors, mask)
        return self._plans[key]

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

        Each list of columns gives an array, as ``read_outputs`` reads it. A call
        whose shot words do not fit in a quarter of the free memory runs in parts
        of whole noise blocks, which draw the samples of the whole call.

        Args:
            shots: Number of shots.
            columns: The output columns of each array.
            bit_packed: Whether the arrays are bit-packed.
            batch_size: The most shots a thread runs at once; it bounds memory and
                does not change the samples.
            mask: The postselected detectors: only shots where they all read 0
                are returned. A shot with a postselected direct detector of 1
                skips the components, and the components prune shots at their
                postselected outputs.
            flip: A reference bit per output, XOR-ed in before the tests.

        """
        plan = self.plan(mask)
        num_outputs, part = self._layout.num_outputs, self._part_shots()
        if shots == 0:
            # NOTE: an empty call draws no seeds, so it leaves the next samples alone.
            words = np.empty((0, self._layout.num_words), dtype=np.uint64)
            return [
                read_outputs(words, c, num_outputs, bit_packed=bit_packed)
                for c in columns
            ], 0.0
        constant = self._constant(flip)
        noise_seeds = self._channels.block_seeds(shots)
        seeds = self._next_seeds(self._seeds)

        def read(words: np.ndarray) -> list[np.ndarray]:
            return [
                read_outputs(words, c, num_outputs, bit_packed=bit_packed)
                for c in columns
            ]

        if shots <= part:
            words, deviation = self._sample_part(
                0, shots, noise_seeds, seeds, plan, constant, batch_size
            )
            return read(words), deviation
        outputs = [
            (
                np.empty((shots, -(-len(c) // 8)), dtype=np.uint8)
                if bit_packed
                else np.empty((shots, len(c)), dtype=np.bool_)
            )
            for c in columns
        ]
        filled = 0
        deviation = 0.0
        for start in range(0, shots, part):
            stop = min(start + part, shots)
            words, part_deviation = self._sample_part(
                start, stop, noise_seeds, seeds, plan, constant, batch_size
            )
            arrays = read(words)
            count = len(arrays[0]) if arrays else stop - start
            for out, array in zip(outputs, arrays, strict=True):
                out[filled : filled + count] = array
            filled += count
            # NOTE: written so that a NaN deviation sticks.
            if part_deviation > deviation or part_deviation != part_deviation:
                deviation = part_deviation
        return [out[:filled] for out in outputs], deviation

    def _part_shots(self) -> int:
        """Return the shots of a part: whole noise blocks whose words fit in memory.

        That is a quarter of the free memory: half of the sampling budget.
        """
        fits = memory_budget() // (2 * 8 * max(1, self._layout.num_words))
        return _BLOCK_SHOTS * max(1, fits // _BLOCK_SHOTS)

    def _sample_part(
        self,
        start: int,
        stop: int,
        noise_seeds: np.ndarray,
        seeds: np.ndarray,
        plan: ShotPlan,
        constant: np.ndarray,
        batch_size: int | None,
    ) -> tuple[np.ndarray, float]:
        """Return the shot words of a call's shots ``start … stop-1`` and their deviation.

        With a postselected detector, only the shots that pass postselection
        are returned, and only their output words. ``noise_seeds`` seed the
        call's noise blocks, so ``start`` is a whole number of blocks, and
        ``seeds`` its components.
        """
        shots = stop - start
        width = self._layout.output_words
        blocks = noise_seeds[start // _BLOCK_SHOTS : -(-stop // _BLOCK_SHOTS)]
        if shots and not self._layout.components and plan.rejected:
            return self._sample_screened(shots, blocks, plan, constant), 0.0
        words = np.empty((shots, self._layout.num_words), dtype=np.uint64)
        if shots == 0:
            return words, 0.0
        self._channels.sample_words(words, self._layout.noise, blocks)
        if not self._layout.components and not (constant.any() or plan.screen.any()):
            # NOTE: direct outputs without constants are their noise bits.
            return words, 0.0
        threads = numba.get_num_threads()
        chunk, workers = self._schedule(shots, batch_size, threads)
        part_stats: list[tuple] = []
        alive_all: np.ndarray | None = None
        if self._conditioned:
            # NOTE: conditioned components run on the whole part at once, a key's
            # shots on one thread, so that no key's lanes are built twice.
            alive_all = np.empty(shots, dtype=np.bool_)
            part_stats = self._run_conditioned(
                words, alive_all, plan, constant, start, seeds, threads
            )

        def run(offset: int) -> tuple[list[tuple], int]:
            rows = words[offset : offset + chunk]
            if alive_all is not None:
                alive = alive_all[offset : offset + chunk]
                kept = _compact(rows, alive, width) if plan.rejected else len(rows)
                return [], kept
            alive = np.empty(len(rows), dtype=np.bool_)
            stats = self._run_chunk(
                rows, alive, plan, constant, start + offset, seeds, threads
            )
            # NOTE: each chunk moves its survivors to its front on its own thread.
            kept = _compact(rows, alive, width) if plan.rejected else len(rows)
            return stats, kept

        offsets = range(0, shots, chunk)
        results = run_tasks(run, offsets, workers)
        deviation = self._record(
            part_stats + [stats for result, _ in results for stats in result]
        )
        if not plan.rejected:
            return words, deviation
        kept = np.array([k for _, k in results], dtype=np.int64)
        filled = _join(words, kept, chunk, width)
        # NOTE: the output words alone, so that bit-packed outputs can be views.
        if width == words.shape[1]:
            return _shrink(words, filled), deviation
        # NOTE: a copy even of one row, which a view would not release.
        return words[:filled, :width].copy(), deviation

    def __getstate__(self) -> dict:
        """Pickle without the scratch words and their lock."""
        state = self.__dict__.copy()
        state["_scratch"] = None
        del state["_scratch_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Unpickle, with a new lock for the scratch words."""
        self.__dict__.update(state)
        self._scratch_lock = threading.Lock()

    def _sample_screened(
        self, shots: int, blocks: np.ndarray, plan: ShotPlan, constant: np.ndarray
    ) -> np.ndarray:
        """Return the output words of the shots that pass postselection, of ``shots``.

        For a program without components, which postselects direct detectors
        only, all of them in the screen. Each noise block is screened while it is
        in cache, and the survivors are copied out of words that later calls
        reuse: fresh memory would cost a page fault per 4 kB.
        """
        width = self._layout.num_words
        assert width == self._layout.output_words
        # NOTE: a call that finds the scratch taken (another thread's) allocates.
        with self._scratch_lock:
            scratch, self._scratch = self._scratch, None
        if scratch is None or len(scratch) < shots:
            scratch = np.empty((shots, width), dtype=np.uint64)
        words = scratch[:shots]
        keep, kept = self._channels.sample_screened_words(
            words, self._layout.noise, blocks, constant, plan.screen
        )
        starts = np.zeros(len(kept), dtype=np.int64)
        np.cumsum(kept[:-1], out=starts[1:])
        out = np.empty((int(kept.sum()), width), dtype=np.uint64)
        _copy_kept(len(out) * width, words, keep, _BLOCK_SHOTS, kept, starts, out)
        if scratch.nbytes <= MAX_SCRATCH_BYTES:
            with self._scratch_lock:
                self._scratch = scratch
        return out

    def direct_bools(self, shots: int, flip: np.ndarray | None = None) -> np.ndarray:
        """Return every output bit of a program without components, as booleans.

        The noise sampler writes them straight into the rows. The samples equal
        those of ``sample``. ``flip`` is as in ``sample``.
        """
        assert self._direct_tables is not None
        out = np.empty((shots, self._layout.num_outputs), dtype=np.uint8)
        self._channels.sample_words(out, self._direct_tables)
        constant = self._layout.constant
        if flip is not None:
            constant = constant ^ flip
        if constant.any():
            out ^= constant
        return out.view(np.bool_)

    def reference(self) -> tuple[np.ndarray, float]:
        """Return a noiseless sample of every output and its largest norm deviation."""
        words = np.zeros((1, self._layout.num_words), dtype=np.uint64)
        stats = self._run_chunk(
            words,
            np.empty(1, dtype=np.bool_),
            self.plan(None),
            self._constant(None),
            0,
            self._next_seeds(self._reference_seeds),
            1,
        )
        outputs = bool_outputs(words, np.arange(self._layout.num_outputs))[0]
        return outputs, self._record(stats)

    def _constant(self, flip: np.ndarray | None) -> np.ndarray:
        """Return the outputs' constants XOR the reference bits ``flip``, as words."""
        constant = self._layout.constant
        if flip is not None:
            constant = constant ^ flip
        return shared.pack_bits(constant, self._layout.output_words)

    def _next_seeds(self, stream: np.random.Generator) -> np.ndarray:
        """Return a seed per component for one call, from ``stream``."""
        count = len(self._layout.components)
        return stream.integers(0, 2**64, size=count, dtype=np.uint64)

    def _schedule(
        self, shots: int, batch_size: int | None, threads: int
    ) -> tuple[int, int]:
        """Return the shots of a chunk and how many chunks run at once.

        A chunk per thread, of at least ``MIN_CHUNK_SHOTS`` or the fewer shots
        whose amplitude work reaches ``PARALLEL_WORK``, and at most
        ``MAX_CHUNK_SHOTS`` and ``batch_size``. A call that fits in one chunk runs
        alone, on parallel amplitude loops; otherwise as many chunks run at once
        as there are threads, if they fit in half of the free memory, and
        smaller ones, or fewer, if not.
        """
        fits = max(1, memory_budget() // self.bytes_per_shot())
        # NOTE: the least size outweighs a chunk's fixed cost; a wide component's
        # shots do so sooner, so that its small calls still fill every thread.
        least = max(1, min(MIN_CHUNK_SHOTS, PARALLEL_WORK >> self._widest))
        chunk = min(max(-(-shots // threads), least), MAX_CHUNK_SHOTS)
        if batch_size is not None:
            chunk = min(chunk, batch_size)
        if shots <= min(chunk, fits):
            return chunk, 1
        workers = min(threads, fits)
        chunk = max(1, min(chunk, fits // workers))
        return chunk, min(workers, -(-shots // chunk))

    def _run_conditioned(
        self,
        words: np.ndarray,
        alive: np.ndarray,
        plan: ShotPlan,
        constant: np.ndarray,
        first_shot: int,
        seeds: np.ndarray,
        threads: int,
    ) -> list[tuple]:
        """Screen a part's shots and run its conditioned components on all of them."""
        _screen(words, constant, plan.screen, alive)
        stats = []
        for component, program, seed in zip(
            self._layout.components, plan.programs, seeds, strict=True
        ):
            if not alive.any():
                break
            stats.append(
                conditioned.run_part(
                    component, program, words, alive, first_shot, seed, threads
                )
            )
        return stats

    def _run_chunk(
        self,
        rows: np.ndarray,
        alive: np.ndarray,
        plan: ShotPlan,
        constant: np.ndarray,
        first_shot: int,
        seeds: np.ndarray,
        threads: int,
    ) -> list[tuple]:
        """Screen the rows' shots and run the components on them.

        ``alive`` receives whether each shot passed the screen and every
        component's postselected outputs: a shot that one prunes is skipped by
        the later ones, and its outputs are left as they are, since it is not
        returned. Alone on its thread, a chunk runs each component's amplitude
        loops on ``threads`` threads.
        """
        _screen(rows, constant, plan.screen, alive)
        stats = []
        for component, program, seed in zip(
            self._layout.components, plan.programs, seeds, strict=True
        ):
            if not alive.any():
                break
            if isinstance(component, conditioned.ConditionedComponent):
                stats.append(
                    conditioned.run_chunk(
                        component, program, rows, alive, first_shot, seed
                    )
                )
                continue
            stats.append(
                shared.run_lanes(
                    len(rows) << component.width,
                    rows,
                    alive,
                    first_shot,
                    seed,
                    self._share,
                    program,
                    shared._POISON,
                    PARALLEL_WORK,
                    threads,
                )
            )
        return stats

    def _record(self, stats: list[tuple]) -> float:
        """Keep the lane counts; return the largest normalization deviation."""
        for _, groups, peak, size, dead in stats:
            self.sharing_stats.append(
                {"groups": groups, "peak lanes": peak, "lanes": size, "dead": dead}
            )
        # NOTE: a NaN deviation sticks.
        return float(np.max([stat[0] for stat in stats], initial=0.0))
