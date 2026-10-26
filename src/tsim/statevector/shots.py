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
from collections import deque
from dataclasses import dataclass

import numba
import numpy as np
import psutil

import tsim.statevector.shared as shared
from tsim.core.types import CompiledProgram
from tsim.noise.channels import _BLOCK_SHOTS, ChannelSampler, WordTables
from tsim.statevector.program import STABILIZER_RANK_HINT, StatevectorComponent
from tsim.statevector.shared import KeyBasis, LaneProgram, key_basis, lane_program
from tsim.utils.kernels import PARALLEL_WORK, jit, kernel, run_tasks

# NOTE: beyond 2^16 shots a chunk costs memory without sharing more, and below
# 2^8 its fixed costs dominate.
MAX_CHUNK_SHOTS = 2**16
MIN_CHUNK_SHOTS = 2**8


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
    components: tuple[StatevectorComponent, ...]
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
        assert isinstance(component, StatevectorComponent)
        tape = component.tape
        selection = np.asarray(tape.f_selection, dtype=np.int64)
        output_bits = shared.unpack_symbols(tape.output_row, component.num_words)
        slot, f = np.nonzero(output_bits[:, : len(selection)])
        f_index.append(selection[f])
        bit.append(np.asarray(component.output_indices, dtype=np.int64)[slot])
        constant[list(component.output_indices)] = tape.output_const
        key = key_basis(component)
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
        kept: The outputs a discarded shot keeps (its direct detectors), as words.
        programs: Each component's lane program.

    """

    screen: np.ndarray
    kept: np.ndarray
    programs: tuple[LaneProgram, ...]


def shot_plan(
    layout: ShotLayout, num_detectors: int, mask: np.ndarray | None
) -> ShotPlan:
    """Return the plan of ``layout`` for a postselection ``mask`` over the detectors."""
    num_outputs, words = layout.num_outputs, layout.output_words
    postselected = np.zeros(num_outputs, dtype=bool)
    if mask is not None:
        postselected[:num_detectors] = mask
    is_detector = np.arange(num_outputs) < num_detectors
    programs = tuple(
        lane_program(
            component, key, key_start, postselected[list(component.output_indices)]
        )
        for component, key, key_start in zip(
            layout.components, layout.keys, layout.key_starts, strict=True
        )
    )
    return ShotPlan(
        screen=shared.pack_bits(postselected & layout.direct, words),
        kept=shared.pack_bits(layout.direct & is_detector, words),
        programs=programs,
    )


@jit
def _screen(words, constant, screen, kept, alive):
    """XOR the constants into the outputs; a screened shot keeps only ``kept``."""
    for s in range(words.shape[0]):
        hit = np.uint64(0)
        for w in range(len(constant)):
            words[s, w] ^= constant[w]
            hit |= words[s, w] & screen[w]
        alive[s] = hit == 0
        if hit:
            for w in range(len(kept)):
                words[s, w] &= kept[w]


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
    them, or whole bytes, the result is a view of ``words``; else a masked copy.
    """
    num_bytes = -(-count // 8)
    packed = words.view(np.uint8)[:, :num_bytes]
    if count % 8 == 0 or count == num_outputs:
        return packed
    packed = packed.copy()
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
        seed: int,
        *,
        share: bool,
    ) -> None:
        """Lay out the program's shot words; ``share`` lets shots share lanes.

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
        self._share = share
        self._seeds = np.random.default_rng([seed, 1])
        # NOTE: reference samples draw from their own stream, so that asking for
        # one leaves the samples of a seed alone.
        self._reference_seeds = np.random.default_rng([seed, 2])
        self._plans: dict[bytes, ShotPlan] = {}
        self._bool_tables: dict[bytes, WordTables] = {}
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
            map(shared.lane_bytes, self._layout.components, self._layout.keys),
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
            mask: The postselected detectors: a shot with a postselected direct
                detector of 1 keeps only its direct detectors, and the components
                prune shots at their postselected outputs.
            flip: A reference bit per output, XOR-ed in before the tests.

        """
        plan = self.plan(mask)
        constant = self._constant(flip)
        noise_seeds = self._channels.block_seeds(shots)
        seeds = self._next_seeds(self._seeds)
        num_outputs, part = self._layout.num_outputs, self._part_shots()

        def read(words: np.ndarray, c: np.ndarray) -> np.ndarray:
            return read_outputs(words, c, num_outputs, bit_packed=bit_packed)

        if shots <= part:
            words, deviation = self._sample_part(
                0, shots, noise_seeds, seeds, plan, constant, batch_size
            )
            return [read(words, c) for c in columns], deviation
        outputs = [
            (
                np.empty((shots, -(-len(c) // 8)), dtype=np.uint8)
                if bit_packed
                else np.empty((shots, len(c)), dtype=np.bool_)
            )
            for c in columns
        ]
        deviation = 0.0
        for start in range(0, shots, part):
            stop = min(start + part, shots)
            words, part_deviation = self._sample_part(
                start, stop, noise_seeds, seeds, plan, constant, batch_size
            )
            for out, c in zip(outputs, columns, strict=True):
                out[start:stop] = read(words, c)
            # NOTE: written so that a NaN deviation sticks.
            if part_deviation > deviation or part_deviation != part_deviation:
                deviation = part_deviation
        return outputs, deviation

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

        ``noise_seeds`` seed the call's noise blocks, so ``start`` is a whole
        number of blocks, and ``seeds`` its components.
        """
        shots = stop - start
        words = np.empty((shots, self._layout.num_words), dtype=np.uint64)
        if shots == 0:
            return words, 0.0
        blocks = noise_seeds[start // _BLOCK_SHOTS : -(-stop // _BLOCK_SHOTS)]
        self._channels.sample_words(words, self._layout.noise, blocks)
        if not self._layout.components and not (constant.any() or plan.screen.any()):
            # NOTE: direct outputs without constants are their noise bits.
            return words, 0.0
        threads = numba.get_num_threads()
        chunk, workers = self._schedule(shots, batch_size, threads)

        def run(offset: int) -> list[tuple]:
            rows = words[offset : offset + chunk]
            return self._run_chunk(rows, plan, constant, start + offset, seeds, threads)

        results = run_tasks(run, range(0, shots, chunk), workers)
        return words, self._record([stats for result in results for stats in result])

    def direct_bools(
        self, shots: int, columns: list[np.ndarray], flip: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Return the output bits ``columns`` of a program without components.

        The noise sampler writes them as booleans: a row holds every list of
        columns in turn, so the lists' arrays share each shot's faults. The
        samples equal those of ``sample``. ``flip`` is as in ``sample``.
        """
        assert not self._layout.components
        order = np.concatenate(columns).astype(np.int64)
        key = order.tobytes()
        if key not in self._bool_tables:
            f = self._layout.direct_f[order]
            sets = np.flatnonzero(f >= 0)
            self._bool_tables[key] = self._channels.word_tables(f[sets], sets, bits=1)
        out = np.empty((shots, len(order)), dtype=np.uint8)
        self._channels.sample_words(out, self._bool_tables[key])
        constant = (
            self._layout.constant if flip is None else self._layout.constant ^ flip
        )
        if constant[order].any():
            out ^= constant[order]
        # NOTE: the arrays are views of one array, as copies would cost more than
        # sampling them.
        bools = out.view(np.bool_)
        return np.split(bools, np.cumsum([len(c) for c in columns])[:-1], axis=1)

    def reference(self) -> tuple[np.ndarray, float]:
        """Return a noiseless sample of every output and its largest norm deviation."""
        words = np.zeros((1, self._layout.num_words), dtype=np.uint64)
        stats = self._run_chunk(
            words,
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

        A chunk per thread, of ``MIN_CHUNK_SHOTS`` or more, and at most
        ``MAX_CHUNK_SHOTS`` and ``batch_size``. A call that fits in one chunk runs
        alone, on parallel amplitude loops; otherwise as many chunks run at once
        as there are threads, if they fit in half of the free memory, and
        smaller ones, or fewer, if not.
        """
        fits = max(1, memory_budget() // self.bytes_per_shot())
        chunk = min(max(-(-shots // threads), MIN_CHUNK_SHOTS), MAX_CHUNK_SHOTS)
        if batch_size is not None:
            chunk = min(chunk, batch_size)
        if shots <= min(chunk, fits):
            return chunk, 1
        workers = min(threads, fits)
        chunk = max(1, min(chunk, fits // workers))
        return chunk, min(workers, -(-shots // chunk))

    def _run_chunk(
        self,
        rows: np.ndarray,
        plan: ShotPlan,
        constant: np.ndarray,
        first_shot: int,
        seeds: np.ndarray,
        threads: int,
    ) -> list[tuple]:
        """Screen the rows' shots and run the components on them.

        Alone on its thread, a chunk runs each component's amplitude loops on
        ``threads`` threads.
        """
        alive = np.empty(len(rows), dtype=np.bool_)
        _screen(rows, constant, plan.screen, plan.kept, alive)
        if not alive.any():
            return []
        return [
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
            for component, program, seed in zip(
                self._layout.components, plan.programs, seeds, strict=True
            )
        ]

    def _record(self, stats: list[tuple]) -> float:
        """Keep the lane counts; return the largest normalization deviation."""
        for _, groups, peak, size, dead in stats:
            self.sharing_stats.append(
                {"groups": groups, "peak lanes": peak, "lanes": size, "dead": dead}
            )
        # NOTE: a NaN deviation sticks.
        return float(np.max([stat[0] for stat in stats], initial=0.0))
