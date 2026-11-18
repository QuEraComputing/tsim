"""CUDA version of ``fault_join.scan`` (prototype).

Same join as the numba kernel: a stored left side of ``w // 2`` faults, sorted by
a linear hash of their direct detectors and then by their last fault, and a
streamed right side of ``r`` faults. A thread takes one prefix ``(c0, c1)`` of
the right side and loops over the rest. A 2^29-bit filter of the left hashes,
small enough for the L2 cache, answers most lookups without touching the index.

``COLLECT`` gathers the distinct rotation signs in a hash set of 64-bit
fingerprints (a fingerprint collision would drop a sign vector: odds about
k^2 / 2^65 for k signs). ``EMIT`` keeps the sets that the table marks.
"""

from __future__ import annotations

import functools

import cupy as cp
import numpy as np

from tsim.noise import fault_join

MAX_WORDS = 4
FILTER_BITS = 29

_SOURCE = r"""
typedef unsigned long long u64;
#define MAXW 4

__device__ __forceinline__ u64 mix(u64 x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

struct Args {
    int mode, r, n, wd, ws, wo, small;
    const u64 *hashes, *dw, *sw, *ow;
    const int *lsets; const long long *llast; const u64 *ld, *ls, *lo;
    const u64 *run_keys; const long long *run_start, *run_end, *run_table; long long run_cap;
    const unsigned int *filter;
    u64 *set_fp; u64 *set_keys; int *set_reps; long long set_cap; unsigned long long *set_count;
    const u64 *tkeys, *tmasked, *det_mask, *obs_mask, *tobs; const unsigned char *tmulti;
    const long long *ttable; long long tcap;
    int *out_sets; unsigned long long *out_count; long long out_cap;
};

__device__ void visit(const Args &a, u64 h, const int *c) {
    u64 m = mix(h);
    unsigned int bit = (unsigned int)(m >> (64 - %(FILTER_BITS)d));
    if (!((a.filter[bit >> 5] >> (bit & 31)) & 1u)) return;
    long long s = (long long)(m & (u64)(a.run_cap - 1));
    long long i = -1;
    while (a.run_table[s] >= 0) {
        long long j = a.run_table[s];
        if (a.run_keys[j] == h) { i = j; break; }
        s = (s + 1) & (a.run_cap - 1);
    }
    if (i < 0) return;
    for (long long e = a.run_start[i]; e < a.run_end[i]; e++) {
        if (a.llast[e] >= c[0]) break;
        bool ok = true;
        for (int k = 0; k < a.wd; k++) {
            u64 word = a.ld[e * a.wd + k];
            for (int j = 0; j < a.r; j++) word ^= a.dw[(long long)c[j] * a.wd + k];
            if (word) { ok = false; break; }
        }
        if (!ok) continue;
        u64 S[MAXW], O[MAXW];
        for (int k = 0; k < a.ws; k++) {
            u64 word = a.ls[e * a.ws + k];
            for (int j = 0; j < a.r; j++) word ^= a.sw[(long long)c[j] * a.ws + k];
            S[k] = word;
        }
        for (int k = 0; k < a.wo; k++) {
            u64 word = a.lo[e * a.wo + k];
            for (int j = 0; j < a.r; j++) word ^= a.ow[(long long)c[j] * a.wo + k];
            O[k] = word;
        }
        int w = a.small + a.r;
        if (a.mode == 0) {
            u64 fp = 0x9E3779B97F4A7C15ULL;
            for (int k = 0; k < a.ws; k++) fp = mix(fp ^ S[k]);
            if (fp == 0) fp = 1;
            long long slot = (long long)(fp & (u64)(a.set_cap - 1));
            while (true) {
                // NOTE: a plain read first: most signs are already in the set,
                // and an atomic on a hot slot serializes the threads.
                u64 seen = *(volatile u64 *)&a.set_fp[slot];
                if (seen == fp) break;
                if (seen != 0ULL) { slot = (slot + 1) & (a.set_cap - 1); continue; }
                u64 old = atomicCAS(&a.set_fp[slot], 0ULL, fp);
                if (old == 0ULL) {
                    for (int k = 0; k < a.ws; k++) a.set_keys[slot * a.ws + k] = S[k];
                    for (int j = 0; j < a.small; j++) a.set_reps[slot * w + j] = a.lsets[e * a.small + j];
                    for (int j = 0; j < a.r; j++) a.set_reps[slot * w + a.small + j] = c[j];
                    atomicAdd(a.set_count, 1ULL);
                    break;
                }
                if (old == fp) break;
                slot = (slot + 1) & (a.set_cap - 1);
            }
        } else {
            u64 fp = 0x9E3779B97F4A7C15ULL;
            for (int k = 0; k < a.ws; k++) fp = mix(fp ^ S[k]);
            for (int k = 0; k < a.wo; k++) fp = mix(fp ^ (O[k] & a.det_mask[k]));
            long long slot = (long long)(fp & (u64)(a.tcap - 1));
            long long hit = -1;
            while (a.ttable[slot] >= 0) {
                long long j = a.ttable[slot];
                bool same = true;
                for (int k = 0; k < a.ws && same; k++) same = a.tkeys[j * a.ws + k] == S[k];
                for (int k = 0; k < a.wo && same; k++) same = a.tmasked[j * a.wo + k] == (O[k] & a.det_mask[k]);
                if (same) { hit = j; break; }
                slot = (slot + 1) & (a.tcap - 1);
            }
            if (hit < 0) continue;
            if (!a.tmulti[hit]) {
                bool same = true;
                for (int k = 0; k < a.wo; k++) same = same && ((O[k] & a.obs_mask[k]) == a.tobs[hit * a.wo + k]);
                if (same) continue;
            }
            unsigned long long pos = atomicAdd(a.out_count, 1ULL);
            if ((long long)pos < a.out_cap) {
                for (int j = 0; j < a.small; j++) a.out_sets[pos * w + j] = a.lsets[e * a.small + j];
                for (int j = 0; j < a.r; j++) a.out_sets[pos * w + a.small + j] = c[j];
            }
        }
    }
}

extern "C" __global__ void scan(Args a, long long first_prefix, long long num_prefix) {
    long long t = first_prefix + blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= first_prefix + num_prefix) return;
    int c[4];
    if (a.r == 1) {
        c[0] = (int)t;
        visit(a, a.hashes[c[0]], c);
        return;
    }
    c[0] = (int)(t / a.n);
    c[1] = (int)(t % a.n);
    if (c[1] <= c[0]) return;
    u64 h01 = a.hashes[c[0]] ^ a.hashes[c[1]];
    if (a.r == 2) { visit(a, h01, c); return; }
    for (c[2] = c[1] + 1; c[2] < a.n; c[2]++) {
        u64 h2 = h01 ^ a.hashes[c[2]];
        if (a.r == 3) { visit(a, h2, c); continue; }
        for (c[3] = c[2] + 1; c[3] < a.n; c[3]++) visit(a, h2 ^ a.hashes[c[3]], c);
    }
}
""".replace("%(FILTER_BITS)d", str(FILTER_BITS))


@functools.cache
def _kernel():
    return cp.RawModule(code=_SOURCE, options=("-std=c++17",)).get_function("scan")


def _mix(x: np.ndarray) -> np.ndarray:
    """NumPy version of the kernels' splitmix64 finalizer, on uint64 arrays."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _table_index(
    keys: np.ndarray, masked: np.ndarray, det_mask: np.ndarray
) -> np.ndarray:
    """Return the open-addressing index of the table rows, hashed as the kernel does."""
    fp = np.full(len(keys), 0x9E3779B97F4A7C15, dtype=np.uint64)
    for k in range(keys.shape[1]):
        fp = _mix(fp ^ keys[:, k])
    for k in range(masked.shape[1]):
        fp = _mix(fp ^ (masked[:, k] & det_mask[k]))
    capacity = 1 << max(1, (2 * len(keys)).bit_length())
    table = np.full(capacity, -1, dtype=np.int64)
    slots = (fp & np.uint64(capacity - 1)).astype(np.int64)
    for i, s in enumerate(slots.tolist()):
        while table[s] >= 0:
            s = (s + 1) & (capacity - 1)
        table[s] = i
    return table


class CudaJoin:
    """The device copy of a ``faults._Join``, scanned on the GPU."""

    def __init__(self, join) -> None:
        """Copy ``join``'s faults, stored sets and run index to the device."""
        d, s, o, hashes = join._faults
        for words in (d, s, o):
            if words.shape[1] > MAX_WORDS:
                raise ValueError("too many words for the CUDA prototype")
        sets, last, ld, ls, lo = join._left
        keys, start, end, table = join._runs
        self._n = len(hashes)
        self._r = join._large
        self._weight = join._weight
        self._widths = (d.shape[1], s.shape[1], o.shape[1])
        self._small = sets.shape[1]
        up = cp.asarray
        self._faults = tuple(up(np.ascontiguousarray(a)) for a in (hashes, d, s, o))
        self._left = (
            (
                up(np.ascontiguousarray(sets, dtype=np.int32).reshape(-1))
                if self._small
                else up(np.zeros(1, np.int32))
            ),
            up(last.astype(np.int64)),
            up(np.ascontiguousarray(ld)),
            up(np.ascontiguousarray(ls)),
            up(np.ascontiguousarray(lo)),
        )
        self._runs = (up(keys), up(start), up(end), up(table), len(table))
        bits = (_mix(keys) >> np.uint64(64 - FILTER_BITS)).astype(np.int64)
        words = np.zeros(1 << (FILTER_BITS - 5), dtype=np.uint32)
        np.bitwise_or.at(
            words, bits >> 5, (np.uint32(1) << (bits & 31).astype(np.uint32))
        )
        self._filter = up(words)

    def _args(self, mode, collect, emit):
        """Return the kernel's ``Args`` struct as a NumPy structured scalar."""
        hashes, d, s, o = self._faults
        lsets, llast, ld, ls, lo = self._left
        run_keys, run_start, run_end, run_table, run_cap = self._runs
        wd, ws, wo = self._widths
        set_fp, set_keys, set_reps, set_cap, set_count = collect
        (
            tkeys,
            tmasked,
            det,
            obs,
            tmulti,
            tobs,
            ttable,
            tcap,
            out,
            out_count,
            out_cap,
        ) = emit
        ints = ["mode", "r", "n", "wd", "ws", "wo", "small"]
        fields: list[tuple[str, type]] = [(name, np.int32) for name in ints]
        values = [mode, self._r, self._n, wd, ws, wo, self._small]
        pointers = [
            ("hashes", hashes),
            ("dw", d),
            ("sw", s),
            ("ow", o),
            ("lsets", lsets),
            ("llast", llast),
            ("ld", ld),
            ("ls", ls),
            ("lo", lo),
            ("run_keys", run_keys),
            ("run_start", run_start),
            ("run_end", run_end),
            ("run_table", run_table),
            ("run_cap", run_cap),
            ("filter", self._filter),
            ("set_fp", set_fp),
            ("set_keys", set_keys),
            ("set_reps", set_reps),
            ("set_cap", set_cap),
            ("set_count", set_count),
            ("tkeys", tkeys),
            ("tmasked", tmasked),
            ("det_mask", det),
            ("obs_mask", obs),
            ("tobs", tobs),
            ("tmulti", tmulti),
            ("ttable", ttable),
            ("tcap", tcap),
            ("out_sets", out),
            ("out_count", out_count),
            ("out_cap", out_cap),
        ]
        for name, value in pointers:
            if isinstance(value, int):
                fields.append((name, np.int64))
                values.append(value)
            else:
                fields.append((name, np.uint64))
                values.append(value.data.ptr)
        dtype = np.dtype(fields, align=True)
        return np.array([tuple(values)], dtype=dtype)[0]

    progress = None
    """Called with (prefixes done, prefixes in all, seconds) after each launch."""

    stop_after = None
    """If set, stop after this many prefixes (for timing a sample)."""

    def _launch(self, args) -> None:
        import time

        total = self._n if self._r == 1 else self._n * self._n
        block = 128
        # NOTE: launches of a bounded size keep each under the display watchdog.
        chunk = 1 << 20 if self._r >= 4 else 1 << 26
        start = time.monotonic()
        # NOTE: prefixes (c0, c1) are visited from the largest c0 down, so a
        # sample spans the cheap and the expensive ones evenly when shuffled.
        firsts = list(range(0, total, chunk))
        if self.stop_after is not None:
            rng = np.random.default_rng(0)
            firsts = sorted(
                rng.permutation(firsts)[: max(1, -(-self.stop_after // chunk))].tolist()
            )
        for i, first in enumerate(firsts):
            count = min(chunk, total - first)
            grid = -(-count // block)
            _kernel()((grid,), (block,), (args, np.int64(first), np.int64(count)))
            if self.progress is not None:
                cp.cuda.Device().synchronize()
                self.progress(
                    (i + 1) * chunk,
                    len(firsts) * chunk,
                    total,
                    time.monotonic() - start,
                )
        cp.cuda.Device().synchronize()

    def _dummy_emit(self):
        z64 = cp.zeros(1, cp.uint64)
        return (
            z64,
            z64,
            z64,
            z64,
            cp.zeros(1, cp.uint8),
            z64,
            cp.full(1, -1, cp.int64),
            1,
            cp.zeros(1, cp.int32),
            cp.zeros(1, cp.uint64),
            0,
        )

    def collect(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the distinct signs of the sets, and one set of each."""
        ws = self._widths[1]
        capacity = 1 << 16
        while True:
            set_fp = cp.zeros(capacity, cp.uint64)
            set_keys = cp.zeros(capacity * ws, cp.uint64)
            set_reps = cp.zeros(capacity * self._weight, cp.int32)
            set_count = cp.zeros(1, cp.uint64)
            collect = (set_fp, set_keys, set_reps, capacity, set_count)
            self._launch(self._args(fault_join.COLLECT, collect, self._dummy_emit()))
            if int(set_count.get()[0]) * 2 < capacity:
                break
            capacity *= 8
        used = cp.flatnonzero(set_fp != 0)
        signs = set_keys.reshape(capacity, ws)[used].get()
        sets = set_reps.reshape(capacity, self._weight)[used].get()
        return signs, sets

    def emit(self, table: tuple) -> np.ndarray:
        """Return the sets whose signs and output flips ``table`` marks."""
        keys, masked, det_mask, obs_mask, multi, obs, _ = table
        index = _table_index(keys, masked, det_mask)
        up = cp.asarray
        capacity = 1 << 20
        z64 = cp.zeros(1, cp.uint64)
        collect = (z64, z64, cp.zeros(1, cp.int32), 1, z64)
        while True:
            out = cp.zeros(capacity * self._weight, cp.int32)
            count = cp.zeros(1, cp.uint64)
            emit = (
                up(np.ascontiguousarray(keys)),
                up(np.ascontiguousarray(masked)),
                up(det_mask),
                up(obs_mask),
                up(multi.astype(np.uint8)),
                up(np.ascontiguousarray(obs)),
                up(index),
                len(index),
                out,
                count,
                capacity,
            )
            self._launch(self._args(fault_join.EMIT, collect, emit))
            total = int(count.get()[0])
            if total <= capacity:
                return out.reshape(capacity, self._weight)[:total].get()
            capacity = 1 << total.bit_length()
