"""A lane's state, and the operations on it.

The stored state is ``F^dag (|a>_active (x) |0>_dormant)``:

- ``frame``: the Clifford ``F`` as ``2n`` Pauli rows (``paulis``), so the virtual
  image of an op's Pauli ``P`` is ``F P F^dag``, and a last row whose first word
  counts the frame's changes (its *version*);
- ``active``: the virtual qubits of the dense register, bit ``i`` of an amplitude
  index is qubit ``active[i]``; ``pos[q]`` is ``i``, or -1 for a dormant qubit;
- ``amps``: the ``2^width`` amplitudes ``a``;
- ``cache``, ``cache_ver``: the virtual images of the ops, each valid while its
  version matches the frame's (most steps change no frame).

Gates act on virtual qubits (``F <- G F``, ``a <- G a``) and must keep the dormant
qubits in ``|0>``. Measurements take two calls: a ``prepare_*`` that does every
outcome-independent change and returns the probability of outcome 1, and a
``finish_*`` per outcome, so that a lane can split before finishing.
"""

from __future__ import annotations

import numpy as np

from tsim.statevector.conditioned.paulis import (
    gate_cx,
    gate_cz,
    gate_h,
    gate_pauli,
    gate_s,
    get_bit,
    image,
    popcount,
)
from tsim.utils.kernels import jit

H, S, S_DAG, X, Z, CX, CZ, CY = 0, 1, 2, 3, 4, 5, 6, 7

_ONE = np.uint64(1)
_THREE = np.uint64(3)
_SQRT_HALF = 0.7071067811865476
_I_POWERS = np.array([1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j])
# NOTE: an outcome whose probability is this close to 0 or 1 is certain.
CERTAIN = 1e-12


class TooWide(Exception):
    """A lane would need more amplitudes than allowed."""


# -- state ---------------------------------------------------------------------


@jit
def new_state(n: int, num_words: int, num_ops: int):
    """Return ``(frame, active, pos, amps, cache, cache_ver)`` of ``|0...0>``, identity frame."""
    frame = np.zeros((2 * n + 1, 2 * num_words + 1), dtype=np.uint64)
    for q in range(n):
        frame[q, q >> 6] = _ONE << np.uint64(q & 63)
        frame[n + q, num_words + (q >> 6)] = _ONE << np.uint64(q & 63)
    frame[2 * n, 0] = _ONE
    active = np.full(n, -1, dtype=np.int64)
    pos = np.full(n, -1, dtype=np.int64)
    amps = np.ones(1, dtype=np.complex128)
    cache = np.empty((num_ops, 2 * num_words + 1), dtype=np.uint64)
    cache_ver = np.zeros(num_ops, dtype=np.uint64)
    return frame, active, pos, amps, cache, cache_ver


@jit
def op_image(frame, cache, cache_ver, r: int, op_x, op_z, op_e, num_words: int):
    """Return the virtual image of op ``r``'s Pauli (a row of the cache)."""
    version = frame[frame.shape[0] - 1, 0]
    if cache_ver[r] != version:
        image(frame, op_x[r], op_z[r], op_e[r], cache[r], num_words)
        cache_ver[r] = version
    return cache[r]


@jit
def width_of(amps) -> int:
    """Return the number of active qubits, from the register's size."""
    return popcount(np.uint64(amps.shape[0] - 1))


# -- amplitude kernels on register bits ----------------------------------------


@jit
def _amp_gate(amps, code: int, i: int, j: int) -> None:
    """Apply a gate to register bits ``i`` (and ``j``) in place."""
    size = amps.shape[0]
    bi = 1 << i
    if code == H:
        for k in range(size):
            if not k & bi:
                a0, a1 = amps[k], amps[k | bi]
                amps[k] = (a0 + a1) * _SQRT_HALF
                amps[k | bi] = (a0 - a1) * _SQRT_HALF
    elif code in (S, S_DAG):
        f = 1j if code == S else -1j
        for k in range(size):
            if k & bi:
                amps[k] *= f
    elif code == X:
        for k in range(size):
            if not k & bi:
                amps[k], amps[k | bi] = amps[k | bi], amps[k]
    elif code == Z:
        for k in range(size):
            if k & bi:
                amps[k] = -amps[k]
    else:
        bj = 1 << j
        for k in range(size):
            if not k & bi:
                continue
            if code == CZ:
                if k & bj:
                    amps[k] = -amps[k]
            elif not k & bj:
                a0, a1 = amps[k], amps[k | bj]
                if code == CX:
                    amps[k], amps[k | bj] = a1, a0
                else:  # CY: |0> -> i|1>, |1> -> -i|0>
                    amps[k], amps[k | bj] = -1j * a1, 1j * a0


@jit
def apply_pauli(amps, xm: int, zm: int, e: int, out) -> None:
    """Write ``i^e X^xm Z^zm amps`` to ``out``, for masks over register bits."""
    phase = _I_POWERS[e & 3]
    for k in range(amps.shape[0]):
        sign = -phase if popcount(np.uint64(k & zm)) & 1 else phase
        out[k ^ xm] = sign * amps[k]


# -- gates ---------------------------------------------------------------------


@jit
def rows_gate(rows, code: int, a: int, b: int, num_words: int) -> None:
    """Conjugate every Pauli row by a gate (``R <- G R G^dag``)."""
    if code == H:
        gate_h(rows, a, num_words)
    elif code == S:
        gate_s(rows, a, num_words, False)
    elif code == S_DAG:
        gate_s(rows, a, num_words, True)
    elif code == X:
        gate_pauli(rows, a, num_words, 1)
    elif code == Z:
        gate_pauli(rows, a, num_words, 2)
    elif code == CX:
        gate_cx(rows, a, b, num_words)
    elif code == CZ:
        gate_cz(rows, a, b, num_words)
    else:  # CY = S_b CX S_b^dag, applied right to left
        gate_s(rows, b, num_words, True)
        gate_cx(rows, a, b, num_words)
        gate_s(rows, b, num_words, False)


@jit
def frame_gate(frame, code: int, a: int, b: int, num_words: int) -> None:
    """Apply a gate to the frame only (``F <- G F``), and bump its version."""
    last = frame.shape[0] - 1
    rows_gate(frame[:last], code, a, b, num_words)
    frame[last, 0] += _ONE


@jit
def gate(frame, pos, amps, code: int, a: int, b: int, num_words: int) -> None:
    """Apply a gate to the frame and the register; it must keep the dormant qubits in ``|0>``.

    On dormant qubits only phases, Z-type gates and dormant controls are allowed,
    which act trivially on ``|0>``.
    """
    frame_gate(frame, code, a, b, num_words)
    pa = pos[a]
    if code >= CX:
        pb = pos[b]
        if pa < 0 or (pb < 0 and code == CZ):
            return  # a dormant control, or a CZ onto a dormant |0>: trivial
        if pb < 0:
            raise ValueError("gate would leave a dormant qubit in a superposition")
        _amp_gate(amps, code, pa, pb)
        return
    if pa < 0:
        if code in (H, X):
            raise ValueError("gate would move a dormant qubit off |0>")
        return
    _amp_gate(amps, code, pa, 0)


@jit
def row_gate(row, code: int, a: int, b: int, num_words: int) -> None:
    """Apply a gate to one Pauli row (to follow a Pauli through gates)."""
    rows_gate(row.reshape(1, row.shape[0]), code, a, b, num_words)


# -- register size -------------------------------------------------------------


@jit
def promote(active, pos, amps, q: int, max_width: int):
    """Make the dormant qubit ``q`` (in ``|0>``) active; return the new amplitudes."""
    w = width_of(amps)
    if w + 1 > max_width:
        raise TooWide()
    out = np.zeros(2 * amps.shape[0], dtype=np.complex128)
    out[: amps.shape[0]] = amps
    active[w] = q
    pos[q] = w
    return out


@jit
def drop(frame, active, pos, amps, q: int, bit: int, num_words: int):
    """Remove the active qubit ``q``, whose bit is ``bit`` in every amplitude.

    Returns the new amplitudes. A dropped 1 is turned into a dormant 0 by an X in
    the frame.
    """
    w = width_of(amps)
    i = pos[q]
    out = np.empty(amps.shape[0] >> 1, dtype=np.complex128)
    low = (1 << i) - 1
    lost = 0.0
    for k in range(out.shape[0]):
        base = ((k & ~low) << 1) | (k & low)
        out[k] = amps[base | (bit << i)]
        other = amps[base | ((1 - bit) << i)]
        lost += other.real**2 + other.imag**2
    if lost > 1e-20:
        raise ValueError("dropped a qubit that is not in a basis state")
    for j in range(i, w - 1):
        active[j] = active[j + 1]
        pos[active[j]] = j
    active[w - 1] = -1
    pos[q] = -1
    if bit:
        frame_gate(frame, X, q, 0, num_words)
    return out


@jit
def masks(row, active, width: int) -> tuple[int, int, bool]:
    """Return the X and Z masks of ``row`` over the register, and whether X hits a dormant qubit."""
    xm = 0
    zm = 0
    for i in range(width):
        q = active[i]
        xm |= get_bit(row, 0, q) << i
    num_words = (row.shape[0] - 1) // 2
    for i in range(width):
        zm |= get_bit(row, num_words, active[i]) << i
    count = 0
    for w in range(num_words):
        count += popcount(row[w])
    return xm, zm, count != popcount(np.uint64(xm))


# -- stabilizers back into the frame -------------------------------------------


@jit
def free_stabilizer(frame, active, pos, amps, s, num_words: int):
    """Free one qubit given a stabilizer ``s`` of the register (letters on active qubits).

    Gates map ``s`` to ``+-Z_p`` and qubit ``p`` is dropped. Returns the new amplitudes.
    """
    w = width_of(amps)
    p = -1
    for i in range(w):
        q = active[i]
        if get_bit(s, 0, q):
            if p < 0:
                p = q
            else:
                gate(frame, pos, amps, CX, p, q, num_words)
                row_gate(s, CX, p, q, num_words)
    if p >= 0:
        for i in range(w):
            q = active[i]
            if q != p and get_bit(s, num_words, q):
                gate(frame, pos, amps, CZ, p, q, num_words)
                row_gate(s, CZ, p, q, num_words)
        if get_bit(s, num_words, p):
            gate(frame, pos, amps, S_DAG, p, 0, num_words)
            row_gate(s, S_DAG, p, 0, num_words)
        gate(frame, pos, amps, H, p, 0, num_words)
        row_gate(s, H, p, 0, num_words)
    else:
        for i in range(w):
            q = active[i]
            if get_bit(s, num_words, q):
                if p < 0:
                    p = q
                else:
                    gate(frame, pos, amps, CX, q, p, num_words)
                    row_gate(s, CX, q, p, num_words)
    bit = 1 if (s[2 * num_words] & _THREE) == 2 else 0
    return drop(frame, active, pos, amps, p, bit, num_words)


@jit
def compress(frame, active, pos, amps, num_words: int):
    """Free every qubit that a Pauli stabilizer of the register fixes; return the amplitudes.

    Z-type stabilizers first (cheap, from the support), then any other Pauli
    stabilizer that ``_pauli_stabilizer`` finds, one at a time.
    """
    s_row = np.zeros(2 * num_words + 1, dtype=np.uint64)
    while True:
        amps = _compress_z(frame, active, pos, amps, num_words)
        if width_of(amps) == 0 or not _pauli_stabilizer(amps, active, num_words, s_row):
            return amps
        amps = free_stabilizer(frame, active, pos, amps, s_row, num_words)


@jit
def _wht(values) -> None:
    """In-place Walsh-Hadamard transform (unnormalized)."""
    h = 1
    size = values.shape[0]
    while h < size:
        for start in range(0, size, 2 * h):
            for k in range(start, start + h):
                a, b = values[k], values[k + h]
                values[k] = a + b
                values[k + h] = a - b
        h *= 2


@jit
def _pauli_stabilizer(amps, active, num_words: int, s_row) -> bool:
    """Look for a Pauli stabilizer ``i^e X^x Z^z`` (x != 0) of the register; write it to ``s_row``.

    A stabilizer's ``x`` keeps ``|a|^2`` invariant under ``y -> y ^ x``, so ``x`` lies in
    the orthogonal complement of the support of the Walsh transform of ``|a|^2``.
    For such an ``x``, ``h(y) = a(y ^ x) conj(a(y))`` has a Walsh coefficient of modulus
    ``sum |a|^2 = 1`` exactly when ``a(y ^ x) / a(y)`` is a constant times ``(-1)^{z.y}``:
    that coefficient's index is ``z`` and its phase gives ``e``. The candidates are a
    basis of the complement, and all its elements when it is small.
    """
    w = width_of(amps)
    size = amps.shape[0]
    prob = np.empty(size, dtype=np.float64)
    for k in range(size):
        prob[k] = amps[k].real ** 2 + amps[k].imag ** 2
    _wht(prob)
    basis = np.zeros(64, dtype=np.int64)
    rank = 0
    for k in range(1, size):
        if abs(prob[k]) <= 1e-9:
            continue
        d = k
        for b in range(w - 1, -1, -1):
            if not (d >> b) & 1:
                continue
            if basis[b]:
                d ^= basis[b]
            else:
                basis[b] = d
                rank += 1
                break
        if rank == w:
            return False
    # The complement: reduce, then one vector per free coordinate.
    for b in range(w - 1, -1, -1):
        if basis[b]:
            for c in range(b - 1, -1, -1):
                if basis[c] and (basis[b] >> c) & 1:
                    basis[b] ^= basis[c]
    shifts = np.empty(w, dtype=np.int64)
    dim = 0
    for f in range(w):
        if basis[f]:
            continue
        v = 1 << f
        for b in range(w):
            if basis[b] and (basis[b] >> f) & 1:
                v |= 1 << b
        shifts[dim] = v
        dim += 1
    num_candidates = (1 << dim) - 1 if dim <= 6 else dim
    h = np.empty(size, dtype=np.complex128)
    for c in range(1, num_candidates + 1):
        if dim <= 6:
            x = 0
            for i in range(dim):
                if (c >> i) & 1:
                    x ^= shifts[i]
        else:
            x = shifts[c - 1]
        for k in range(size):
            h[k] = amps[k ^ x] * np.conj(amps[k])
        _wht(h)
        best, best_abs = 0, 0.0
        for k in range(size):
            v = abs(h[k])
            if v > best_abs:
                best, best_abs = k, v
        if best_abs < 1.0 - 1e-12:
            continue
        z = best
        coef = h[z]
        if popcount(np.uint64(z & x)) & 1:
            coef = -coef
        # coef = i^{-e}
        if abs(coef.real) > abs(coef.imag):
            k_pow = 0 if coef.real > 0 else 2
        else:
            k_pow = 1 if coef.imag > 0 else 3
        e = (4 - k_pow) & 3
        for k in range(2 * num_words + 1):
            s_row[k] = 0
        for i in range(w):
            q = active[i]
            if (x >> i) & 1:
                s_row[q >> 6] |= _ONE << np.uint64(q & 63)
            if (z >> i) & 1:
                s_row[num_words + (q >> 6)] |= _ONE << np.uint64(q & 63)
        s_row[2 * num_words] = np.uint64(e)
        return True
    return False


@jit
def _compress_z(frame, active, pos, amps, num_words: int):
    """Free every qubit fixed by a Z-type stabilizer of the register; return the amplitudes.

    ``Z^s`` stabilizes the register iff ``s`` is orthogonal to the differences of its
    support, which span the support's affine hull.
    """
    n = frame.shape[0] // 2
    s_row = np.zeros(2 * num_words + 1, dtype=np.uint64)
    while True:
        w = width_of(amps)
        if w == 0:
            return amps
        top = 0.0
        for k in range(amps.shape[0]):
            v = amps[k].real ** 2 + amps[k].imag ** 2
            if v > top:
                top = v
        y0 = -1
        basis = np.zeros(
            64, dtype=np.int64
        )  # basis[b]: vector with leading bit b, or 0
        rank = 0
        for k in range(amps.shape[0]):
            v = amps[k].real ** 2 + amps[k].imag ** 2
            if v <= 1e-24 * top:
                continue
            if y0 < 0:
                y0 = k
                continue
            d = k ^ y0
            for b in range(w - 1, -1, -1):
                if not (d >> b) & 1:
                    continue
                if basis[b]:
                    d ^= basis[b]
                else:
                    basis[b] = d
                    rank += 1
                    break
            if rank == w:
                break
        if rank == w:
            return amps
        # A vector s orthogonal to every basis vector: reduce to echelon form and
        # take a free coordinate.
        for b in range(w - 1, -1, -1):
            if basis[b]:
                for c in range(b - 1, -1, -1):
                    if basis[c] and (basis[b] >> c) & 1:
                        basis[b] ^= basis[c]
        free = -1
        for b in range(w):
            if not basis[b]:
                free = b
                break
        s = 1 << free
        for b in range(w):
            if basis[b] and (basis[b] >> free) & 1:
                s |= 1 << b
        for k in range(2 * num_words + 1):
            s_row[k] = 0
        for i in range(w):
            if (s >> i) & 1:
                q = active[i]
                s_row[num_words + (q >> 6)] |= _ONE << np.uint64(q & 63)
        s_row[2 * num_words] = np.uint64(2 * (popcount(np.uint64(s & y0)) & 1))
        amps = free_stabilizer(frame, active, pos, amps, s_row, num_words)
        _ = n


# -- rotations -----------------------------------------------------------------


@jit
def rotate(
    frame,
    active,
    pos,
    amps,
    cache,
    cache_ver,
    r: int,
    op_x,
    op_z,
    op_e,
    theta: float,
    num_words: int,
    max_width: int,
):
    """Apply ``exp(-i theta P)`` for op ``r``'s Pauli ``P``; return the amplitudes."""
    n = frame.shape[0] // 2
    v = op_image(frame, cache, cache_ver, r, op_x, op_z, op_e, num_words)
    # NOTE: gather the X support on dormant qubits onto one of them and make it active.
    d = -1
    for q in range(n):
        if pos[q] < 0 and get_bit(v, 0, q):
            if d < 0:
                d = q
            else:
                frame_gate(frame, CX, d, q, num_words)
    if d >= 0:
        amps = promote(active, pos, amps, d, max_width)
        v = op_image(frame, cache, cache_ver, r, op_x, op_z, op_e, num_words)
    xm, zm, _ = masks(v, active, width_of(amps))
    if xm == 0 and zm == 0:
        return amps  # a global phase
    tmp = np.empty_like(amps)
    apply_pauli(amps, xm, zm, np.int64(v[2 * num_words]), tmp)
    c, s = np.cos(theta), np.sin(theta)
    for k in range(amps.shape[0]):
        amps[k] = c * amps[k] - 1j * s * tmp[k]
    return amps


# -- plain measurements --------------------------------------------------------
# prepare_measure returns (kind, p1, d): kind 0 is a fair coin on the dormant
# qubit d, kind 1 a certain outcome (p1 is 0 or 1), kind 2 a measurement of the
# register.


@jit
def prepare_measure(
    frame,
    active,
    pos,
    amps,
    cache,
    cache_ver,
    m: int,
    op_x,
    op_z,
    op_e,
    num_words: int,
    v,
):
    """Prepare the measurement of op ``m``'s Pauli; ``v`` receives its virtual image."""
    n = frame.shape[0] // 2
    v[:] = op_image(frame, cache, cache_ver, m, op_x, op_z, op_e, num_words)
    d = -1
    for q in range(n):
        if pos[q] < 0 and get_bit(v, 0, q):
            d = q
            break
    if d >= 0:
        if get_bit(v, num_words, d):
            gate(frame, pos, amps, S_DAG, d, 0, num_words)
        v[:] = op_image(frame, cache, cache_ver, m, op_x, op_z, op_e, num_words)
        for q in range(n):
            if q == d:
                continue
            xq, zq = get_bit(v, 0, q), get_bit(v, num_words, q)
            if xq and zq:
                gate(frame, pos, amps, CY, d, q, num_words)
            elif xq:
                gate(frame, pos, amps, CX, d, q, num_words)
            elif zq:
                gate(frame, pos, amps, CZ, d, q, num_words)
        v[:] = op_image(frame, cache, cache_ver, m, op_x, op_z, op_e, num_words)
        return 0, 0.5, d
    xm, zm, _ = masks(v, active, width_of(amps))
    e = np.int64(v[2 * num_words] & _THREE)
    if xm == 0 and zm == 0:
        return 1, (0.0 if e == 0 else 1.0), -1
    tmp = np.empty_like(amps)
    apply_pauli(amps, xm, zm, e, tmp)
    expect = 0.0
    for k in range(amps.shape[0]):
        expect += (np.conj(amps[k]) * tmp[k]).real
    p1 = min(max((1.0 - expect) / 2.0, 0.0), 1.0)
    return 2, p1, -1


@jit
def finish_measure(
    frame, active, pos, amps, kind: int, d: int, bit: int, v, num_words: int
):
    """Project onto outcome ``bit`` of a prepared measurement; return the amplitudes."""
    if kind == 0:
        # v is +-X_d: d holds |+> or |->, which H (and X) map back to |0>.
        plus = ((v[2 * num_words] & _THREE) == 0) == (bit == 0)
        frame_gate(frame, H, d, 0, num_words)
        if not plus:
            frame_gate(frame, X, d, 0, num_words)
        return amps
    if kind == 1:
        return amps
    w = width_of(amps)
    xm, zm, _ = masks(v, active, w)
    e = np.int64(v[2 * num_words] & _THREE)
    tmp = np.empty_like(amps)
    apply_pauli(amps, xm, zm, e, tmp)
    lam = 1.0 if bit == 0 else -1.0
    norm = 0.0
    for k in range(amps.shape[0]):
        amps[k] = 0.5 * (amps[k] + lam * tmp[k])
        norm += amps[k].real ** 2 + amps[k].imag ** 2
    scale = 1.0 / np.sqrt(norm)
    for k in range(amps.shape[0]):
        amps[k] *= scale
    # lam * v restricted to the register stabilizes it now.
    s = np.zeros(2 * num_words + 1, dtype=np.uint64)
    for i in range(w):
        q = active[i]
        if (xm >> i) & 1:
            s[q >> 6] |= _ONE << np.uint64(q & 63)
        if (zm >> i) & 1:
            s[num_words + (q >> 6)] |= _ONE << np.uint64(q & 63)
    s[2 * num_words] = np.uint64((e + (0 if bit == 0 else 2)) & 3)
    return free_stabilizer(frame, active, pos, amps, s, num_words)
