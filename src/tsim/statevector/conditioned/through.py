"""Measure-through: measure ``M`` behind a pending layer of T-like rotations.

The layer ``Q = prod_k exp(-i theta_k P_k)`` (pairwise commuting, all anticommuting
with ``M``, every ``2 theta_k`` an odd multiple of ``pi/4``) stays pending, and the
stored state is projected onto the eigenspaces of

    O = Q^dag M Q = prod_k exp(2 i theta_k P_k) M,

a Clifford involution (each factor is a Clifford rotation). In virtual coordinates
call it ``K``, with the stored state ``|a, 0>`` (register ``a``, dormant qubits in 0):

1. A dormant Z-string ``Z^t`` with ``K^dag Z^t K = +Z^u``, ``u`` dormant, stabilizes
   ``K|y, 0>`` for every ``y``, hence the projected state. These ``t`` form a group
   ``H`` (GF(2) elimination on the images ``K Z_d K^dag``; ``K`` is Hermitian).
2. CX gates among dormant qubits (trivial on the state) turn a basis of ``H`` into
   single ``Z_d``: those qubits stay dormant, the others (``Dp``) join the register.
3. ``K|0...0>`` on the register is a stabilizer state: its amplitudes follow from
   its stabilizers ``K Z_j K^dag`` up to one phase, which one Gauss sum gives.
4. ``K|a, 0> = sum_y a_y (K X^y K^dag) K|0...0>``: each of the ``2^r`` support points
   of ``K|0...0>`` adds to ``2^w`` points, ``2^(w + r)`` work.

Nothing wider than the projected register is ever stored. The outcome-``b``
state is ``(|a, 0> + (-1)^b K|a, 0>) / 2``.
"""

from __future__ import annotations

import numpy as np

from tsim.statevector.conditioned.gauss import layer_amplitude
from tsim.statevector.conditioned.lane import (
    CX,
    op_image,
    rows_gate,
    TooWide,
    apply_pauli,
    frame_gate,
    width_of,
)
from tsim.statevector.conditioned.paulis import (
    anticommutes,
    copy_row,
    get_bit,
    image,
    multiply_right,
    popcount,
    set_identity,
)
from tsim.utils.kernels import jit

_ONE = np.uint64(1)
_THREE = np.uint64(3)
_I_POWERS = np.array([1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j])


@jit
def _layer_rows(frame, cache, cache_ver, op_x, op_z, op_e, layer, m: int, num_words: int):
    """Return the virtual rows of the layer Paulis and of ``M``."""
    L = layer.shape[0]
    rows = np.empty((L, 2 * num_words + 1), dtype=np.uint64)
    for k in range(L):
        rows[k] = op_image(frame, cache, cache_ver, layer[k], op_x, op_z, op_e, num_words)
    m_row = op_image(frame, cache, cache_ver, m, op_x, op_z, op_e, num_words).copy()
    return rows, m_row


@jit
def _restrict(row, keep, num_keep: int, num_words: int) -> tuple[int, int, int]:
    """Return the X and Z masks of ``row`` over ``keep`` and its phase exponent.

    Letters off ``keep`` must be Z (they act on dormant ``|0>``) and are dropped.
    """
    xm = 0
    zm = 0
    for i in range(num_keep):
        q = keep[i]
        xm |= get_bit(row, 0, q) << i
        zm |= get_bit(row, num_words, q) << i
    return xm, zm, np.int64(row[2 * num_words] & _THREE)


@jit
def _stabilizer_state(gx, gz, ge, size: int):
    """Return the stabilizer state with generators ``i^ge X^gx Z^gz`` on ``log2(size)`` qubits.

    Up to a global phase. Also returns the index ``y0`` of a nonzero amplitude
    (the anchor for the phase) and the support, in the order it was filled.
    """
    num = gx.shape[0]
    x = gx.copy()
    z = gz.copy()
    e = ge.copy()
    # Generators with independent X parts (pivots), and Z-type ones (constraints).
    piv = np.empty(num, dtype=np.int64)
    num_piv = 0
    lead = np.empty(num, dtype=np.int64)
    cons_z = np.empty(num, dtype=np.int64)
    cons_b = np.empty(num, dtype=np.int64)
    num_cons = 0
    for g in range(num):
        for p in range(num_piv):
            h = piv[p]
            if (x[g] >> lead[p]) & 1:
                # g <- g * h, with the phase of the product in this form
                twist = popcount(np.uint64(z[g] & x[h]))
                e[g] = (e[g] + e[h] + 2 * twist) & 3
                x[g] ^= x[h]
                z[g] ^= z[h]
        if x[g]:
            low = x[g] & -x[g]
            lead[num_piv] = popcount(np.uint64(low - 1))
            piv[num_piv] = g
            num_piv += 1
        elif z[g]:
            cons_z[num_cons] = z[g]
            cons_b[num_cons] = e[g] >> 1  # i^e Z^z stabilizes: (-1)^{z.y} = i^{-e}
            num_cons += 1
    # y0 with z.y0 = b for every constraint.
    # Gauss-Jordan: each kept constraint owns a pivot bit no other one has, so
    # y0 = sum of the pivots whose constraint reads 1.
    rz = cons_z[:num_cons].copy()
    rb = cons_b[:num_cons].copy()
    pivot_bit = np.full(num_cons, -1, dtype=np.int64)
    for c in range(num_cons):
        if rz[c] == 0:
            continue
        low = rz[c] & -rz[c]
        bit = popcount(np.uint64(low - 1))
        pivot_bit[c] = bit
        for c2 in range(num_cons):
            if c2 != c and (rz[c2] >> bit) & 1:
                rz[c2] ^= rz[c]
                rb[c2] ^= rb[c]
    y0 = 0
    for c in range(num_cons):
        if pivot_bit[c] >= 0 and rb[c]:
            y0 |= 1 << pivot_bit[c]
    out = np.zeros(size, dtype=np.complex128)
    support = np.empty(1 << num_piv, dtype=np.int64)
    amp = 2.0 ** (-0.5 * num_piv)
    y = y0
    out[y] = amp
    support[0] = y
    # Gray code over the pivots: psi(y ^ x_g) = i^e_g (-1)^{z_g . y} psi(y).
    for count in range(1, 1 << num_piv):
        p = popcount(np.uint64((count ^ (count >> 1)) ^ ((count - 1) ^ ((count - 1) >> 1))) - 1)
        h = piv[p]
        sign = _I_POWERS[e[h] & 3]
        if popcount(np.uint64(z[h] & y)) & 1:
            sign = -sign
        amp = amp * sign
        y ^= x[h]
        out[y] = amp
        support[count] = y
    return out, y0, support


@jit
def _factor_signs(angles):
    """Return per factor whether cos 2 theta < 0, sin 2 theta < 0 and sin 4 theta < 0."""
    L = angles.shape[0]
    eps_neg = np.empty(L, dtype=np.int64)
    eta_neg = np.empty(L, dtype=np.int64)
    s4_neg = np.empty(L, dtype=np.int64)
    for k in range(L):
        two = 2.0 * angles[k]
        eps_neg[k] = 1 if np.cos(two) < 0 else 0
        eta_neg[k] = 1 if np.sin(two) < 0 else 0
        s4_neg[k] = eps_neg[k] ^ eta_neg[k]
    return eps_neg, eta_neg, s4_neg


@jit
def _images(rows, s4_neg, m_row, active, w: int, n: int, num_words: int):
    """Return ``K Z_j K^dag`` for every qubit j and ``K X_q K^dag`` for the register.

    ``Z_j`` anticommutes with the factors whose X part holds j, and ``X_q`` with
    those whose Z part holds q, so each generator's factors are a column of the
    layer's bit matrix (one uint64 mask, as a layer has at most 64 factors).
    """
    W2 = 2 * num_words
    L = rows.shape[0]
    by_x = np.zeros(n, dtype=np.uint64)  # bit k: factor k has X on qubit j
    by_z = np.zeros(n, dtype=np.uint64)
    for k in range(L):
        bit = _ONE << np.uint64(k)
        for wd in range(num_words):
            xs, zs = rows[k, wd], rows[k, num_words + wd]
            while xs:
                low = xs & (~xs + _ONE)
                by_x[64 * wd + popcount(low - _ONE)] |= bit
                xs ^= low
            while zs:
                low = zs & (~zs + _ONE)
                by_z[64 * wd + popcount(low - _ONE)] |= bit
                zs ^= low
    prod = np.empty(W2 + 1, dtype=np.uint64)
    zimg = np.zeros((n, W2 + 1), dtype=np.uint64)
    for j in range(n):
        zimg[j, num_words + (j >> 6)] = _ONE << np.uint64(j & 63)
        _apply_factors(zimg[j], rows, s4_neg, m_row, by_x[j], num_words, prod)
    ximg = np.zeros((max(w, 1), W2 + 1), dtype=np.uint64)
    for j in range(w):
        q = active[j]
        ximg[j, q >> 6] = _ONE << np.uint64(q & 63)
        _apply_factors(ximg[j], rows, s4_neg, m_row, by_z[q], num_words, prod)
    return zimg, ximg


@jit
def _apply_factors(row, rows, s4_neg, m_row, factors: np.uint64, num_words: int, prod) -> None:
    """Map ``row`` to ``K row K^dag``, given the mask of layer factors it anticommutes with.

    ``prod`` is scratch.
    """
    W2 = 2 * num_words
    phase = 2 if anticommutes(row, m_row, num_words) else 0
    set_identity(prod, num_words)
    bits = factors
    while bits:
        low = bits & (~bits + _ONE)
        k = popcount(low - _ONE)
        multiply_right(prod, rows[k], num_words)
        phase += 1 + 2 * s4_neg[k]
        bits ^= low
    multiply_right(prod, row, num_words)
    prod[W2] = (prod[W2] + np.uint64(phase)) & _THREE
    copy_row(row, prod, num_words)


@jit
def _frozen_basis(zimg, pos, active, w: int, num_words: int):
    """Return a basis of the dormant Z-strings ``Z^t`` with ``K Z^t K^dag = +Z^u``, u dormant.

    As bit words over the qubits, one per row (the first ``count`` rows), and ``count``.
    Found by elimination on ``[x | z on the register]`` of the images, tagging rows
    by the dormant qubits they combine; a combination with sign - is paired with
    the first such one, so that the products have sign +.
    """
    n = zimg.shape[0]
    W2 = 2 * num_words
    nd = n - w
    dormant = np.empty(nd, dtype=np.int64)
    i = 0
    for q in range(n):
        if pos[q] < 0:
            dormant[i] = q
            i += 1
    num_cols = num_words + 1
    mat = np.zeros((nd, num_cols), dtype=np.uint64)
    tag_words = max((nd + 63) >> 6, 1)
    tags = np.zeros((nd, tag_words), dtype=np.uint64)
    for i in range(nd):
        row = zimg[dormant[i]]
        for wd in range(num_words):
            mat[i, wd] = row[wd]
        zr = 0
        for j in range(w):
            zr |= get_bit(row, num_words, active[j]) << j
        mat[i, num_words] = np.uint64(zr)
        tags[i, i >> 6] = _ONE << np.uint64(i & 63)
    r = 0
    for col in range(64 * num_cols):
        if r == nd:
            break
        cw, cb = col >> 6, np.uint64(col & 63)
        pivot = -1
        for i in range(r, nd):
            if (mat[i, cw] >> cb) & _ONE:
                pivot = i
                break
        if pivot < 0:
            continue
        for wd in range(num_cols):
            mat[r, wd], mat[pivot, wd] = mat[pivot, wd], mat[r, wd]
        for wd in range(tag_words):
            tags[r, wd], tags[pivot, wd] = tags[pivot, wd], tags[r, wd]
        for i in range(nd):
            if i != r and (mat[i, cw] >> cb) & _ONE:
                for wd in range(num_cols):
                    mat[i, wd] ^= mat[r, wd]
                for wd in range(tag_words):
                    tags[i, wd] ^= tags[r, wd]
        r += 1
    count = 0
    basis = np.zeros((nd, num_words), dtype=np.uint64)
    neg = np.zeros(num_words, dtype=np.uint64)
    have_neg = False
    prod = np.empty(W2 + 1, dtype=np.uint64)
    vec = np.empty(num_words, dtype=np.uint64)
    for i in range(r, nd):
        set_identity(prod, num_words)
        vec[:] = 0
        for j in range(nd):
            if (tags[i, j >> 6] >> np.uint64(j & 63)) & _ONE:
                multiply_right(prod, zimg[dormant[j]], num_words)
                d = dormant[j]
                vec[d >> 6] ^= _ONE << np.uint64(d & 63)
        if (prod[W2] & _THREE) == 0:
            basis[count] = vec
            count += 1
        elif not have_neg:
            neg[:] = vec
            have_neg = True
        else:
            for wd in range(num_words):
                basis[count, wd] = vec[wd] ^ neg[wd]
            count += 1
    return basis, count


@jit
def _freeze(frame, zimg, ximg, rows, m_row, basis, count: int, active, pos, w: int, num_words: int):
    """Turn each basis Z-string into one ``Z_d`` by CX gates among dormant qubits.

    Returns the projected register: the current one, then the dormant qubits that
    are not frozen. A gate ``g = CX(c, t)`` changes the frame to ``g F``, so ``K``
    becomes ``g K g^dag``: the images follow (``K Z_t K^dag`` picks up ``K Z_c K^dag``,
    as ``g^dag Z_t g = Z_c Z_t``) and are conjugated by ``g``, and so are the layer rows.
    """
    n = zimg.shape[0]
    frozen = np.zeros(num_words, dtype=np.uint64)
    m_rows = m_row.reshape(1, m_row.shape[0])
    for i in range(count):
        for wd in range(num_words):
            basis[i, wd] &= ~frozen[wd]
        pj = -1
        for q in range(n):
            if get_bit(basis[i], 0, q):
                pj = q
                break
        for q in range(pj + 1, n):
            if get_bit(basis[i], 0, q):
                frame_gate(frame, CX, q, pj, num_words)  # Z_p Z_q -> Z_p
                multiply_right(zimg[pj], zimg[q], num_words)
                rows_gate(zimg, CX, q, pj, num_words)
                rows_gate(ximg, CX, q, pj, num_words)
                rows_gate(rows, CX, q, pj, num_words)
                rows_gate(m_rows, CX, q, pj, num_words)
                for i2 in range(i + 1, count):
                    if get_bit(basis[i2], 0, pj):
                        basis[i2, q >> 6] ^= _ONE << np.uint64(q & 63)
        frozen[pj >> 6] |= _ONE << np.uint64(pj & 63)
    keep = np.empty(n, dtype=np.int64)
    for j in range(w):
        keep[j] = active[j]
    wa = w
    for q in range(n):
        if pos[q] < 0 and not get_bit(frozen, 0, q):
            keep[wa] = q
            wa += 1
    return keep[:wa].copy()


@jit
def _image_of_zero(zimg, keep, rows, m_row, eps_neg, eta_neg, num_words: int):
    """Return ``K|0...0>`` on the register ``keep``, and its support.

    Its amplitudes follow from its stabilizers ``K Z_j K^dag`` up to a phase; one
    Gauss sum gives the exact amplitude at one support point.
    """
    n = zimg.shape[0]
    wa = keep.shape[0]
    gx = np.empty(n, dtype=np.int64)
    gz = np.empty(n, dtype=np.int64)
    ge = np.empty(n, dtype=np.int64)
    for j in range(n):
        gx[j], gz[j], ge[j] = _restrict(zimg[j], keep, wa, num_words)
    psi0, y0, support = _stabilizer_state(gx, gz, ge, 1 << wa)
    target = np.zeros(num_words, dtype=np.uint64)
    for i in range(wa):
        if (y0 >> i) & 1:
            q = keep[i]
            target[q >> 6] |= _ONE << np.uint64(q & 63)
    exact = layer_amplitude(rows, rows.shape[0], eps_neg, eta_neg, m_row, target, num_words)
    scale = exact / psi0[y0]
    for k in range(psi0.shape[0]):
        psi0[k] *= scale
    return psi0, support


@jit
def _image_of_register(amps, ximg, keep, psi0, support, num_words: int):
    """Return ``K|a, 0> = sum_y a_y W_y K|0>`` with ``W_y = K X^y K^dag``.

    For ``W_y = i^E_y X^X_y Z^Z_y``, ``(W_y psi0)[j0 ^ X_y] = i^E_y (-1)^{Z_y . j0} psi0[j0]``:
    each support point ``j0`` of ``psi0`` adds to the ``2^w`` points ``j0 ^ X_y``.
    """
    w = width_of(amps)
    wa = keep.shape[0]
    wx = np.empty(max(w, 1), dtype=np.int64)
    wz = np.empty(max(w, 1), dtype=np.int64)
    we = np.empty(max(w, 1), dtype=np.int64)
    for j in range(w):
        wx[j], wz[j], we[j] = _restrict(ximg[j], keep, wa, num_words)
    num_y = 1 << w
    xs = np.zeros(num_y, dtype=np.int64)
    zs = np.zeros(num_y, dtype=np.int64)
    es = np.zeros(num_y, dtype=np.int64)
    prev = 0
    for count in range(1, num_y):
        g = count ^ (count >> 1)
        q = popcount(np.uint64((g ^ prev) - 1))
        xs[g] = xs[prev] ^ wx[q]
        zs[g] = zs[prev] ^ wz[q]
        es[g] = (es[prev] + we[q] + 2 * popcount(np.uint64(zs[prev] & wx[q]))) & 3
        prev = g
    coef = np.empty(num_y, dtype=np.complex128)
    for y in range(num_y):
        coef[y] = amps[y] * _I_POWERS[es[y]]
    kv = np.zeros(1 << wa, dtype=np.complex128)
    for i in range(support.shape[0]):
        j0 = support[i]
        c = psi0[j0]
        for y in range(num_y):
            if popcount(np.uint64(zs[y] & j0)) & 1:
                kv[j0 ^ xs[y]] -= coef[y] * c
            else:
                kv[j0 ^ xs[y]] += coef[y] * c
    return kv


@jit
def prepare_through(
    frame, active, pos, amps, cache, cache_ver, op_x, op_z, op_e, layer, angles, m: int, num_words: int, max_width: int
):
    """Prepare the measurement of op ``m`` through the layer of rotation ops ``layer``.

    ``angles`` are the layer's concrete angles (``exp(-i theta P)``). Applies the
    outcome-independent frame changes and returns ``(p1, keep, phi, kv)``: the
    probability of outcome 1, the qubits of the projected register (the current
    register first), and ``|a, 0>`` and ``K|a, 0>`` on it. For a certain outcome
    ``keep`` is the current register and ``phi`` and ``kv`` are empty.
    """
    n = frame.shape[0] // 2
    w = width_of(amps)
    eps_neg, eta_neg, s4_neg = _factor_signs(angles)
    rows, m_row = _layer_rows(frame, cache, cache_ver, op_x, op_z, op_e, layer, m, num_words)
    empty = np.empty(0, dtype=np.complex128)
    keep_now = active[:w].copy()
    if w == 0:
        # Fast path: <0|K|0> alone decides a certain outcome.
        zero = np.zeros(num_words, dtype=np.uint64)
        e0 = layer_amplitude(rows, layer.shape[0], eps_neg, eta_neg, m_row, zero, num_words)
        if abs(abs(e0.real) - 1.0) < 1e-12:
            return (0.0 if e0.real > 0 else 1.0), keep_now, empty, empty
    zimg, ximg = _images(rows, s4_neg, m_row, active, w, n, num_words)
    basis, count = _frozen_basis(zimg, pos, active, w, num_words)
    keep = _freeze(frame, zimg, ximg, rows, m_row, basis, count, active, pos, w, num_words)
    if keep.shape[0] > max_width:
        raise TooWide()
    psi0, support = _image_of_zero(zimg, keep, rows, m_row, eps_neg, eta_neg, num_words)
    kv = _image_of_register(amps, ximg, keep, psi0, support, num_words)
    phi = np.zeros(kv.shape[0], dtype=np.complex128)
    phi[: amps.shape[0]] = amps
    expect = 0.0
    for k in range(kv.shape[0]):
        expect += (np.conj(phi[k]) * kv[k]).real
    if abs(abs(expect) - 1.0) < 1e-12:
        return (0.0 if expect > 0 else 1.0), keep_now, empty, empty
    p1 = min(max((1.0 - expect) / 2.0, 0.0), 1.0)
    return p1, keep, phi, kv


@jit
def finish_through(active, pos, keep, phi, kv, bit: int):
    """Return the register of outcome ``bit``: ``(phi + (-1)^bit kv) / 2``, normalized.

    Makes the qubits of ``keep`` active.
    """
    lam = 1.0 if bit == 0 else -1.0
    out = np.empty_like(phi)
    norm = 0.0
    for k in range(phi.shape[0]):
        out[k] = 0.5 * (phi[k] + lam * kv[k])
        norm += out[k].real ** 2 + out[k].imag ** 2
    scale = 1.0 / np.sqrt(norm)
    for k in range(out.shape[0]):
        out[k] *= scale
    for i in range(keep.shape[0]):
        active[i] = keep[i]
        pos[keep[i]] = i
    return out
