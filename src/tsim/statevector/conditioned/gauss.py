"""Z4 Gauss sums, and one amplitude of ``O|0...0>`` for a measure-through.

A Z4 quadratic form on ``GF(2)^d`` (``d <= 64``) is

    q(t) = c + sum_j lin[j] t_j + 2 sum_{j<l} A[j, l] t_j t_l   (mod 4),

stored as ``c``, ``lin`` (int64, values mod 4) and ``A`` as one uint64 bit mask per
variable (symmetric, zero diagonal). ``gauss_sum`` returns ``sum_t i^q(t)`` by
eliminating one variable at a time, in ``O(d^2)`` word operations.

``layer_amplitude`` returns ``<p| O |0...0>`` for ``O = prod_k exp(2 i theta_k V_k) M``,
where the ``V_k`` commute pairwise, anticommute with ``M`` and every ``2 theta_k`` is
an odd multiple of ``pi/4``. Expanding each factor as ``(eps_k + i eta_k V_k)/sqrt 2``
gives one term per subset ``S`` of the layer; the terms that reach ``|p>`` form an
affine subspace of subsets, and on it the term's phase is a Z4 quadratic form.
"""

from __future__ import annotations

import numpy as np

from tsim.statevector.conditioned.paulis import popcount
from tsim.utils.kernels import jit

_ONE = np.uint64(1)
_I_POWERS = np.array([1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j])


@jit
def _lowest(bits: np.uint64) -> int:
    """Return the index of the lowest set bit of a nonzero ``bits``."""
    return popcount((bits & (~bits + _ONE)) - _ONE)


@jit
def _parity(bits: np.uint64) -> int:
    return popcount(bits) & 1


@jit
def gauss_sum(c: int, lin, quad) -> complex:
    """Return ``sum_t i^q(t)`` for the form ``(c, lin, quad)``; ``lin`` and ``quad`` are overwritten."""
    d = lin.shape[0]
    total = _I_POWERS[c & 3]
    alive = np.uint64(0) if d == 0 else (~np.uint64(0)) >> np.uint64(64 - d)
    while alive:
        v = _lowest(alive)
        alive &= ~(_ONE << np.uint64(v))
        l0 = lin[v] & 3
        row = quad[v] & alive
        if l0 & 1:
            # sum_{u_v} i^{u_v (l0 + 2 row.u)} = (1 + i^l0) i^{-l0 [row.u]}, and the parity
            # [row.u] = sum_k u_k - 2 sum_{j<k} u_j u_k over the row.
            total *= 1.0 + _I_POWERS[l0]
            bits = row
            while bits:
                k = _lowest(bits)
                bits &= bits - _ONE
                lin[k] = (lin[k] - l0) & 3
                quad[k] ^= row
                quad[k] &= ~(_ONE << np.uint64(k))
            continue
        if row == 0:
            if l0 == 2:
                return 0.0j
            total *= 2.0
            continue
        # l0 even: the sum over u_v is 2 [row.u = l0/2 (mod 2)], so u_k = b ^ g.u
        # for the lowest k of the row and g the rest of it; substitute u_k.
        total *= 2.0
        k = _lowest(row)
        g = row & ~(_ONE << np.uint64(k))
        b = l0 >> 1
        alive &= ~(_ONE << np.uint64(k))
        lk = lin[k] & 3
        ak = quad[k] & alive
        # lin_k u_k = lin_k (b + (1 - 2b) [g.u]) with [g.u] = sum - 2 sum of pairs.
        total *= _I_POWERS[(lk * b) & 3]
        shift = (lk * (1 - 2 * b)) & 3
        bits = g
        while bits:
            j = _lowest(bits)
            bits &= bits - _ONE
            lin[j] = (lin[j] + shift) & 3
            if lk & 1:
                quad[j] ^= g
                quad[j] &= ~(_ONE << np.uint64(j))
        # 2 u_k (A_k.u) = 2 b (A_k.u) + 2 [g.u](A_k.u): linear 2b on A_k, linear 2 on
        # g & A_k (u_i^2 = u_i), and pairs i in g, j in A_k.
        bits = ak
        while bits:
            j = _lowest(bits)
            bits &= bits - _ONE
            lin[j] = (lin[j] + 2 * b) & 3
            quad[j] ^= g
        bits = g
        while bits:
            i = _lowest(bits)
            bits &= bits - _ONE
            quad[i] ^= ak
        bits = g & ak
        while bits:
            i = _lowest(bits)
            bits &= bits - _ONE
            lin[i] = (lin[i] + 2) & 3
        bits = g | ak
        while bits:
            i = _lowest(bits)
            bits &= bits - _ONE
            quad[i] &= ~(_ONE << np.uint64(i))
    return total


@jit
def _form_value(s: np.uint64, lin, upper) -> int:
    """Return ``sum_{k in s} lin_k + 2 sum_{j<k in s} u(j, k)`` (mod 4)."""
    value = 0
    bits = s
    while bits:
        k = _lowest(bits)
        bits &= bits - _ONE
        value += lin[k] + 2 * popcount(upper[k] & s)
    return value & 3


@jit
def layer_amplitude(
    layer, num_layer: int, eps_neg, eta_neg, m_row, target, num_words: int
) -> complex:
    """Return ``<target| O |0...0>`` for ``O = prod_k exp(2 i theta_k V_k) M``.

    Args:
        layer: Rows of the virtual ``V_k`` (first ``num_layer`` rows used, at most 64).
        num_layer: Number of layer rotations L.
        eps_neg: Whether ``cos 2 theta_k < 0``, per factor.
        eta_neg: Whether ``sin 2 theta_k < 0``, per factor.
        m_row: The virtual ``M``.
        target: The basis state ``p`` as ``num_words`` bit words.
        num_words: Words per bit vector W.

    """
    W = num_words
    L = num_layer
    # Phase of the term S, applying V_k in ascending k to M|0> = i^e_M |x_M>:
    # V_k |y> = i^e_k (-1)^{z_k.y} |y ^ x_k>, with y = x_M ^ sum_{j<k in S} x_j.
    lin = np.empty(L, dtype=np.int64)
    upper = np.zeros(L, dtype=np.uint64)  # bit k of upper[j]: u(j, k) = z_k.x_j, j < k
    c0 = np.int64(m_row[2 * W])
    for k in range(L):
        zx = 0
        for w in range(W):
            zx += popcount(layer[k, W + w] & m_row[w])
        lin[k] = (
            1 + 2 * eta_neg[k] + np.int64(layer[k, 2 * W]) + 2 * zx + 2 * eps_neg[k]
        ) & 3
        c0 += 2 * eps_neg[k]
        for j in range(k):
            t = 0
            for w in range(W):
                t += popcount(layer[k, W + w] & layer[j, w])
            if t & 1:
                upper[j] |= _ONE << np.uint64(k)
    # Subsets S with sum_{k in S} x_k = target ^ x_M: elimination, tagging rows by subset.
    rows = np.empty((L, W), dtype=np.uint64)
    tags = np.empty(L, dtype=np.uint64)
    pivot_word = np.empty(L, dtype=np.int64)
    pivot_bit = np.empty(L, dtype=np.uint64)
    num_pivots = 0
    null = np.empty(L, dtype=np.uint64)
    num_null = 0
    for k in range(L):
        for w in range(W):
            rows[num_pivots, w] = layer[k, w]
        tag = _ONE << np.uint64(k)
        for p in range(num_pivots):
            if (rows[num_pivots, pivot_word[p]] >> pivot_bit[p]) & _ONE:
                for w in range(W):
                    rows[num_pivots, w] ^= rows[p, w]
                tag ^= tags[p]
        lead = -1
        for w in range(W):
            if rows[num_pivots, w]:
                lead = w
                break
        if lead < 0:
            null[num_null] = tag
            num_null += 1
            continue
        tags[num_pivots] = tag
        pivot_word[num_pivots] = lead
        pivot_bit[num_pivots] = np.uint64(_lowest(rows[num_pivots, lead]))
        num_pivots += 1
    want = np.empty(W, dtype=np.uint64)
    for w in range(W):
        want[w] = target[w] ^ m_row[w]
    s0 = np.uint64(0)
    for p in range(num_pivots):
        if (want[pivot_word[p]] >> pivot_bit[p]) & _ONE:
            for w in range(W):
                want[w] ^= rows[p, w]
            s0 ^= tags[p]
    for w in range(W):
        if want[w]:
            return 0.0j
    # The form on s = s0 ^ sum_j t_j N_j. With B(s) = sum_{j<k in s} u(j, k) (mod 2) and
    # its polar form beta(x, y) = B(x ^ y) - B(x) - B(y):
    #   lin'_k = +-lin_k (minus where s0_k), L_j = sum_{k in N_j} lin'_k + 2 (B(N_j) + beta(s0, N_j)),
    #   A_jl = |N_j & N_l & odd lin| + beta(N_j, N_l)  (mod 2).
    sym = upper.copy()
    for j in range(L):
        bits = upper[j]
        while bits:
            k = _lowest(bits)
            bits &= bits - _ONE
            sym[k] |= _ONE << np.uint64(j)
    odd = np.uint64(0)
    for k in range(L):
        if lin[k] & 1:
            odd |= _ONE << np.uint64(k)
    d = num_null
    form_lin = np.empty(d, dtype=np.int64)
    form_quad = np.zeros(d, dtype=np.uint64)
    polar = np.empty(d, dtype=np.uint64)  # bit a of polar[j]: (sym N_j)_a
    for j in range(d):
        v = np.uint64(0)
        for a in range(L):
            if _parity(sym[a] & null[j]):
                v |= _ONE << np.uint64(a)
        polar[j] = v
    for j in range(d):
        nj = null[j]
        value = 0
        bits = nj
        while bits:
            k = _lowest(bits)
            bits &= bits - _ONE
            lk = lin[k]
            if (s0 >> np.uint64(k)) & _ONE:
                lk = (4 - lk) & 3
            value += lk
        b_self = 0
        bits = nj
        while bits:
            k = _lowest(bits)
            bits &= bits - _ONE
            b_self += popcount(upper[k] & nj)
        value += 2 * (b_self + _parity(s0 & polar[j]))
        form_lin[j] = value & 3
        for k2 in range(j + 1, d):
            if (popcount(nj & null[k2] & odd) + _parity(nj & polar[k2])) & 1:
                form_quad[j] |= _ONE << np.uint64(k2)
                form_quad[k2] |= _ONE << np.uint64(j)
    c = (c0 + _form_value(s0, lin, upper)) & 3
    return gauss_sum(c, form_lin, form_quad) * 2.0 ** (-0.5 * L)
