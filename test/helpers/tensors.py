"""Doubled tensors laid out like zxparam's ``Graph.to_tensor``.

zxparam orders open indices as outputs then inputs, by qubit, and gives each
quantum boundary two adjacent axes (the copy, then its conjugate):
(o0, o0*, o1, o1*, ..., i0, i0*, i1, i1*, ...).
"""

import numpy as np


def doubled(u: np.ndarray) -> np.ndarray:
    """Return ``u ⊗ conj(u)`` for a 2ⁿ×2ⁿ matrix, in zxparam's axis order."""
    n = u.shape[0].bit_length() - 1
    t = np.multiply.outer(u.reshape([2] * 2 * n), u.conj().reshape([2] * 2 * n))
    outs = [a for q in range(n) for a in (q, 2 * n + q)]
    ins = [a for q in range(n) for a in (n + q, 3 * n + q)]
    return t.transpose(outs + ins)


def channel(*kraus: np.ndarray) -> np.ndarray:
    """Return the doubled tensor of ρ ↦ Σ K ρ K† for square Kraus operators."""
    return np.sum([doubled(k) for k in kraus], axis=0)


def doubled_state(psi: np.ndarray) -> np.ndarray:
    """Return ``psi ⊗ conj(psi)`` for a 2ⁿ vector (outputs only)."""
    n = psi.shape[0].bit_length() - 1
    t = np.multiply.outer(psi.reshape([2] * n), psi.conj().reshape([2] * n))
    return t.transpose([a for q in range(n) for a in (q, n + q)])
