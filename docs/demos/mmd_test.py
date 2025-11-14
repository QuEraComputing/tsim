import numpy as np
from scipy.spatial.distance import cdist


def hamming_kernel(X, Y, gamma):
    # X, Y: (n,d) binary arrays (0/1); returns (n,n') Gram matrix
    # scipy cdist with metric="hamming" returns fraction of mismatched bits
    H = cdist(X, Y, metric="hamming")
    return np.exp(-gamma * H)


def mmd2_unbiased(Kxx, Kyy, Kxy):
    # Unbiased MMD^2 (Gretton et al. 2012)
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_x = Kxx.sum() / (m * (m - 1))
    term_y = Kyy.sum() / (n * (n - 1))
    term_xy = Kxy.mean()
    return term_x + term_y - 2 * term_xy


def median_heuristic_gamma(X, Y, n_pairs=2000):
    rng = np.random.default_rng(0)
    ix = rng.choice(len(X), size=min(n_pairs, len(X)), replace=False)
    iy = rng.choice(len(Y), size=min(n_pairs, len(Y)), replace=False)
    H = cdist(X[ix], Y[iy], metric="hamming").ravel()
    med = np.median(H[H > 0]) if np.any(H > 0) else 1.0
    return 1.0 / med if med > 0 else 1.0


def mmd_test(X, Y, n_perm=1000, rng_seed=0):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    gamma = median_heuristic_gamma(X, Y)
    Kxx = hamming_kernel(X, X, gamma)
    Kyy = hamming_kernel(Y, Y, gamma)
    Kxy = hamming_kernel(X, Y, gamma)
    stat = mmd2_unbiased(Kxx.copy(), Kyy.copy(), Kxy)
    # permutation null
    Z = np.vstack([X, Y])
    n = len(X)
    rng = np.random.default_rng(rng_seed)
    perm_stats = []
    for _ in range(n_perm):
        idx = rng.permutation(len(Z))
        Xp, Yp = Z[idx[:n]], Z[idx[n:]]
        Kxx_p = hamming_kernel(Xp, Xp, gamma)
        Kyy_p = hamming_kernel(Yp, Yp, gamma)
        Kxy_p = hamming_kernel(Xp, Yp, gamma)
        perm_stats.append(mmd2_unbiased(Kxx_p, Kyy_p, Kxy_p))
    perm_stats = np.array(perm_stats)
    pval = (1 + np.sum(perm_stats >= stat)) / (1 + n_perm)
    return {"mmd2": float(stat), "p_value": float(pval), "gamma": float(gamma)}
