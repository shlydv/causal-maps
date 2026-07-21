"""Null controls and map-similarity stats.

- Spearman over the flattened layer x position grid (map similarity, Gate P1).
- Permutation p-value with +1 smoothing.
- Random-position null (Gate P0): compare the effect at expected token positions
  against the effect at random positions drawn from the SAME layers (matched
  layer distribution). Computed by resampling positions from the already-computed
  full IE sweep -> no extra forward passes, and no refitting inside the loop.
"""
import numpy as np
from scipy.stats import spearmanr


def spearman_grid(map_a, map_b):
    """Spearman rho over the flattened (layer x position) grids."""
    a = np.asarray(map_a, dtype=float).ravel()
    b = np.asarray(map_b, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(a[m], b[m])
    return float(rho)


def mean_map(ie_all, pair_idx=None):
    """Mean IE over pairs. ie_all: [nL, nP, B]."""
    ie = np.asarray(ie_all, dtype=float)
    if pair_idx is None:
        return np.nanmean(ie, axis=2)
    return np.nanmean(ie[:, :, list(pair_idx)], axis=2)


def pair_partition_null(ie_all, n_a, n_b, n_draws=1000, seed=0):
    """Null for Gate P1: random pair partitions with the same sizes as the
    real template-disjoint halves. Returns null Spearman rhos [n_draws]."""
    ie = np.asarray(ie_all, dtype=float)
    B = ie.shape[2]
    if n_a + n_b > B:
        raise ValueError(f"n_a+n_b={n_a+n_b} > B={B}")
    rng = np.random.default_rng(seed)
    null = np.empty(n_draws, dtype=float)
    idx = np.arange(B)
    for d in range(n_draws):
        rng.shuffle(idx)
        a, b = idx[:n_a], idx[n_a:n_a + n_b]
        null[d] = spearman_grid(mean_map(ie, a), mean_map(ie, b))
    return null


def localization_stats(ie_mean, top_frac=0.05):
    """Contrast 1: do the top top_frac of sites carry >50% of total |IE|?"""
    w = np.abs(np.asarray(ie_mean, dtype=float)).ravel()
    w = w[np.isfinite(w)]
    total = float(w.sum())
    if total <= 0 or w.size == 0:
        return {"localized": False, "top_mass": float("nan"), "total": total,
                "n_top": 0, "n_sites": int(w.size)}
    n_top = max(1, int(np.ceil(top_frac * w.size)))
    top = np.sort(w)[-n_top:]
    top_mass = float(top.sum() / total)
    return {"localized": bool(top_mass > 0.5), "top_mass": top_mass,
            "total": total, "n_top": n_top, "n_sites": int(w.size)}


def map_entropy_norm(ie_mean):
    """Contrast 3: normalized entropy of |IE| mass over sites ∈ [0, 1]."""
    w = np.abs(np.asarray(ie_mean, dtype=float)).ravel()
    w = w[np.isfinite(w)]
    total = float(w.sum())
    if total <= 0 or w.size < 2:
        return float("nan")
    p = w / total
    H = float(-np.sum(p * np.log(p + 1e-12)))
    return H / float(np.log(w.size))


def mass_weighted_mean_layer(ie_mean, layers):
    """Contrast 2 helper."""
    ie = np.asarray(ie_mean, dtype=float)
    layers = np.asarray(layers, dtype=float)
    w = np.abs(ie)
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    # w[l, p]; broadcast layers over positions
    return float((w * layers[:, None]).sum() / total)


def permutation_pvalue(real, null_samples, alternative="greater"):
    """p = P(null at least as extreme as real), +1 smoothed."""
    null = np.asarray(null_samples, dtype=float)
    null = null[np.isfinite(null)]
    n = null.size
    if n == 0:
        return float("nan")
    if alternative == "greater":
        k = int(np.sum(null >= real))
    elif alternative == "less":
        k = int(np.sum(null <= real))
    else:
        raise ValueError(alternative)
    return (k + 1) / (n + 1)


def _stat_maxlayer(ie_per_pair, position_idx):
    """max over layers of (mean over pairs, over the given positions) of IE."""
    # ie_per_pair: [nL, nP, B]; position_idx: 1D indices into nP
    per_layer = np.nanmean(ie_per_pair[:, position_idx, :], axis=(1, 2))  # [nL]
    return float(np.nanmax(per_layer)), int(np.nanargmax(per_layer))


def random_position_null(ie_per_pair, expected_positions, candidate_positions=None,
                         n_draws=1000, seed=0):
    """P0 statistic + matched random-position null.

    real = max_L mean_{pairs, p in expected_positions} IE(L, p)
    null[d] = same statistic with |expected_positions| positions sampled (without
              replacement) from candidate_positions, independently per layer
              (matched layer distribution).

    Returns (real_stat, real_layer, null_stats[n_draws]).
    candidate_positions defaults to all positions EXCEPT the expected ones (so
    the null is 'other positions', not a mixture that includes the signal)."""
    ie = np.asarray(ie_per_pair, dtype=float)
    nL, nP, _B = ie.shape
    expected_positions = np.asarray(expected_positions, dtype=int)
    k = expected_positions.size
    if candidate_positions is None:
        candidate_positions = np.array([p for p in range(nP)
                                        if p not in set(expected_positions.tolist())],
                                       dtype=int)
    else:
        candidate_positions = np.asarray(candidate_positions, dtype=int)
    if candidate_positions.size < k:
        raise ValueError(
            f"need >= {k} candidate positions for the null, got {candidate_positions.size}")

    real_stat, real_layer = _stat_maxlayer(ie, expected_positions)

    rng = np.random.default_rng(seed)
    null = np.empty(n_draws, dtype=float)
    for d in range(n_draws):
        per_layer = np.empty(nL, dtype=float)
        for li in range(nL):
            idx = rng.choice(candidate_positions, size=k, replace=False)
            per_layer[li] = np.nanmean(ie[li, idx, :])
        null[d] = np.nanmax(per_layer)
    return real_stat, real_layer, null
