"""SVQX-style Monte-Carlo Shapley estimators (the inexact baselines we compare against).

These mirror the qshaptools / SVQX coalition-sampling estimators: they evaluate ordinary
masked circuits and converge to the exact Shapley values only in the sampling limit.
Ported verbatim (semantics preserved) from the TN-SHAP-Q verification core
(svqx_core.py :: svqx_mc_shapley / permutation_shapley).
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def _shapley_weight(s, n):
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def svqx_mc_shapley(game, alpha=0.1, K=1, shots=None, rng=None, max_coalitions=None):
    """SVQX subsampled-coalition estimator: per player, average marginals over a
    Shapley-weighted sample of coalitions.  alpha is the fraction of the 2^(n-1)
    coalitions sampled per player; alpha>=1 with exact values reduces to exact Shapley."""
    rng = rng or np.random.default_rng()
    n = game.n
    phi = np.zeros(n)
    if alpha >= 1.0:
        for i in range(n):
            others = [j for j in range(n) if j != i]
            for r in range(len(others) + 1):
                w = _shapley_weight(r, n)
                for S in combinations(others, r):
                    zS = np.zeros(n, dtype=np.int8)
                    for j in S:
                        zS[j] = 1
                    zSi = zS.copy(); zSi[i] = 1
                    vS = np.mean([game.value(zS, shots=shots, rng=rng) for _ in range(K)])
                    vSi = np.mean([game.value(zSi, shots=shots, rng=rng) for _ in range(K)])
                    phi[i] += w * (vSi - vS)
        return phi
    for i in range(n):
        others = [j for j in range(n) if j != i]
        n_total = 1 << (n - 1)
        n_samp = max(1, int(np.ceil(alpha * n_total)))
        if max_coalitions:
            n_samp = min(n_samp, max_coalitions)
        sizes = np.arange(n)
        wsize = np.array([math.comb(n - 1, s) * _shapley_weight(s, n) for s in sizes])
        wsize = wsize / wsize.sum()
        acc = 0.0
        for _ in range(n_samp):
            s = rng.choice(sizes, p=wsize)
            S = list(rng.choice(others, size=s, replace=False)) if s > 0 else []
            zS = np.zeros(n, dtype=np.int8); zS[S] = 1
            zSi = zS.copy(); zSi[i] = 1
            vS = np.mean([game.value(zS, shots=shots, rng=rng) for _ in range(K)])
            vSi = np.mean([game.value(zSi, shots=shots, rng=rng) for _ in range(K)])
            acc += (vSi - vS)
        phi[i] = acc / n_samp
    return phi


def svqx_eval_count(game, alpha):
    """Number of masked-circuit value evaluations svqx_mc_shapley spends at this alpha."""
    n = game.n
    n_samp = max(1, int(np.ceil(alpha * (1 << (n - 1)))))
    return int(2 * n * n_samp)


def permutation_shapley(game, n_perms=100, shots=None, rng=None):
    """Permutation-sampling (ApproShapley) baseline."""
    rng = rng or np.random.default_rng()
    n = game.n
    phi = np.zeros(n)
    for _ in range(n_perms):
        perm = rng.permutation(n)
        z = np.zeros(n, dtype=np.int8)
        prev = game.value(z, shots=shots, rng=rng)
        for i in perm:
            z[i] = 1
            cur = game.value(z, shots=shots, rng=rng)
            phi[i] += (cur - prev)
            prev = cur
    return phi / n_perms
