"""Comparison metrics: MAE, cosine similarity, and rank-agreement measures.

Rank measures (Spearman, Kendall-tau, top-1 hit) are implemented in pure NumPy so
the demo carries no SciPy ranking dependency; they match scipy.stats to machine
precision on tie-free inputs.
"""
from __future__ import annotations

import numpy as np


def mae(a, b):
    """Mean absolute error between two flattened arrays."""
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    return float(np.mean(np.abs(a - b)))


def max_abs_err(a, b):
    """Maximum absolute error between two flattened arrays."""
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    return float(np.max(np.abs(a - b)))


def cosine(a, b):
    """Cosine similarity between two flattened arrays."""
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _rankdata(a):
    """Average-rank of each element (1-based), ties averaged (scipy 'average' method)."""
    a = np.ravel(np.asarray(a, float))
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # average 1-based rank
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation coefficient (Pearson correlation of ranks)."""
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    if den == 0:
        return float("nan")
    return float(np.dot(ra, rb) / den)


def kendall_tau(a, b):
    """Kendall's tau-b rank correlation coefficient."""
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    n = len(a)
    nc = nd = 0
    t_a = t_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            s = np.sign(da) * np.sign(db)
            if s > 0:
                nc += 1
            elif s < 0:
                nd += 1
            else:
                if da == 0:
                    t_a += 1
                if db == 0:
                    t_b += 1
    n0 = n * (n - 1) / 2
    den = np.sqrt((n0 - t_a) * (n0 - t_b))
    if den == 0:
        return float("nan")
    return float((nc - nd) / den)


def top1_agreement(a, b):
    """1.0 if the top-magnitude index agrees between a and b, else 0.0."""
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    return float(int(np.argmax(np.abs(a)) == np.argmax(np.abs(b))))
