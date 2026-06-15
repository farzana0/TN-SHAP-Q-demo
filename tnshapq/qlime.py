"""Q-LIME: a local linear surrogate over feature coalitions (Pira & Ferrie 2024).

This is a ranking baseline, NOT a Shapley method: it fits a kernel-weighted ridge
regression of the masked-circuit value on the binary coalition vector and uses the
coefficients as feature importances.  Ported verbatim from the TN-SHAP-Q verification
core (experiments/e3_tn_qlime_qnn.py).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def _kernel(dist, kernel_width):
    return np.exp(-(dist ** 2) / (kernel_width ** 2 + 1e-12))


def fit_weighted_ridge(Z, y, w, lam=1e-3) -> Tuple[float, np.ndarray]:
    """Weighted ridge regression y ~ b + Z @ beta (intercept unpenalized)."""
    Z = np.asarray(Z, float)
    y = np.asarray(y, float).reshape(-1)
    w = np.asarray(w, float).reshape(-1)
    X = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
    XtW = X.T @ np.diag(w)
    A = XtW @ X
    A[1:, 1:] += lam * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, XtW @ y)
    return float(coef[0]), coef[1:]


def qlime_explain(game, *, shots: Optional[int], seed: int, M: int = 512,
                  kernel_width: float = 0.35, lam: float = 1e-3) -> Tuple[np.ndarray, float]:
    """Return (beta, weighted_r2) for a Q-LIME surrogate with M sampled coalitions."""
    rng = np.random.default_rng(seed)
    d = game.n
    Z = rng.integers(0, 2, size=(M, d), dtype=np.int8)
    Z[0, :] = 1
    Z[1, :] = 0
    dist = np.sum(1 - Z, axis=1) / math.sqrt(d)
    w = _kernel(dist, kernel_width=kernel_width)
    y = np.array([game.value(z, shots=shots, rng=rng) for z in Z], dtype=float)
    b0, beta = fit_weighted_ridge(Z, y, w, lam=lam)
    pred = b0 + Z @ beta
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - pred) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    return np.asarray(beta, float), r2
