"""Exact 2^P coalition enumeration: value table, Shapley values, and order-2 interactions.

This is the ground-truth reference the Owen estimators are validated against.  Ported
verbatim (semantics preserved) from the TN-SHAP-Q verification core (svqx_core.py ::
exact_value_table / exact_shapley / exact_interactions).
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .qnn import FeatureGame


def exact_value_table(game, shots=None, rng=None):
    """Enumerate all 2^n coalitions -> dict {bitmask:int -> value}.  Costs 2^n evaluations."""
    n = game.n
    table = {}
    for mask in range(1 << n):
        z = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        table[mask] = game.value(z, shots=shots, rng=rng)
    return table


def _mask_of(indices):
    m = 0
    for i in indices:
        m |= (1 << i)
    return m


def exact_shapley(table, n):
    """Exact first-order Shapley values from a full 2^n value table."""
    phi = np.zeros(n)
    fact = [math.factorial(k) for k in range(n + 1)]
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(len(others) + 1):
            w = fact[r] * fact[n - r - 1] / fact[n]
            for S in combinations(others, r):
                mS = _mask_of(S)
                phi[i] += w * (table[mS | (1 << i)] - table[mS])
    return phi


def exact_interactions(table, n):
    """Grabisch-Roubens order-2 Shapley interaction index (symmetric [n,n], diag 0)."""
    I = np.zeros((n, n))
    fact = [math.factorial(k) for k in range(n + 1)]
    for i, j in combinations(range(n), 2):
        others = [k for k in range(n) if k not in (i, j)]
        acc = 0.0
        for r in range(len(others) + 1):
            w = fact[r] * fact[n - r - 2] / fact[n - 1]
            for S in combinations(others, r):
                mS = _mask_of(S)
                mSi = mS | (1 << i)
                mSj = mS | (1 << j)
                mSij = mSi | (1 << j)
                acc += w * (table[mSij] - table[mSi] - table[mSj] + table[mS])
        I[i, j] = I[j, i] = acc
    return I


def feature_value_table(qnn, n, x, baseline, *, shots=None, rng=None):
    """nu(z) = P(class 1 | x_z) over all 2^n feature coalitions (exact statevector)."""
    game = FeatureGame(qnn, x, baseline, target="class1")
    return exact_value_table(game, shots=shots, rng=rng)
