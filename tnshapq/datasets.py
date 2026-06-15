"""Reproducible model/data configs used by the demo notebooks.

Every QNN is trained deterministically (fixed seed, parameter-shift + Adam) so figures
are byte-stable; trained weights are additionally cached under tnshapq/_cache/ to avoid
recomputation (the cache is reproducible, not a source of nondeterminism).
"""
from __future__ import annotations

import os

import numpy as np

from .qnn import QNN

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
os.makedirs(_CACHE, exist_ok=True)

SEED = 0


def _scaled_xy(data, target, features, pos_classes):
    """Binary subset on `pos_classes`, columns `features`, MinMax-scaled to [0, pi]."""
    from sklearn.preprocessing import MinMaxScaler
    m = np.isin(target, pos_classes)
    Xf = np.asarray(data)[m][:, features]
    y = (np.asarray(target)[m] == pos_classes[1]).astype(float)
    Xf = MinMaxScaler((0, np.pi)).fit_transform(Xf)
    return Xf, y


def iris_qnn(n_features=4, *, n_layers=2, epochs=80, lr=0.15, seed=SEED):
    """Trained single-R_Y QNN on Iris (setosa vs versicolor).

    n_features=4 uses all four features; n_features=2 uses petal length/width ([2,3]).
    Returns (X, y, baseline, qnn, acc).  baseline = feature-wise dataset mean.
    """
    from sklearn.datasets import load_iris
    iris = load_iris()
    feats = [2, 3] if n_features == 2 else list(range(n_features))
    X, y = _scaled_xy(iris.data, iris.target, feats, pos_classes=[0, 1])
    qnn = QNN(n_features=n_features, n_layers=n_layers, seed=seed)
    cache = os.path.join(_CACHE, f"iris{n_features}_L{n_layers}_e{epochs}_seed{seed}.npy")
    if os.path.exists(cache):
        qnn.weights = np.load(cache)
    else:
        qnn.fit(X, y, epochs=epochs, lr=lr, seed=seed)
        np.save(cache, qnn.weights)
    acc = float(np.mean((qnn.predict(X) > 0.5) == y))
    return X, y, baseline_of(X), qnn, acc


def baseline_of(X):
    """Feature-wise dataset mean baseline."""
    return np.asarray(X, float).mean(0)


def pick_instances(X, n_instances, *, seed=123):
    """Deterministic instance cohort (indices into X)."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X), size=min(n_instances, len(X)), replace=False)
    return [(int(i), np.asarray(X[i], float)) for i in sorted(idxs)]


def synthetic_qnn(d, *, n_layers=2, seed=0):
    """Untrained synthetic single-R_Y QNN with a random instance/baseline in [0, pi]^d.

    Exactness of the multilinear extension is weight- and instance-independent, so the
    scaling study uses random (seed-fixed) weights and points.  Returns (qnn, x, baseline).
    """
    qnn = QNN(n_features=d, n_layers=n_layers, seed=seed)
    rng = np.random.default_rng(1000 + d)
    x = rng.uniform(0.0, np.pi, size=d)
    baseline = rng.uniform(0.0, np.pi, size=d)
    return qnn, x, baseline


def breast_cancer_qnn(n_features=6, *, n_layers=2, epochs=60, lr=0.15, seed=SEED,
                      n_train=160):
    """Trained single-R_Y QNN on the (reduced) breast-cancer task.

    The `n_features` most discriminative columns (ANOVA F-score) are kept so exact 2^d
    enumeration stays feasible; training uses a seed-fixed subsample of size `n_train`.
    Returns (X, y, baseline, qnn, acc, feat_idx, feat_names).
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import MinMaxScaler
    bc = load_breast_cancer()
    sel = SelectKBest(f_classif, k=n_features).fit(bc.data, bc.target)
    feat_idx = np.sort(sel.get_support(indices=True))
    Xf = MinMaxScaler((0, np.pi)).fit_transform(bc.data[:, feat_idx])
    y = bc.target.astype(float)                       # 0=malignant, 1=benign
    qnn = QNN(n_features=n_features, n_layers=n_layers, seed=seed)
    cache = os.path.join(_CACHE, f"bc{n_features}_L{n_layers}_e{epochs}_n{n_train}_seed{seed}.npy")
    if os.path.exists(cache):
        qnn.weights = np.load(cache)
    else:
        rng = np.random.default_rng(seed)
        tr = rng.choice(len(Xf), size=min(n_train, len(Xf)), replace=False)
        qnn.fit(Xf[tr], y[tr], epochs=epochs, lr=lr, seed=seed)
        np.save(cache, qnn.weights)
    acc = float(np.mean((qnn.predict(Xf) > 0.5) == y))
    names = [str(bc.feature_names[i]) for i in feat_idx]
    return Xf, y, baseline_of(Xf), qnn, acc, feat_idx, names
