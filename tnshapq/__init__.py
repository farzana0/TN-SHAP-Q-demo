"""TN-SHAP-Q demo package: exact multilinear-extension Shapley attribution for QNNs.

A self-contained consolidation of the TN-SHAP-Q verification core so every demo notebook
imports the *same* verified math instead of re-deriving it:

  - qnn        : single-R_Y QNN, feature game, white-box mixed-state contraction
  - gate_game  : gate-coalition game (CircuitGame) and the qshaptools QAOA example
  - exact      : 2^P enumeration reference (value table, Shapley, order-2 interactions)
  - owen       : Owen-integral Shapley/interactions (feature + gate, exact + finite-shot)
  - datasets   : reproducible Iris / synthetic / breast-cancer configs
  - qlime      : Q-LIME local-linear ranking baseline
  - metrics    : MAE, cosine, Spearman/Kendall/top-1

CPU is forced (CUDA_VISIBLE_DEVICES="") and BLAS pinned to one thread on import, matching
the project environment; everything runs on the Qiskit statevector simulator (numpy only).
"""
from __future__ import annotations

import os as _os

for _v in ("CUDA_VISIBLE_DEVICES",):
    _os.environ.setdefault(_v, "")
# Pin every thread pool to one worker: numpy/BLAS (OMP/MKL/OPENBLAS) AND qiskit's Rust
# accelerators (RAYON). Single-threaded reductions are deterministic to the last bit, so the
# near-machine-epsilon error values and the figures rendered from them are byte-stable.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from . import datasets, exact, gate_game, metrics, owen, qlime, qnn, svqx  # noqa: E402
from .qnn import QNN, FeatureGame, apply_mask  # noqa: E402
from .gate_game import CircuitGame, build_qaoa_example  # noqa: E402
from .exact import (  # noqa: E402
    exact_value_table, exact_shapley, exact_interactions, feature_value_table,
)
from .owen import (  # noqa: E402
    gl_nodes_weights, threshold_M, query_counts,
    owen_integral_shapley, owen_integral_shapley_shots, owen_integral_shapley_shots_hardware,
    owen_integral_interactions,
    gate_owen_shapley, gate_owen_shapley_shots, gate_owen_shapley_shots_hardware,
    gate_owen_interactions,
    gate_extension_value,
)

__all__ = [
    "datasets", "exact", "gate_game", "metrics", "owen", "qlime", "qnn", "svqx",
    "QNN", "FeatureGame", "apply_mask", "CircuitGame", "build_qaoa_example",
    "exact_value_table", "exact_shapley", "exact_interactions", "feature_value_table",
    "gl_nodes_weights", "threshold_M", "query_counts",
    "owen_integral_shapley", "owen_integral_shapley_shots", "owen_integral_shapley_shots_hardware",
    "owen_integral_interactions",
    "gate_owen_shapley", "gate_owen_shapley_shots", "gate_owen_shapley_shots_hardware",
    "gate_owen_interactions",
    "gate_extension_value",
]
