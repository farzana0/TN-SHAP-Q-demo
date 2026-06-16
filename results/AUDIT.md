# results/AUDIT.md — independent calculation audit

Exact Shapley/interactions recomputed with fresh reference implementations (coalition-weighted formula; permutation average as a second method at $d{=}4$), independent of `tnshapq.exact`. Owen estimators (`tnshapq.owen`) are the methods under test, compared against this ground truth. Metrics recomputed from raw arrays.

**Reproduce all notebooks (byte-stable):**
```bash
cd TN-SHAP-Q-demo
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
RAYON_NUM_THREADS=1 \
  /network/scratch/f/farzaneh.heidari/tn-svqx/env/bin/python -m nbconvert \
  --to notebook --execute --inplace notebooks/[0-9][0-9]_*.ipynb
# audit (independent recomputation, writes this file):
RAYON_NUM_THREADS=1 /network/scratch/f/farzaneh.heidari/tn-svqx/env/bin/python audit.py
```

## 01 — feature first-order Shapley (Iris, d=4)
| quantity | recomputed | JSON/manuscript | status |
|---|---|---|---|
| train accuracy | 0.9700 | 0.97 | OK |
| M threshold ceil(d/2) | 2 | 2 | OK |
| evals = 2dM | 16 | 16 | OK |
| enum evals 2^d | 16 | 16 | OK |
| MAE Owen(M=2) vs exact | 4.25e-17 | 4.25e-17 | OK |
| cosine(M=2) | 1.000000 | 1.000000 | OK |
| MAE Owen(M=1) (inexact) | 1.82e-04 | 1.82e-04 | OK |
| ref vs permutation Shapley | 5.6e-17 | (cross-check) | OK |
| assert_allclose(M=2,exact,1e-10) | PASS | exact claim | PASS |

## 02 — feature order-2 interactions (Iris, d=4)
| quantity | recomputed | JSON/manuscript | status |
|---|---|---|---|
| M threshold ceil((d-1)/2) | 2 | 2 | OK |
| evals/pair = 2^2 M | 8 | 8 | OK |
| #pairs C(4,2) | 6 | 6 | OK |
| MAE Owen(M=2) vs GR exact | 9.51e-17 | 9.51e-17 | OK |
| MAE Owen(M=1) (inexact) | 2.62e-06 | 2.62e-06 | OK |
| dominant pair | [2, 3] | [2, 3] | OK |
| assert_allclose(M=2,GR,1e-10) | PASS | exact claim | PASS |

## 03 — dimension scaling (synthetic single-RY QNNs)
| dim | M=ceil(d/2) | Owen evals | 2^d | recomputed max err | JSON max err | status |
|---|---|---|---|---|---|---|
| d=4 | M=2 | 16 (2dM) | 16 | 8.3e-17 | 8.3e-17 | OK |
| d=6 | M=3 | 36 (2dM) | 64 | 3.6e-17 | 4.2e-17 | OK |
| d=8 | M=4 | 64 (2dM) | 256 | 3.5e-17 | 3.3e-17 | OK |
| d=10 | M=5 | 100 (2dM) | 1024 | 4.5e-17 | 4.5e-17 | OK |
| d=12 | M=6 | 144 (2dM) | 4096 | 9.7e-17 | 1.0e-16 | OK |

## 05 — gate first-order Shapley (qshaptools QAOA, m=13)
| quantity | recomputed | JSON/manuscript | status |
|---|---|---|---|
| players m | 13 | 13 | OK |
| M threshold ceil(m/2) | 7 | 7 | OK |
| Owen evals 2mM | 182 | 182 | OK |
| enum evals 2^m | 8192 | 8192 | OK |
| speedup 2^m/2mM | 45.0 | 45.0 | OK |
| max err Owen vs exact | 1.83e-15 | 4.05e-15 | OK |
| assert_allclose(1e-12) | PASS | exact claim | PASS |

## 06 — gate order-2 interactions (qshaptools QAOA, m=13)
| quantity | recomputed | JSON/manuscript | status |
|---|---|---|---|
| M threshold ceil((m-1)/2) | 6 | 6 | OK |
| evals/pair 2^2 M | 24 | 24 | OK |
| #pairs C(13,2) | 78 | 78 | OK |
| max err Owen vs GR exact | 4.00e-15 | 4.44e-15 | OK |
| v_full (recomputed table) | 1.8128 | 1.8128 | OK |
| assert_allclose(1e-12) | PASS | exact claim | PASS |

## 04 — shot-noise robustness (feature + gate)
Seeds: feature `range(24)`, gate `range(1000,1000+24)`; repeats per N = **24**; N = [10, 100, 1000, 10000]. Finite-shot estimates are NOT claimed exact (all MAE >> machine precision).

| quantity | recomputed | JSON/manuscript | status |
|---|---|---|---|
| feature log-log MAE slope | -0.494 | -0.494 | OK |
| gate log-log MAE slope | -0.504 | -0.504 | OK |
| close to -1/2 ? | yes | target -0.5 | OK |
| feature MAE not ~0 (not exact) | 3.5e-03..1.1e-01 | finite-shot | OK |
| gate MAE not ~0 (not exact) | 1.3e-02..4.4e-01 | finite-shot | OK |

## 07 — Q-LIME (ranking-only)
JSON keys: `['budgets', 'cohort_size', 'd', 'kendall', 'kendall_saturation', 'ranking_only', 'seeds', 'spearman', 'spearman_saturation', 'task', 'top1', 'top1_saturation']` — no MAE-vs-Shapley-magnitude field present (ranking-only confirmed).

| quantity | recomputed | JSON/manuscript | status |
|---|---|---|---|
| ranking-only (no magnitude MAE) | yes | required | OK |
| Spearman saturation | 0.95 | 0.95 | OK |
| Kendall saturation | 0.917 | 0.917 | OK |
| top-1 saturation | 0.875 | 0.875 | OK |
| saturates below 1 | yes | <1 | OK |

## 08 — breast-cancer VQC benchmark (d=6)
| quantity | recomputed | JSON/manuscript | status |
|---|---|---|---|
| feature dim d | 6 | 6 | OK |
| M = ceil(d/2) | 3 | 3 | OK |
| enum 2^d feasible & used | 64 (built) | 64 | OK |
| Owen evals 2dM | 36 | 36 | OK |
| train accuracy | 0.9174 | 0.9174 | OK |
| MAE Owen(M=3) vs exact | 5.54e-17 | 5.66e-17 | OK |
| cosine(M=3) | 1.000000 | 1.000000 | OK |
| MAE Owen(M=2) threshold FAIL | 7.50e-07 | 7.50e-07 | OK (inexact) |

## Summary
| experiment | audit status |
|---|---|
| 01 | PASS |
| 02 | PASS |
| 03 | PASS |
| 04 | PASS |
| 05 | PASS |
| 06 | PASS |
| 07 | PASS |
| 08 | PASS |

**Overall: ALL PASS** — 8/8 experiments validated against independent recomputation.

**Discrepancies with the manuscript:** none in any scientific claim. Every paper number is generated from these `results/*.json` by `docs/make_macros.py` -> `results_macros.tex` (verified zero drift); the cost-table order-$k$ row and the interaction proposition use $2^kM$ (= the recomputed 24 evals/pair at $k{=}2,M{=}6$).

**Note on machine-precision error values.** The headline *exactness* claims (thresholds, evaluation counts, "machine precision") all reproduce. The exact decimal of a $\sim10^{-15}$ max-error is summation-order dependent, however: an independent exact-Shapley implementation gives gate max-error $1.8\times10^{-15}$ (this audit) vs the notebook's $4.1\times10^{-15}$ (paper), and likewise $4.0$ vs $4.4\times10^{-15}$ for gate interactions, with $\le10^{-16}$ wiggles in the feature/scaling rows. Both sit far below the $10^{-12}$ exactness bound and both pass `assert_allclose`; the manuscript reports the notebook's measured value. No claim depends on the last digits of a near-zero quantity.
