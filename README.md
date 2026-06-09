# TN-SHAP-Q — demo

Exact, query-efficient **Shapley attribution for quantum neural networks (QNNs)** via multilinear
extensions, demonstrated in two standalone notebooks.

A single-$R_Y$ QNN is **exactly multilinear** in the lifted features $[1,\cos x_i,\sin x_i]$ (the
single-frequency case of the finite Fourier structure of Schuld–Sweke–Meyer). Consequently, when you
mask features (or gates) against a baseline, **the multilinear extension of the induced cooperative
game is the very function the circuit computes — not a fitted surrogate.** First-order Shapley values
are then the **Owen diagonal integral**

$$\phi_i=\int_0^1 \partial_i F(t\mathbf 1)\,dt,\qquad
\partial_i F(t\mathbf 1)=F(z_i{=}1,z_{-i}{=}t)-F(z_i{=}0,z_{-i}{=}t),$$

which $M$-point Gauss–Legendre quadrature evaluates **exactly once $M\ge\lceil P/2\rceil$** for $P$
players, using $2PM$ evaluations of the extension — versus $2^P$ for exact coalition enumeration.

| notebook | players | compared against | external dependency |
|---|---|---|---|
| [`qlime_vs_tnshapq_features.ipynb`](qlime_vs_tnshapq_features.ipynb) | **input features** | **Q-LIME** (local linear surrogate) | none |
| [`tnshapq_vs_qshaptools_gates.ipynb`](tnshapq_vs_qshaptools_gates.ipynb) | **circuit gates** | **SVQX** (Heese et al.'s `qshaptools`) | a `qshaptools` clone |

Each notebook inlines a compact QNN / gate game, exact $2^P$ enumeration (ground truth), and the Owen
route, so notebook 1 is fully self-contained and notebook 2 needs only the external `qshaptools`
clone it compares against.

---

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Notebook 1 (feature importance) — no external deps:
jupyter nbconvert --to notebook --execute --inplace qlime_vs_tnshapq_features.ipynb

# Notebook 2 (gate importance) — clone the reference toolbox first:
git clone https://github.com/RaoulHeese/qshaptools.git
jupyter nbconvert --to notebook --execute --inplace tnshapq_vs_qshaptools_gates.ipynb
```

The committed notebooks already contain executed outputs and figures. (Developed against Qiskit
2.4.1; any Qiskit ≥ 1.0 works.)

---

## Notebook 1 — feature importance: Q-LIME vs TN-SHAP-Q

**Steps:** (1) define a single-$R_Y$ QNN; (2) train it on Iris (setosa vs versicolor, 4 features
scaled to $[0,\pi]$), pick a dataset-mean baseline and a few instances; (3) compute the **exact**
feature Shapley values by enumerating all $2^d$ masked coalitions; (4) compute **TN-SHAP-Q** Shapley
values by the Owen integral; (5) compute **Q-LIME** by fitting a kernel-weighted linear surrogate to
sampled feature masks; (6) compare attributions and accuracy vs. evaluation count.

**Results** (representative run, $d{=}4$, train accuracy ≈ 0.98):

| method | MAE vs exact | cosine | Spearman | evaluations |
|---|---|---|---|---|
| **TN-SHAP-Q (Owen, $M{=}2$)** | **5 × 10⁻¹⁷** | **1.000** | **1.00** | $2dM = 16$ |
| Q-LIME (256 perturbations) | 2.8 × 10⁻² | 0.99 | 1.00 | 256 |
| exact enumeration (reference) | — | — | — | $2^4 = 16$ |

- **Exactness threshold:** Owen with $M{=}1$ is inexact (MAE $1.8\times10^{-3}$); $M{=}2=\lceil d/2\rceil$
  reaches machine precision — exactly as the theory predicts.
- **Q-LIME has a bias floor:** its MAE **plateaus at ≈ 5 × 10⁻²** regardless of how many perturbations
  it draws (16 → 512). A local *linear* model cannot represent the exact Shapley value; it gives a good
  *ranking* (Spearman 1.0) but biased magnitudes.

Figures: `feature_importance_bars.pdf`, `qlime_vs_tnshapq_query_efficiency.pdf`.

---

## Notebook 2 — gate importance: TN-SHAP-Q vs SVQX (`qshaptools`)

Runs **Heese et al.'s own quantum-Shapley toolbox on their own example** and compares to TN-SHAP-Q on
the identical gate game.

**Steps:** (1) load the `qshaptools` Shapley engine (`ushap`/`qshap`); (2) build **their README
experiment** — a QAOA ansatz for $H = Z_0Z_1 + 2Z_0 - 3Z_2$, decomposed to elementary gates (13 gate
**players**) and parameter-bound; (3) run **their exact** gate Shapley and **their subsampled**
estimator, counting value-function evaluations; (4) run **TN-SHAP-Q** Owen on the same game; (5)
verify agreement and compare evaluation counts.

**Results** (QAOA, $m{=}13$ gate players):

| method | gate Shapley accuracy | evaluations |
|---|---|---|
| **TN-SHAP-Q (Owen, $M{=}7$)** | exact, max err **4 × 10⁻¹⁵** vs enumeration | **182** extension evals |
| SVQX **exact** (`qshaptools`) | reference | **8192** value evals |
| SVQX subsampled ($K{=}1$ / $K{=}64$) | MAE 0.66 / 0.11 | ≈ 25 / 1093 value evals |

- **Same answer:** TN-SHAP-Q reproduces the `qshaptools` exact gate Shapley to $\sim10^{-15}$ (and our
  independent enumeration matches theirs to $3\times10^{-16}$).
- **≈ 45× fewer evaluations** than their exact mode (**182 Owen extension evaluations vs. 8192 SVQX
  value evaluations**), and the gap grows from $O(m^2)$ vs. $O(2^m)$ with circuit size. Their
  subsampled estimator needs far more value evaluations to approach the exact values.

Figure: `tnshapq_vs_qshaptools_gates.pdf`.

### Running `qshaptools` here
`qshaptools` targets Qiskit < 1.0 (`Aer`, `qiskit.opflow`, `qiskit.utils.QuantumInstance`, removed in
≥ 1.0). The notebook runs their Shapley **engine verbatim** (`ushap`/`qshap`, pure NumPy) and adapts
only the deprecated circuit glue (`extract_from_circuit`/`build_circuit`) plus a value function
computing the identical $\langle H\rangle$. To run their package completely unmodified, use a separate
environment pinned to their original Qiskit version. Point `QSHAPTOOLS_PATH` at your clone if it is not
at `./qshaptools/src/qshaptools`.

---

## Related work and papers

- **SVQX — quantum gate Shapley values.** R. Heese, T. Gerlach, S. Mücke, S. Müller, M. Jakobs, N.
  Piatkowski, *Explaining Quantum Circuits with Shapley Values*, arXiv:[2301.09138](https://arxiv.org/abs/2301.09138)
  (Quantum, 2025). Code: [github.com/RaoulHeese/qshaptools](https://github.com/RaoulHeese/qshaptools).
  Reproduced directly in notebook 2.
- **TN-SHAP / TN-SHAP-G — multilinear / tensor-network attribution.** F. Heidari, C. Li, G. Rabusseau,
  *Tractable Shapley Values and Interactions via Tensor Networks*, AISTATS 2026,
  arXiv:[2510.22138](https://arxiv.org/abs/2510.22138); F. Heidari, G. Rabusseau, *TN-SHAP-G*, ICML
  2026, arXiv:[2606.01540](https://arxiv.org/abs/2606.01540). TN-SHAP-Q is the exact, quantum instance
  of this multilinear-attribution framework.
- **Q-LIME — local linear QNN explanations.** L. Pira, C. Ferrie, *On the Interpretability of Quantum
  Neural Networks*, arXiv:[2308.11098](https://arxiv.org/abs/2308.11098) (QMI, 2024).
- **Encoding structure.** M. Schuld, R. Sweke, J. J. Meyer, *Effect of data encoding on the expressive
  power of variational quantum-machine-learning models*, Phys. Rev. A 103, 032430 (2021),
  arXiv:[2008.08605](https://arxiv.org/abs/2008.08605).
- **QNN model.** E. Farhi, H. Neven, *Classification with Quantum Neural Networks on Near Term
  Processors*, arXiv:[1802.06002](https://arxiv.org/abs/1802.06002).
- **Game theory.** L. S. Shapley, *A Value for n-Person Games* (1953); G. Owen, *Multilinear Extensions
  of Games*, Management Science (1972).

---

## Summary

- **Accuracy:** TN-SHAP-Q is **exact** — it reproduces exact coalition-enumeration Shapley values (and
  the `qshaptools` exact gate Shapley) to machine precision ($\sim10^{-15}$–$10^{-16}$), for both
  feature and gate players. Q-LIME is a good ranker but carries a local-linear bias floor.
- **Efficiency:** the Owen route is exact at $2PM$ evaluations with $M=\lceil P/2\rceil$ — polynomial
  in the number of players — versus $2^P$ for enumeration (≈ 45× fewer on the 13-gate QAOA example, and
  exponentially more as circuits grow).

*Caveat:* TN-SHAP-Q evaluates the multilinear extension, while SVQX / Q-LIME evaluate ordinary masked
circuits; we compare Shapley outputs and value-evaluation counts.
