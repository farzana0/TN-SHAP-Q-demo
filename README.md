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
players, using only $2PM$ evaluations — versus $2^P$ for exact coalition enumeration.

This repo shows that on two player choices:

| notebook | players | compared against | external dependency |
|---|---|---|---|
| [`qlime_vs_tnshapq_features.ipynb`](qlime_vs_tnshapq_features.ipynb) | **input features** | **Q-LIME** (local linear surrogate) | none |
| [`tnshapq_vs_qshaptools_gates.ipynb`](tnshapq_vs_qshaptools_gates.ipynb) | **circuit gates** | **SVQX** (Heese et al.'s `qshaptools`) | a `qshaptools` clone |

Each notebook inlines a compact QNN / gate game, exact $2^P$ enumeration (ground truth), and the Owen
route — so notebook 1 is fully self-contained, and notebook 2 only needs the external `qshaptools`
clone it is explicitly comparing against.

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

The committed notebooks already contain executed outputs and figures, so you can also just open and
read them. (Developed against Qiskit 2.4.1; any Qiskit ≥ 1.0 works.)

---

## Notebook 1 — feature importance: Q-LIME vs TN-SHAP-Q

**What we do, step by step:**
1. Define a single-$R_Y$ QNN (encode each feature by $R_Y(x_i)$, then $L{=}2$ variational layers of
   $R_Y(w)$ + a CX ring, readout $P(\text{class }1)=(1-\langle Z_0\rangle)/2$).
2. Train it on Iris (setosa vs versicolor, 4 features scaled to $[0,\pi]$); pick a dataset-mean
   baseline and a few instances.
3. Compute the **exact** feature Shapley values by enumerating all $2^d$ masked coalitions.
4. Compute **TN-SHAP-Q** Shapley values by the Owen integral (mixed lifted state
   $(1-z_i)\rho(b_i)+z_i\rho(x_i)$ per feature; Gauss–Legendre quadrature).
5. Compute **Q-LIME**: fit a kernel-weighted linear surrogate to sampled binary feature masks.
6. Compare attributions, then sweep accuracy vs. query count for both methods.

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
- **Identical-query-type control:** estimating the Owen interior values by *randomized masking* (ordinary
  masked-circuit queries, just like Q-LIME) still **converges** to exact (MAE $4.6\times10^{-2}\to
  1.0\times10^{-2}$ as samples grow), while Q-LIME stays at its floor — so TN-SHAP-Q's accuracy
  advantage is intrinsic, not a query-type artifact.

Figures: `feature_importance_bars.pdf`, `qlime_vs_tnshapq_query_efficiency.pdf`.

---

## Notebook 2 — gate importance: TN-SHAP-Q vs SVQX (`qshaptools`)

This runs **Heese et al.'s own quantum-Shapley toolbox on their own example** and compares to
TN-SHAP-Q on the identical gate game.

**What we do, step by step:**
1. Clone `qshaptools` and load its **Shapley engine** (`ushap` / `qshap`).
2. Build **their README experiment**: a QAOA ansatz for the cost Hamiltonian
   $H = Z_0Z_1 + 2Z_0 - 3Z_2$, decomposed to elementary gates (13 gate **players**) and
   parameter-bound. The gate game removes the gates outside a coalition and reads the energy
   $\langle H\rangle$.
3. Run **their exact** gate Shapley (`QuantumShapleyValues`, all coalitions) and **their subsampled**
   estimator, counting masked-circuit evaluations.
4. Run **TN-SHAP-Q** Owen on the same game (per-gate mixed-unitary channel
   $\Phi_i(\rho)=z_iU_i\rho U_i^\dagger+(1-z_i)\rho$; Gauss–Legendre, $M=\lceil m/2\rceil$).
5. Verify they agree and compare query efficiency.

**Results** (QAOA, $m{=}13$ gate players):

| method | gate Shapley accuracy | evaluations |
|---|---|---|
| **TN-SHAP-Q (Owen, $M{=}7$)** | exact, max err **4 × 10⁻¹⁵** vs enumeration | $2m\lceil m/2\rceil = $ **182** |
| `qshaptools` **exact** | reference | $2^{13} = $ **8192** |
| `qshaptools` subsampled ($K{=}1$ / $K{=}64$) | MAE 0.66 / 0.11 | ≈ 25 / 1093 |

- **Same answer:** TN-SHAP-Q reproduces the `qshaptools` exact gate Shapley to $\sim10^{-15}$ (and our
  independent enumeration matches theirs to $3\times10^{-16}$).
- **≈ 45× fewer queries** than their exact mode (182 vs. 8192), and the gap grows from $O(m^2)$ vs.
  $O(2^m)$ with circuit size.
- Their **subsampled** estimator needs far more ordinary evaluations to approach the exact values and
  never reaches machine precision.

Figure: `tnshapq_vs_qshaptools_gates.pdf`.

### Note on running `qshaptools` here
`qshaptools` targets **Qiskit < 1.0** (`Aer`, `qiskit.opflow`, `qiskit.utils.QuantumInstance`, all
removed in ≥ 1.0). The notebook runs their **Shapley engine verbatim** (`ushap` / `qshap`, pure NumPy)
and replaces only the deprecated circuit glue (`extract_from_circuit` / `build_circuit`) with a small
Qiskit-≥1.0 shim plus a value function computing the **identical** $\langle H\rangle$ (their
`qvalues.value_H` used the removed backend). To run their package completely unmodified instead, use a
separate environment pinned to their original Qiskit version. Point `QSHAPTOOLS_PATH` at your clone if
it is not at `./qshaptools/src/qshaptools`.

---

## Reproducibility of related work and papers

This demo reproduces / builds directly on:

- **SVQX — quantum gate Shapley values.** R. Heese, T. Gerlach, S. Mücke, S. Müller, M. Jakobs, N.
  Piatkowski, *Explaining Quantum Circuits with Shapley Values: Towards Explainable Quantum Machine
  Learning*, arXiv:[2301.09138](https://arxiv.org/abs/2301.09138) (Quantum, 2025). Code:
  [github.com/RaoulHeese/qshaptools](https://github.com/RaoulHeese/qshaptools). **Reproduced**
  directly in notebook 2 (their engine, their experiment).
- **TN-SHAP / TN-SHAP-G — multilinear / tensor-network attribution.** F. Heidari, C. Li, G. Rabusseau,
  *Tractable Shapley Values and Interactions via Tensor Networks*, AISTATS 2026,
  arXiv:[2510.22138](https://arxiv.org/abs/2510.22138); F. Heidari, G. Rabusseau, *TN-SHAP-G:
  Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions*, ICML 2026,
  arXiv:[2606.01540](https://arxiv.org/abs/2606.01540). TN-SHAP-Q is the **exact, quantum instance** of
  this multilinear-attribution framework (here the multilinear object is determined by the circuit, not
  learned).
- **Q-LIME — local linear QNN explanations.** L. Pira, C. Ferrie, *On the Interpretability of Quantum
  Neural Networks*, arXiv:[2308.11098](https://arxiv.org/abs/2308.11098) (Quantum Machine Intelligence,
  2024). Re-implemented as the baseline in notebook 1.
- **Encoding structure.** M. Schuld, R. Sweke, J. J. Meyer, *Effect of data encoding on the expressive
  power of variational quantum-machine-learning models*, Phys. Rev. A 103, 032430 (2021),
  arXiv:[2008.08605](https://arxiv.org/abs/2008.08605).
- **QNN model.** E. Farhi, H. Neven, *Classification with Quantum Neural Networks on Near Term
  Processors*, arXiv:[1802.06002](https://arxiv.org/abs/1802.06002).
- **Game theory.** L. S. Shapley, *A Value for n-Person Games* (1953); G. Owen, *Multilinear Extensions
  of Games*, Management Science (1972).

---

## Headline takeaways

- **Accuracy:** TN-SHAP-Q is **exact** — it reproduces exact coalition-enumeration Shapley values (and
  the `qshaptools` exact gate Shapley) to machine precision ($\sim10^{-15}$–$10^{-16}$), for both
  feature and gate players. Q-LIME is a good ranker but carries a local-linear bias floor.
- **Efficiency:** the Owen route is exact at $2PM$ evaluations with $M=\lceil P/2\rceil$ — polynomial
  in the number of players — versus $2^P$ for enumeration (≈ 45× fewer on the 13-gate QAOA example, and
  exponentially more as circuits grow). Sampling estimators need far more queries to merely approach
  this accuracy.

### Honest caveat
TN-SHAP-Q's "evaluations" are **extension evaluations** — interior values of the multilinear extension
(mixed lifted states for features; per-gate mixed-unitary channels for gates). In simulation these are
computed directly; on hardware they are estimated by randomized coalition masking (a factor $R$ more
ordinary masked-circuit shots, $\mathrm{MAE}\propto R^{-1/2}$). Q-LIME's and `qshaptools`'s counts are
ordinary masked-circuit evaluations. The query reduction therefore reflects **exploiting the
multilinear structure of the game**, not a generic shot-count speedup — and the identical-query-type
control in notebook 1 shows TN-SHAP-Q still wins on accuracy even when restricted to ordinary queries,
because Q-LIME is biased.
