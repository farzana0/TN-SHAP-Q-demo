# Experimental settings and provenance

For every experiment, this documents **where the setting comes from in the quantum-computing
literature** and **exactly what we replicate**. Each experiment is one self-contained notebook in
[`notebooks/`](notebooks/); headline numbers are written to [`results/`](results/) and independently
re-checked in [`results/AUDIT.md`](results/AUDIT.md). The same provenance appears in the paper
appendix and its standalone companion `experimental_settings.tex`.

| notebook | players | setting source (QC reference) | what we replicate |
|---|---|---|---|
| `01`, `02` | features | single-$R_Y$ finite-lift encoding (Schuld–Sweke–Meyer); variational QNN classifier (Farhi–Neven) | the finite-lift QNN; Owen / Grabisch–Roubens attribution vs exact $2^d$ enumeration |
| `03` | features | same encoding, synthetic seed-fixed weights | weight-independent exactness for $d=4,6,8,10,12$ |
| `04` | feat./gates | projective shot-noise (standard measurement model); finite-shot sensitivity motivated by Q-LIME | $N$-shot estimates, $N\in\{10,10^2,10^3,10^4\}$, both games |
| `05`, `06` | gates | SVQX gate game + the QAOA example (Heese et al., `qshaptools`); QAOA (Farhi–Goldstone–Gutmann); Qiskit `QAOAAnsatz` | their gate-removal game and $H=Z_0Z_1+2Z_0-3Z_2$ example, verbatim |
| `07` | features | Q-LIME local-linear surrogate (Pira–Ferrie) | their surrogate; **ranking-only** comparison (no magnitude MAE) |
| `08` | features | single-$R_Y$ VQC on a UCI benchmark | finite-lift VQC on Wisconsin breast cancer; exact recovery vs $2^6$ |

## Details

### Finite-lift QNN and the feature game — `01`, `02`, `03`, `08`
**Source.** Encoding each feature by one $R_Y(x_i)$ gives $\rho_i(x_i)=\tfrac12(I+\cos x_i\,Z+\sin x_i\,X)$,
i.e. the finite lift $\phi(x_i)=[1,\cos x_i,\sin x_i]$ — the *single-frequency* case of the data-encoding
Fourier spectrum of **Schuld, Sweke & Meyer**, with the variational-classifier form (input-independent
block, measured observable) of **Farhi & Neven**.

**Setting we use.** $R_Y(x_i)$ encoding on $d$ qubits; variational block of $L{=}2$ layers, each
$R_Y(w_q)$ on every qubit then a CX ring $q\to(q{+}1)\bmod d$; readout $Z_0$,
$P(\text{class }1)=(1-\langle Z_0\rangle)/2$. Weights $\sim\mathcal U(0,2\pi)$ (seed 0), trained by
parameter-shift + Adam on BCE. Inputs MinMax-scaled to $[0,\pi]$; coalition baseline = feature-wise
dataset mean; feature game $v_x(S)=P(\text{class }1\mid x_S)$. Datasets: Iris (setosa vs versicolor,
$d{=}4$) and Wisconsin breast cancer (top-6 ANOVA features, $d{=}6$). The hardware-efficient ansatz and
trained weights are ours; the **replicated** part is the finite-lift single-$R_Y$ encoding regime.

### Gate game and the QAOA benchmark — `05`, `06`
**Source.** The gate-coalition game (players are gates; a coalition keeps its gates and removes the rest;
value = energy $\langle\psi_S|H|\psi_S\rangle$) is the **SVQX** setting of **Heese et al.** and their
`qshaptools` toolbox. We reproduce **their README example**: $H=Z_0Z_1+2Z_0-3Z_2$ and a depth-one
**QAOA** ansatz (Farhi–Goldstone–Gutmann) built with Qiskit's `QAOAAnsatz`.

**Setting we use.** `QAOAAnsatz(cost_operator=H, reps=1)` decomposed to elementary gates and
parameter-bound ($\gamma{=}\beta{=}0.3$) → 13 gate players on 3 qubits. Gate-removal masking and the
statevector energy value function exactly as in SVQX (finite-shot via qubit-wise-commuting Pauli
grouping). We compare against SVQX exact $2^{13}=8192$ enumeration and the SVQX subsampled estimator;
only the attribution algorithm (channel-mixture Owen integration) is ours.

### Q-LIME baseline — `07`
**Source.** **Q-LIME** (Pira & Ferrie): a kernel-weighted local-*linear* surrogate over sampled binary
feature masks — the quantum analogue of LIME. **Setting we use.** Run on the same Iris QNN with
$16$–$512$ masks; because its coefficients are not Shapley values we compare **only the feature ranking**
(Spearman, Kendall-$\tau$, top-1).

### Shot-noise model — `04`
**Source.** Standard projective-measurement shot noise; explanation sensitivity to it is the motivation
of Q-LIME. **Setting we use.** Replace each exact expectation by an $N$-shot estimate — a readout
probability by $\mathrm{Binomial}(N,p)/N$, the gate energy by multinomial sampling per
qubit-wise-commuting Pauli group — and report MAE / cosine / rank correlation, averaged over fixed seeds.

### Attribution math
Classical cooperative game theory: Shapley values (Shapley), the Owen multilinear-extension diagonal
integral (Owen), Grabisch–Roubens interaction indices and Harsanyi dividends, with Gauss–Legendre
quadrature. TN-SHAP-Q is the quantum instance of the TN-SHAP multilinear-attribution family.

## References
- Schuld, Sweke & Meyer, *Effect of data encoding…*, Phys. Rev. A 103, 032430 (2021), [arXiv:2008.08605](https://arxiv.org/abs/2008.08605)
- Farhi & Neven, *Classification with QNNs on Near-Term Processors*, [arXiv:1802.06002](https://arxiv.org/abs/1802.06002)
- Heese et al., *Explaining Quantum Circuits with Shapley Values*, QMI 2025, [arXiv:2301.09138](https://arxiv.org/abs/2301.09138); `qshaptools`
- Farhi, Goldstone & Gutmann, *A Quantum Approximate Optimization Algorithm*, [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
- Javadi-Abhari et al., *Quantum computing with Qiskit*, [arXiv:2405.08810](https://arxiv.org/abs/2405.08810) (`QAOAAnsatz`, `Statevector`)
- Pira & Ferrie, *On the Interpretability of QNNs*, QMI 2024, [arXiv:2308.11098](https://arxiv.org/abs/2308.11098)
- Shapley (1953); Owen, *Multilinear Extensions of Games*, Manag. Sci. (1972); Grabisch & Roubens, IJGT (1999); Harsanyi (1963)
- Fisher, *Ann. Eugenics* (1936) — Iris (UCI); Wolberg, Street & Mangasarian — Breast Cancer Wisconsin (UCI, 1995)
