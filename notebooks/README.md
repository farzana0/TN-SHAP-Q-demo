# TN-SHAP-Q demo notebooks

One self-contained notebook per experiment, each reproducing **exactly one paper figure**
and writing its headline numbers to `../results/<NN>.json`. Every notebook:

* starts with a markdown cell stating what it computes and which figure it produces,
* pins a fixed seed (figures are byte-stable),
* imports the shared [`tnshapq`](../tnshapq) package — the Owen integration, the finite-lift
  feature extension, the gate channel-mixture extension, and the exact $2^P$ enumeration
  reference live there, **not** inside the notebooks,
* ends with a verification cell asserting machine-precision agreement against $2^P$
  enumeration via `np.testing.assert_allclose` (where exact ground truth exists),
* saves its figure to `../figures/` at ≥150 dpi.

## Run

```bash
# all figures + results, headless (CPU only, no torch/GPU)
for nb in notebooks/[0-9][0-9]_*.ipynb; do
  python -m nbconvert --to notebook --execute --inplace "$nb"
done
```

## Notebook → figure → paper reference

| Notebook | Figure(s) | Paper element |
|---|---|---|
| `01_feature_attribution.ipynb` | `fig_feature_first.png` | Fig. 1(a) (first-order); Table 1; §Experiments "Feature attribution and the exactness threshold" ($M\ge\lceil d/2\rceil$) |
| `02_feature_interactions.ipynb` | `fig_feature_int.png`, **`fig_feature.png`** | Fig. 1(a) composite; Prop. *Interaction exactness*; dominant pair (features 2–3) |
| `03_dimension_scaling.ipynb` | `fig_cost.png` | Fig. 1(b); Prop. *First-order exactness and cost*; §"Dimension scaling" ($d=4,\dots,12$) |
| `04_shot_noise.ipynb` | `fig_shots.png` | Fig. 1(c); §"Shot-noise robustness" ($\propto 1/\sqrt N$, feature + gate); Hardware realizability |
| `05_gate_attribution.ipynb` | `fig_gate_first.png` | §"Gate attribution and interactions" (182 vs 8192); cost table (SVQX rows) |
| `06_gate_interactions.ipynb` | `fig_gate_int.png` | Prop. *Interaction exactness*; §"Gate attribution and interactions" (78 pairs, $M=6$) |
| `07_qlime_ranking.ipynb` | `fig_qlime.png` | §"Gate attribution and interactions" (Q-LIME ranking sentence); cost table (Q-LIME row) |
| `08_benchmark_breast_cancer.ipynb` | `fig_benchmark.png` | §Experiments (stronger, non-toy benchmark) |

The composite paper figure is the three panels `fig_feature.png` (a), `fig_cost.png` (b),
`fig_shots.png` (c). `results/*.json` carries every quantitative claim used in the abstract,
Section 5, and the cost table.
