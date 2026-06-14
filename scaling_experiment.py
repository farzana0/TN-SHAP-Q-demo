"""Asymptotic cost of exact gate Shapley attribution: TN-SHAP-Q vs full enumeration.

For a gate game with m gate players, TN-SHAP-Q (Owen integral, M = ceil(m/2) nodes)
needs 2 * m * ceil(m/2) extension evaluations -- O(m^2) -- while exact coalition
enumeration needs 2^m value evaluations -- O(2^m). The m=13 QAOA example used in the
paper sits at 182 vs 8192 evaluations.

Produces figures/scaling_comparison.pdf (and .png).
"""
import math
import os
import numpy as np
import matplotlib.pyplot as plt

# Styling consistent with the other demo figures
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
})
BLUE, RED = "#3b6fb0", "#c0392b"

M_GRID = [4, 6, 8, 10, 12, 14, 16, 18, 20]


def tnshapq_cost(m):
    """Owen extension evaluations: 2 * m * ceil(m/2)."""
    return 2 * m * math.ceil(m / 2)


def enum_cost(m):
    """Exact coalition enumeration: 2^m value evaluations."""
    return 2 ** m


tn = [tnshapq_cost(m) for m in M_GRID]
en = [enum_cost(m) for m in M_GRID]

# m = 13 QAOA example actually used in the paper
M_STAR = 13
tn_star, en_star = tnshapq_cost(M_STAR), enum_cost(M_STAR)
assert (tn_star, en_star) == (182, 8192), (tn_star, en_star)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.semilogy(M_GRID, en, "s-", color=RED, label=r"Exact enum. $O(2^m)$")
ax.semilogy(M_GRID, tn, "o-", color=BLUE, label=r"TN-SHAP-Q $O(m^2)$")

# m = 13 QAOA example: vertical marker + actual values
ax.axvline(M_STAR, ls="--", color="0.5", lw=1.0)
ax.text(M_STAR + 0.15, en[0] * 1.5, "QAOA example", rotation=90,
        va="bottom", ha="left", color="0.4", fontsize=9)
ax.scatter([M_STAR], [en_star], marker="*", s=220, color=RED, zorder=6,
           edgecolor="white", linewidth=0.5)
ax.scatter([M_STAR], [tn_star], marker="*", s=220, color=BLUE, zorder=6,
           edgecolor="white", linewidth=0.5)
ax.annotate(f"{en_star}", (M_STAR, en_star), textcoords="offset points",
            xytext=(6, 4), color=RED, fontsize=9)
ax.annotate(f"{tn_star}", (M_STAR, tn_star), textcoords="offset points",
            xytext=(6, -12), color=BLUE, fontsize=9)

ax.set_xlabel("number of gate players $m$")
ax.set_ylabel("evaluations required")
ax.set_title("Exact gate Shapley: cost vs circuit size")
ax.set_xticks(M_GRID)
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/scaling_comparison.pdf", bbox_inches="tight")
fig.savefig("figures/scaling_comparison.png", dpi=150, bbox_inches="tight")
print("wrote figures/scaling_comparison.pdf")
for m, a, b in zip(M_GRID, tn, en):
    print(f"  m={m:2d}: TN-SHAP-Q={a:5d}  enum={b}")
print(f"  m={M_STAR} (QAOA): TN-SHAP-Q={tn_star}  enum={en_star}")
