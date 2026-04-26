"""
Scatter plot: structural score vs. actual sensitivity for all 48 experts.
Saves to docs/MoE-Prune-Overleaf/iclr2026/plots/structural_vs_sensitivity.png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
METRICS_PATH = ROOT / "experiments/baseline/pruning_metrics.json"
SENSITIVITY_PATH = ROOT / "experiments/pruning/sensitivity_results.json"
OUT_PATH = ROOT / "docs/MoE-Prune-Overleaf/iclr2026/plots/structural_vs_sensitivity.png"

with open(METRICS_PATH) as f:
    metrics = json.load(f)
with open(SENSITIVITY_PATH) as f:
    sens_data = json.load(f)

sensitivity = sens_data["sensitivity"]
num_layers = metrics["layers"]

scores, sensitivities, expert_ids = [], [], []

for layer_str, layer_data in num_layers.items():
    layer = int(layer_str)
    for expert in range(layer_data["num_experts"]):
        score = layer_data["structural_score"][expert]
        key = f"layer_{layer}_expert_{expert}"
        sens = sensitivity[key]
        scores.append(score)
        sensitivities.append(sens)
        expert_ids.append(expert)

scores = np.array(scores)
sensitivities = np.array(sensitivities)
expert_ids = np.array(expert_ids)

fig, ax = plt.subplots(figsize=(5.5, 4))

colors = {0: "#4878CF", 1: "#6ACC65", 2: "#D65F5F", 3: "#B47CC7"}
labels = {0: "Expert 0", 1: "Expert 1", 2: "Expert 2 (routing sink)", 3: "Expert 3"}
sizes = {0: 55, 1: 55, 2: 90, 3: 55}
markers = {0: "o", 1: "s", 2: "*", 3: "^"}

for exp in [0, 1, 3, 2]:  # draw Expert 2 last so it's on top
    mask = expert_ids == exp
    ax.scatter(
        scores[mask],
        sensitivities[mask],
        c=colors[exp],
        s=sizes[exp],
        marker=markers[exp],
        label=labels[exp],
        zorder=3 if exp == 2 else 2,
        edgecolors="white" if exp != 2 else "black",
        linewidths=0.5 if exp != 2 else 0.8,
        alpha=0.9,
    )

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

ax.set_xlabel("Structural Score (higher = prune first)", fontsize=11)
ax.set_ylabel("Sensitivity: $\\Delta$PPL when disabled", fontsize=11)
ax.set_title("Structural Score vs. Actual Sensitivity (all 48 experts)", fontsize=11)

ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

# Annotate the Expert 2 cluster
e2_mask = expert_ids == 2
ax.annotate(
    "Expert 2:\nharmful in all 12 layers",
    xy=(scores[e2_mask].mean(), sensitivities[e2_mask].mean()),
    xytext=(0.62, -50),
    fontsize=8,
    color="#D65F5F",
    arrowprops=dict(arrowstyle="->", color="#D65F5F", lw=1.0),
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
