# Pruning Results Analysis

This document analyzes the results of expert pruning experiments on TinyMixtral-4x248M-MoE.

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Model | TinyMixtral-4x248M-MoE |
| Layers | 12 |
| Experts per layer | 4 (independent per layer) |
| Total experts | 48 |
| Routing | Top-2 |
| Calibration data | WikiText-2 train (128 samples, 19,903 tokens) |
| Evaluation data | WikiText-2 test (663,040 tokens) |
| Baseline Perplexity | 800.38 |

---

## Executive Summary

We conducted two types of experiments:
1. **Structural score-based pruning**: Used co-activation metrics to rank experts for pruning
2. **Sensitivity analysis**: Measured actual PPL impact of disabling each expert individually

**Key Findings:**
- Structural scores **do not reliably predict** actual pruning impact
- **Expert 2 is harmful** in ALL 12 layers (disabling it *improves* perplexity)
- **Edge layers (0, 11) are critical** - highest sensitivity to expert removal
- High co-activation redundancy does not guarantee safe pruning

---

## Part 1: Structural Score-Based Pruning

### Methodology

Experts were ranked using a structural score combining:
```
Score = 0.3 × (1 - utilization) + 0.4 × redundancy + 0.3 × (1 - centrality)
```

Higher score = more "prunable" according to the heuristic.

### Expert Selected: Layer 0, Expert 1

| Expert | Utilization | Redundancy | Centrality | Structural Score |
|--------|-------------|------------|------------|------------------|
| 0 | 37.4% | 43.2% | 0.643 | 0.012 (keep) |
| **1** | **22.3%** | **72.3%** | **0.486** | **0.789 (prune)** |
| 2 | 17.2% | 42.3% | 0.356 | 0.600 |
| 3 | 23.0% | 60.7% | 0.473 | 0.637 |

**Rationale:** Expert 1 had highest redundancy (72.3%), moderate-low utilization, and was not the central hub.

### Results

| Metric | Value |
|--------|-------|
| Baseline Perplexity | 800.38 |
| Pruned Perplexity | 890.15 |
| **PPL Increase** | **+89.77 (+11.2%)** |
| Experts Pruned | 1 / 48 (2.1%) |

**Observation:** Despite 72% redundancy, removing Expert 1 caused significant degradation. High co-activation does not mean the partner expert can compensate.

---

## Part 2: Sensitivity Analysis

### Methodology

For each of the 48 experts, we:
1. Disabled that single expert
2. Ran full perplexity evaluation on WikiText-2 test set
3. Computed: `Sensitivity = PPL(disabled) - PPL(baseline)`

**Interpretation:**
- Positive = expert is important (removing it hurts)
- Negative = expert is harmful (removing it helps)
- Total evaluations: 49 (1 baseline + 48 experts)

### Full Sensitivity Table

| Layer | Expert 0 | Expert 1 | Expert 2 | Expert 3 | Most Sensitive |
|-------|----------|----------|----------|----------|----------------|
| 0 | **+188.20** | +89.77 | -155.68 | +87.85 | Expert 0 |
| 1 | +31.65 | **+44.26** | -79.13 | +38.52 | Expert 1 |
| 2 | +18.06 | +41.05 | -80.89 | **+47.87** | Expert 3 |
| 3 | +21.99 | **+67.20** | -110.63 | +60.22 | Expert 1 |
| 4 | +43.62 | **+115.04** | -152.45 | +50.00 | Expert 1 |
| 5 | +36.07 | **+101.31** | -162.20 | +60.80 | Expert 1 |
| 6 | +55.56 | +58.33 | -156.94 | **+91.69** | Expert 3 |
| 7 | +36.59 | +105.27 | -162.96 | **+108.90** | Expert 3 |
| 8 | +42.71 | +43.22 | -94.33 | **+80.85** | Expert 3 |
| 9 | **+54.46** | +29.21 | -84.71 | +51.73 | Expert 0 |
| 10 | **+123.48** | +14.46 | -107.59 | +50.48 | Expert 0 |
| 11 | **+190.38** | -1.18 | -89.11 | +32.07 | Expert 0 |

### Key Finding 1: Expert 2 is Harmful Across ALL Layers

Expert 2 shows **negative sensitivity in all 12 layers** (range: -79 to -163 PPL).

| Layer | Expert 2 Sensitivity | Interpretation |
|-------|---------------------|----------------|
| 0 | -155.68 | Removing improves PPL by 155 |
| 5 | -162.20 | Removing improves PPL by 162 |
| 7 | -162.96 | Removing improves PPL by 163 |
| Avg | **-119.72** | Average improvement per layer |

**Possible explanations:**
1. Expert 2 learned spurious/noisy patterns during pretraining
2. Router systematically misroutes tokens to Expert 2
3. Expert 2's outputs interfere destructively with the residual stream

### Key Finding 2: Edge Layers are Critical

| Layer Region | Most Important | Max Sensitivity |
|--------------|----------------|-----------------|
| **Layer 0** (first) | Expert 0 | +188.20 |
| **Layer 11** (last) | Expert 0 | +190.38 |
| Middle (1-10) | Varies | +44 to +123 |

First and last MoE layers are most sensitive to perturbation. Expert 0 at these edges is critical.

### Key Finding 3: Sensitivity Patterns by Expert

| Expert | Avg Sensitivity | Role |
|--------|-----------------|------|
| Expert 0 | +62.6 | Critical at edges, moderate in middle |
| Expert 1 | +56.6 | Important in layers 3-7 |
| Expert 2 | **-119.7** | Harmful everywhere |
| Expert 3 | +64.3 | Important in layers 6-8 |

---

## Part 3: Structural Score vs Sensitivity Comparison

### The Core Problem

Structural scores failed to identify Expert 2 as the best pruning candidate.

| Expert | Structural Score Rank | Actual Best Action |
|--------|----------------------|-------------------|
| Layer 0, Expert 1 | #1 (prune first) | Keep (sens: +89.77) |
| Layer 0, Expert 2 | #25 (middle) | **Prune (sens: -155.68)** |
| Layer 0, Expert 0 | #37 (keep) | Keep (sens: +188.20) |

### Structural Score Rankings for Expert 2

Expert 2 was ranked in the **middle** across layers, never flagged as top pruning candidate:

| Layer | Expert 2 Rank | Structural Score | Actual Sensitivity |
|-------|---------------|------------------|-------------------|
| 0 | 25 | 0.600 | -155.68 |
| 4 | 41 | 0.390 | -152.45 |
| 7 | 44 | 0.166 | -162.96 |
| 9 | 46 | 0.000 | -84.71 |

**Conclusion:** Structural score based on co-activation patterns cannot identify harmful experts. Sensitivity measurement is essential.

### Why Does This Happen?

1. **Redundancy ≠ Replaceability**: High co-activation may indicate complementary (not redundant) computation
2. **Harmful experts may have normal statistics**: Expert 2 has typical utilization and redundancy but produces harmful outputs
3. **Layer position not captured**: Structural score ignores that layer 0/11 are more critical

---

## Pruning Recommendations

### Safe to Prune (Low/Negative Sensitivity)

| Priority | Expert | Layers | Sensitivity | Expected Impact |
|----------|--------|--------|-------------|-----------------|
| 1 | **Expert 2** | All (0-11) | -79 to -163 | **Improves PPL** |
| 2 | Expert 1 | 10, 11 | +14.46, -1.18 | Minimal |
| 3 | Expert 0 | 2 | +18.06 | Low impact |

### Do NOT Prune (High Sensitivity)

| Expert | Layers | Sensitivity | Risk |
|--------|--------|-------------|------|
| Expert 0 | 0, 10, 11 | +123 to +190 | Critical |
| Expert 1 | 3, 4, 5, 7 | +67 to +115 | High |
| Expert 3 | 6, 7, 8 | +81 to +109 | High |

### Recommended Next Experiment

Prune Expert 2 from all 12 layers:

```bash
python scripts/evaluate_pruning.py --prune_experts "0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:2"
```

**Expected outcome:** PPL decrease of ~80-160 points (model improves).

---

## Conclusions

1. **Expert masking works** - Successfully disabled experts and measured impact
2. **Structural scores are insufficient** - Co-activation metrics missed the harmful Expert 2 entirely
3. **Expert 2 is harmful** - Negative sensitivity in ALL 12 layers; should be pruned first
4. **Edge layers are critical** - Layers 0 and 11 have highest sensitivity (Expert 0: +188-190)
5. **Redundancy ≠ safe pruning** - 72% redundancy still caused 11% PPL increase
6. **Sensitivity analysis is essential** - Direct measurement required to identify true pruning candidates

---

## Future Work

1. **Validate Expert 2 pruning**: Run full evaluation with Expert 2 disabled globally
2. **Investigate Expert 2**: Why is it harmful? Analyze weight norms, activation patterns
3. **Improve structural metrics**: Add sensitivity-aware terms or layer-position weighting
4. **Test on other models**: Is the "harmful expert" pattern specific to TinyMixtral?
5. **Gradient-based alternatives**: Faster sensitivity estimation without full evaluation

---

## Data Files

| File | Description |
|------|-------------|
| `sensitivity_results.json` | Full sensitivity scores for all 48 experts |
| `pruning_results.json` | Results from --num_prune 1 experiment |
| `sensitivity_heatmap.png` | Visual heatmap of sensitivity scores |
| `../baseline/pruning_ranking.json` | Structural score rankings |
| `../baseline/pruning_metrics.json` | Per-expert structural metrics |

---

*Last updated: 2026-04-02*
