# FINDING.md - Key Research Findings

This document tracks significant findings from the MoE-Place research project.

## Summary

| # | Finding | Date | Impact |
|---|---------|------|--------|
| 1 | Expert 2 is harmful across all 12 layers | 2026-04-02 | Discovered prunable experts exist |
| 2 | Structural metrics fail (<17% agreement with sensitivity) | 2026-04-17 | Ruled out co-activation approach |
| 3 | Expert 2 is a "routing sink" (near-zero C_i, not negative S_i) | 2026-04-19 | C_i threshold identifies prune candidates |
| 4 | C_i threshold generalizes across domains | 2026-04-19 | Expert 2 is universal sink (code, math, scientific) |

---

## Finding 1: Expert 2 is Harmful Across All Layers

**Date:** 2026-04-02
**Source:** `experiments/pruning/sensitivity_results.json`

Expert 2 has **negative sensitivity in all 12 layers** (-79 to -163 PPL). Disabling Expert 2 improves model performance rather than degrading it.

| Layer | Expert 2 Sensitivity |
|-------|---------------------|
| 0 | -155.68 |
| 5 | -162.20 |
| 7 | -162.96 |

**Implication:** Some experts in pretrained MoE models are actively harmful and should be pruned.

---

## Finding 2: Structural Metrics Fail to Predict Pruning Sensitivity

**Date:** 2026-04-17
**Source:** `experiments/pruning/ablation/structural_ablation_results.json`

### Summary

All structural metrics (utilization, redundancy, centrality) fail to identify which experts are safe to prune. The combined structural score performs **worse** than random chance.

| Metric | Top-1 Agreement | Spearman ρ | Verdict |
|--------|-----------------|------------|---------|
| Utilization | 17% | 0.18 | Fails |
| Redundancy | 17% | **-0.35** | **Inverted** |
| Centrality | 17% | 0.23 | Fails |
| Structural Score | **8%** | -0.03 | **Worst** |

*Note: Random baseline for 4 experts = 25% top-1 agreement*

### Root Cause Analysis

**The fundamental flaw:** All structural metrics measure **selection patterns**, not **output quality**.

#### Why Utilization Fails

- **Assumption:** Rarely-used experts contribute less → safe to prune
- **Reality:** The router can be wrong. Expert 2 has high utilization in later layers (41% in layer 11) but is still harmful.
- **Conclusion:** High selection frequency ≠ high quality contribution

#### Why Redundancy is Inverted

- **Assumption:** If expert A always fires with expert B, then A is "covered" by B → redundant
- **Reality:** High co-activation means experts work **together** effectively. They are complementary, not substitutes.
- **Conclusion:** Co-activation signals collaboration, not redundancy

#### Why Centrality Fails

- **Assumption:** Peripheral experts in the co-activation graph are less critical
- **Reality:** Expert 2 has moderate-to-high centrality (0.54 → 0.67) in later layers, yet is harmful.
- **Conclusion:** Graph position ≠ output quality

### Summary Table

| Metric | Measures | Assumes | Reality |
|--------|----------|---------|---------|
| Utilization | How often selected | Router is optimal | Router can be miscalibrated |
| Redundancy | Co-firing frequency | Co-firing = substitutable | Co-firing = collaborative |
| Centrality | Graph connectivity | Central = important | Central ≠ useful output |

### Implication

Structural metrics capture *who is selected* but not *who is useful*. To identify prunable experts without sensitivity measurement, we need metrics that assess **output contribution**:

- Output magnitude/norm when expert fires
- Gradient flow through expert
- Activation similarity (true functional redundancy)
- Impact on hidden state trajectory

---

## Finding 3: Contribution Metrics Identify Expert 2 as "Zero Contributor"

**Date:** 2026-04-19
**Source:** `experiments/pruning/contribution_metrics.json`

### Summary

Contribution metrics (C_i, S_i) successfully identify Expert 2 as anomalous, but **not** in the way originally hypothesized.

**Original Hypothesis:** Harmful experts would have **negative S_i** (contribution opposes output direction).

**Actual Finding:** Expert 2 has **near-zero positive S_i** (~0.003 to 0.55) while other experts have S_i values of 23-315.

### Detailed Results

| Layer | Expert 2 Sensitivity | Expert 2 C_i | Expert 2 S_i | Other Experts C_i | Other Experts S_i |
|-------|---------------------|--------------|--------------|-------------------|-------------------|
| 0 | -155.68 | 0.004 | 0.003 | 0.56-0.64 | 53-56 |
| 5 | -162.20 | 0.024 | 0.037 | 0.71-0.74 | 58-70 |
| 7 | -162.96 | 0.029 | 0.101 | 0.78-0.86 | 88-103 |
| 11 | -89.11 | 0.066 | 0.55 | 0.81-0.96 | 87-315 |

**Key observation:** Expert 2's contribution magnitude is **30-150x smaller** than other experts.

### The "Routing Sink" Phenomenon

Expert 2 is not "actively harmful" (negative S_i). Instead, it acts as a **routing sink**:

1. **High Selection Rate:** Expert 2 receives 15K-37K tokens per layer (comparable to other experts)
2. **Near-Zero Contribution:** Despite being selected, Expert 2 contributes almost nothing to the output
3. **Negative Sensitivity:** Removing Expert 2 *improves* model performance

**Interpretation:** The router selects Expert 2, "spending" one of the top-k slots, but Expert 2 fails to contribute meaningfully. This wastes routing capacity that could go to useful experts.

### Why Does Removing a Zero-Contributor Help?

Possible mechanisms (not yet validated):

1. **Routing Reallocation:** With Expert 2 masked, its tokens go to experts 0, 1, 3 who actually contribute
2. **Interference Removal:** Even small contributions can interfere with the residual stream
3. **Load Balancing:** Expert 2's removal may improve utilization of remaining experts

### Contribution Metrics vs Sensitivity Correlation

For non-Expert-2 experts, S_i roughly correlates with importance:

**Layer 0:**
| Expert | S_i | Sensitivity |
|--------|-----|-------------|
| 0 | 55.2 | +188 (critical) |
| 1 | 55.7 | +90 (important) |
| 3 | 52.9 | +88 (important) |
| 2 | 0.003 | -156 (harmful) |

**Layer 11:**
| Expert | S_i | Sensitivity |
|--------|-----|-------------|
| 0 | 315.3 | +190 (critical) |
| 1 | 283.7 | -1.2 (neutral) |
| 3 | 86.6 | +32 (somewhat important) |
| 2 | 0.55 | -89 (harmful) |

Note: Layer 11 Expert 1 shows a discrepancy (high S_i but neutral sensitivity), requiring further investigation.

### Practical Implications

1. **Pruning Criterion:** Experts with C_i < threshold (e.g., < 0.1) are prune candidates
2. **No Need for Negative S_i:** The signal is "near-zero contribution," not "negative contribution"
3. **Simpler Detection:** C_i magnitude alone may suffice; S_i sign not required

### Open Questions

1. **Why is Expert 2 selected if it contributes nothing?** Router miscalibration? Training artifact?
2. **Is this pattern model-specific?** Need to test on other MoE models
3. **Can we predict routing sinks without forward pass?** Weight analysis?

---

## Finding 4: C_i Threshold Generalizes Across Domains

**Date:** 2026-04-19
**Source:** `experiments/pruning/domain_generalization/` (SLURM job 5017884)

### Summary

The C_i < 0.1 threshold for identifying routing sinks **generalizes universally** across all tested domains. Expert 2 is a routing sink not just for WikiText-2, but also for code, math, and scientific text.

### Cross-Domain Results

| Domain | Expert 2 Avg C_i | Min C_i | Max C_i | Sink Layers | Is Sink? |
|--------|-----------------|---------|---------|-------------|----------|
| WikiText-2 (general) | 0.0317 | 0.0044 | 0.0739 | 12/12 | **YES** |
| Code (CodeParrot) | 0.0202 | 0.0041 | 0.0261 | 12/12 | **YES** |
| Math (GSM8K) | 0.0255 | 0.0059 | 0.0349 | 12/12 | **YES** |
| Scientific (PubMed) | 0.0209 | 0.0044 | 0.0288 | 12/12 | **YES** |

**Key observation:** Expert 2's C_i is consistently 0.02-0.03 across all domains, while other experts have C_i > 0.5.

### All Experts Comparison

| Expert | WikiText-2 | Code | Math | Scientific |
|--------|------------|------|------|------------|
| 0 | 0.79 | 0.53 | 0.65 | 0.70 |
| 1 | 0.79 | 0.75 | 0.67 | 0.60 |
| **2** | **0.03** | **0.02** | **0.03** | **0.02** |
| 3 | 0.80 | 0.51 | 0.61 | 0.55 |

Expert 2 is **25-40x smaller** than other experts across all domains.

### Implications

1. **Universal Threshold:** C_i < 0.1 works across text, code, math, and scientific domains
2. **Domain-Independent:** Expert 2's "routing sink" behavior is intrinsic to the model, not data-dependent
3. **Robust Pruning Criterion:** Can compute C_i on any calibration data and apply threshold universally
4. **No Domain-Specific Tuning:** Single threshold works for all domains tested

### Significance

This finding validates that:
- The routing sink phenomenon is a **model property**, not a data artifact
- C_i-based pruning can use **any calibration data** without loss of generality
- The approach is ready for scaling to larger models (Mixtral-8x7B)

### Next Steps

1. Test on larger models (Mixtral-8x7B) to confirm routing sinks exist in production MoE models
2. Investigate why Expert 2 is selected despite contributing nothing (router weight analysis)

---

## Finding 5: [Reserved for Future Findings]

*To be added as research progresses.*
