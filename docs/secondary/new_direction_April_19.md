# Next Research Directions for MoE Compression (Project Pivot)

## Overview

Based on our findings, the key limitation of current approaches is:

> **All existing metrics (including ours) measure selection patterns, not actual contribution to model performance.**

This leads to two critical failures:

* Inability to identify **harmful experts**
* Poor correlation with true pruning sensitivity

We propose three next directions, with **Direction A as the primary focus**.

---

# 🔥 Direction A (PRIMARY): Contribution-Aware Expert Evaluation

## 🎯 Goal

Develop **metrics that approximate expert contribution to model output quality**, enabling pruning decisions without expensive sensitivity analysis.

---

## 🧠 Key Insight

> **Selection ≠ Contribution**

An expert can:

* Be frequently selected (high utilization)
* Be central in co-activation graph
* Participate in many trajectories

…and still be **harmful** (as shown by Expert 2).

---

## 🔧 Proposed Methods

### 1. Output Contribution Magnitude

Measure how much an expert contributes to the forward pass:

[
C_i = \mathbb{E}_x \left[ | G_i(x) \cdot E_i(x) |_2 \right]
]

* Captures **actual signal injected into model**
* Weighted by router importance

---

### 2. Signed Contribution (Detect Harmful Experts) ⭐

Measure whether the expert pushes representations in a helpful direction:

[
S_i = \mathbb{E}*x \left[ \langle G_i(x) E_i(x), h*{\text{residual}}(x) \rangle \right]
]

Interpretation:

* **Positive** → aligned with model’s direction → useful
* **Negative** → misaligned → potentially harmful

👉 This is the **most important metric** for your project.

---

### 3. Leave-One-Out Approximation (Cheap Sensitivity Proxy)

Approximate impact of removing expert (i):

[
\Delta y \approx - G_i(x) \cdot E_i(x)
]

Instead of recomputing forward pass:

* Estimate change directly from expert output

---

### 4. Gradient-Based Contribution (Optional Extension)

Measure gradient flow:

[
G_i^{grad} = \mathbb{E}*x \left[ | \nabla*{E_i} \mathcal{L}(x) | \right]
]

* High gradient → expert influences loss strongly
* Can detect both helpful and harmful roles

---

## 🧪 Experiments

### (1) Correlation Study (Core Result)

Compute correlation with true sensitivity:

| Metric                    | Spearman ρ | Expected |
| ------------------------- | ---------- | -------- |
| Utilization               | Low        | ❌        |
| Redundancy                | Negative   | ❌        |
| Centrality                | Low        | ❌        |
| Contribution (C_i)        | Moderate   | ✅        |
| Signed Contribution (S_i) | High       | 🔥       |

---

### (2) Harmful Expert Detection

* Identify experts with **negative S_i**
* Compare with sensitivity results

Expected:

* Expert 2 → strongly negative score

---

### (3) Pruning Evaluation

Compare:

* Random
* Utilization-based
* Structural score
* **Contribution-based (yours)**

Metric:

* Perplexity change

---

## 💡 Expected Contribution

* First method to **systematically identify harmful experts**
* Replace brute-force sensitivity with **cheap proxy**
* Demonstrate **functional metrics > structural metrics**

---

# 🚀 Direction B: Trajectory-Aware + Contribution Hybrid

## 🎯 Goal

Combine your work with **MoE Pathfinder-style global pruning**

---

## 🧠 Motivation

Pathfinder improves over local metrics by using:

* Cross-layer trajectories
* Global path importance

BUT:

> It does not explicitly filter harmful experts

---

## 🔧 Proposed Idea

Define path score:

[
\text{Score(path)} =
\underbrace{\text{Path importance}}*{\text{Pathfinder}}
+
\lambda \cdot
\underbrace{\text{Contribution quality}}*{\text{Direction A}}
]

---

## 🔥 Key Innovation

> Only keep paths that are both:

* Frequently used
* **Functionally useful**

---

## 🧪 Experiments

* Pathfinder baseline
* Pathfinder + contribution filtering

Compare:

* Performance
* Number of experts kept

---

## 💡 Contribution

* Bridges **structure (Pathfinder)** and **function (your work)**
* Prevents retaining harmful experts in strong paths

---

# 🧪 Direction C: Understanding Harmful Experts

## 🎯 Goal

Explain **why harmful experts exist**

---

## 🔥 Key Finding to Investigate

From your results:

* Expert 2 is harmful in ALL layers

This is unusual and highly publishable.

---

## 🔍 Hypotheses

### 1. Misaligned Representation

* Output direction opposes useful features

### 2. Over-amplification

* Produces large activations → destabilizes model

### 3. Router Miscalibration

* Router over-selects bad expert

### 4. Training Artifact

* Expert learned spurious correlations

---

## 🧪 Experiments

* Activation norm analysis
* Cosine similarity with residual stream
* Routing probability distribution
* Layer-wise behavior comparison

---

## 💡 Contribution

* First detailed analysis of **harmful experts in MoE**
* Explains failure of routing-based pruning

---

# 🎯 Recommended Direction

## Focus:

👉 **Direction A (Contribution-Aware Metrics)**

## Why:

* Directly addresses your main finding
* Strong theoretical + empirical contribution
* Independent of model scale
* Easy to validate with current setup

---

## Suggested Paper Framing

> **“From Selection to Contribution: Identifying Harmful Experts in Mixture-of-Experts Models”**

---

# 💬 Final Takeaway

> MoE pruning should not be based on **who gets selected**, but on **who actually helps the model**.

---

## Next Steps Checklist

* [ ] Implement contribution metrics (C_i, S_i)
* [ ] Run correlation with sensitivity
* [ ] Validate harmful expert detection
* [ ] Compare pruning methods
* [ ] (Optional) Extend to trajectory-based pruning
