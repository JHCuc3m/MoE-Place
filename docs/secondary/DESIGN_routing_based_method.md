# DESIGN.md - Technical Design Documentation

This document provides detailed technical explanations for the algorithms and metrics used in MoE-Place.

## Table of Contents

1. [Pruning Metrics](#pruning-metrics)
2. [Metrics Usefulness Analysis](#metrics-usefulness-analysis)
3. [CAEP Algorithm](#caep-algorithm)
4. [Implementation Notes](#implementation-notes)

---

## Pruning Metrics

We compute several metrics to identify which experts can be pruned with minimal performance impact. Each metric captures a different aspect of expert importance.

### 1. Utilization (Selection Frequency)

**Formula:**
```
Utilization(i) = count(expert_i selected) / total_selections
```

**Intuition:** How often is each expert selected by the router? Experts that are rarely selected may be less critical to model performance.

**For pruning:**
- **Low utilization** → Expert is rarely needed → Candidate for pruning
- **High utilization** → Expert handles many tokens → Likely important

**Limitations:**
- Doesn't account for *when* the expert is used (rare but critical cases)
- Load imbalance in MoE training can skew this metric

---

### 2. Redundancy Score

**Formula:**
```
Redundancy(i) = max_j [ P(expert j fires | expert i fires) ]
              = max_j [ C[i,j] / C[i,i] ]
```

Where:
- `C[i,j]` = co-activation count (times experts i and j fire together)
- `C[i,i]` = self-activation count (total times expert i fires)

**Intuition:** If expert `i` always fires together with expert `j`, then `P(j|i) ≈ 1`. This means expert `i` is "covered" by expert `j` — whenever you need `i`, you also have `j`. Removing `i` might be safe because `j` is always there to handle similar inputs.

**Example:**
```
If expert 0 fires 1000 times, and 950 of those times expert 2 also fires:
  Redundancy(0) = 950/1000 = 0.95  → Very redundant

If expert 3 fires 1000 times, but max co-activation with any other is 400:
  Redundancy(3) = 400/1000 = 0.40  → Less redundant, more unique
```

**For pruning:**
- **High redundancy** → Expert is always accompanied by another → Safe to prune
- **Low redundancy** → Expert activates independently → Likely unique function

---

### 3. Degree Centrality

**Formula:**
```
Degree(i) = Σ_j w[i,j] / (N-1)
```

Where `w[i,j]` is the edge weight (co-activation count) between experts i and j, normalized.

**Intuition:** How "connected" is this expert to others in the co-activation graph?
- High degree = frequently co-activates with many different experts
- Low degree = isolated, rarely co-activates with others

**For pruning:**
- **Low degree centrality** → Expert is peripheral → Possibly unimportant
- **High degree centrality** → Expert is a "hub" → Likely important

**Graph interpretation:**
```
    E0 -------- E1
    |  \      / |
    |    \  /   |
    |     \/    |
    |     /\    |
    |   /    \  |
    E3 -------- E2

High degree: E0, E2 (connected to everyone)
Low degree: E1, E3 (fewer connections)
```

---

### 4. Eigenvector Centrality

**Formula:**
```
x_i = (1/λ) Σ_j A[i,j] * x_j
```

Where:
- `A` is the adjacency (co-activation) matrix
- `λ` is the largest eigenvalue
- `x` is the eigenvector (centrality scores)

This is solved iteratively: your centrality depends on the centrality of your neighbors.

**Intuition:** It's not just about how many connections you have, but whether you're connected to *important* experts. An expert is important if it co-activates with other important experts.

**Analogy:** PageRank for web pages. A page is important if it's linked by other important pages.

**For pruning:**
- **Low eigenvector centrality** → Connected mainly to unimportant experts → Prunable
- **High eigenvector centrality** → Connected to important experts → Critical

**Difference from degree centrality:**
```
Scenario: Expert A connects to experts B, C, D
          Expert E connects only to expert F

If B, C, D are all unimportant (low centrality themselves):
  - Degree(A) = high (3 connections)
  - Eigenvector(A) = low (connected to unimportant nodes)

If F is very important:
  - Degree(E) = low (1 connection)
  - Eigenvector(E) = moderate (connected to important node)
```

---

### 5. Betweenness Centrality

**Formula:**
```
Betweenness(i) = Σ_{s≠i≠t} σ_st(i) / σ_st
```

Where:
- `σ_st` = number of shortest paths from s to t
- `σ_st(i)` = number of those paths passing through i

**Intuition:** How often does this expert lie on the "path" between other expert pairs? Experts with high betweenness are bridges connecting different clusters.

**For pruning:**
- **Low betweenness** → Not a bridge → Removing won't disconnect the graph
- **High betweenness** → Critical bridge → Removing may break important patterns

---

### 6. Clustering Coefficient

**Formula:**
```
Clustering(i) = (actual triangles through i) / (possible triangles through i)
```

**Intuition:** Are the expert's neighbors also connected to each other? High clustering means the expert is part of a tight-knit group.

**For pruning:**
- **High clustering** → Expert is in a dense cluster → Others in cluster may cover its function
- **Low clustering** → Expert connects disparate groups → May have unique bridging role

---

### 7. Structural Score (Combined)

**Formula:**
```
Score(i) = α * (1 - utilization_norm)
         + β * redundancy_norm
         + γ * (1 - centrality_norm)

Default weights: α=0.3, β=0.4, γ=0.3
```

All values are normalized to [0, 1] before combining.

**Intuition:** Combines three complementary signals:
1. **Low utilization** (α=0.3): Rarely selected → probably not critical
2. **High redundancy** (β=0.4): Covered by others → safe to remove
3. **Low centrality** (γ=0.3): Not well-connected → peripheral

**Higher structural score = More suitable for pruning**

**Why these weights?**
- Redundancy gets highest weight (0.4) because it directly measures functional overlap
- Utilization and centrality split the remainder equally
- Weights can be tuned via ablation studies

---

## Metrics Usefulness Analysis

The effectiveness of each metric depends on the MoE architecture, particularly the **number of experts per layer** and **top-k routing**.

### Small Expert Count (4-8 experts, e.g., TinyMixtral)

| Metric | Usefulness | Reasoning |
|--------|------------|-----------|
| **Utilization** | HIGH | Clear signal - some experts may be rarely used |
| **Redundancy** | HIGH | With top-2 routing, pairs form naturally; redundancy is meaningful |
| **Degree Centrality** | LOW | All experts likely connect to all others; not discriminative |
| **Eigenvector Centrality** | MODERATE | May still identify hub vs peripheral experts |
| **Betweenness Centrality** | LOW | With 4 nodes fully connected, all paths are short |
| **Structural Score** | HIGH | Combines signals; robust even with few experts |

**Recommended approach:** Focus on **utilization + redundancy**. Graph metrics have limited discriminative power with few nodes.

### Medium Expert Count (16-32 experts)

| Metric | Usefulness | Reasoning |
|--------|------------|-----------|
| **Utilization** | HIGH | Load imbalance becomes more pronounced |
| **Redundancy** | HIGH | More pair combinations; redundancy patterns emerge |
| **Degree Centrality** | MODERATE | Some experts may specialize and have fewer connections |
| **Eigenvector Centrality** | HIGH | Hub-and-spoke patterns more likely |
| **Betweenness Centrality** | MODERATE | Bridges between expert clusters may exist |
| **Structural Score** | HIGH | All components contribute meaningfully |

**Recommended approach:** Use **full structural score** with all metrics.

### Large Expert Count (64+ experts, e.g., DeepSeek-MoE, Mixtral 8x7B)

| Metric | Usefulness | Reasoning |
|--------|------------|-----------|
| **Utilization** | HIGH | Significant load imbalance expected |
| **Redundancy** | MODERATE | Many experts means sparser co-activation |
| **Degree Centrality** | HIGH | Clear differentiation between specialists and generalists |
| **Eigenvector Centrality** | HIGH | Expert hierarchies emerge |
| **Betweenness Centrality** | HIGH | Critical bridges between expert communities |
| **Clustering Coefficient** | HIGH | Expert clusters/communities form |
| **Structural Score** | HIGH | All components highly informative |

**Recommended approach:** Use **full structural score** and consider **community detection** to identify expert clusters.

### Impact of Top-K Routing

| Top-K | Effect on Metrics |
|-------|-------------------|
| **Top-1** | Utilization is primary signal; no co-activation (redundancy=0) |
| **Top-2** | Co-activation meaningful; redundancy is key metric |
| **Top-4+** | Dense co-activation; degree centrality less discriminative |

---

## CAEP Algorithm

**Coactivation-Aware Expert Pruning (CAEP)** uses the structural score to systematically remove experts.

### Algorithm

```
Input:
  - Model M with N experts per layer, L layers
  - Target: remove K experts total (or K per layer)
  - Co-activation statistics from calibration data

Procedure:
1. Compute structural score for all experts in all layers
2. Rank experts by score (descending = most prunable first)
3. For i = 1 to K:
   a. Select expert with highest score
   b. Disable/remove expert from model
   c. (Optional) Re-route tokens to remaining experts
4. (Optional) Fine-tune remaining experts
5. Evaluate pruned model

Output: Pruned model M' with (N-K) experts
```

### Variants

**CAEP-Fast:** Skip sensitivity analysis, use only structural metrics
- Pro: Very fast, no additional forward passes
- Con: May miss experts that are rare but critical

**CAEP-Merge:** Instead of removing, merge co-activated experts
```
merged_expert.weight = (expert_a.weight + expert_b.weight) / 2
```
- Pro: Preserves capacity better
- Con: More complex, merged expert may not work well

**CAEP-Reroute:** After pruning, update router to redistribute tokens
```
# Tokens that would go to pruned expert → most co-activated remaining expert
reroute_target[pruned_i] = argmax_j(C[pruned_i, j]) for j in remaining
```
- Pro: No tokens are dropped
- Con: Remaining experts may be overloaded

---

## Implementation Notes

### File Structure

```
src/pruning/
  metrics.py      # All metric computations
  algorithms.py   # CAEP and variants (TODO)

scripts/
  compute_pruning_metrics.py  # Compute metrics from co-activation stats
```

### Computational Complexity

| Metric | Complexity | Notes |
|--------|------------|-------|
| Utilization | O(N) | Simple counting |
| Redundancy | O(N²) | Requires full co-activation matrix |
| Degree Centrality | O(N²) | Sum over adjacency matrix |
| Eigenvector Centrality | O(N³) | Eigenvalue computation |
| Betweenness Centrality | O(N³) | All-pairs shortest paths |

For small N (4-64 experts), all metrics are fast. For very large N (1000+ experts), consider approximate algorithms.

### Numerical Stability

- Co-activation matrix may have zeros → add small epsilon before division
- Eigenvector computation may fail on disconnected graphs → use numpy's robust solver
- Normalize all metrics to [0,1] before combining into structural score
