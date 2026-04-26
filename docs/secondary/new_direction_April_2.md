# Expert Pruning Based on Co-Activation Patterns

## Overview

This document outlines a new research direction: **systematic expert pruning in Mixture-of-Experts (MoE) models using co-activation patterns**. The core idea is to leverage the expert collaboration structure to identify which experts can be removed with minimal performance degradation.

## Motivation

MoE models achieve parameter efficiency by activating only a subset of experts per token. However, they still incur:
- Memory overhead from storing all expert weights
- Latency from routing decisions and expert execution
- Deployment complexity on memory-constrained devices

**Key Insight**: Co-activation patterns reveal structural redundancy. If experts frequently activate together, they may encode overlapping information and be candidates for consolidation or removal.

## Theoretical Foundation

### Co-Activation Matrix

Given a dataset and trained MoE model, the co-activation matrix $C \in \mathbb{R}^{N \times N}$ where $N$ is the number of experts:

$$C_{ij} = \frac{|\{t : \text{expert } i \text{ and } j \text{ both selected for token } t\}|}{|\{t : \text{expert } i \text{ selected for token } t\}|}$$

This represents $P(\text{expert } j \text{ fires} | \text{expert } i \text{ fires})$.

### Graph Interpretation

The co-activation matrix defines a weighted graph where:
- Nodes = experts
- Edge weights = co-activation frequency

Graph-theoretic metrics then quantify expert importance and redundancy.

## Proposed Metrics

### 1. Expert Utilization (Baseline)

```python
utilization[i] = count(expert_i_selected) / total_tokens
```

- **Interpretation**: How often each expert is used
- **Pruning strategy**: Remove least-utilized experts
- **Limitation**: Ignores structural relationships

### 2. Redundancy Score

```python
redundancy[i] = max_j(P(j fires | i fires))
```

- **Interpretation**: High score means expert is always accompanied by another
- **Pruning strategy**: Remove highly redundant experts (covered by others)

### 3. Graph Centrality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Degree Centrality | $\frac{\sum_j C_{ij}}{N-1}$ | Overall connectivity |
| Betweenness Centrality | Fraction of shortest paths through node | Bridge between expert clusters |
| Eigenvector Centrality | Principal eigenvector of $C$ | Importance from important neighbors |
| PageRank | Iterative importance propagation | Robust importance measure |

- **Pruning strategy**: Remove low-centrality (peripheral) experts

### 4. Clustering Coefficient

```python
clustering[i] = (actual triangles through i) / (possible triangles through i)
```

- **Interpretation**: Experts in tight clusters may be internally redundant
- **Pruning strategy**: Keep one representative per cluster

### 5. Output Contribution

```python
contribution[i] = sum(routing_weights for tokens routed to expert i)
```

- **Interpretation**: Weighted importance based on router confidence
- **Pruning strategy**: Remove low-contribution experts

### 6. Sensitivity Analysis (Ground Truth)

```python
sensitivity[i] = performance(full_model) - performance(model_without_expert_i)
```

- **Interpretation**: Direct measurement of expert importance
- **Pruning strategy**: Remove least-sensitive experts
- **Cost**: Expensive (requires N forward passes on eval set)

## Proposed Algorithm

### Coactivation-Aware Expert Pruning (CAEP)

```
Input:
  - Trained MoE model M with N experts
  - Target number of experts K < N
  - Evaluation dataset D
  - Balance parameter α ∈ [0, 1]

Algorithm:
1. Collect routing statistics on D
2. Construct co-activation matrix C
3. Compute structural metrics:
   - centrality = eigenvector_centrality(C)
   - redundancy = redundancy_score(C)
4. Compute structural pruning score:
   - structural_score = (1 - centrality) * redundancy
5. (Optional) Compute sensitivity on subset of D
6. Combine scores:
   - prune_score = α * structural_score + (1-α) * (1 - sensitivity)
7. Iteratively prune N - K experts with highest prune_score
8. (Optional) Fine-tune remaining experts

Output: Pruned model M' with K experts
```

### Variants

**CAEP-Fast**: Skip step 5, use only structural metrics (α = 1)

**CAEP-Merge**: Instead of pruning, merge co-activated experts:
```python
merged_expert.weight = (expert_a.weight + expert_b.weight) / 2
```

**CAEP-Reroute**: After pruning, update router to redistribute tokens:
```python
# Tokens that would go to pruned expert → most co-activated remaining expert
reroute_map[pruned_i] = argmax_j(C[pruned_i, j]) for j in remaining
```

## Evaluation Plan

### Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Quality | Perplexity | Language modeling performance |
| Quality | Task accuracy | Downstream task performance |
| Efficiency | Latency | End-to-end inference time |
| Efficiency | Memory | Peak GPU memory usage |
| Efficiency | Throughput | Tokens per second |
| Balance | Routing entropy | Distribution of tokens across experts |
| Balance | Load factor | Max/mean tokens per expert |

### Baselines

1. **Random pruning**: Remove experts randomly
2. **Utilization-based**: Remove least-used experts
3. **Magnitude-based**: Remove experts with smallest weight norms
4. **Gradient-based**: Remove experts with smallest gradient norms

### Ablations

1. Effect of α (structural vs sensitivity weighting)
2. Layer-wise vs global pruning
3. Iterative vs one-shot pruning
4. With vs without fine-tuning after pruning
5. Different centrality metrics

## Research Questions

### Primary Questions

1. **Does co-activation structure predict pruning sensitivity?**
   - Hypothesis: Low-centrality experts can be pruned with less performance loss
   - Test: Correlation(centrality, sensitivity)

2. **How much pruning is achievable before significant degradation?**
   - Hypothesis: 20-30% experts removable with <5% perplexity increase
   - Test: Pruning curve (experts removed vs perplexity)

3. **Is merging better than pruning?**
   - Hypothesis: Merging preserves more capacity
   - Test: Compare merge vs prune at same compression ratio

### Secondary Questions

4. **Do different layers have different pruning tolerance?**
5. **Does the optimal pruning strategy vary by model size?**
6. **Can we predict prunable experts without running inference?**

## Implementation Plan

### Phase 1: Infrastructure (Week 1-2)
- [ ] Implement co-activation matrix collection
- [ ] Implement graph metric computations
- [ ] Implement expert disabling/removal mechanism
- [ ] Set up evaluation pipeline

### Phase 2: Metric Validation (Week 3-4)
- [ ] Compute all metrics on TinyMixtral
- [ ] Run sensitivity analysis (ground truth)
- [ ] Analyze correlation between structural metrics and sensitivity
- [ ] Identify best metric combination

### Phase 3: Pruning Experiments (Week 5-6)
- [ ] Implement CAEP algorithm
- [ ] Run pruning experiments at various compression ratios
- [ ] Compare against baselines
- [ ] Ablation studies

### Phase 4: Extensions (Week 7-8)
- [ ] Implement CAEP-Merge variant
- [ ] Implement CAEP-Reroute variant
- [ ] Scale experiments to larger models (if compute allows)
- [ ] Write up results

## Expected Contributions

1. **Novel pruning criterion**: First to use co-activation graph structure for expert pruning
2. **Efficient approximation**: Structural metrics as proxy for expensive sensitivity analysis
3. **Empirical analysis**: Comprehensive study of expert redundancy in MoE models
4. **Practical method**: Deployable technique for MoE model compression

## Related Work

- **Occult** (Luo et al., 2025): Co-activation for expert placement (inter-GPU)
- **Expert Choice Routing** (Zhou et al., 2022): Load balancing in MoE
- **Switch Transformer** (Fedus et al., 2022): Sparse MoE scaling
- **ST-MoE** (Zoph et al., 2022): Stable MoE training
- **Structured Pruning**: General neural network compression literature

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Structural metrics don't correlate with sensitivity | Medium | Fall back to sensitivity-based pruning |
| Pruning causes training instability | Low | Use gradual pruning with fine-tuning |
| Compute constraints limit experiments | Medium | Focus on smaller models, efficient evaluation |
| Results don't generalize across models | Medium | Test on multiple MoE architectures |

## Notes

- This direction pivots from the original intra-GPU placement focus
- Co-activation analysis infrastructure can be reused from original plan
- Pruning is more practically impactful than placement optimization
- Easier to demonstrate clear wins (memory/latency reduction is measurable)
