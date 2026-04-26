# PLAN.md - MoE-Place Implementation Roadmap

## Project Direction

**Original Focus:** Hardware-aware expert placement for reducing intra-GPU communication

**Previous Focus:** Expert pruning based on co-activation patterns (structural metrics)

**Current Focus (Post-Ablation Pivot):** **Contribution-Aware Expert Evaluation** - measuring actual expert contribution to model output, not just selection patterns

> **Key Insight:** Selection ≠ Contribution. An expert can be frequently selected but still harmful.

See [direction.md](direction.md) for detailed methodology on the new direction.

---

## Phase 1: Environment Setup & Infrastructure

### 1.1 Development Environment
- [x] Set up Python environment (conda/venv) with PyTorch + CUDA
- [x] Install dependencies: transformers, accelerate, triton
- [x] Verify GPU access on PACE cluster
- [ ] Set up experiment logging (wandb or tensorboard)

**Resolved:**
- Q1.1: **Primary: H100 80GB HBM3** (also have A100, V100 available)
- Q1.2: **CUDA 12.4** (Pytorch 2.6.0+cu124)

### 1.2 Project Structure
- [x] Create directory structure

**Deliverable:** Working development environment

---

## Phase 2: Baseline MoE Implementation

### 2.1 Pretrained Model Support
- [x] Add TinyMixtral-4x248M-MoE loader (`src/models/pretrained_moe.py`)
- [x] Implement MixtralRoutingCollector for hook-based routing capture
- [x] Create pretrained routing collection script (`scripts/collect_pretrained_routing.py`)

**Resolved:**
- Q2.1: **Pretrained TinyMixtral-4x248M-MoE** for realistic routing patterns
- Q2.2: **Configuration:** 12 layers, 4 experts, top-2 routing

### 2.2 Routing Statistics Collection
- [x] Add hooks to capture routing decisions per layer
- [x] Store per-token expert assignments (`src/routing/statistics.py`)
- [x] Implement batch/dataset-level routing aggregation

**Deliverable:** Working MoE that can run inference and log routing decisions

---

## Phase 3: Expert Collaboration Analysis

### 3.1 Collaboration Matrix Construction
- [x] Implement co-activation counting (how often expert pairs activate together)
- [x] Build per-layer collaboration matrices (`RoutingStatisticsCollector.coactivation_counts`)
- [x] Implement matrix normalization/weighting schemes (`get_collaboration_matrix()`)

### 3.2 Collaboration Graph Visualization
- [x] Create visualization tools for collaboration patterns (`src/routing/visualization.py`)
- [x] Identify frequently co-activated expert clusters (`get_top_collaborations()`)
- [x] Analyze routing distribution (load balancing) (`plot_expert_load()`, `print_collaboration_summary()`)

**Resolved:**
- Q3.1: Using diverse text samples (code, science, literature, etc.)
- Q3.2: Default 500 samples (configurable via --num_samples)

**Deliverable:** Collaboration matrices and analysis tools

---

## Phase 3.5: Benchmark Selection & Data Pipeline

### 3.5.1 Co-activation Collection Datasets
- [x] Research standard benchmarks used in MoE pruning/analysis literature
- [x] Select calibration dataset for co-activation collection
- [ ] Determine required sample size for stable co-activation estimates
- [x] Implement data loading pipeline

**Decision: WikiText-2 for Quick Iteration**

For fast iteration and idea validation, using WikiText-2:
- **Calibration**: 128 samples from WikiText-2 train split
- **Evaluation**: Full WikiText-2 test set (~4K samples)

| Dataset | Role | Split | Samples |
|---------|------|-------|---------|
| WikiText-2 | Calibration | train | 128 (configurable) |
| WikiText-2 | Evaluation | test | Full (~4358 texts) |

### 3.5.2 Evaluation Benchmarks
- [x] Select perplexity evaluation datasets (held-out from calibration)
- [ ] (Optional) Select downstream task benchmarks
- [x] Implement evaluation data loaders

**Primary Evaluation: WikiText-2 Perplexity**
- Standard LM benchmark, small and fast
- Perplexity is continuous metric (good for measuring degradation)
- Easy to compare before/after pruning

### 3.5.3 Data Pipeline Design
- [x] Ensure train/calibration/eval splits are disjoint (train vs test)
- [x] Implement sliding window for perplexity (stride=256)
- [x] Add collate function with proper padding
- [x] Document data sources and preprocessing

**Implementation:**
- `src/data/benchmarks.py` - WikiText-2 loading
- `src/evaluation/perplexity.py` - Perplexity computation
- `scripts/analysis/benchmark_baseline.py` - Unified benchmark script

### 3.5.4 Domain Generalization Analysis - COMPLETED
- [x] Compute C_i across domains (WikiText, code, math, scientific)
- [x] Test if C_i < 0.1 threshold generalizes across domains - **YES, universally**
- [x] Identify domain-specific vs universal "routing sink" experts - **Expert 2 is universal sink**

**Result:** Expert 2 is a routing sink in ALL 4 domains (C_i 0.02-0.03). See [FINDING.md#finding-4](FINDING.md#finding-4-ci-threshold-generalizes-across-domains).

**Run domain generalization:**
```bash
# Quick test (C_i only)
python scripts/analysis/benchmark_contribution_domains.py --quick

# Full C_i benchmark (all domains)
python scripts/analysis/benchmark_contribution_domains.py

# Full analysis with sensitivity (recommended)
python scripts/analysis/domain_sensitivity_analysis.py --quick           # Quick test
python scripts/analysis/domain_sensitivity_analysis.py                   # Full analysis
python scripts/analysis/domain_sensitivity_analysis.py --skip_sensitivity  # C_i only

# SLURM jobs
sbatch scripts/jobs/submit_domain_generalization.sh   # C_i only (~2 hours)
sbatch scripts/jobs/submit_domain_sensitivity.sh      # Full sensitivity (~6 hours)
```

**Key Questions:**
- Q3.5.1: Does co-activation pattern depend heavily on calibration data domain? → **To test later**
- Q3.5.2: How many samples needed for stable co-activation matrix? → **Start with 128, ablate**
- Q3.5.3: Should we use same or different data for calibration vs evaluation? → **Different (train vs test)**

**Deliverable:** Data pipeline with calibration and evaluation datasets - **DONE**

**Run baseline benchmark:**
```bash
# Quick test (64 calibration samples, 50 eval batches)
python scripts/analysis/benchmark_baseline.py --quick

# Full baseline (128 calibration, full eval)
python scripts/analysis/benchmark_baseline.py
```

---

## Phase 4: Pruning Metrics Implementation

> **See [docs/DESIGN.md](docs/DESIGN.md) for detailed metric explanations and usefulness analysis.**

### 4.1 Basic Metrics
- [x] Expert utilization (selection frequency)
- [x] Output contribution (routing-weight-weighted importance) → Implemented as C_i in 4.5
- [ ] Expert weight magnitude (L2 norm of parameters) - may explain routing sink phenomenon

### 4.2 Co-activation Based Metrics
- [x] Redundancy score: `max_j(P(expert_j fires | expert_i fires))`
- [x] Conditional co-activation matrix (normalized)
- [x] Expert clustering coefficient

### 4.3 Graph Centrality Metrics
- [x] Implement graph construction from co-activation matrix
- [x] Degree centrality
- [x] Betweenness centrality
- [x] Eigenvector centrality
- [x] PageRank

### 4.4 Sensitivity Analysis (Ground Truth)
- [x] Implement expert disabling mechanism (`src/pruning/expert_masking.py`)
- [x] Implement per-expert sensitivity measurement (`compute_sensitivity()`)
- [ ] Cache sensitivity results for efficiency

### 4.5 Contribution Metrics (NEW - Primary Focus) 🔥

> **Motivation:** Structural metrics failed (ablation showed <17% agreement with sensitivity).
> We need metrics that measure **actual contribution**, not just selection patterns.

#### 4.5.1 Contribution Collector Infrastructure
- [x] Create `ExpertContributionCollector` class (`src/pruning/contribution_metrics.py`)
- [x] Hook MoE layers to capture:
  - `E_i(x)`: Raw expert output (before gating)
  - `G_i(x)`: Gate weight for expert i
  - `h_post(x)`: MoE layer output (post-residual)
  - Expert selection indices (which experts are active per token)

#### 4.5.2 Metric 1: Output Contribution Magnitude (C_i)
```
C_i = (1/N_i) * Σ_x [ |G_i(x) · E_i(x)|_2 / |h_post(x)|_2 ]
```
- [x] Implement magnitude computation
- [x] Aggregate over tokens where expert i is active (in top-k)
- [x] Normalize by post-MoE output norm

#### 4.5.3 Metric 2: Signed Contribution (S_i) - PRIMARY
```
S_i = (1/N_i) * Σ_x [ ⟨G_i(x) · E_i(x), h_post(x)⟩ / |h_post(x)|_2 ]
```
- [x] Implement signed contribution (cosine-like alignment)
- [x] **Interpretation:**
  - Positive S_i → aligned with output → helpful expert
  - Negative S_i → opposed to output → harmful expert
- [x] Validate: Expert 2 identified via **near-zero C_i** (not negative S_i as hypothesized)

#### 4.5.4 Combined Score
- [x] Use S_i as primary metric (NOT C_i alone)
- [x] C_i provides magnitude context for S_i

**Implementation:**
- `src/pruning/contribution_metrics.py` - ExpertContributionCollector, compute_contribution_scores()
- `scripts/analysis/compute_contribution_metrics.py` - Run contribution analysis

**Run contribution metrics:**
```bash
# Local run
python scripts/analysis/compute_contribution_metrics.py

# SLURM job (with sensitivity comparison)
sbatch scripts/jobs/submit_contribution.sh
```

### Metrics Summary (Updated)

| Metric | Type | Usefulness | Notes |
|--------|------|------------|-------|
| Utilization | Structural | LOW | Doesn't predict sensitivity |
| Redundancy | Structural | INVERTED | High redundancy = important, not prunable |
| Centrality | Structural | LOW | <17% agreement with sensitivity |
| **Magnitude (C_i)** | Functional | **HIGH** | Identifies "routing sinks" (C_i < 0.1) |
| **Signed Contribution (S_i)** | Functional | SUPPORTING | Not negative as expected; use C_i instead |

**Recommendation:** Use **C_i magnitude** as primary metric. Experts with C_i < 0.1 are prune candidates.

**Implementation:**
- `src/pruning/metrics.py` - All structural metrics
- `scripts/analysis/compute_pruning_metrics.py` - Compute and rank experts

**Run metrics computation:**
```bash
python scripts/analysis/compute_pruning_metrics.py --stats_path experiments/baseline/coactivation_stats.json
```

**Deliverable:** `src/pruning/metrics.py` with all metric implementations - **DONE**

---

## Phase 5: Pruning Algorithm Implementation

### 5.1 Expert Removal Infrastructure
- [x] Implement expert masking (soft removal, keeps weights) - `MixtralExpertMasker`
- [ ] Implement expert deletion (hard removal, reduces model size)
- [x] Implement routing modification (zeros routing weights for disabled experts)

**Implementation Details:**
- Router returns `(probs, weights, indices)` tuple
- Masking zeros out `weights` where `indices` matches disabled expert
- Weights are renormalized after masking
- Verified working via `test_masking()` in evaluate_pruning.py

### 5.2 Contribution-Aware Expert Pruning (CAP)
- [ ] Implement C_i-based scoring (prune experts with C_i < threshold)
- [ ] Implement global pruning (rank all experts across layers by C_i)
- [ ] Implement per-layer pruning (prune lowest C_i expert per layer)
- [ ] Validate: Pruning low-C_i experts should improve or maintain PPL

**Reframed** - Originally "CAEP" based on structural metrics (failed). Now based on contribution magnitude (C_i).

### 5.3 CAEP Variants
- [ ] CAEP-Fast: Structural metrics only (no sensitivity computation)
- [ ] CAEP-Merge: Merge co-activated experts instead of pruning
- [ ] CAEP-Reroute: Update router to redistribute pruned expert tokens

### 5.4 Baseline
The baseline for all experiments is the **unpruned model** (800.38 PPL on WikiText-2).
All pruning methods are evaluated by perplexity change relative to this baseline.

---

## Phase 6: Evaluation Infrastructure

### 6.1 Quality Metrics
- [x] Perplexity evaluation on validation set (`src/evaluation/perplexity.py`)
- [ ] (Optional) Downstream task accuracy

### 6.2 Efficiency Metrics
- [ ] Inference latency measurement
- [ ] Throughput (tokens/second)
- [ ] Memory usage (peak GPU memory)
- [ ] Model size (parameter count, disk size)

### 6.3 Balance Metrics
- [ ] Routing entropy (distribution across remaining experts)
- [ ] Load factor (max/mean tokens per expert)
- [ ] Expert utilization after pruning

### 6.4 Profiling Integration
- [ ] PyTorch Profiler setup
- [ ] NVIDIA Nsight Compute scripts
- [ ] Automated profiling wrapper

**Deliverable:** `src/evaluation/` with comprehensive evaluation pipeline

---

## Phase 7: Experiments

### 7.1 Metric Validation
- [x] Compute all metrics on TinyMixtral
- [x] Run full sensitivity analysis (ground truth) - `scripts/analysis/evaluate_pruning.py --sensitivity`
- [x] Correlation analysis: structural metrics vs sensitivity
- [x] Structural metrics ablation - **FAILED** (all metrics <17% agreement)
- [x] **Contribution metrics validation** - COMPLETED

**See [FINDING.md](FINDING.md) for detailed analysis of all findings.**

### 7.1.5 Contribution Metrics Validation - COMPLETED
- [x] Compute contribution metrics (C_i, S_i) for all experts
- [x] **Expert 2 detection:** C_i identifies Expert 2 as near-zero contributor (C_i < 0.07 vs others 0.5-0.96)
- [x] **Hypothesis revised:** S_i is NOT negative; Expert 2 is a "routing sink" with near-zero contribution
- [x] **Practical criterion:** C_i < 0.1 threshold identifies prune candidates

**Result:** Contribution metrics successfully identify Expert 2 as anomalous. See [FINDING.md#finding-3](FINDING.md#finding-3-contribution-metrics-identify-expert-2-as-zero-contributor) for details.

### 7.1.6 Cross-Domain Sensitivity Analysis - COMPLETED
- [x] For each domain (WikiText-2, code/CodeParrot, math/GSM8K, scientific/PubMed):
  - Compute C_i contribution metrics on calibration data
  - Run per-expert sensitivity analysis on evaluation data
  - Compute Spearman correlation between C_i and sensitivity per domain
- [x] **Result:** C_i < 0.1 identifies Expert 2 in every domain; Expert 2 avg C_i 0.020–0.032 across domains vs 0.51+ for others

**Script:** `scripts/analysis/domain_sensitivity_analysis.py`
```bash
python scripts/analysis/domain_sensitivity_analysis.py --quick           # Quick test
python scripts/analysis/domain_sensitivity_analysis.py                   # Full (all domains)
python scripts/analysis/domain_sensitivity_analysis.py --skip_sensitivity  # C_i only
sbatch scripts/jobs/submit_domain_sensitivity.sh                         # SLURM (~6 hours)
```
**Output:** `experiments/pruning/domain_generalization/`

### 7.2 Pruning Experiments
- [ ] Pruning curves: experts removed vs perplexity
- [ ] Layer-wise pruning analysis (which layers tolerate pruning?)

### 7.3 Ablation Studies
- [x] **Structural score component ablation:** DONE
  - Utilization-only ranking vs sensitivity ground truth
  - Redundancy-only ranking vs sensitivity ground truth
  - Centrality-only ranking vs sensitivity ground truth
- [ ] Effect of alpha/beta/gamma weighting in combined score
- [ ] Different centrality metrics comparison (degree, betweenness, eigenvector, PageRank)

**Ablation Study Findings** (see `experiments/pruning/ablation/structural_ablation_results.json`):
1. **All structural metrics fail** - No metric achieves >17% top-1 agreement with sensitivity (random = 25%)
2. **Redundancy is INVERTED** - Negative correlation (ρ = -0.35, p = 0.015); high redundancy predicts *important* experts, not prunable ones
3. **Combined structural score is WORST** - Only 8% top-1 agreement, worse than any individual metric
4. **Expert 2 invisible to all metrics** - Despite being harmful in all 12 layers, no metric consistently ranks it #1 for pruning

**Conclusion:** Structural metrics (utilization, redundancy, centrality) are fundamentally insufficient for pruning decisions. Sensitivity measurement remains necessary.

### 7.4 Variant Experiments
- [ ] CAEP-Merge vs CAEP-Prune comparison
- [ ] CAEP-Reroute effectiveness
- [ ] Combined approach (merge some, prune others)

### 7.5 Model Scaling Experiments 🔥

Test if routing sink phenomenon generalizes to larger/different MoE architectures.

#### 7.5.1 DeepSeek-MoE-16B - IN PROGRESS
- [x] Add model support to `src/models/pretrained_moe.py`
- [x] Update `ExpertContributionCollector` for 64 routed + 2 shared experts
- [x] Handle DeepSeek's separate `gate_proj/up_proj/down_proj` (vs Mixtral's fused)
- [x] Create SLURM job: `scripts/jobs/submit_deepseek.sh`
- [ ] Run contribution analysis and identify routing sinks
- [ ] Compare shared experts vs routed experts

**DeepSeek Architecture:**
| Aspect | Value |
|--------|-------|
| Layers | 28 (first is dense) |
| Routed Experts | 64 per layer |
| Shared Experts | 2 per layer (always active) |
| Top-K | 6 (routed) |
| Total Params | 16.4B |
| Active Params | 2.8B |

**Run:** `sbatch scripts/jobs/submit_deepseek.sh`

#### 7.5.2 Mixtral-8x7B (Future)
- [ ] Memory optimization for H100 (INT8 or gradient checkpointing)
- [ ] Same architecture as TinyMixtral (8 experts, top-2) - direct comparison
- [ ] 46x scale from TinyMixtral

#### 7.5.3 Qwen1.5-MoE-A2.7B (Future)
- [ ] 60 experts, top-4 routing
- [ ] Different routing mechanism

**Deliverable:** Experimental results for paper

---

## Phase 8: Analysis & Paper

### 8.1 Result Analysis
- [ ] Statistical significance testing
- [ ] Identify key findings
- [ ] Understand failure cases

### 8.2 Visualization
- [ ] Pruning curves (experts removed vs perplexity)
- [ ] C_i heatmap (layers × experts, highlighting routing sinks)
- [ ] C_i vs sensitivity scatter plot (validate correlation)
- [ ] Per-layer contribution distribution (bar charts)

**Reframed** - Focus on contribution metrics, not co-activation graphs.

### 8.3 Paper Writing
- [x] Mid-report completed (`docs/MoE-Place-Overleaf/iclr2026/mid_report.tex`)
- [ ] Update methodology section (final report)
- [ ] Write results section (final report)
- [x] Create figures and tables (sensitivity heatmap, co-activation matrix)

**Deliverable:** Paper draft with experimental results

---

## Priority Order

**Completed:**
1. Phase 1 (Environment) - DONE
2. Phase 2 (Baseline MoE) - DONE
3. Phase 3 (Collaboration Analysis) - DONE
4. Phase 3.5 (Benchmark Selection) - DONE
5. Phase 4.1-4.4 (Structural Pruning Metrics) - DONE
6. Phase 5.1 (Expert Masking) - DONE
7. Phase 6.1 (Quality Metrics - Perplexity) - DONE
8. Phase 7.1 (Metric Validation + Sensitivity Analysis) - DONE
9. Phase 8.3 (Mid-report) - DONE
10. Phase 7.3 (Structural Score Ablation Study) - DONE → **FAILED: structural metrics insufficient**

**COMPLETED:**
11. ~~**Phase 4.5 (Contribution Metrics)**~~ - DONE: C_i and S_i implemented
12. ~~**Phase 7.1.5 (Contribution Metrics Validation)**~~ - DONE: Expert 2 identified as "routing sink"
13. ~~**Phase 3.5.4 (Domain Generalization)**~~ - DONE: C_i threshold generalizes across all 4 domains
14. ~~**Phase 7.5.1 (DeepSeek-MoE-16B Support)**~~ - DONE: Model loading, contribution metrics, SLURM job

**🔥 CURRENT PRIORITY:**
- **Scale to larger models** - test if routing sinks exist in production MoE models
  - ✅ DeepSeek-MoE-16B support implemented (64 routed + 2 shared experts)
  - ⏳ Run DeepSeek contribution analysis (`sbatch scripts/jobs/submit_deepseek.sh`)
  - Next: Mixtral-8x7B (same architecture as TinyMixtral, 46x scale)
- Investigate "routing sink" phenomenon (why is Expert 2 selected but contributes nothing?)

**Should-have (lower priority):**
- Phase 5.2 (Contribution-Aware Pruning implementation)
- Phase 5.3 (CAEP Variants - pending new approach)
- Phase 7.2 (Additional Pruning Experiments)

**Nice-to-have:**
- Phase 7.4 (Variant Experiments)
- Phase 6.2-6.4 (Efficiency + Balance Metrics + Profiling)

---

## Open Questions Summary

| ID | Question | Status |
|----|----------|--------|
| Q1.1 | Target GPU type on PACE? | **Resolved: H100 80GB primary** |
| Q1.2 | CUDA version on PACE? | **Resolved: CUDA 12.4** |
| Q2.1 | Model choice? | **Resolved: TinyMixtral-4x248M-MoE (pretrained)** |
| Q2.2 | Model configuration? | **Resolved: 12 layers, 4 experts, top-2** |
| Q3.1 | Dataset for routing statistics? | **Resolved: WikiText-2 train** |
| Q3.2 | Tokens needed for stable estimates? | Open - need empirical validation |
| Q3.5.1 | Calibration dataset choice? | **Resolved: WikiText-2 train (128 samples)** |
| Q3.5.2 | Evaluation benchmarks? | **Resolved: WikiText-2 test (perplexity)** |
| Q3.5.3 | Domain generalization of C_i threshold? | **Resolved: YES** - C_i < 0.1 works across all 4 domains |
| Q4.1 | Best structural metric for pruning? | **Resolved: None work** - all <17% top-1 agreement; redundancy is inverted |
| Q4.2 | Optimal alpha for combined scoring? | **Resolved: N/A** - combining metrics (8%) worse than individual metrics |
| Q4.3 | Does S_i correlate with sensitivity? | **Resolved: Partially** - S_i identifies zero-contributors, not negative contributors |
| Q4.4 | Does S_i identify Expert 2 as harmful? | **Resolved: Yes, but differently** - Expert 2 has near-zero C_i/S_i (not negative S_i) |
| Q5.1 | Merge vs prune effectiveness? | Open - experimental comparison |
| Q5.2 | Fine-tuning budget after pruning? | Open - depends on compute |
| Q7.1 | Why is Expert 2 harmful? | **Resolved: "Routing sink"** - selected but contributes nothing; see Finding 3 |
| Q7.2 | Is harmful expert pattern model-specific? | Open - test on other MoE models |
| Q7.3 | Why is Expert 2 selected if it contributes nothing? | **NEW** - router miscalibration? training artifact? |
| Q7.4 | Does C_i < 0.1 threshold generalize to other models? | **IN PROGRESS** - DeepSeek-MoE-16B support added, awaiting results |
| Q7.5 | Do routing sinks exist in DeepSeek's 64-expert architecture? | **NEW** - run `sbatch scripts/jobs/submit_deepseek.sh` |
| Q7.6 | How do shared experts (always-active) compare to routed experts? | **NEW** - DeepSeek has 2 shared experts per layer |

---

## Next Steps

1. ~~Environment setup (Phase 1)~~ DONE
2. ~~Baseline MoE with pretrained model (Phase 2)~~ DONE
3. ~~Collaboration Analysis (Phase 3)~~ DONE
4. ~~Benchmark Selection (Phase 3.5)~~ DONE
5. ~~Structural Pruning Metrics (Phase 4.1-4.4)~~ DONE
6. ~~Expert Masking (Phase 5.1)~~ DONE
7. ~~Sensitivity Analysis (Phase 7.1)~~ DONE
   - **Finding:** Structural scores do NOT correlate with sensitivity
   - **Finding:** Expert 2 is harmful across all layers
8. ~~Mid-report (Phase 8.3)~~ DONE
9. ~~Structural Score Ablation Study (Phase 7.3)~~ DONE
   - **Conclusion:** All structural metrics fail (<17% agreement with sensitivity)

---

**COMPLETED: Contribution Metrics Validation (Phase 7.1.5)**

10. ~~**Implement `ExpertContributionCollector`** (Phase 4.5.1)~~ DONE
11. ~~**Implement contribution metrics** (Phase 4.5.2-4.5.3)~~ DONE
12. ~~**Create `scripts/compute_contribution_metrics.py`**~~ DONE
13. ~~**Validate contribution metrics** (Phase 7.1.5)~~ DONE
    - **Result:** Expert 2 identified as "routing sink" (near-zero C_i/S_i, not negative S_i)
    - **Practical criterion:** C_i < 0.1 identifies prune candidates
    - See [FINDING.md#finding-3](FINDING.md#finding-3-contribution-metrics-identify-expert-2-as-zero-contributor)

---

**🔥 CURRENT PRIORITY: Understanding and Scaling**

14. ~~**DeepSeek-MoE-16B Support**~~ DONE
    - ✅ Updated `src/models/pretrained_moe.py` (model loading, config extraction)
    - ✅ Updated `src/pruning/contribution_metrics.py` (64 routed + 2 shared experts)
    - ✅ Created `scripts/jobs/submit_deepseek.sh` (SLURM job)
    - ✅ Created `scripts/debug/debug_model_structure.py` (model exploration)

15. ⏳ **Run DeepSeek Analysis**
    - Submit job: `sbatch scripts/jobs/submit_deepseek.sh`
    - Check for routing sinks in 64-expert architecture
    - Compare shared experts vs routed experts contribution

16. Investigate "routing sink" phenomenon
    - Why does the router select Expert 2 if it contributes nothing?
    - Weight analysis? Training artifact?

17. Scale to Mixtral-8x7B (same architecture as TinyMixtral, production-scale)
    - Goal: Confirm C_i threshold generalizes across models/domains

---

## Timeline (8 weeks)

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Phase 3.5 | Benchmark selection + data pipeline |
| 2 | Phase 4 | All pruning metrics implemented |
| 3-4 | Phase 5 | CAEP algorithm + baselines |
| 5 | Phase 6 | Evaluation infrastructure |
| 6 | Phase 7.1-7.2 | Core experiments |
| 7 | Phase 7.3-7.4 | Ablations + variants |
| 8 | Phase 8 | Analysis + paper draft |
