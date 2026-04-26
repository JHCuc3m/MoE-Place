# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoE-Place is a research project investigating **expert pruning** in Mixture-of-Experts (MoE) models. After finding that structural metrics (co-activation patterns) fail to predict pruning quality, we pivoted to **contribution-aware expert evaluation** - measuring actual expert contribution to model output.

**Key Insight:** Selection ≠ Contribution. An expert can be frequently selected but still harmful.

See [docs/direction.md](docs/direction.md) for the new direction methodology.

## Current Status

**Completed:**
- Co-activation matrix collection and visualization
- Structural pruning metrics (utilization, redundancy, centrality)
- Expert masking infrastructure
- Full sensitivity analysis (ground truth)
- Structural metrics ablation study (**FAILED: <17% agreement with sensitivity**)
- Mid-report
- Contribution metrics infrastructure (C_i, S_i)
- Contribution metrics validation (**Expert 2 = "routing sink"**)
- Domain generalization (**C_i threshold works across all 4 domains**)

**Key Findings:**
1. **Expert 2 is harmful in ALL 12 layers** - Negative sensitivity (-79 to -163 PPL); disabling improves model
2. **Structural scores do NOT predict sensitivity** - All metrics fail (<17% top-1 agreement)
3. **Expert 2 is a "routing sink"** - Near-zero C_i (< 0.1) while others have C_i 0.5-0.96
4. **C_i threshold works** - C_i < 0.1 identifies prune candidates (not negative S_i as hypothesized)
5. **C_i generalizes across domains** - Expert 2 is a sink in WikiText, code, math, and scientific text

**Current Priority:** Scale to larger models - test if routing sinks exist in production MoE models

**In Progress:**
- DeepSeek-MoE-16B support added (64 routed experts + 2 shared experts)

## Development Setup

```bash
# On PACE cluster, load modules first
module load anaconda3/2023.03

# Create and activate conda environment
bash scripts/setup_env.sh
conda activate moe-place

# Or manually:
pip install -e .
```

## Common Commands

```bash
# Run baseline benchmark (co-activation + perplexity on WikiText-2)
python scripts/benchmark_baseline.py                    # Full evaluation
python scripts/benchmark_baseline.py --quick            # Quick test

# Compute structural pruning metrics from co-activation stats
python scripts/compute_pruning_metrics.py

# Compute contribution metrics (C_i, S_i) - NEW
python scripts/compute_contribution_metrics.py          # Full computation
python scripts/compute_contribution_metrics.py --quick  # Quick test
python scripts/compute_contribution_metrics.py --compare_sensitivity  # Validate vs ground truth

# Submit SLURM job for contribution metrics validation
sbatch scripts/jobs/submit_contribution.sh

# DeepSeek-MoE-16B analysis (larger model, 64 experts + 2 shared)
python scripts/compute_contribution_metrics.py --model deepseek-ai/deepseek-moe-16b-base --quick
sbatch scripts/jobs/submit_deepseek.sh                    # SLURM job (~6 hours)

# Domain generalization - test C_i across multiple domains
python scripts/benchmark_contribution_domains.py --quick           # Quick C_i test
python scripts/domain_sensitivity_analysis.py --quick              # Quick with sensitivity
python scripts/domain_sensitivity_analysis.py                      # Full analysis (C_i + sensitivity)
sbatch scripts/jobs/submit_domain_sensitivity.sh                   # SLURM job (~6 hours)

# Visualize co-activation matrices
python scripts/visualize_coactivation.py --annotate    # All layers
python scripts/visualize_coactivation.py --layer 0     # Single layer

# Evaluate pruning impact
python scripts/evaluate_pruning.py --num_prune 1       # Prune top-1 globally
python scripts/evaluate_pruning.py --prune_per_layer 1 # Prune 1 per layer
python scripts/evaluate_pruning.py --sensitivity       # Full sensitivity analysis
python scripts/evaluate_pruning.py --num_prune 1 --quick  # Quick test

# Debug model structure
python scripts/debug_mixtral_structure.py

# Run tests
pytest tests/
```

## Code Architecture

```
src/
  models/           # Pretrained MoE model loading
    pretrained_moe.py   - Load TinyMixtral, hook routing decisions
  routing/          # Router analysis and statistics
    statistics.py       - RoutingStatisticsCollector for co-activation tracking
    visualization.py    - Collaboration matrix plots, expert load charts
  data/             # Benchmark data loading
    benchmarks.py       - WikiText-2 loading, calibration/eval splits
  pruning/          # Expert pruning (main focus)
    metrics.py          - Structural metrics (centrality, redundancy) - FAILED
    expert_masking.py   - MixtralExpertMasker for disabling experts, sensitivity analysis
    contribution_metrics.py - ExpertContributionCollector, C_i/S_i metrics (NEW)
  evaluation/       # Model evaluation
    perplexity.py       - Perplexity computation on WikiText-2
  kernels/          # Custom Triton/CUDA kernels (if needed)
  profiling/        # PyTorch Profiler and Nsight utilities
  utils/            # Common utilities
scripts/            # Executable scripts for experiments
configs/            # Model and experiment configurations
experiments/        # Results and logs
  baseline/         - Baseline benchmark results, co-activation stats
  pruning/          - Pruning experiment results, sensitivity analysis
docs/               # Documentation
  NEW_DIRECTION.md  - Detailed methodology for pruning approach
  DESIGN.md         - Technical design: metrics, algorithms, implementation notes
  MoE-Place-Overleaf/ - LaTeX reports
```

## Models

### Primary: TinyMixtral-4x248M-MoE
- 12 transformer layers (all with MoE)
- 4 experts per layer (48 total)
- Top-2 routing
- ~1B total parameters

### Scaling Target: DeepSeek-MoE-16B
- 28 transformer layers (first layer dense, rest MoE)
- 64 routed experts + 2 shared experts per layer
- Top-6 routing (routed) + always-active (shared)
- 16.4B total parameters, 2.8B active

See [MODEL.md](MODEL.md) for detailed architecture documentation.

## Hardware Target

- Primary: H100 80GB HBM3, CUDA 12.4
- Model: TinyMixtral-4x248M-MoE (pretrained, realistic routing patterns)

## Key Metrics

### Quality Metrics
- **Perplexity** on WikiText-2 test set (primary metric)
- Baseline (unpruned): 800.38 PPL

### Structural Metrics (FAILED - do not predict sensitivity)
- **Utilization**: Expert selection frequency
- **Redundancy**: Max conditional co-activation probability
- **Centrality**: Eigenvector centrality from co-activation graph

### Contribution Metrics (NEW - Current Focus)
- **C_i (Magnitude)**: `|G_i(x) · E_i(x)|_2 / |h_post(x)|_2` - expert contribution magnitude
- **S_i (Signed)**: `⟨G_i(x) · E_i(x), h_post(x)⟩ / |h_post(x)|_2` - alignment with output
  - Positive S_i → aligned with output → helpful expert
  - Negative S_i → opposed to output → harmful expert (prune target)

### Sensitivity (Ground Truth)
- PPL increase when expert is disabled
- Positive = important, Negative = harmful

## Research Questions

1. **Does co-activation structure predict pruning sensitivity?**
   - **Finding:** NO - Structural scores fail (<17% agreement); all metrics insufficient

2. **Can contribution metrics (S_i) predict sensitivity?**
   - **Current work:** Validate S_i correlation with sensitivity ground truth
   - **Key test:** Does S_i identify Expert 2 as harmful (negative S_i)?

3. **Why are some experts harmful?**
   - Expert 2 degrades model in all layers
   - **Hypothesis:** Negative S_i (contribution opposes output)

## Key Results

| Experiment | Result |
|------------|--------|
| Baseline PPL | 800.38 |
| Prune Layer 0 Expert 1 (structural score) | 890.15 (+11.2%) |
| Expert 2 sensitivity (all layers) | -79 to -163 (harmful) |
| Expert 0 Layer 0/11 sensitivity | +188 to +190 (critical) |

See `experiments/pruning/RESULT_ANALYSIS.md` for detailed analysis.
