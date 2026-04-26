# MoE-Place

Code for **"Co-Activation Aware Expert Pruning for Mixture-of-Experts Models"**.

## Setup

```bash
# On PACE cluster
module load anaconda3/2023.03
bash scripts/setup_env.sh
conda activate moe-place
```

Or manually: `pip install -e .`

**Hardware:** NVIDIA H100 80GB, CUDA 12.4

---

## Reproducing Experiments

### 1. Baseline perplexity

```bash
python scripts/analysis/benchmark_baseline.py          # full
python scripts/analysis/benchmark_baseline.py --quick  # fast check
```

### 2. Structural metrics & ablation study

```bash
# Compute co-activation-based metrics (utilization, redundancy, centrality)
python scripts/analysis/compute_pruning_metrics.py

# Ablation: compare each metric against sensitivity ground truth
python scripts/analysis/ablation_structural_metrics.py \
    --stats_path experiments/baseline/coactivation_stats.json \
    --sensitivity_path experiments/pruning/sensitivity_results.json \
    --output_dir experiments/pruning/ablation
```

### 3. Sensitivity analysis (ground truth)

```bash
# Full sensitivity analysis — disables each of the 48 experts one at a time
python scripts/analysis/evaluate_pruning.py --sensitivity

# Global Expert 2 pruning (all 12 layers simultaneously)
python scripts/analysis/evaluate_pruning.py --prune_experts "0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:2"
```

### 4. Contribution metrics (C_i, S_i)

```bash
python scripts/analysis/compute_contribution_metrics.py                   # full
python scripts/analysis/compute_contribution_metrics.py --quick           # fast check
python scripts/analysis/compute_contribution_metrics.py --compare_sensitivity  # validate vs ground truth
```

### 5. Domain generalization

```bash
# C_i across WikiText-2, code, math, and scientific text
python scripts/analysis/benchmark_contribution_domains.py

# Pruning impact across domains (baseline vs Expert 2 pruned)
python scripts/analysis/eval_pruned_cross_domain.py
```

---

## SLURM Jobs

For the heavier experiments on the PACE cluster:

```bash
sbatch scripts/jobs/submit_contribution.sh          # contribution metrics
sbatch scripts/jobs/submit_cross_domain_pruning.sh  # cross-domain pruning impact
sbatch scripts/jobs/submit_domain_sensitivity.sh    # full domain sensitivity
sbatch scripts/jobs/submit_ablation.sh              # structural metrics ablation
```

All jobs use `--partition=gpu-h100 --account=gts-ur2`.

---

## Results

Experiments output to `experiments/`:
- `baseline/` — co-activation stats, perplexity
- `pruning/` — sensitivity results, contribution metrics, ablation, domain generalization
