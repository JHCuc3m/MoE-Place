#!/bin/bash
#SBATCH --job-name=moe-domain-sens
#SBATCH -N 1
#SBATCH -p gpu-h100
#SBATCH --account=gts-ur2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=experiments/pruning/domain_generalization/sensitivity_job_%j.out
#SBATCH --error=experiments/pruning/domain_generalization/sensitivity_job_%j.err
#SBATCH --mem=64G
#SBATCH -t 6:00:00

# Cross-Domain Sensitivity & Contribution Analysis
#
# For each domain (wikitext2, code, math, scientific):
# 1. Compute C_i contribution metrics
# 2. Run full sensitivity analysis (perplexity-based)
# 3. Compare C_i vs sensitivity correlation
#
# This is compute-intensive: ~48 forward passes per domain (12 layers × 4 experts)

module load anaconda3/2023.03

PYTHON=/storage/scratch1/6/jchen3392/envs/moe-place/bin/python

cd /storage/scratch1/6/jchen3392/MoE-Place

echo "=========================================="
echo "Cross-Domain Sensitivity Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

mkdir -p experiments/pruning/domain_generalization

$PYTHON scripts/analysis/domain_sensitivity_analysis.py \
    --num_samples 128 \
    --batch_size 4 \
    --ci_threshold 0.1 \
    --output_dir experiments/pruning/domain_generalization

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
