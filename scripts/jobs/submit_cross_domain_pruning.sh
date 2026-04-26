#!/bin/bash
#SBATCH --job-name=moe-cross-domain-prune
#SBATCH -N 1
#SBATCH -p gpu-h100
#SBATCH --account=gts-ur2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=experiments/pruning/domain_generalization/cross_domain_pruning_%j.out
#SBATCH --error=experiments/pruning/domain_generalization/cross_domain_pruning_%j.err
#SBATCH --mem=32G
#SBATCH -t 1:00:00

# Evaluate perplexity impact of globally pruning Expert 2 across all domains.
# Only 2 forward passes per domain — should finish in ~20 minutes.

echo "=========================================="
echo "Cross-Domain Pruning Impact Evaluation"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Started:  $(date)"
echo ""

module load anaconda3/2023.03

PYTHON=/storage/scratch1/6/jchen3392/envs/moe-place/bin/python

cd /storage/scratch1/6/jchen3392/MoE-Place

mkdir -p experiments/pruning/domain_generalization

$PYTHON scripts/analysis/eval_pruned_cross_domain.py \
    --num_samples 128 \
    --batch_size 4 \
    --output_dir experiments/pruning/domain_generalization

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
