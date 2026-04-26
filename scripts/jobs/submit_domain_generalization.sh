#!/bin/bash
#SBATCH -J moe_domain_gen
#SBATCH -N 1
#SBATCH -p gpu-h100
#SBATCH --account=gts-ur2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Domain Generalization: Test C_i threshold across multiple data domains
# Goal: Confirm Expert 2 is a routing sink universally, not just for WikiText-2

echo "=========================================="
echo "Domain Generalization Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

module load anaconda3/2023.03

PYTHON=/storage/scratch1/6/jchen3392/envs/moe-place/bin/python

cd /storage/scratch1/6/jchen3392/MoE-Place

mkdir -p experiments/pruning/domain_generalization

$PYTHON scripts/analysis/benchmark_contribution_domains.py \
    --num_samples 128 \
    --batch_size 4 \
    --ci_threshold 0.1 \
    --output_dir experiments/pruning/domain_generalization

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
