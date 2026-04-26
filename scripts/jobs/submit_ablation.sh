#!/bin/bash
#SBATCH -J moe_ablation_study
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

module load anaconda3/2023.03

PYTHON=/storage/scratch1/6/jchen3392/envs/moe-place/bin/python

cd /storage/scratch1/6/jchen3392/MoE-Place

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON"
echo "========================================"

$PYTHON scripts/analysis/ablation_structural_metrics.py \
    --stats_path experiments/baseline/coactivation_stats.json \
    --sensitivity_path experiments/pruning/sensitivity_results.json \
    --output_dir experiments/pruning/ablation

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
