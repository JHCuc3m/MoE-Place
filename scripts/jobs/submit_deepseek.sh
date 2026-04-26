#!/bin/bash
#SBATCH -J deepseek_moe_analysis
#SBATCH -N 1
#SBATCH -p gpu-h100
#SBATCH --account=gts-ur2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH -t 6:00:00
#SBATCH -o slurm-deepseek-%j.out
#SBATCH -e slurm-deepseek-%j.err

# DeepSeek-MoE-16B Contribution Analysis
# Requires: H100 80GB GPU, ~32GB model + overhead

module load anaconda3/2023.03

PYTHON=/storage/scratch1/6/jchen3392/envs/moe-place/bin/python

cd /storage/scratch1/6/jchen3392/MoE-Place

echo "========================================"
echo "DeepSeek-MoE-16B Contribution Analysis"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON"
echo "========================================"

nvidia-smi

mkdir -p experiments/deepseek

echo ""
echo "========================================"
echo "Phase 1: Quick Test (verifying model loads)"
echo "========================================"
$PYTHON scripts/analysis/compute_contribution_metrics.py \
    --model deepseek-ai/deepseek-moe-16b-base \
    --quick \
    --output_dir experiments/deepseek

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Phase 2: Full Contribution Analysis"
    echo "========================================"
    $PYTHON scripts/analysis/compute_contribution_metrics.py \
        --model deepseek-ai/deepseek-moe-16b-base \
        --num_samples 128 \
        --batch_size 2 \
        --output_dir experiments/deepseek
else
    echo "Quick test failed, skipping full analysis"
    exit 1
fi

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
