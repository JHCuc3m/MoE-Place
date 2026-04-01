#!/bin/bash
# Setup script for MoE-Place development environment

set -e

ENV_NAME="moe-place"

# Load conda module on PACE
echo "Loading anaconda module..."
module load anaconda3/2023.03

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.11 -y

echo "Activating environment"
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA 12.4 support"
# Note: PyTorch 2.4+ supports CUDA 12.x, compatible with CUDA 13.0 runtime
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing other dependencies"
pip install -r requirements.txt

echo "Installing package in editable mode"
pip install -e .

echo ""
echo "Setup complete! Activate with: conda activate $ENV_NAME"
