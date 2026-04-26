#!/bin/bash
# Setup script for MoE-Place development environment

set -e

ENV_NAME="moe-place"
ENV_PREFIX="/storage/scratch1/6/jchen3392/envs/$ENV_NAME"
PIP="$ENV_PREFIX/bin/pip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load conda module on PACE
echo "Loading anaconda module..."
module load anaconda3/2023.03

# Remove existing env if present (avoids prefix-already-exists error)
if [ -d "$ENV_PREFIX" ]; then
    echo "Removing existing environment at $ENV_PREFIX..."
    rm -rf "$ENV_PREFIX"
fi

echo "Creating conda environment at: $ENV_PREFIX"
conda create --prefix "$ENV_PREFIX" python=3.11 -y

# Register envs dir so 'conda activate moe-place' works by name in interactive shells
conda config --append envs_dirs "/storage/scratch1/6/jchen3392/envs" 2>/dev/null || true

# Use full path to pip (avoids conda activate issues in script context)
echo "Installing PyTorch with CUDA 12.4 support..."
"$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing project dependencies..."
"$PIP" install -e "$PROJECT_ROOT[dev]"

# Install flash-attn (required by DeepSeek remote model code)
echo "Installing flash-attn (requires GPU node, ~15-20 min to compile)..."
if nvidia-smi &>/dev/null; then
    module load nvhpc-cuda/12.4
    unset CUDA_HOME CUDA_PATH
    "$PIP" install flash-attn --no-build-isolation --cache-dir /storage/scratch1/6/jchen3392/pip_cache
else
    echo "  WARNING: No GPU detected. Skipping flash-attn."
    echo "  On a GPU node, run:"
    echo "    module load nvhpc-cuda/12.4"
    echo "    unset CUDA_HOME CUDA_PATH"
    echo "    $PIP install flash-attn --no-build-isolation"
fi

echo ""
echo "Setup complete! Activate with: conda activate $ENV_NAME"
