- submodule update

```bash
git submodule update --remote --merge docs/MoE-Place-Overleaf
```

- Get partitions on Slurm
```bash
sinfo -o "%P %G %D %T"
```

- Request H100 on Ice
```bash
srun --partition=ice-gpu --account=cse --qos=coc-ice --gres=gpu:h100:1 --ntasks=1 --cpus-per-task=8 --mem=64G --pty bash -i
```

- Check allocated GPU
```bash
nvidia-smi
```

- Set up environment on H100 node: 
```bash
bash scripts/setup_env.sh
module load anaconda3/2023.03
conda activate moe-place
```

- Get Baseline
```bash
# Quick test (64 calibration samples, 50 eval batches, ~2-3 min)
python scripts/benchmark_baseline.py --quick

# Full baseline (128 calibration, full WikiText-2 test eval)
python scripts/benchmark_baseline.py

#Custom settings
python scripts/benchmark_baseline.py --calibration_samples 256 --eval_batch_size 8
```

- Run Metrics Computation
```bash
python scripts/compute_pruning_metrics.py
```

- Visualize

```bash
# Visualize all layers with pruning annotations
python scripts/visualize_coactivation.py --annotate

# Visualize only Layer 0
python scripts/visualize_coactivation.py --layer 0 --annotate

# Show plots interactively (instead of just saving)
python scripts/visualize_coactivation.py --layer 0 --annotate --show
```

- Prune

```bash
# Prune top-N experts globally (based on structural score ranking)
python scripts/evaluate_pruning.py --num_prune 1

# Prune 1 expert per layer
python scripts/evaluate_pruning.py --prune_per_layer 1

# Prune specific experts (layer:expert format)
python scripts/evaluate_pruning.py --prune_experts "0:1,5:2"

# Run full sensitivity analysis (disable each expert one at a time)
python scripts/evaluate_pruning.py --sensitivity

# Quick test (50 eval batches)
python scripts/evaluate_pruning.py --num_prune 1 --quick
```
