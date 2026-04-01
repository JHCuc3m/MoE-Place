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

- Collect routing stats from pretrained model      
  TinyMixtral-4x248M-MoE

```bash
python scripts/collect_pretrained_routing.py --num_samples 500
```
