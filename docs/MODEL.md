# MODEL.md - MoE Architecture Documentation

This document explains the Mixture-of-Experts (MoE) architecture used in this project.

## Model: TinyMixtral-4x248M-MoE

We use a pretrained MoE model from HuggingFace: `Isotonic/TinyMixtral-4x248M-MoE`

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_type` | mixtral | Mixtral-style architecture |
| `num_layers` | 12 | Total transformer layers |
| `hidden_size` | 1024 | Hidden dimension |
| `num_experts` | 4 | Experts per MoE layer |
| `num_experts_per_tok` | 2 | Top-k experts selected per token |
| `intermediate_size` | 4096 | Expert MLP hidden dimension |

---

## Architecture Diagram

### Full Model Structure

```
Input Token IDs (batch, seq_len)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Embedding Layer                                        │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 0: Attention + MoE (4 experts, top-2)            │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Attention + MoE (4 experts, top-2)            │
├─────────────────────────────────────────────────────────┤
│  ...                                                    │
├─────────────────────────────────────────────────────────┤
│  Layer 11: Attention + MoE (4 experts, top-2)           │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  LM Head → Output Logits                                │
└─────────────────────────────────────────────────────────┘
```

All 12 layers use MoE (unlike some models that alternate MoE/FFN).

---

### MoE Layer Structure (MixtralSparseMoeBlock)

```
                    Input: (batch, seq, hidden=1024)
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Router (MixtralTopKRouter)│
                    │      Linear: 1024 → 4 experts  │
                    └───────────────────────────────┘
                                    │
                         Router Logits: (batch*seq, 4)
                                    │
                              Softmax + Top-2
                                    │
                    ┌───────────────┴───────────────┐
                    │  expert_indices: (tokens, 2)  │
                    │  routing_weights: (tokens, 2) │
                    └───────────────┬───────────────┘
                                    │
                              Token Dispatch
                                    │
            ┌───────────┬───────────┼───────────┬───────────┐
            ▼           ▼           ▼           ▼           │
        ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐      │
        │Expert │   │Expert │   │Expert │   │Expert │      │
        │   0   │   │   1   │   │   2   │   │   3   │      │
        │ (MLP) │   │ (MLP) │   │ (MLP) │   │ (MLP) │      │
        └───────┘   └───────┘   └───────┘   └───────┘      │
            │           │           │           │           │
            └───────────┴───────────┴───────────┴───────────┘
                                    │
                              Weighted Combine
                            output = Σ (weight_i × expert_i(x))
                                    │
                                    ▼
                    Output: (batch, seq, hidden=1024)
```

---

## Expert Co-activation (Collaboration)

When a token selects multiple experts, those experts are "co-activated" together.

### Example

```
Token "def"     → selects [Expert 1, Expert 2]  → Pair (1,2) co-activated
Token "function"→ selects [Expert 1, Expert 2]  → Pair (1,2) co-activated
Token "Paris"   → selects [Expert 0, Expert 3]  → Pair (0,3) co-activated
Token "capital" → selects [Expert 0, Expert 2]  → Pair (0,2) co-activated
```

### Collaboration Matrix

A symmetric 4×4 matrix tracking co-activation counts:

```
        Expert 0  1  2  3
Expert 0   [--  12  8 15]
       1   [12  --  45  3]   ← (1,2)=45: frequently co-activated!
       2   [ 8  45  --  8]
       3   [15   3   8 --]
```

### Why Collaboration Matters

Experts that are frequently co-activated together could benefit from:
1. **Memory locality** - Place them adjacent in GPU memory
2. **Kernel fusion** - Process them in the same CUDA kernel
3. **Shared memory reuse** - Load token once, process through both experts

This is the core idea behind the MoE-Place research project.

---

## Code Reference

| Component | File | Class/Function |
|-----------|------|----------------|
| Model Loading | `src/models/pretrained_moe.py` | `load_pretrained_moe()` |
| Config Extraction | `src/models/pretrained_moe.py` | `get_moe_config()` |
| Routing Hooks | `src/models/pretrained_moe.py` | `MixtralRoutingCollector` |
| Statistics | `src/routing/statistics.py` | `RoutingStatisticsCollector` |
| Visualization | `src/routing/visualization.py` | `plot_collaboration_matrix()` |

---

## Running Routing Analysis

```bash
# Collect routing statistics
python scripts/collect_pretrained_routing.py --num_samples 500

# Output saved to experiments/pretrained_routing_stats/
#   - routing_statistics.json
#   - plots/layer_X_collaboration.png
#   - plots/layer_X_load.png
```
