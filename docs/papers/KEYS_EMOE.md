# Key Findings from eMoE (arXiv:2503.06823v1, 10 Mar 2025)

- eMoE targets memory-efficient MoE inference by predicting and loading only required experts. The paper reports MoE models can consume roughly 4x-14x the memory of dense counterparts.
- Token-to-expert routing shows recurring patterns; some experts are "popular" within time windows, making expert selection predictable.
- Consecutive layers exhibit strong expert-selection correlation (about 0.50 for OpenMoE and Mixtral-8x7B), implying predictable cross-layer transitions.
- Consecutive prompts also show notable correlation; reusing the same experts across multiple prompts keeps perplexity nearly unchanged for a while, then degrades as reuse extends too far.
- Dynamic CPU->GPU expert loading reduces memory but drastically increases latency (transfer time can be several seconds and exceed inference time), making per-request loading impractical.
- Task sensitivity matters: some tasks maintain high output similarity even with inaccurate routing in early layers, while open-ended tasks are more sensitive.
- eMoE combines expert prediction, periodic expert invocation (predict every p prompts), task-aware expert loading, and task-aware request scheduling (uses SLOs, task-specific output length, and expert-loading latency).
- Reported outcomes: up to 80% memory reduction, up to 17% latency reduction, 40x longer prompts, 4.5x larger batches, and about 1.5x higher throughput.

