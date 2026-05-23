# Diffusion Tools

This package contains the runtime chunk planner used by Mosaic's diffusion
model paths.

## Runtime Path

Mosaic uses online chunk planning by default. When `VLLM_USE_CHUNKWISE_GRAPH=1`,
the model path calls `search_optimal_config_online(...)` with the current token
shape and activation pool size, then builds the chunk-aware reuse plan for that
request.

No precomputed hardware-specific chunk JSON files are required in the release
tree.

## Public API

```python
from diffusion_tools import search_optimal_config_online

strategy = search_optimal_config_online(
    P=prompt_tokens,
    O=output_tokens,
    pool_size=activation_pool_size,
    model_name="llada",
)
```

The returned strategy contains the selected logits/MLP chunk flags, chunk counts,
estimated activation bytes, and memory component summary used by the runtime
graph builder.

## Integrated Model Paths

- `vllm_add_llada/llada_model.py`
- `vllm_add_dream/dream_model.py`
- `vllm_add_llada_moe/llada_moe_model.py`

