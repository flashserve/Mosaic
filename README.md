# Mosaic Release

Mosaic is a source-run vLLM fork for diffusion language model serving. It keeps
the Python package name and CLI entry point as `vllm`, and adds inference paths
for LLaDA, Dream, and LLaDA-MoE.

> Paper status: **Mosaic: Unlocking Long-Context Inference for Diffusion LLMs via
> Global Memory Planning and Dynamic Peak Taming** has been accepted to
> **ICML 2026**.

This repository is intended to contain code only. Model weights, compiled shared
objects, benchmark CSV/JSON reports, server logs, traces, and other runtime
artifacts are generated locally and should not be committed.

## Highlights

- vLLM V1 API server with a `/generate` endpoint for diffusion LLM requests.
- Model adapters for LLaDA, Dream, and LLaDA-MoE.
- CUDA sampling kernels, LLaDA fused ops, optional CUDA VMM activation allocator,
  and Triton gather/GEMM kernels.
- Unified single-request smoke client.
- Context-length benchmark runner that sweeps linearly to a target length and
  binary searches after the first failure.

## Supported Models

| Family | Expected local directory | Server script | Default port | Mask id |
| --- | --- | --- | ---: | ---: |
| LLaDA | `llada-8b-instruct` | `scripts/run_llada_server.sh` | 8901 | 126336 |
| Dream | `dream-v0-instruct-7b` | `scripts/run_dream_server.sh` | 8701 | 151666 |
| LLaDA-MoE | `llada-moe-7b-a1b` | `scripts/run_llada_moe_server.sh` | 10001 | 156895 |

Model weights are not included. Place them under `models/` in the repository, or
set environment variables to point to external model directories.

## Validated Environment

The release path was validated in a Debian 12 container with NVIDIA CUDA GPUs:

| Component | Validated version |
| --- | --- |
| Python | 3.12.12 |
| PyTorch | 2.7.1+cu126 |
| Transformers | 4.57.3 |
| Triton | 3.3.1 |
| CUDA toolkit | 12.x with `nvcc` available |

Other recent NVIDIA CUDA environments may work, but the custom CUDA extensions
must be rebuilt in the target Python environment.

## Repository Setup

Clone the repository and enter the release tree:

```bash
git clone https://github.com/flashserve/Mosaic Mosaic-release
cd Mosaic-release
```

Create or activate a Python environment. The validated environment uses Python
3.12 and PyTorch 2.7.1 with CUDA 12.x:

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
```

Install build tools and Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements/build.txt
python -m pip install -r requirements/cuda.txt
```

If your PyTorch/CUDA stack is already managed by the cluster image or container,
use the versions supplied by that environment and install only the missing Python
packages. The key requirement is that `python`, `torch`, and `nvcc` agree on a
compatible CUDA toolchain.

Mosaic is validated as a source-run tree. Export `PYTHONPATH` from the repository
root before starting servers or benchmarks:

```bash
export MOSAIC_HOME="$PWD"
export PYTHONPATH="$MOSAIC_HOME:$MOSAIC_HOME/flash_sample:$MOSAIC_HOME/vmm_allocator:$MOSAIC_HOME/vllm_add_llada/cuda_kernels:${PYTHONPATH:-}"
```

The server scripts run `python -m vllm.entrypoints.api_server`, so make sure the
intended environment is active or put its `bin` directory first in `PATH`.

## Build Custom Operators

Build the diffusion sampling kernels:

```bash
cd "$MOSAIC_HOME/flash_sample"
python setup.py build_ext --inplace
```

Build the LLaDA fused CUDA ops:

```bash
cd "$MOSAIC_HOME/vllm_add_llada/cuda_kernels"
python setup.py build_ext --inplace
```

Build the optional CUDA VMM allocator. This allocator is recommended for long
LLaDA context benchmarks and is enabled by default in the provided server and
benchmark scripts:

```bash
cd "$MOSAIC_HOME/vmm_allocator"
python setup.py build_ext --inplace
```

Return to the repository root and verify imports:

```bash
cd "$MOSAIC_HOME"
python - <<'PY'
import flash_sample
from flash_sample import entropy, entropy_witht, low_confidence
from vllm_add_llada.cuda_kernels import fused_ops
import vmm_allocator

print("flash_sample ok", bool(low_confidence), bool(entropy), bool(entropy_witht))
print("fused_ops ok", fused_ops.__name__)
print("vmm_allocator ok", vmm_allocator.__name__)
PY
```

`gather_gemm` uses Triton JIT compilation and does not need a separate build
step.

## Model Paths

The scripts first look under `MOSAIC_MODEL_ROOT`, then allow each model family to
be overridden independently.

Recommended layout:

```text
Mosaic-release/
  models/
    llada-8b-instruct/
    dream-v0-instruct-7b/
    llada-moe-7b-a1b/
```

Use this layout with:

```bash
export MOSAIC_MODEL_ROOT="$MOSAIC_HOME/models"
```

Or point to external weight directories:

```bash
export LLADA_MODEL_DIR=/path/to/llada-8b-instruct
export DREAM_MODEL_DIR=/path/to/dream-v0-instruct-7b
export LLADA_MOE_MODEL_DIR=/path/to/llada-moe-7b-a1b
```

`MODEL_DIR` can be used as a one-off override for any server script:

```bash
MODEL_DIR=/path/to/llada-8b-instruct scripts/run_llada_server.sh
```

## Start A Server

Choose the GPU with `CUDA_VISIBLE_DEVICES`. The examples below bind to localhost;
set `HOST=0.0.0.0` when serving to other machines.

Start LLaDA:

```bash
cd "$MOSAIC_HOME"
CUDA_VISIBLE_DEVICES=0 \
HOST=127.0.0.1 \
PORT=8901 \
MAX_MODEL_LEN=204800 \
GPU_MEMORY_UTILIZATION=0.92 \
scripts/run_llada_server.sh
```

Start Dream:

```bash
cd "$MOSAIC_HOME"
CUDA_VISIBLE_DEVICES=0 \
HOST=127.0.0.1 \
PORT=8701 \
MAX_MODEL_LEN=4096 \
scripts/run_dream_server.sh
```

Start LLaDA-MoE:

```bash
cd "$MOSAIC_HOME"
CUDA_VISIBLE_DEVICES=0 \
HOST=127.0.0.1 \
PORT=10001 \
MAX_MODEL_LEN=8192 \
scripts/run_llada_moe_server.sh
```

Useful runtime overrides:

```bash
export VLLM_USE_VMM=1
export VLLM_VMM_CHUNK_SIZE=2097152
export VLLM_USE_CHUNKWISE_GRAPH=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export EXTRA_VLLM_ARGS="--enforce-eager"
```

## Single Request Test

With a server running, send a single `/generate` request:

```bash
python scripts/test_api_request.py \
  --model-family llada \
  --host http://127.0.0.1:8901 \
  --steps 8 \
  --gen-length 32 \
  --block-length 32
```

Dream:

```bash
python scripts/test_api_request.py \
  --model-family dream \
  --host http://127.0.0.1:8701 \
  --steps 8 \
  --gen-length 32 \
  --block-length 32
```

LLaDA-MoE:

```bash
python scripts/test_api_request.py \
  --model-family llada-moe \
  --host http://127.0.0.1:10001 \
  --steps 8 \
  --gen-length 32 \
  --block-length 32
```

The client prints the raw JSON response. It defaults to `cfg_scale=0`,
`temperature=0`, and the correct mask id for each model family.

## Context Benchmark

The benchmark scripts start a server, wait for `/health`, send a long `/generate`
request, parse `execute_diffusion_once time = ... ms` from the server log, write
CSV/JSON results, stop the server, and continue to the next length.

Mosaic benchmarks use online chunk planning. The benchmark runner starts the
server with `VLLM_USE_CHUNKWISE_GRAPH=1`; the model then calls the online planner
for the current prompt/output shape and the active activation pool budget. The
release no longer ships or consumes precomputed hardware-specific chunk config
JSON files.

The activation pool budget can be fixed with `--activation-pool-size-bytes`,
which is forwarded to the server as `VLLM_ACTIVATION_POOL_SIZE_BYTES`. If this is
omitted, the server sizes the activation pool from currently available memory at
startup. `--gpu-memory-utilization` remains the vLLM startup option for the
standard vLLM memory budget; it is separate from the Mosaic activation pool
budget used by online chunk planning.

The length sweep has two phases:

1. Linear scan from `--start` to `--max` by `--step`.
2. Binary search between the last success and first failure, using `--precision`,
   only if a failure occurs before `--max`.

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `--start` | First total context length to test. |
| `--step` | Linear-scan increment. Use a large value for fast upper-limit checks. |
| `--max` | Target upper bound. If all linear points pass, this is the best length. |
| `--precision` | Binary-search granularity after the first failure. |
| `--steps` | Number of diffusion refinement steps. Use `1` for a fast reachability test. |
| `--alpha` | Prompt length / total length. `0.5` means half prompt, half generated tokens. |
| `--gpu` | GPU ids exposed to the server process. |
| `--output-dir` | Where CSV, JSON, and server logs are written. Use `/tmp/...` for scratch runs. |
| `--activation-pool-size-bytes` | Optional fixed activation pool budget for online chunk planning. |

### LLaDA Upper-Limit Benchmark

The validated LLaDA upper-limit run used `alpha=0.5`, `steps=1`, VMM enabled, and
a large 51200-token step so the sweep reached 204800 quickly:

```bash
cd "$MOSAIC_HOME"
export VLLM_USE_VMM=1
export VLLM_VMM_CHUNK_SIZE=2097152
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python benchmarks/llada/run_benchmark.py \
  --project-path "$MOSAIC_HOME" \
  --model-path "$LLADA_MODEL_DIR" \
  --output-dir /tmp/mosaic_benchmark_llada_204800 \
  --start 51200 \
  --step 51200 \
  --max 204800 \
  --precision 8192 \
  --steps 1 \
  --alpha 0.5 \
  --gpu 0 \
  --port 8916 \
  --startup-timeout 1800 \
  --request-timeout 64800 \
  --shutdown-timeout 300 \
  --gpu-memory-utilization 0.92
```

The benchmark writes CSV and JSON summaries to `--output-dir`.

### Dream Benchmark

```bash
python benchmarks/dream/run_benchmark.py \
  --project-path "$MOSAIC_HOME" \
  --model-path "$DREAM_MODEL_DIR" \
  --output-dir /tmp/mosaic_benchmark_dream \
  --start 1024 \
  --step 1024 \
  --max 4096 \
  --precision 128 \
  --steps 10 \
  --gpu 0 \
  --port 8701
```

### LLaDA-MoE Benchmark

```bash
python benchmarks/llada_moe/run_benchmark.py \
  --project-path "$MOSAIC_HOME" \
  --model-path "$LLADA_MOE_MODEL_DIR" \
  --output-dir /tmp/mosaic_benchmark_llada_moe \
  --start 1024 \
  --step 1024 \
  --max 8192 \
  --precision 128 \
  --steps 10 \
  --gpu 0 \
  --port 10001
```

## Request Payload Reference

The server endpoint is `/generate`. The smoke client and benchmark runner send a
payload with these fields:

```json
{
  "prompt": "Explain diffusion language models in one short paragraph.",
  "max_tokens": 32,
  "gen_length": 32,
  "output_length": 32,
  "steps": 8,
  "block_length": 32,
  "temperature": 0.0,
  "cfg_scale": 0.0,
  "remasking": "low_confidence",
  "mask_id": 126336
}
```

For benchmark runs, `gen_length` and `output_length` are derived from the total
length and `--alpha`. When `--alpha 0.5` and `--max 204800`, the request uses a
102400-token prompt and a 102400-token diffusion output.

## Runtime Environment Variables

| Variable | Purpose |
| --- | --- |
| `MOSAIC_MODEL_ROOT` | Base directory for all local model weights. |
| `MODEL_DIR` | One-off model path override for a server script. |
| `LLADA_MODEL_DIR` | LLaDA model path. |
| `DREAM_MODEL_DIR` | Dream model path. |
| `LLADA_MOE_MODEL_DIR` | LLaDA-MoE model path. |
| `VLLM_USE_VMM` | Enable the optional CUDA VMM activation allocator. |
| `VLLM_VMM_CHUNK_SIZE` | VMM physical mapping chunk size in bytes. |
| `VLLM_USE_CHUNKWISE_GRAPH` | Enable chunk-aware graph execution helpers. |
| `VLLM_ACTIVATION_POOL_SIZE_BYTES` | Optional fixed activation pool budget used by online chunk planning. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Allow long `--max-model-len` values. |
| `GPU_MEMORY_UTILIZATION` | Forwarded to vLLM server startup scripts. |
| `EXTRA_VLLM_ARGS` | Extra command-line flags appended by server scripts. |

## Troubleshooting

If imports fail for `flash_sample`, `fused_ops`, or `vmm_allocator`, rebuild the
custom operators in the active Python environment and confirm that `PYTHONPATH`
contains the operator directories.

If a long LLaDA run fails around long context lengths, confirm that the server is
started with a large enough `MAX_MODEL_LEN` or benchmark `--max-model-len`, VMM is
enabled, and the current tree includes the LLaDA max-sequence-length patch in the
model wrapper.

If `/health` succeeds but the client gets connection errors, check that the client
uses the same `HOST` and `PORT` as the server and that no old server process is
occupying the port.

If GPU memory appears to remain allocated after a failed run, stop the server
process group and confirm with:

```bash
nvidia-smi
ps -eo pid,pgid,stat,comm,args | grep -E 'vllm.entrypoints.api_server|benchmarks/.*/run_benchmark' | grep -v grep
```

## Cleanup

Generated files are ignored by `.gitignore`. To return the source tree to a
code-only state after local validation, remove build products and Python caches:

```bash
find . -type d \( -name build -o -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
find . -type f \( -name '*.so' -o -name '*.pyc' -o -name '*.pyo' -o -name '*.o' -o -name '*.log' -o -name '*.csv' \) -delete
rm -rf benchmarks/results logs benchmark_results
```

Prefer `/tmp/mosaic-benchmark-run` for benchmark `--output-dir` when validating release
candidates so benchmark reports and logs never enter the repository tree.

## Citation

If you use Mosaic in research, please cite:

```bibtex
@article{zheng2026mosaic,
  title={Mosaic: Unlocking Long-Context Inference for Diffusion LLMs via Global Memory Planning and Dynamic Peak Taming},
  author={Zheng, Liang and Shi, Bowen and Hu, Yitao and Zhang, Jiawei and Li, Ruofan and Chen, Sheng and Li, Wenxin and Li, Keqiu},
  journal={arXiv preprint arXiv:2601.06562},
  year={2026}
}
```
