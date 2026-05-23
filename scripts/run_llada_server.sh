#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_ROOT="${MOSAIC_MODEL_ROOT:-${REPO_DIR}/models}"
MODEL_DIR="${MODEL_DIR:-${LLADA_MODEL_DIR:-${MODEL_ROOT}/llada-8b-instruct}}"

# Ensure Python can import plugins, source-tree custom ops, and local model code
# in worker subprocesses.
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/flash_sample:${REPO_DIR}/vmm_allocator:${REPO_DIR}/vllm_add_llada/cuda_kernels:${MODEL_DIR}:${PYTHONPATH:-}"
export LLADA_MODEL_DIR="${MODEL_DIR}"

export VLLM_USAGE_COLLECTION_DISABLED=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export VLLM_USE_VMM="${VLLM_USE_VMM:-1}"
export VLLM_USE_CHUNKWISE_GRAPH="${VLLM_USE_CHUNKWISE_GRAPH:-1}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8901}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-204800}"

ARGS=(python -m vllm.entrypoints.api_server \
  --model "${MODEL_DIR}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}")

if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
  ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${EXTRA_VLLM_ARGS})
  ARGS+=("${EXTRA_ARGS[@]}")
fi

exec "${ARGS[@]}"

