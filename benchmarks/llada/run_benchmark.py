#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(CURRENT_DIR.parent / "common"))

from benchmark_core import BenchmarkRunner


def default_model_path() -> str:
    model_root = os.environ.get("MOSAIC_MODEL_ROOT", str(REPO_ROOT / "models"))
    return os.environ.get("LLADA_MODEL_DIR",
                          str(Path(model_root) / "llada-8b-instruct"))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLaDA context benchmark")
    parser.add_argument("--project-path", default=str(REPO_ROOT))
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Optional conda environment name or absolute environment path. "
        "Using --python is preferred for reproducible runs.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start", type=int, default=51200)
    parser.add_argument("--step", type=int, default=51200)
    parser.add_argument("--max", type=int, default=204800)
    parser.add_argument("--precision", type=int, default=8192)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8901)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Prompt length / total length ratio")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=64800)
    parser.add_argument("--shutdown-timeout", type=int, default=180)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--activation-pool-size-bytes",
        type=int,
        default=None,
        help="Optional activation pool budget passed to the server for online "
        "chunk planning. If omitted, the server sizes the pool from available "
        "memory at startup.")
    args = parser.parse_args()

    output_dir = args.output_dir or str(
        REPO_ROOT / "benchmarks" / "results" /
        f"llada_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    config = {
        "repo_root": args.project_path,
        "python_executable": args.python_executable,
        "conda_env": args.conda_env,
        "model_path": args.model_path,
        "model_env_var": "LLADA_MODEL_DIR",
        "output_dir": output_dir,
        "server_port": args.port,
        "max_len": args.max,
        "gpu_ids": [int(item) for item in args.gpu.split(",") if item],
        "alpha": args.alpha,
        "test_mask_id": 126336,
        "test_prompt": "What is Diffusion-LM?",
        "test_steps": args.steps,
        "server_startup_timeout": args.startup_timeout,
        "request_timeout": args.request_timeout,
        "server_shutdown_timeout": args.shutdown_timeout,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "extra_env": ({
            "VLLM_ACTIVATION_POOL_SIZE_BYTES": args.activation_pool_size_bytes,
        } if args.activation_pool_size_bytes is not None else {}),
    }
    best = BenchmarkRunner(config).run_smart_benchmark(
        args.start, args.step, args.max, args.precision)
    print(f"Best supported length: {best}")


if __name__ == "__main__":
    main()