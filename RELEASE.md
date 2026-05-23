# Mosaic Release Checklist

Mosaic is distributed as a source-run vLLM fork for diffusion language model
serving. A release should be a clean source tree that another developer can
clone, build, and validate without local benchmark artifacts or machine-specific
paths.

## Source Tree Policy

Keep these in the repository:

- Source code, scripts, configuration JSON files, documentation, and tests.
- Small documentation assets already inherited from vLLM.
- Build recipes for `flash_sample`, `vllm_add_llada/cuda_kernels`, and
	`vmm_allocator`.

Do not commit these:

- Model weights or tokenizer files.
- Compiled extensions such as `*.so` and object files such as `*.o`.
- `build/`, `dist/`, `*.egg-info/`, `__pycache__/`, `.pytest_cache/`, and
	`.mypy_cache/`.
- Benchmark CSV/JSON reports, server logs, traces, and profiler outputs.
- Local paths, credentials, cluster inventory, or private run notes.

## Pre-Release Checks

From the repository root:

```bash
find . -type d \( -name build -o -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
find . -type f \( -name '*.so' -o -name '*.pyc' -o -name '*.pyo' -o -name '*.o' -o -name '*.log' -o -name '*.csv' \) -delete
python tools/check_python_syntax.py
```

Then scan for release hazards:

```bash
grep -RInI -E '/workspace|/opt/conda|/root/|/home/[^ ]+' README.md scripts benchmarks diffusion_tools vllm_add_dream vllm_add_llada vllm_add_llada_moe vmm_allocator || true
grep -RInI -E 'AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|sk-[0-9A-Za-z]{20,}|hf_[0-9A-Za-z]{20,}|BEGIN (RSA|DSA|EC|OPENSSH|PRIVATE) KEY' . || true
```

Expected notes:

- Dockerfiles and inherited vLLM docs may contain container paths such as
	`/workspace` and `/root/.cache`; those are documentation/build examples.
- Tests and examples may contain dummy API keys such as `EMPTY` or `sk-fake-key`.
	These are placeholders, not credentials.

## Functional Validation

Use the root `README.md` as the canonical validation path:

1. Create or activate the target Python/CUDA environment.
2. Install `requirements/build.txt` and `requirements/cuda.txt`.
3. Build `flash_sample`, `vllm_add_llada/cuda_kernels`, and `vmm_allocator`.
4. Export `MOSAIC_HOME`, `PYTHONPATH`, and model path variables.
5. Start one server script and run `scripts/test_api_request.py`.
6. Run at least one benchmark with `--output-dir /tmp/mosaic-benchmark-run`.

Current release scope includes runnable entry points for:

| Model family | Included path |
| --- | --- |
| LLaDA | Long-context server path and benchmark entry point. |
| Dream | Smoke server path and benchmark entry point. |
| LLaDA-MoE | Smoke server path and benchmark entry point. |

## Versioning

For private or public GitHub releases, use tags that identify both the Mosaic
release and the validation date, for example:

```bash
git tag -a mosaic-2026-05-22 -m "Mosaic source release 2026-05-22"
```

Avoid publishing wheels until the custom CUDA extension build matrix is defined
for the target CUDA, PyTorch, Python, and GPU architectures.
