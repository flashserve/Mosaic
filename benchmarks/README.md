# Mosaic Context Benchmarks

These scripts run a source-tree Mosaic server, send long-context diffusion
requests, and write CSV/JSON reports plus server logs. If `--output-dir` is not
provided, outputs are written under `benchmarks/results/`; for release
validation, prefer an explicit scratch directory such as `/tmp/mosaic-benchmark-run`.

`alpha` is the prompt-to-total ratio: `prompt_len = total_len * alpha`, and
`output_len = total_len - prompt_len`.

Example:

```bash
python benchmarks/llada/run_benchmark.py \
  --model-path /path/to/llada-8b-instruct \
  --start 1024 --step 1024 --max 4096 --steps 10 --alpha 0.5
```

The LLaDA-MoE benchmark uses mask id `156895`; LLaDA uses `126336`; Dream uses
`151666`.