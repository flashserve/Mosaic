#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path


DEFAULT_PATHS = [
    "benchmarks",
    "scripts",
    "diffusion_tools",
    "flash_sample",
    "gather_gemm",
    "vllm_add_dream",
    "vllm_add_llada",
    "vllm_add_llada_moe",
    "vmm_allocator",
    "sitecustomize.py",
]

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
}


def iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in paths:
        if input_path.is_file() and input_path.suffix == ".py":
            files.append(input_path)
            continue
        if not input_path.is_dir():
            continue
        for source_path in input_path.rglob("*.py"):
            if any(part in SKIP_DIRS for part in source_path.parts):
                continue
            files.append(source_path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse Python files without writing __pycache__ artifacts.")
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    args = parser.parse_args()

    root = Path.cwd()
    paths = [Path(item) if Path(item).is_absolute() else root / item
             for item in args.paths]
    errors: list[str] = []

    for source_path in iter_python_files(paths):
        try:
            source = source_path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(source_path))
        except SyntaxError as exc:
            errors.append(f"{source_path}:{exc.lineno}:{exc.offset}: {exc.msg}")
        except UnicodeDecodeError as exc:
            errors.append(f"{source_path}: decode error: {exc}")

    if errors:
        print("Python syntax check failed:")
        for error in errors:
            print(error)
        return 1

    print(f"Python syntax check passed for {len(iter_python_files(paths))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())