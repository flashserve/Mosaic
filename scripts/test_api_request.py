#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

MODEL_DEFAULTS = {
    "llada": {
        "port": 8901,
        "model_env": "LLADA_MODEL_DIR",
        "model_name": "llada-8b-instruct",
        "mask_id": 126336,
        "block_length": 32,
    },
    "dream": {
        "port": 8701,
        "model_env": "DREAM_MODEL_DIR",
        "model_name": "dream-v0-instruct-7b",
        "mask_id": 151666,
        "block_length": 32,
    },
    "llada-moe": {
        "port": 10001,
        "model_env": "LLADA_MOE_MODEL_DIR",
        "model_name": "llada-moe-7b-a1b",
        "mask_id": 156895,
        "block_length": 32,
    },
}


def default_model_dir(model_family: str) -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    defaults = MODEL_DEFAULTS[model_family]
    model_root = os.environ.get("MOSAIC_MODEL_ROOT",
                                os.path.join(repo_root, "models"))
    return os.environ.get(defaults["model_env"],
                          os.path.join(model_root, defaults["model_name"]))


def build_prompt(args: argparse.Namespace) -> str:
    if not args.chat_template:
        return args.prompt

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("--chat-template requires transformers") from exc

    tokenizer_path = args.tokenizer or default_model_dir(args.model_family)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              trust_remote_code=True)
    messages = [{"role": "user", "content": args.prompt}]
    return tokenizer.apply_chat_template(messages,
                                         add_generation_prompt=True,
                                         tokenize=False)


def extract_text(data: Any) -> str:
    if isinstance(data, dict):
        value = data.get("text") or data.get("output") or data.get("outputs")
        if isinstance(value, list) and value:
            return str(value[0])
        if value is not None:
            return str(value)
    if isinstance(data, list) and data:
        return extract_text(data[0])
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one /generate request to a Mosaic diffusion server.")
    parser.add_argument("--model-family",
                        choices=sorted(MODEL_DEFAULTS),
                        default="llada")
    parser.add_argument("--host", default=None)
    parser.add_argument("--prompt",
                        default="Explain diffusion language models in one short paragraph.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    defaults = MODEL_DEFAULTS[args.model_family]
    host = args.host or f"http://127.0.0.1:{defaults['port']}"
    prompt = build_prompt(args)

    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": args.gen_length,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "output_length": args.gen_length,
        "block_length": args.block_length or defaults["block_length"],
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
        "mask_id": args.mask_id if args.mask_id is not None else defaults["mask_id"],
    }

    response = requests.post(f"{host.rstrip('/')}/generate",
                             json=payload,
                             timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    text = extract_text(data)
    if not text:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        raise RuntimeError("response did not contain generated text")

    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())