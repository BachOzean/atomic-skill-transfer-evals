#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PATCH_SUFFIX = ".bak_before_ast_eval_patch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch and validate HF checkpoints exported from veRL FSDP shards.")
    parser.add_argument("model_path", help="HF checkpoint directory.")
    parser.add_argument("--no-patch", action="store_true", help="Only validate; do not modify config files.")
    parser.add_argument("--validate", action="store_true", default=True, help="Validate AutoConfig and AutoTokenizer loading.")
    parser.add_argument("--no-validate", dest="validate", action="store_false", help="Skip transformers validation.")
    parser.add_argument("--vllm-smoke", action="store_true", help="Run a tiny vLLM generation smoke test.")
    parser.add_argument("--max-model-len", type=int, default=2048, help="vLLM smoke max_model_len.")
    parser.add_argument("--max-tokens", type=int, default=8, help="vLLM smoke max_tokens.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.20")))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + PATCH_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)


def write_json_if_changed(path: Path, payload: dict[str, Any], original: dict[str, Any]) -> bool:
    if payload == original:
        return False
    backup_once(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return True


def patch_config(model_path: Path) -> list[str]:
    changed: list[str] = []

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    config_original = load_json(config_path)
    config = dict(config_original)
    for key, value in (
        ("rope_theta", 1000000),
        ("bos_token_id", 151643),
        ("torch_dtype", "bfloat16"),
    ):
        if config.get(key) != value:
            config[key] = value
    if write_json_if_changed(config_path, config, config_original):
        changed.append(str(config_path))

    tokenizer_path = model_path / "tokenizer_config.json"
    if tokenizer_path.exists():
        tokenizer_original = load_json(tokenizer_path)
        tokenizer_config = dict(tokenizer_original)
        if isinstance(tokenizer_config.get("extra_special_tokens"), list):
            tokenizer_config.pop("extra_special_tokens", None)
        if write_json_if_changed(tokenizer_path, tokenizer_config, tokenizer_original):
            changed.append(str(tokenizer_path))

    return changed


def validate_transformers(model_path: Path) -> dict[str, str]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return {
        "config_class": config.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": str(getattr(tokenizer, "vocab_size", "")),
    }


def smoke_vllm(model_path: Path, args: argparse.Namespace) -> dict[str, str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    outputs = llm.generate(["What is 1+1? Answer:"], SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    text = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
    return {"prompt": "What is 1+1? Answer:", "output_prefix": text[:80]}


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    changed = [] if args.no_patch else patch_config(model_path)
    validation = validate_transformers(model_path) if args.validate else {}
    vllm_result = smoke_vllm(model_path, args) if args.vllm_smoke else {}

    payload = {
        "model_path": str(model_path),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "changed_files": changed,
        "validation": validation,
        "vllm_smoke": vllm_result,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
