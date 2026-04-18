#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from external_math_eval_utils import ensure_dir, slugify
from repo_config import REPO_ROOT, resolve_cache_root, resolve_lm_eval_python, resolve_outputs_root, resolve_tmp_root


LM_EVAL_CLI_SHIM = Path(__file__).with_name("run_lm_eval_cli.py")
DEFAULT_OUTPUT_ROOT = resolve_outputs_root() / "musr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuSR with the portable lm-eval stack.")
    parser.add_argument("--lm-eval-python", default=str(resolve_lm_eval_python()))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default=datetime.now().strftime("musr_%Y%m%d_%H%M%S"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"), required=os.environ.get("MODEL_PATH") is None)
    parser.add_argument("--model-tag", default=os.environ.get("MODEL_TAG"))
    parser.add_argument("--task-name", default="leaderboard_musr")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="vllm")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-gen-toks", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--apply-chat-template", dest="apply_chat_template", action="store_true", default=True)
    parser.add_argument("--no-apply-chat-template", dest="apply_chat_template", action="store_false")
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--extra-model-arg", action="append", default=[])
    parser.add_argument("--extra-cli-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_cache_env(dry_run: bool) -> None:
    cache_root = Path(os.environ.setdefault("XDG_CACHE_HOME", str(resolve_cache_root())))
    hf_home = Path(os.environ.setdefault("HF_HOME", str(cache_root / "hf")))
    hf_datasets_cache = Path(os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets")))
    tmp_root = Path(os.environ.setdefault("TMPDIR", str(resolve_tmp_root())))
    if not dry_run:
        ensure_dir(cache_root)
        ensure_dir(hf_home)
        ensure_dir(hf_datasets_cache)
        ensure_dir(tmp_root)


def build_model_args(args: argparse.Namespace) -> str:
    fragments = [
        f"pretrained={args.model_path}",
        f"dtype={args.dtype}",
        "trust_remote_code=True",
        f"enable_thinking={args.enable_thinking}",
    ]
    if args.backend == "vllm":
        fragments.append(f"gpu_memory_utilization={os.environ.get('GPU_MEMORY_UTILIZATION', '0.90')}")
        fragments.extend(
            [
                f"tensor_parallel_size={os.environ.get('TENSOR_PARALLEL_SIZE', '1')}",
                f"max_model_len={os.environ.get('MAX_MODEL_LEN', '8192')}",
                f"max_num_seqs={os.environ.get('MAX_NUM_SEQS', '16')}",
                f"swap_space={os.environ.get('SWAP_SPACE', '8')}",
            ]
        )
    fragments.extend(args.extra_model_arg)
    return ",".join(fragments)


def build_gen_kwargs(args: argparse.Namespace) -> str:
    do_sample = "true" if args.temperature > 0 else "false"
    return ",".join(
        [
            f"temperature={args.temperature}",
            f"top_p={args.top_p}",
            f"top_k={args.top_k}",
            f"min_p={args.min_p}",
            f"do_sample={do_sample}",
            f"max_gen_toks={args.max_gen_toks}",
        ]
    )


def build_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    cmd = [
        args.lm_eval_python,
        str(LM_EVAL_CLI_SHIM),
        "--model",
        args.backend,
        "--model_args",
        build_model_args(args),
        "--tasks",
        args.task_name,
        "--num_fewshot",
        str(args.num_fewshot),
        "--batch_size",
        str(args.batch_size),
        "--gen_kwargs",
        build_gen_kwargs(args),
        "--seed",
        str(args.seed),
        "--output_path",
        str(run_dir),
        "--log_samples",
    ]
    if args.apply_chat_template:
        cmd.append("--apply_chat_template")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    cmd.extend(args.extra_cli_arg)
    return cmd


def main() -> None:
    args = parse_args()
    configure_cache_env(args.dry_run)

    model_tag = args.model_tag or slugify(Path(args.model_path).name)
    run_root = Path(args.output_root) / args.run_tag
    if not args.dry_run:
        run_root = ensure_dir(run_root)
    run_dir = run_root / model_tag
    if not args.dry_run:
        run_dir = ensure_dir(run_dir)

    cmd = build_command(args, run_dir)
    metadata = {
        "repo_root": str(REPO_ROOT),
        "run_tag": args.run_tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task_name": args.task_name,
        "backend": args.backend,
        "model_path": args.model_path,
        "model_tag": model_tag,
        "batch_size": args.batch_size,
        "num_fewshot": args.num_fewshot,
        "max_gen_toks": args.max_gen_toks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "seed": args.seed,
        "limit": args.limit,
        "apply_chat_template": args.apply_chat_template,
        "enable_thinking": args.enable_thinking,
        "output_dir": str(run_dir),
        "command": shlex.join(cmd),
    }

    print(f"Run root: {run_root}")
    print(shlex.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        metadata_path = run_root / "run_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        print(f"Saved run metadata to {metadata_path}")


if __name__ == "__main__":
    main()
