#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from external_math_eval_utils import (
    BENCHMARK_SPECS_BY_SLUG,
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_EVAL_ROOT,
    DEFAULT_GPQA_LOCAL_DATASET_DIR,
    DEFAULT_LM_EVAL_PYTHON,
    DEFAULT_MATH_SYSTEM_INSTRUCTION,
    DEFAULT_TASK_INCLUDE_PATH,
    SYSTEM_INSTRUCTION_AUTO,
    all_benchmark_slugs,
    benchmark_default_system_instruction,
    benchmark_local_task_name,
    benchmark_slugs,
    benchmark_task_candidates,
    discover_origin_only_model_path,
    ensure_dir,
    parse_csv_arg,
    slugify,
)
from repo_config import REPO_ROOT, resolve_cache_root, resolve_tmp_root


MODEL_ORDER = ("base", "origin_only")
LM_EVAL_CLI_SHIM = Path(__file__).with_name("run_lm_eval_cli.py")
GPQA_HF_DATASET = ("Idavidrein/gpqa", "gpqa_diamond")
GPQA_LOCAL_METADATA_KEY = "gpqa_local_dataset_dir"
GPQA_LOCAL_DATASET_DEFAULT = str(DEFAULT_GPQA_LOCAL_DATASET_DIR) if DEFAULT_GPQA_LOCAL_DATASET_DIR.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external_math benchmarks with a portable lm-eval stack.")
    parser.add_argument("--lm-eval-python", default=str(DEFAULT_LM_EVAL_PYTHON), help="Python interpreter inside the lm-eval overlay.")
    parser.add_argument("--output-root", default=str(DEFAULT_EVAL_ROOT), help="Root directory for benchmark outputs.")
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Run tag under the output root.")
    parser.add_argument("--models", default="base", help="Comma-separated model tags to run. Supported: base,origin_only.")
    parser.add_argument("--benchmarks", default=",".join(benchmark_slugs()), help=f"Comma-separated benchmark slugs. All available: {','.join(all_benchmark_slugs())}.")
    parser.add_argument("--base-model-path", default=str(DEFAULT_BASE_MODEL_PATH), help="HF-loadable base model path.")
    parser.add_argument("--origin-only-model-path", default=os.environ.get("ORIGIN_ONLY_MODEL_PATH"), help="HF-loadable origin-only model path.")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="vllm", help="lm-eval model backend to use.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype passed to lm-eval model_args.")
    parser.add_argument("--batch-size", default="auto", help="lm-eval batch size. Use a number or 'auto'.")
    parser.add_argument("--max-gen-toks", type=int, default=38912, help="Maximum generation tokens per sample.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for generation.")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot count for lm-eval tasks.")
    parser.add_argument("--apply-chat-template", dest="apply_chat_template", action="store_true", default=True, help="Apply the tokenizer chat template.")
    parser.add_argument("--no-apply-chat-template", dest="apply_chat_template", action="store_false", help="Disable chat templating.")
    parser.add_argument("--chat-template", default=None, help="Optional chat template name to pass through to lm-eval.")
    parser.add_argument(
        "--system-instruction",
        default=SYSTEM_INSTRUCTION_AUTO,
        help=f"System instruction passed to lm-eval. Defaults to automatic per-benchmark behavior (math uses '{DEFAULT_MATH_SYSTEM_INSTRUCTION}', non-math uses no extra instruction).",
    )
    parser.add_argument("--no-system-instruction", dest="system_instruction", action="store_const", const="", help="Disable the automatic system instruction for all benchmarks.")
    parser.add_argument("--fewshot-as-multiturn", action="store_true", help="Render few-shot examples as a multi-turn conversation.")
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True, help="Enable Qwen3 thinking tags.")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false", help="Disable Qwen3 thinking tags.")
    parser.add_argument("--limit", type=float, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--task-override", action="append", default=[], metavar="BENCHMARK=TASK", help="Override resolved lm-eval task name for a benchmark slug.")
    parser.add_argument("--include-path", action="append", default=[str(DEFAULT_TASK_INCLUDE_PATH)], help="Additional lm-eval task include paths.")
    parser.add_argument("--extra-model-arg", action="append", default=[], help="Extra key=value fragments appended to model_args.")
    parser.add_argument("--extra-cli-arg", action="append", default=[], help="Extra raw CLI arguments appended after the standard lm-eval flags.")
    parser.add_argument(
        "--gpqa-local-dataset-dir",
        default=os.environ.get("GPQA_LOCAL_DATASET_DIR", GPQA_LOCAL_DATASET_DEFAULT),
        help="Optional local dataset path for GPQA Diamond.",
    )
    parser.add_argument("--device", default=None, help="Optional device identifier for the hf backend.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--list-tasks", action="store_true", help="List visible lm-eval tasks and exit.")
    return parser.parse_args()


def parse_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"Invalid override: {item}. Expected BENCHMARK=TASK.")
        overrides[key.strip()] = value.strip()
    return overrides


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


def list_tasks(python_bin: str, include_paths: list[str]) -> set[str]:
    cmd = [python_bin, str(LM_EVAL_CLI_SHIM), "--tasks", "list"]
    for include_path in include_paths:
        cmd.extend(["--include_path", include_path])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return set()
    visible_tasks: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("tasks"):
            continue
        for part in re_split_task_line(line):
            visible_tasks.add(part)
    return visible_tasks


def prepare_include_paths(include_paths: list[str]) -> list[str]:
    prepared: list[str] = []
    seen: set[str] = set()
    for raw_path in include_paths:
        if not raw_path:
            continue
        include_path = str(Path(raw_path).expanduser().resolve())
        if not Path(include_path).exists():
            raise FileNotFoundError(f"Include path does not exist: {include_path}")
        if include_path not in seen:
            prepared.append(include_path)
            seen.add(include_path)
    return prepared


def prepare_optional_dataset_dir(raw_path: str | None, label: str) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir() and not path.is_file():
        raise ValueError(f"{label} must be a directory or file: {path}")
    return str(path)


def re_split_task_line(line: str) -> list[str]:
    if line.startswith("|"):
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells and cells[0] and set(cells[0]) != {"-"}:
            return [cells[0]]
    parts = [piece.strip() for piece in line.split(",") if piece.strip()]
    return parts if len(parts) > 1 else line.split()


def resolve_task_name(benchmark_slug: str, overrides: dict[str, str], visible_tasks: set[str]) -> str:
    if benchmark_slug in overrides:
        return overrides[benchmark_slug]
    for candidate in benchmark_task_candidates(benchmark_slug):
        if candidate in visible_tasks:
            return candidate
    return benchmark_task_candidates(benchmark_slug)[0]


def resolve_system_instruction(args: argparse.Namespace, benchmark_slug: str) -> str | None:
    if args.system_instruction == SYSTEM_INSTRUCTION_AUTO:
        return benchmark_default_system_instruction(benchmark_slug)
    if args.system_instruction == "":
        return None
    return args.system_instruction


def resolve_task_name_and_metadata(
    args: argparse.Namespace,
    benchmark_slug: str,
    overrides: dict[str, str],
    visible_tasks: set[str],
) -> tuple[str, dict[str, str]]:
    task_name = resolve_task_name(benchmark_slug, overrides, visible_tasks)
    metadata: dict[str, str] = {}
    if benchmark_slug == "gpqa_diamond" and args.gpqa_local_dataset_dir:
        local_task_name = benchmark_local_task_name(benchmark_slug)
        if not local_task_name:
            raise ValueError("GPQA local dataset override requested, but no local task is registered for gpqa_diamond.")
        task_name = local_task_name
        metadata[GPQA_LOCAL_METADATA_KEY] = args.gpqa_local_dataset_dir
    return task_name, metadata


def verify_gpqa_hf_access(python_bin: str) -> None:
    dataset_path, dataset_name = GPQA_HF_DATASET
    code = """
from datasets import load_dataset
import sys
dataset_path, dataset_name = sys.argv[1], sys.argv[2]
load_dataset(dataset_path, dataset_name, split="train[:1]")
"""
    proc = subprocess.run([python_bin, "-c", code, dataset_path, dataset_name], capture_output=True, text=True, timeout=60)
    if proc.returncode == 0:
        return
    details = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(
        "GPQA Diamond requires authenticated HF access to Idavidrein/gpqa unless "
        f"--gpqa-local-dataset-dir is provided. Probe failed with: {details}"
    )


def resolve_models(args: argparse.Namespace) -> dict[str, Path]:
    requested = parse_csv_arg(args.models)
    unknown = sorted(set(requested) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown model tags: {unknown}. Supported: {list(MODEL_ORDER)}")

    model_paths = {"base": Path(args.base_model_path), "origin_only": None}
    if "origin_only" in requested:
        origin_path = Path(args.origin_only_model_path) if args.origin_only_model_path else discover_origin_only_model_path()
        model_paths["origin_only"] = origin_path

    resolved: dict[str, Path] = {}
    for tag in requested:
        path = model_paths.get(tag)
        if path is None:
            raise FileNotFoundError("Could not resolve origin-only model path. Pass --origin-only-model-path or set ORIGIN_ONLY_MODEL_PATH.")
        resolved[tag] = path
    return resolved


def build_model_args(args: argparse.Namespace, model_path: Path) -> str:
    fragments = [
        f"pretrained={model_path}",
        f"dtype={args.dtype}",
        "trust_remote_code=True",
        f"enable_thinking={args.enable_thinking}",
    ]
    if args.device and args.backend == "hf":
        fragments.append(f"device={args.device}")
    if args.backend == "vllm":
        fragments.append(f"gpu_memory_utilization={os.environ.get('GPU_MEMORY_UTILIZATION', '0.90')}")
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


def build_command(
    args: argparse.Namespace,
    model_path: Path,
    benchmark_slug: str,
    task_name: str,
    benchmark_out_dir: Path,
    task_metadata: dict[str, str],
) -> list[str]:
    cmd = [
        args.lm_eval_python,
        str(LM_EVAL_CLI_SHIM),
        "--model",
        args.backend,
        "--model_args",
        build_model_args(args, model_path),
        "--tasks",
        task_name,
        "--num_fewshot",
        str(args.num_fewshot),
        "--batch_size",
        str(args.batch_size),
        "--gen_kwargs",
        build_gen_kwargs(args),
        "--seed",
        str(args.seed),
        "--output_path",
        str(benchmark_out_dir),
        "--log_samples",
    ]
    system_instruction = resolve_system_instruction(args, benchmark_slug)
    if system_instruction:
        cmd.extend(["--system_instruction", system_instruction])
    if args.apply_chat_template:
        cmd.append("--apply_chat_template")
        if args.chat_template:
            cmd.append(args.chat_template)
    if args.fewshot_as_multiturn:
        cmd.append("--fewshot_as_multiturn")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    for include_path in args.include_path:
        if include_path:
            cmd.extend(["--include_path", include_path])
    if task_metadata:
        cmd.extend(["--metadata", json.dumps(task_metadata)])
    cmd.extend(args.extra_cli_arg)
    return cmd


def main() -> None:
    args = parse_args()
    if args.chat_template and not args.apply_chat_template:
        raise ValueError("--chat-template requires --apply-chat-template.")

    configure_cache_env(args.dry_run)
    raw_include_paths = list(args.include_path)
    args.include_path = prepare_include_paths(args.include_path)

    run_root = Path(args.output_root) / args.run_tag
    if not args.dry_run:
        run_root = ensure_dir(run_root)
    models = resolve_models(args)
    benchmarks = parse_csv_arg(args.benchmarks)
    unknown_benchmarks = sorted(set(benchmarks) - set(BENCHMARK_SPECS_BY_SLUG))
    if unknown_benchmarks:
        raise ValueError(f"Unknown benchmark slugs: {unknown_benchmarks}")
    if "gpqa_diamond" in benchmarks:
        args.gpqa_local_dataset_dir = prepare_optional_dataset_dir(args.gpqa_local_dataset_dir, "GPQA local dataset dir")
    else:
        args.gpqa_local_dataset_dir = None

    overrides = parse_overrides(args.task_override)
    if args.list_tasks:
        visible_tasks = list_tasks(args.lm_eval_python, args.include_path)
        for task_name in sorted(visible_tasks):
            print(task_name)
        return

    visible_tasks = list_tasks(args.lm_eval_python, args.include_path)
    resolved_task_info = {slug: resolve_task_name_and_metadata(args, slug, overrides, visible_tasks) for slug in benchmarks}
    resolved_tasks = {slug: task_name for slug, (task_name, _metadata) in resolved_task_info.items()}
    if not args.dry_run and "gpqa_diamond" in benchmarks and not args.gpqa_local_dataset_dir:
        verify_gpqa_hf_access(args.lm_eval_python)

    metadata = {
        "repo_root": str(REPO_ROOT),
        "run_tag": args.run_tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lm_eval_python": args.lm_eval_python,
        "backend": args.backend,
        "models": {tag: str(path) for tag, path in models.items()},
        "benchmarks": benchmarks,
        "resolved_tasks": resolved_tasks,
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
        "chat_template": args.chat_template,
        "system_instruction": args.system_instruction,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "enable_thinking": args.enable_thinking,
        "gpqa_local_dataset_dir": args.gpqa_local_dataset_dir,
        "original_include_paths": raw_include_paths,
        "include_paths": args.include_path,
        "extra_model_arg": args.extra_model_arg,
        "extra_cli_arg": args.extra_cli_arg,
        "commands": [],
    }

    print(f"Run root: {run_root}")
    for model_tag, model_path in models.items():
        model_out_dir = run_root / slugify(model_tag)
        if not args.dry_run:
            model_out_dir = ensure_dir(model_out_dir)
        for benchmark_slug in benchmarks:
            benchmark_out_dir = model_out_dir / benchmark_slug
            if not args.dry_run:
                benchmark_out_dir = ensure_dir(benchmark_out_dir)
            task_name, task_metadata = resolved_task_info[benchmark_slug]
            cmd = build_command(args, model_path, benchmark_slug, task_name, benchmark_out_dir, task_metadata)
            metadata["commands"].append(
                {
                    "model_tag": model_tag,
                    "model_path": str(model_path),
                    "benchmark_slug": benchmark_slug,
                    "task_name": task_name,
                    "task_metadata": task_metadata,
                    "system_instruction": resolve_system_instruction(args, benchmark_slug),
                    "output_dir": str(benchmark_out_dir),
                    "command": shlex.join(cmd),
                }
            )
            print(f"[run] {model_tag} :: {benchmark_slug} -> {task_name}")
            print(shlex.join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)

    if not args.dry_run:
        metadata_path = run_root / "run_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        print(f"Saved run metadata to {metadata_path}")


if __name__ == "__main__":
    main()
