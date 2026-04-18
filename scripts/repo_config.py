#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, default: Path | str) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser()
    if isinstance(default, Path):
        return default.expanduser()
    return Path(default).expanduser()


def resolve_run_root() -> Path:
    return env_path("RUN_ROOT", REPO_ROOT)


def resolve_outputs_root() -> Path:
    return env_path("OUTPUTS_ROOT", resolve_run_root() / "outputs")


def resolve_cache_root() -> Path:
    return env_path("CACHE_ROOT", resolve_run_root() / ".cache")


def resolve_tmp_root() -> Path:
    return env_path("TMPDIR", resolve_run_root() / "tmp")


def resolve_lm_eval_python() -> Path:
    return env_path("LM_EVAL_PYTHON", REPO_ROOT / ".venvs" / "lm_eval_overlay" / "bin" / "python")


def resolve_task_include_path() -> Path:
    return REPO_ROOT / "lm_eval_tasks"


def resolve_base_model_path() -> Path:
    return env_path("BASE_MODEL_PATH", resolve_run_root() / "models" / "Qwen3-1.7B")


def resolve_gpqa_local_dataset_dir() -> Path:
    return env_path("GPQA_LOCAL_DATASET_DIR", resolve_run_root() / "data" / "gpqa")


def resolve_flash_attn_overlay() -> Path:
    return env_path("FLASH_ATTN_OVERLAY", resolve_run_root() / "vendor" / "flash_attn_overlay")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
