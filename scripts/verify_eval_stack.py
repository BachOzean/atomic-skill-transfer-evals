#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from repo_config import REPO_ROOT, resolve_lm_eval_python, resolve_task_include_path


EXPECTED_VERSIONS = {
    "lm_eval": "0.4.9.2",
    "transformers": "4.57.1",
    "vllm": "0.17.1",
    "datasets": "4.6.0",
    "flashinfer-python": "0.6.6",
}

REQUIRED_TASKS = {
    "leaderboard_musr",
    "leaderboard_gpqa_diamond_local",
    "ext_math_olympiadbench",
    "ext_math_omni_math_rule",
    "ext_math_matharena_hmmt_feb_2025",
    "ext_math_matharena_hmmt_nov_2025",
    "ext_math_matharena_brumo_2025",
    "ext_math_amc23",
    "ext_math_omni_math_500",
    "hendrycks_math500",
    "medqa_4options",
    "medmcqa",
    "pubmedqa",
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_medicine",
    "mmlu_medical_genetics",
    "mmlu_professional_medicine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the portable eval stack against the expected reference versions.")
    parser.add_argument("--lm-eval-python", default=str(resolve_lm_eval_python()))
    parser.add_argument("--task-include-path", default=str(resolve_task_include_path()))
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--gpqa-local-dataset-dir", default=os.environ.get("GPQA_LOCAL_DATASET_DIR"))
    parser.add_argument("--allow-version-drift", action="store_true")
    return parser.parse_args()


def query_versions(python_bin: str) -> dict[str, str | None]:
    code = """
import importlib.metadata
import json
import sys

packages = sys.argv[1:]
payload = {}
for package in packages:
    try:
        payload[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        payload[package] = None
print(json.dumps(payload))
"""
    proc = subprocess.run([python_bin, "-c", code, *EXPECTED_VERSIONS], check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def list_tasks(python_bin: str, include_path: str) -> set[str]:
    shim = REPO_ROOT / "scripts" / "run_lm_eval_cli.py"
    proc = subprocess.run(
        [python_bin, str(shim), "--tasks", "list", "--include_path", include_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "lm_eval task discovery failed with exit code "
            f"{proc.returncode}: {(proc.stderr or proc.stdout).strip()}"
        )
    tasks: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("tasks"):
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if cells and cells[0] and set(cells[0]) != {"-"}:
                tasks.add(cells[0])
            continue
        if "," in line:
            tasks.update(part.strip() for part in line.split(",") if part.strip())
        else:
            tasks.update(line.split())
    return tasks


def validate_local_path(path_str: str | None, label: str) -> tuple[bool, str | None]:
    if not path_str:
        return True, None
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return False, f"{label} does not exist: {path}"
    if path.is_dir() and label == "base_model_path":
        config_path = path / "config.json"
        if not config_path.exists():
            return False, f"{label} is missing config.json: {path}"
    return True, None


def main() -> None:
    args = parse_args()
    lm_eval_python = Path(args.lm_eval_python).expanduser()
    include_path = Path(args.task_include_path).expanduser().resolve()

    failures: list[str] = []
    if not lm_eval_python.exists():
        failures.append(f"lm_eval_python does not exist: {lm_eval_python}")
    if not include_path.exists():
        failures.append(f"task include path does not exist: {include_path}")

    versions = {}
    visible_tasks: set[str] = set()
    if not failures:
        versions = query_versions(str(lm_eval_python))
        if not args.allow_version_drift:
            for package, expected in EXPECTED_VERSIONS.items():
                actual = versions.get(package)
                if actual != expected:
                    failures.append(f"version mismatch for {package}: expected {expected}, got {actual}")
        visible_tasks = list_tasks(str(lm_eval_python), str(include_path))
        missing_tasks = sorted(REQUIRED_TASKS - visible_tasks)
        if missing_tasks:
            failures.append(f"missing lm-eval tasks: {missing_tasks}")

    ok, message = validate_local_path(args.base_model_path, "base_model_path")
    if not ok and message:
        failures.append(message)
    ok, message = validate_local_path(args.gpqa_local_dataset_dir, "gpqa_local_dataset_dir")
    if not ok and message:
        failures.append(message)

    payload = {
        "repo_root": str(REPO_ROOT),
        "lm_eval_python": str(lm_eval_python),
        "task_include_path": str(include_path),
        "versions": versions,
        "visible_tasks_checked": sorted(REQUIRED_TASKS),
        "status": "ok" if not failures else "failed",
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
