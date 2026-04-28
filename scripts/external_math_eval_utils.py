#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from math_answer_utils import last_boxed_only_string, normalize_final_answer, remove_boxed, strip_string
from repo_config import (
    REPO_ROOT,
    resolve_base_model_path,
    resolve_gpqa_local_dataset_dir,
    resolve_lm_eval_python,
    resolve_outputs_root,
    resolve_run_root,
    resolve_task_include_path,
)


SYSTEM_INSTRUCTION_AUTO = "__AUTO__"
DEFAULT_MATH_SYSTEM_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
RUN_ROOT = resolve_run_root()
DEFAULT_EVAL_ROOT = resolve_outputs_root() / "external_math"
DEFAULT_LM_EVAL_PYTHON = resolve_lm_eval_python()
DEFAULT_BASE_MODEL_PATH = resolve_base_model_path()
DEFAULT_GPQA_LOCAL_DATASET_DIR = resolve_gpqa_local_dataset_dir()
DEFAULT_TASK_INCLUDE_PATH = resolve_task_include_path()


@dataclass(frozen=True)
class BenchmarkSpec:
    slug: str
    display_name: str
    candidate_task_names: tuple[str, ...]
    task_group: str = "math"
    enabled_by_default: bool = True
    default_system_instruction: str | None = DEFAULT_MATH_SYSTEM_INSTRUCTION
    sample_scoring_mode: str = "math"
    local_task_name: str | None = None


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        slug="math500",
        display_name="MATH500",
        candidate_task_names=("hendrycks_math500", "minerva_math500", "math_500", "math500", "math-500"),
    ),
    BenchmarkSpec(
        slug="aime24",
        display_name="AIME24",
        candidate_task_names=("aime24", "aime_2024", "math_aime24", "ext_math_aime24"),
    ),
    BenchmarkSpec(
        slug="aime25",
        display_name="AIME25",
        candidate_task_names=("aime25", "aime_2025", "math_aime25", "ext_math_aime25"),
    ),
    BenchmarkSpec(
        slug="aime24_repeat8",
        display_name="AIME24 repeat8",
        candidate_task_names=("ext_math_aime24_repeat8",),
        enabled_by_default=False,
    ),
    BenchmarkSpec(
        slug="aime25_repeat8",
        display_name="AIME25 repeat8",
        candidate_task_names=("ext_math_aime25_repeat8",),
        enabled_by_default=False,
    ),
    BenchmarkSpec(
        slug="olympiadbench",
        display_name="OlympiadBench",
        candidate_task_names=("ext_math_olympiadbench", "olympiadbench_math_en", "olympiadbench", "olympiad_bench_math"),
    ),
    BenchmarkSpec(
        slug="omni_math",
        display_name="Omni-MATH",
        candidate_task_names=("ext_math_omni_math_rule", "omni_math_rule", "omni_math", "omnimath"),
    ),
    BenchmarkSpec(
        slug="mmlu_pro",
        display_name="MMLU-Pro",
        candidate_task_names=("mmlu_pro",),
        task_group="knowledge",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="gpqa_diamond",
        display_name="GPQA Diamond",
        candidate_task_names=("leaderboard_gpqa_diamond",),
        task_group="science",
        enabled_by_default=True,
        default_system_instruction=None,
        sample_scoring_mode="metric",
        local_task_name="leaderboard_gpqa_diamond_local",
    ),
    BenchmarkSpec(
        slug="musr",
        display_name="MuSR",
        candidate_task_names=("leaderboard_musr",),
        task_group="reasoning",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="ifeval",
        display_name="IFEval",
        candidate_task_names=("leaderboard_ifeval",),
        task_group="instruction",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
)

BENCHMARK_SPECS_BY_SLUG = {spec.slug: spec for spec in BENCHMARK_SPECS}


def benchmark_slugs() -> list[str]:
    return [spec.slug for spec in BENCHMARK_SPECS if spec.enabled_by_default]


def all_benchmark_slugs() -> list[str]:
    return [spec.slug for spec in BENCHMARK_SPECS]


def benchmark_display_name(slug: str) -> str:
    spec = BENCHMARK_SPECS_BY_SLUG.get(slug)
    return spec.display_name if spec else slug


def benchmark_task_candidates(slug: str) -> tuple[str, ...]:
    spec = BENCHMARK_SPECS_BY_SLUG.get(slug)
    if spec is None:
        raise KeyError(f"Unknown benchmark slug: {slug}")
    return spec.candidate_task_names


def benchmark_default_system_instruction(slug: str) -> str | None:
    spec = BENCHMARK_SPECS_BY_SLUG.get(slug)
    return spec.default_system_instruction if spec else DEFAULT_MATH_SYSTEM_INSTRUCTION


def benchmark_sample_scoring_mode(slug: str) -> str:
    spec = BENCHMARK_SPECS_BY_SLUG.get(slug)
    return spec.sample_scoring_mode if spec else "math"


def benchmark_local_task_name(slug: str) -> str | None:
    spec = BENCHMARK_SPECS_BY_SLUG.get(slug)
    return spec.local_task_name if spec else None


def benchmark_slug_from_task_name(task_name: str) -> str | None:
    for spec in BENCHMARK_SPECS:
        for candidate in spec.candidate_task_names:
            if task_name == candidate or task_name.startswith(f"{candidate}_"):
                return spec.slug
        if spec.local_task_name and (task_name == spec.local_task_name or task_name.startswith(f"{spec.local_task_name}_")):
            return spec.slug
    return None


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower())
    return lowered.strip("-") or "unnamed"


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_origin_only_model_path() -> Path | None:
    direct_candidates = [
        RUN_ROOT / "models" / "qwen3-1.7B-origin-only",
        RUN_ROOT / "models" / "qwen3-1.7B-origin_only",
        RUN_ROOT / "models" / "Qwen3-1.7B-origin-only",
        RUN_ROOT / "models" / "Qwen3-1.7B-origin_only",
    ]
    match = find_first_existing(direct_candidates)
    if match is not None:
        return match
    glob_candidates = [*RUN_ROOT.glob("**/*origin*only*/config.json"), *RUN_ROOT.glob("**/*origin*only*/tokenizer_config.json")]
    return None if not glob_candidates else glob_candidates[0].parent


def maybe_boxed(text: str) -> str | None:
    try:
        boxed = last_boxed_only_string(text)
        if boxed is None:
            return None
        return remove_boxed(boxed)
    except Exception:
        return None


def extract_prediction_text(response_text: str) -> tuple[str, bool]:
    raw = str(response_text or "")
    text = strip_json_fence(raw)
    if not text:
        return "", False
    boxed = maybe_boxed(text)
    if boxed:
        return boxed.strip(), True

    regexes = (
        r"(?is)(?:final\s+answer|answer)\s*[:：]\s*(.+?)(?:\n\s*\n|\Z)",
        r"(?is)(?:therefore|thus|so)\s+(?:the\s+)?answer\s+is\s+(.+?)(?:[.\n]|\Z)",
        r"(?is)\$([^$]+)\$",
    )
    for pattern in regexes:
        matches = re.findall(pattern, text)
        if matches:
            candidate = str(matches[-1]).strip()
            if candidate:
                return candidate, True

    non_empty = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty:
        return "", False
    return non_empty[-1], True


def normalize_math_answer(text: Any) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return ""
    boxed = maybe_boxed(candidate)
    if boxed:
        candidate = boxed
    try:
        candidate = normalize_final_answer(candidate)
    except Exception:
        pass
    try:
        candidate = strip_string(candidate)
    except Exception:
        candidate = re.sub(r"\s+", "", candidate)
    return candidate.strip()


def flatten_truths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        out: list[str] = []
        for key in ("final_answer", "answer", "value", "text"):
            if key in value:
                out.extend(flatten_truths(value[key]))
        if out:
            return out
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            out.extend(flatten_truths(item))
        return out
    return [str(value)]


def candidate_ground_truths(doc: dict[str, Any] | None, target: Any) -> list[str]:
    truths: list[str] = []
    if target not in (None, "", []):
        truths.extend(flatten_truths(target))
    if doc:
        for key in ("final_answer", "answer", "answers", "ground_truth", "target", "expected_answer", "solution_key"):
            if key in doc:
                truths.extend(flatten_truths(doc[key]))
    deduped: list[str] = []
    seen: set[str] = set()
    for truth in truths:
        if truth not in seen:
            deduped.append(truth)
            seen.add(truth)
    return deduped


def score_prediction(prediction: str, truths: Iterable[str]) -> bool:
    normalized_pred = normalize_math_answer(prediction)
    if not normalized_pred:
        return False
    for truth in truths:
        normalized_truth = normalize_math_answer(truth)
        if normalized_truth and normalized_pred == normalized_truth:
            return True
    return False


def flatten_logged_response(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "output", "response"):
            if key in value:
                return flatten_logged_response(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        if len(value) == 1:
            return flatten_logged_response(value[0])
        if isinstance(value[0], str):
            return value[0]
        return flatten_logged_response(value[0])
    return str(value)


def split_thinking_content(text: Any) -> tuple[str, str, bool]:
    raw = str(text or "").strip()
    if not raw:
        return "", "", False
    start = raw.find("<think>")
    end = raw.rfind("</think>")
    if start == -1 or end == -1 or end < start:
        return "", raw, False
    thinking = raw[start + len("<think>") : end].strip()
    final_output = raw[end + len("</think>") :].strip()
    return thinking, final_output, True


def _looks_like_numeric_score(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?", candidate))


def extract_logged_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return "" if _looks_like_numeric_score(value) else value
    if isinstance(value, dict):
        for key in ("text", "output", "response"):
            if key in value:
                return extract_logged_text(value[key])
        return ""
    if isinstance(value, (int, float, bool)):
        return ""
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        if len(value) > 1 and all(isinstance(item, (list, tuple)) for item in value):
            return ""
        if len(value) == 1:
            return extract_logged_text(value[0])
        first = value[0]
        if isinstance(first, str):
            return "" if _looks_like_numeric_score(first) else first
        return extract_logged_text(first)
    return ""


def extract_sample_metric(row: dict[str, Any]) -> float | None:
    for key in (
        "exact_match",
        "exact_match,none",
        "acc",
        "acc,none",
        "acc_norm",
        "acc_norm,none",
        "score",
        "score,none",
        "pass@1",
        "pass@1,none",
        "pass_at_1",
    ):
        value = row.get(key)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def guess_sample_files(run_root: Path) -> list[Path]:
    paths = {
        *run_root.glob("**/samples*.jsonl"),
        *run_root.glob("**/samples*.json"),
        *run_root.glob("**/*samples*.jsonl"),
        *run_root.glob("**/*samples*.json"),
    }
    return sorted(paths)


def guess_results_files(run_root: Path) -> list[Path]:
    return sorted(set(run_root.glob("**/results*.json")))


def infer_model_and_benchmark_from_path(path: Path, run_root: Path) -> tuple[str, str]:
    rel = path.relative_to(run_root)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return parts[0], parts[0]
    return "unknown_model", "unknown_benchmark"


def infer_task_name_from_sample_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("samples_"):
        stem = stem[len("samples_") :]
    match = re.match(r"(.+?)_\d{4}-\d{2}-\d{2}T\d{2}[-:]\d{2}[-:]\d{2}", stem)
    return match.group(1) if match else stem
