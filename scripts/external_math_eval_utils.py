#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

try:
    import sympy as _sympy
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application as _sympy_implicit_multiplication,
    )
    from sympy.parsing.sympy_parser import parse_expr as _sympy_parse_expr
    from sympy.parsing.sympy_parser import standard_transformations as _sympy_standard_transformations
except Exception:  # pragma: no cover
    _sympy = None
    _sympy_parse_expr = None
    _sympy_standard_transformations = ()
    _sympy_implicit_multiplication = None

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
        slug="aime24_repeat64",
        display_name="AIME24 repeat64",
        candidate_task_names=("ext_math_aime24_repeat64",),
        enabled_by_default=False,
    ),
    BenchmarkSpec(
        slug="aime25_repeat64",
        display_name="AIME25 repeat64",
        candidate_task_names=("ext_math_aime25_repeat64",),
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
        slug="matharena_hmmt_feb_2025",
        display_name="MathArena HMMT Feb 2025",
        candidate_task_names=("ext_math_matharena_hmmt_feb_2025", "matharena_hmmt_feb_2025", "hmmt_feb_2025"),
        enabled_by_default=False,
    ),
    BenchmarkSpec(
        slug="matharena_hmmt_nov_2025",
        display_name="MathArena HMMT Nov 2025",
        candidate_task_names=("ext_math_matharena_hmmt_nov_2025", "matharena_hmmt_nov_2025", "hmmt_nov_2025"),
        enabled_by_default=False,
    ),
    BenchmarkSpec(
        slug="matharena_brumo_2025",
        display_name="MathArena BRUMO 2025",
        candidate_task_names=("ext_math_matharena_brumo_2025", "matharena_brumo_2025", "brumo_2025"),
        enabled_by_default=False,
    ),
    BenchmarkSpec(
        slug="amc23",
        display_name="AMC23",
        candidate_task_names=("ext_math_amc23", "amc23"),
        enabled_by_default=False,
    ),
    BenchmarkSpec(
        slug="omni_math_500",
        display_name="Omni-MATH-500",
        candidate_task_names=("ext_math_omni_math_500", "omni_math_500", "omnimath_500"),
        enabled_by_default=False,
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
        slug="medqa",
        display_name="MedQA-USMLE",
        candidate_task_names=("medqa_4options",),
        task_group="medicine",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="medmcqa",
        display_name="MedMCQA",
        candidate_task_names=("medmcqa",),
        task_group="medicine",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="pubmedqa",
        display_name="PubMedQA",
        candidate_task_names=("pubmedqa",),
        task_group="medicine",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="mmlu_anatomy",
        display_name="MMLU Anatomy",
        candidate_task_names=("mmlu_anatomy",),
        task_group="medicine",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="mmlu_clinical_knowledge",
        display_name="MMLU Clinical Knowledge",
        candidate_task_names=("mmlu_clinical_knowledge",),
        task_group="medicine",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="mmlu_college_medicine",
        display_name="MMLU College Medicine",
        candidate_task_names=("mmlu_college_medicine",),
        task_group="medicine",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="mmlu_medical_genetics",
        display_name="MMLU Medical Genetics",
        candidate_task_names=("mmlu_medical_genetics",),
        task_group="medicine",
        enabled_by_default=False,
        default_system_instruction=None,
        sample_scoring_mode="metric",
    ),
    BenchmarkSpec(
        slug="mmlu_professional_medicine",
        display_name="MMLU Professional Medicine",
        candidate_task_names=("mmlu_professional_medicine",),
        task_group="medicine",
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
        local_task_name="leaderboard_gpqa_diamond_local_gen",
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


_TEXT_COMMANDS = ("text", "mathrm", "textrm", "mathbf", "operatorname", "mbox", "textbf")
_UNIT_WORDS = (
    "degree",
    "degrees",
    "unit",
    "units",
    "inch",
    "inches",
    "foot",
    "feet",
    "cm",
    "m",
    "meter",
    "meters",
    "hour",
    "hours",
    "minute",
    "minutes",
    "way",
    "ways",
    "dollar",
    "dollars",
    "cent",
    "cents",
)


def unwrap_latex_text_commands(text: Any) -> str:
    out = str(text or "")
    previous = None
    while previous != out:
        previous = out
        for command in _TEXT_COMMANDS:
            out = re.sub(r"\\" + command + r"\{([^{}]*)\}", r"\1", out)
    return out


def remove_thousands_commas(text: str) -> str:
    raw = str(text or "")
    out = raw.replace("\\!", "")
    if "\\!" in raw or re.fullmatch(r"\$?[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", out.strip()):
        return re.sub(r"(?<=\d),(?=\d{3}\b)", "", out)
    return out


def normalize_base_subscripts(text: str) -> str:
    return re.sub(r"(?<=\d)_\{?\d+\}?", "", text)


def strip_plain_units(text: str) -> str:
    unit_pattern = "|".join(re.escape(unit) for unit in _UNIT_WORDS)
    out = re.sub(r"\^\{?(?:st|nd|rd|th)\}?", "", text, flags=re.IGNORECASE)
    out = re.sub(rf"(?i)\b(?:{unit_pattern})\b(?:\^\{{?\d+\}}?)?", "", out)
    return out


def canonical_math_answer(text: Any) -> str:
    candidate = str(text or "").strip()
    boxed = maybe_boxed(candidate)
    if boxed:
        candidate = boxed
    candidate = unwrap_latex_text_commands(candidate)
    candidate = remove_thousands_commas(candidate)
    candidate = normalize_base_subscripts(candidate)
    candidate = strip_plain_units(candidate)
    candidate = candidate.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    candidate = re.sub(r"\\frac\{([^{}]+)\}([A-Za-z0-9]+)", r"\\frac{\1}{\2}", candidate)
    candidate = candidate.replace("\\left", "").replace("\\right", "")
    candidate = candidate.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    candidate = candidate.replace("\\,", "").replace("\\!", "").replace("{,}", ",")
    candidate = candidate.strip("$").strip()
    if not re.fullmatch(r"[+-]?\.\d+", candidate):
        candidate = candidate.rstrip(".")
    try:
        candidate = strip_string(candidate)
    except Exception:
        candidate = re.sub(r"\s+", "", candidate)
    return candidate.strip()


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
        else:
            current.append(char)
    item = "".join(current).strip()
    if item:
        parts.append(item)
    return parts


def vector_items(text: Any) -> list[str] | None:
    raw = unwrap_latex_text_commands(str(text or "").strip())
    matrix_match = re.search(r"\\begin\{[bpv]?matrix\}(.+?)\\end\{[bpv]?matrix\}", raw, flags=re.DOTALL)
    if matrix_match:
        parts = [part.strip() for part in re.split(r"\\\\|&", matrix_match.group(1)) if part.strip()]
        return parts if len(parts) > 1 else None

    compact = remove_thousands_commas(raw)
    if "\\infty" in compact or "\\cup" in compact or "\\cap" in compact:
        return None
    if compact.startswith("[") and compact.endswith("]"):
        parts = split_top_level_commas(compact[1:-1])
        return parts if len(parts) > 1 else None
    if compact.startswith("(") and compact.endswith(")") and "," in compact:
        parts = split_top_level_commas(compact[1:-1])
        return parts if len(parts) > 1 else None
    if "," in compact and not re.search(r"[\[\]\(\)]", compact):
        parts = split_top_level_commas(compact)
        return parts if len(parts) > 1 else None
    return None


def has_plus_minus(text: Any) -> bool:
    candidate = str(text or "")
    return bool(re.search(r"\\(?:pm|mp)(?![A-Za-z])", candidate))


def replace_latex_frac(text: str) -> str | None:
    out: list[str] = []
    index = 0
    while index < len(text):
        start = text.find("\\frac", index)
        if start == -1:
            out.append(text[index:])
            break
        out.append(text[index:start])
        pos = start + len("\\frac")
        if pos >= len(text):
            return None

        def read_group(offset: int) -> tuple[str, int] | None:
            if offset >= len(text):
                return None
            if text[offset] != "{":
                return text[offset], offset + 1
            depth = 0
            for cursor in range(offset, len(text)):
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[offset + 1 : cursor], cursor + 1
            return None

        numerator = read_group(pos)
        if numerator is None:
            return None
        denominator = read_group(numerator[1])
        if denominator is None:
            return None
        out.append(f"(({numerator[0]})/({denominator[0]}))")
        index = denominator[1]
    return "".join(out)


def latexish_to_sympy_expr(text: str):
    if _sympy is None or _sympy_parse_expr is None or _sympy_implicit_multiplication is None:
        return None
    candidate = str(text or "").strip()
    if not candidate or len(candidate) > 200 or has_plus_minus(candidate):
        return None
    if any(
        token in candidate
        for token in (
            "\\cup",
            "\\cap",
            "\\infty",
            "\\le",
            "\\ge",
            "\\neq",
            "\\equiv",
            "\\pmod",
            "\\mod",
            "\\not",
            "<",
            ">",
            "=",
        )
    ):
        return None
    if "\\begin" in candidate or "\\end" in candidate:
        return None

    candidate = replace_latex_frac(candidate)
    if candidate is None:
        return None
    previous = None
    while previous != candidate:
        previous = candidate
        candidate = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", candidate)
    candidate = candidate.replace("\\sqrt", "sqrt")
    candidate = candidate.replace("\\pi", "pi")
    candidate = candidate.replace("^", "**")
    candidate = candidate.replace("{", "(").replace("}", ")")
    candidate = candidate.replace("\\", "")
    candidate = candidate.replace(",", "")
    if "__" in candidate or not re.fullmatch(r"[0-9A-Za-z_+\-*/(). ]+", candidate):
        return None
    if not re.search(r"[+\-*/()]|sqrt|pi", candidate):
        return None
    try:
        return _sympy_parse_expr(
            candidate,
            transformations=_sympy_standard_transformations + (_sympy_implicit_multiplication,),
            evaluate=True,
        )
    except Exception:
        return None


def safe_symbolic_equiv(prediction: str, truth: str) -> bool:
    if _sympy is None:
        return False
    left = latexish_to_sympy_expr(prediction)
    right = latexish_to_sympy_expr(truth)
    if left is None or right is None:
        return False
    try:
        return bool(_sympy.simplify(left - right) == 0)
    except Exception:
        return False


def simple_numeric_value(text: str) -> Fraction | None:
    candidate = str(text or "").strip()
    frac_match = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", candidate)
    if frac_match:
        denominator = int(frac_match.group(3))
        if denominator == 0:
            return None
        numerator = int(frac_match.group(2))
        if frac_match.group(1) == "-":
            numerator *= -1
        return Fraction(numerator, denominator)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", candidate):
        try:
            return Fraction(candidate)
        except Exception:
            return None
    return None


def math_answers_equivalent(prediction: Any, truth: Any) -> bool:
    normalized_pred = canonical_math_answer(prediction)
    normalized_truth = canonical_math_answer(truth)
    if normalized_pred and normalized_truth and normalized_pred == normalized_truth:
        return True
    if has_plus_minus(normalized_pred) or has_plus_minus(normalized_truth):
        return False
    if (
        normalized_pred
        and normalized_truth
        and re.fullmatch(r"[A-Za-z]+", normalized_pred)
        and re.fullmatch(r"[A-Za-z]+", normalized_truth)
        and normalized_pred.lower() == normalized_truth.lower()
    ):
        return True

    pred_items = vector_items(prediction)
    truth_items = vector_items(truth)
    if pred_items is not None or truth_items is not None:
        if pred_items is None or truth_items is None or len(pred_items) != len(truth_items):
            return False
        return all(math_answers_equivalent(pred_item, truth_item) for pred_item, truth_item in zip(pred_items, truth_items))

    pred_number = simple_numeric_value(normalized_pred)
    truth_number = simple_numeric_value(normalized_truth)
    if pred_number is not None or truth_number is not None:
        return pred_number is not None and truth_number is not None and pred_number == truth_number

    return safe_symbolic_equiv(normalized_pred or str(prediction or ""), normalized_truth or str(truth or ""))


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
    if not canonical_math_answer(prediction):
        return False
    for truth in truths:
        if canonical_math_answer(truth) and math_answers_equivalent(prediction, truth):
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
