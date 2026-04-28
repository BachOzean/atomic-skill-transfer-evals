#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from external_math_eval_utils import (
    benchmark_display_name,
    benchmark_sample_scoring_mode,
    benchmark_slug_from_task_name,
    candidate_ground_truths,
    ensure_dir,
    extract_logged_text,
    extract_prediction_text,
    extract_sample_metric,
    flatten_logged_response,
    guess_results_files,
    guess_sample_files,
    infer_model_and_benchmark_from_path,
    infer_task_name_from_sample_path,
    load_json,
    load_jsonl,
    score_prediction,
    split_thinking_content,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize lm-eval benchmark outputs and build per-sample error slices.")
    parser.add_argument("--run-root", required=True, help="Run root produced by run_external_math_eval.py")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to <run-root>/summary")
    return parser.parse_args()


def extract_primary_metric(result_blob: dict[str, Any], task_name: str) -> tuple[str | None, float | None]:
    task_metrics = result_blob.get("results", {}).get(task_name, {})
    preferred_keys = ("exact_match,none", "acc,none", "acc_norm,none", "score,none", "pass@1,none")
    for key in preferred_keys:
        if key in task_metrics:
            return key, float(task_metrics[key])
    for key, value in task_metrics.items():
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def summarize_results(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in guess_results_files(run_root):
        blob = load_json(path)
        model_tag, benchmark_slug = infer_model_and_benchmark_from_path(path, run_root)
        task_names = sorted((blob.get("results") or {}).keys())
        task_name = task_names[0] if task_names else benchmark_slug
        benchmark_slug = benchmark_slug_from_task_name(task_name) or benchmark_slug
        metric_name, metric_value = extract_primary_metric(blob, task_name)
        rows.append(
            {
                "model_tag": model_tag,
                "benchmark_slug": benchmark_slug,
                "benchmark_name": benchmark_display_name(benchmark_slug),
                "task_name": task_name,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "results_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_sample_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    blob = load_json(path)
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for key in ("samples", "rows", "data"):
            value = blob.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported sample file format: {path}")


def pull_prompt(row: dict[str, Any]) -> str:
    arguments = row.get("arguments")
    if isinstance(arguments, dict):
        for key in sorted(arguments):
            value = arguments.get(key)
            if isinstance(value, dict) and isinstance(value.get("arg_0"), str):
                return value["arg_0"]
    if isinstance(arguments, list) and arguments:
        first = arguments[0]
        if isinstance(first, list) and first:
            return str(first[0])
        if isinstance(first, str):
            return first
    return ""


def pull_doc(row: dict[str, Any]) -> dict[str, Any]:
    doc = row.get("doc")
    return doc if isinstance(doc, dict) else {}


def pull_target(row: dict[str, Any]) -> Any:
    if "target" in row:
        return row.get("target")
    if "targets" in row:
        return row.get("targets")
    return None


def build_sample_frame(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in guess_sample_files(run_root):
        model_tag, benchmark_slug = infer_model_and_benchmark_from_path(path, run_root)
        task_name = infer_task_name_from_sample_path(path)
        benchmark_slug = benchmark_slug_from_task_name(task_name) or benchmark_slug
        scoring_mode = benchmark_sample_scoring_mode(benchmark_slug)
        rows = load_sample_rows(path)
        parsed_rows: list[dict[str, Any]] = []
        for row in rows:
            doc = pull_doc(row)
            target = pull_target(row)
            truths = candidate_ground_truths(doc, target)
            raw_output = extract_logged_text(row.get("resps"))
            filtered_output = extract_logged_text(row.get("filtered_resps"))
            thinking_content, final_output, has_think_tags = split_thinking_content(raw_output)
            sample_metric = extract_sample_metric(row)
            if scoring_mode == "math":
                extraction_source = filtered_output or final_output or raw_output or flatten_logged_response(row.get("filtered_resps") or row.get("resps"))
                extracted_prediction, parse_success = extract_prediction_text(extraction_source)
                correct = float(score_prediction(extracted_prediction, truths))
                parse_success_value: float | None = float(parse_success)
            else:
                extracted_prediction = filtered_output or final_output
                parse_success_value = float(bool(extracted_prediction)) if extracted_prediction else None
                correct = sample_metric
            parsed_rows.append(
                {
                    "model_tag": model_tag,
                    "benchmark_slug": benchmark_slug,
                    "benchmark_name": benchmark_display_name(benchmark_slug),
                    "task_name": task_name,
                    "sample_file": str(path),
                    "doc_id": row.get("doc_id"),
                    "prompt": pull_prompt(row),
                    "raw_output": raw_output,
                    "thinking_content": thinking_content,
                    "final_output": final_output,
                    "has_think_tags": float(has_think_tags),
                    "extracted_prediction": extracted_prediction,
                    "parse_success": parse_success_value,
                    "gold_candidates": json.dumps(truths, ensure_ascii=False),
                    "correct": correct,
                }
            )
        if parsed_rows:
            frames.append(pd.DataFrame(parsed_rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def mean_or_none(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float(numeric.mean())
    return None


def avg_length_or_none(series: pd.Series) -> float | None:
    texts = series.fillna("").astype(str)
    non_empty = texts[texts != ""]
    if non_empty.empty:
        return None
    return float(non_empty.map(len).mean())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else run_root / "summary")

    results_df = summarize_results(run_root)
    samples_df = build_sample_frame(run_root)

    if not results_df.empty:
        results_df = results_df.sort_values(["benchmark_slug", "model_tag"]).reset_index(drop=True)
        results_csv = output_dir / "results_summary.csv"
        results_df.to_csv(results_csv, index=False)
    else:
        results_csv = None

    sample_summary_csv = None
    wrong_csv = None
    parse_fail_csv = None
    if not samples_df.empty:
        sample_summary = (
            samples_df.groupby(["benchmark_slug", "benchmark_name", "model_tag"], as_index=False)
            .agg(
                pass_at_1=("correct", mean_or_none),
                parse_success_rate=("parse_success", mean_or_none),
                sample_count=("benchmark_slug", "size"),
                avg_output_chars=("raw_output", avg_length_or_none),
            )
            .sort_values(["benchmark_slug", "model_tag"])
        )
        sample_summary_csv = output_dir / "sample_summary.csv"
        sample_summary.to_csv(sample_summary_csv, index=False)

        wrong = samples_df[(samples_df["parse_success"] == 1.0) & (samples_df["correct"] == 0.0)].copy()
        wrong_csv = output_dir / "wrong_but_parseable.csv"
        wrong.to_csv(wrong_csv, index=False)

        parse_fail = samples_df[samples_df["parse_success"] == 0.0].copy()
        parse_fail_csv = output_dir / "unparseable_outputs.csv"
        parse_fail.to_csv(parse_fail_csv, index=False)

        sample_dump_csv = output_dir / "all_samples.csv"
        samples_df.to_csv(sample_dump_csv, index=False)

    payload = {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "results_summary_csv": str(results_csv) if results_csv else None,
        "sample_summary_csv": str(sample_summary_csv) if sample_summary_csv else None,
        "wrong_but_parseable_csv": str(wrong_csv) if wrong_csv else None,
        "unparseable_outputs_csv": str(parse_fail_csv) if parse_fail_csv else None,
        "result_rows": int(len(results_df)) if not results_df.empty else 0,
        "sample_rows": int(len(samples_df)) if not samples_df.empty else 0,
    }
    with (output_dir / "summary_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
