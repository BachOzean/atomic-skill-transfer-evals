#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from external_math_eval_utils import (
    candidate_ground_truths,
    ensure_dir,
    extract_logged_text,
    extract_prediction_text,
    extract_sample_metric,
    flatten_logged_response,
    infer_model_and_benchmark_from_path,
    score_prediction,
    split_thinking_content,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score saved MATH500 generations without re-running inference.")
    parser.add_argument("--run-root", required=True, help="Run root containing saved lm-eval samples.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <run-root>/summary/math500_rescore.")
    parser.add_argument("--report-doc", default=None, help="Optional markdown report path.")
    return parser.parse_args()


def load_sample_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for key in ("samples", "rows", "data"):
            value = blob.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported sample file format: {path}")


def pull_doc(row: dict[str, Any]) -> dict[str, Any]:
    doc = row.get("doc")
    return doc if isinstance(doc, dict) else {}


def pull_target(row: dict[str, Any]) -> Any:
    if "target" in row:
        return row.get("target")
    if "targets" in row:
        return row.get("targets")
    return None


def answer_extraction_window(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    boxed_idx = text.rfind("\\boxed")
    if boxed_idx != -1:
        return text[max(0, boxed_idx - 2000) : boxed_idx + max_chars]
    tail = text[-max_chars:]
    if "</think>" not in text:
        return tail
    final_after_think = text.rsplit("</think>", 1)[-1].strip()
    if final_after_think:
        return final_after_think[-max_chars:]
    return tail


def find_math500_sample_files(run_root: Path) -> list[Path]:
    paths = {
        *run_root.glob("**/math500/**/samples*.jsonl"),
        *run_root.glob("**/math500/**/samples*.json"),
    }
    return sorted(path for path in paths if not path.name.startswith("run_metadata"))


def rescore_sample_file(path: Path, run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_tag, benchmark_slug = infer_model_and_benchmark_from_path(path, run_root)
    rows = load_sample_rows(path)
    detail_rows: list[dict[str, Any]] = []
    old_correct = 0.0
    new_correct = 0.0
    old_count = 0
    rescued = 0
    regressions = 0

    for row in rows:
        doc = pull_doc(row)
        target = pull_target(row)
        truths = candidate_ground_truths(doc, target)
        raw_output = extract_logged_text(row.get("resps"))
        filtered_output = extract_logged_text(row.get("filtered_resps"))
        _thinking, final_output, _has_think_tags = split_thinking_content(raw_output)
        extraction_source = filtered_output or final_output or raw_output or flatten_logged_response(row.get("filtered_resps") or row.get("resps"))
        extraction_source = answer_extraction_window(extraction_source)
        extracted_prediction, parse_success = extract_prediction_text(extraction_source)
        old_value = extract_sample_metric(row)
        new_value = float(score_prediction(extracted_prediction, truths))

        if old_value is not None:
            old_correct += float(old_value)
            old_count += 1
        new_correct += new_value
        if old_value == 0.0 and new_value == 1.0:
            rescued += 1
        if old_value == 1.0 and new_value == 0.0:
            regressions += 1

        detail_rows.append(
            {
                "model_tag": model_tag,
                "benchmark_slug": benchmark_slug,
                "doc_id": row.get("doc_id"),
                "old_correct": old_value,
                "rescored_correct": new_value,
                "parse_success": float(parse_success),
                "target": target,
                "gold_candidates": json.dumps(truths, ensure_ascii=False),
                "extracted_prediction": extracted_prediction,
                "sample_file": str(path),
                "unique_id": doc.get("unique_id"),
                "subject": doc.get("subject"),
                "level": doc.get("level"),
            }
        )

    sample_count = len(rows)
    summary = {
        "model_tag": model_tag,
        "benchmark_slug": benchmark_slug,
        "sample_count": sample_count,
        "old_exact_match": old_correct / old_count if old_count else None,
        "rescored_exact_match": new_correct / sample_count if sample_count else None,
        "rescued_false_negatives": rescued,
        "new_regressions": regressions,
        "sample_file": str(path),
    }
    if summary["old_exact_match"] is not None and summary["rescored_exact_match"] is not None:
        summary["delta"] = summary["rescored_exact_match"] - summary["old_exact_match"]
    else:
        summary["delta"] = None
    return summary, detail_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, run_root: Path, summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    top_rows = sorted(summary_rows, key=lambda row: (row.get("rescored_exact_match") or 0.0), reverse=True)
    family_best: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        model_tag = str(row["model_tag"])
        family = model_tag.split("_", 1)[0]
        if model_tag.startswith("base_"):
            family = "base"
        current = family_best.get(family)
        if current is None or (row.get("rescored_exact_match") or 0.0) > (current.get("rescored_exact_match") or 0.0):
            family_best[family] = row
    total_rescued = sum(int(row["rescued_false_negatives"]) for row in summary_rows)
    total_regressions = sum(int(row["new_regressions"]) for row in summary_rows)
    lines = [
        "# MATH500 Saved-Generation False Negative Rescore",
        "",
        f"- run_root: `{run_root}`",
        f"- updated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- method: reuse saved `samples_*.jsonl`; no model inference. Added normalization for LaTeX text wrappers, omitted base subscripts, units/ordinals, thousands separators, vector/matrix forms, plus conservative SymPy equivalence for short algebraic/radical expressions.",
        f"- validation: rescued `{total_rescued}` old false negatives across `{len(summary_rows)}` sample files; old-correct regressions after the fix: `{total_regressions}`.",
        "",
        "## Family Best",
        "",
        "| family | best model | old exact | rescored exact | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for family in sorted(family_best):
        row = family_best[family]
        old = row.get("old_exact_match")
        new = row.get("rescored_exact_match")
        delta = row.get("delta")
        lines.append(f"| `{family}` | `{row['model_tag']}` | {old * 100:.2f} | {new * 100:.2f} | {delta * 100:.2f} |")
    lines.extend(
        [
            "",
            "## All Checkpoints",
        "",
        "| model | old exact | rescored exact | delta | rescued | regressions |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_rows:
        old = row.get("old_exact_match")
        new = row.get("rescored_exact_match")
        delta = row.get("delta")
        lines.append(
            f"| `{row['model_tag']}` | "
            f"{old * 100:.2f} | {new * 100:.2f} | {delta * 100:.2f} | "
            f"{row['rescued_false_negatives']} | {row['new_regressions']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{output_dir / 'math500_rescore_summary.csv'}`",
            f"- Rescued false negatives: `{output_dir / 'math500_rescued_false_negatives.csv'}`",
            f"- All rows: `{output_dir / 'math500_rescore_rows.csv'}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else run_root / "summary" / "math500_rescore")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for sample_file in find_math500_sample_files(run_root):
        summary, details = rescore_sample_file(sample_file, run_root)
        summary_rows.append(summary)
        detail_rows.extend(details)

    summary_rows = sorted(summary_rows, key=lambda row: row["model_tag"])
    detail_rows = sorted(detail_rows, key=lambda row: (row["model_tag"], int(row["doc_id"]) if row["doc_id"] is not None else -1))
    rescued_rows = [row for row in detail_rows if row["old_correct"] == 0.0 and row["rescored_correct"] == 1.0]

    write_csv(
        output_dir / "math500_rescore_summary.csv",
        summary_rows,
        ["model_tag", "benchmark_slug", "sample_count", "old_exact_match", "rescored_exact_match", "delta", "rescued_false_negatives", "new_regressions", "sample_file"],
    )
    write_csv(
        output_dir / "math500_rescore_rows.csv",
        detail_rows,
        ["model_tag", "benchmark_slug", "doc_id", "old_correct", "rescored_correct", "parse_success", "target", "gold_candidates", "extracted_prediction", "sample_file", "unique_id", "subject", "level"],
    )
    write_csv(
        output_dir / "math500_rescued_false_negatives.csv",
        rescued_rows,
        ["model_tag", "benchmark_slug", "doc_id", "old_correct", "rescored_correct", "parse_success", "target", "gold_candidates", "extracted_prediction", "sample_file", "unique_id", "subject", "level"],
    )

    if args.report_doc:
        write_report(Path(args.report_doc), run_root, summary_rows, output_dir)

    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "output_dir": str(output_dir),
                "summary_csv": str(output_dir / "math500_rescore_summary.csv"),
                "rescued_csv": str(output_dir / "math500_rescued_false_negatives.csv"),
                "detail_csv": str(output_dir / "math500_rescore_rows.csv"),
                "sample_files": len(summary_rows),
                "rescued_rows": len(rescued_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
