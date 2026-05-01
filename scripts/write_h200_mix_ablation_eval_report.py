#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


OFFICIAL_QWEN_REF = "https://qwen.readthedocs.io/en/v3.0/getting_started/quickstart.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the H200 mix/sub/base/ablation evaluation report.")
    parser.add_argument("--run-root", required=True, help="Evaluation run root.")
    parser.add_argument(
        "--output-doc",
        default="/root/atomic-skill-transfer-evals/docs/2026-04-29_h200_mix_ablation_eval_report_zh.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("**/results*.json")):
        try:
            blob = load_json(path)
        except Exception:
            continue
        rel = path.relative_to(run_root)
        model_tag = rel.parts[0] if len(rel.parts) >= 1 else "unknown"
        benchmark = rel.parts[1] if len(rel.parts) >= 2 else "unknown"
        for task_name, metrics in (blob.get("results") or {}).items():
            if not isinstance(metrics, dict):
                continue
            for raw_name, value in metrics.items():
                if not isinstance(value, (int, float)) or raw_name.endswith("_stderr,none"):
                    continue
                rows.append(
                    {
                        "model_tag": model_tag,
                        "benchmark": benchmark,
                        "task_name": task_name,
                        "metric": raw_name.replace(",none", ""),
                        "value": float(value),
                        "results_path": str(path),
                    }
                )
    return rows


def classify_model(tag: str) -> str:
    if tag == "base_qwen3_1_7b" or tag == "qwen3_1_7b":
        return "base"
    if tag.startswith("mix_"):
        return "mix"
    if tag.startswith("sub_"):
        return "sub_only"
    if tag.startswith("origin_") or tag == "origin_only":
        return "origin_only"
    if tag.startswith("ablation_"):
        return "ablation"
    if "_diag_" in tag:
        return "diagnostic"
    return "other"


def step_from_tag(tag: str) -> str:
    for part in tag.split("_"):
        if part.isdigit():
            return part
    return "-"


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def build_metric_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for row in rows:
        out[(row["model_tag"], row["benchmark"], row["metric"])] = row["value"]
    return out


def table_for(rows: list[dict[str, Any]], benchmarks: list[str], metrics: list[str]) -> list[str]:
    metric_map = build_metric_map(rows)
    model_tags = sorted({row["model_tag"] for row in rows if row["benchmark"] in benchmarks})
    lines = ["| family | model | step | " + " | ".join(f"{b}:{m}" for b in benchmarks for m in metrics) + " |"]
    lines.append("|---|---|---|" + "|".join(["---"] * (len(benchmarks) * len(metrics))) + "|")
    for tag in model_tags:
        values = [fmt(metric_map.get((tag, benchmark, metric))) for benchmark in benchmarks for metric in metrics]
        lines.append(f"| {classify_model(tag)} | `{tag}` | {step_from_tag(tag)} | " + " | ".join(values) + " |")
    return lines


def best_by_family(rows: list[dict[str, Any]], benchmark: str, metric: str) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for row in rows:
        if row["benchmark"] != benchmark or row["metric"] != metric:
            continue
        family = classify_model(row["model_tag"])
        current = best.get(family)
        if current is None or row["value"] > current[1]:
            best[family] = (row["model_tag"], row["value"])
    return best


def sample_paths(run_root: Path) -> tuple[int, int, str, str]:
    sample_files = sorted([*run_root.glob("**/samples*.jsonl"), *run_root.glob("**/samples*.json")])
    result_files = sorted(run_root.glob("**/results*.json"))
    wrong = run_root / "summary" / "wrong_but_parseable.csv"
    unparseable = run_root / "summary" / "unparseable_outputs.csv"
    return (
        len(result_files),
        len(sample_files),
        str(wrong) if wrong.exists() else "-",
        str(unparseable) if unparseable.exists() else "-",
    )


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_doc = Path(args.output_doc).expanduser().resolve()
    rows = metric_rows(run_root)
    result_count, sample_count, wrong_path, unparseable_path = sample_paths(run_root)

    lines: list[str] = [
        "# H200 mix / sub_only / base / ablation 评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测目录：`{run_root}`",
        f"- 官方口径参考：Qwen Quickstart / thinking mode，`enable_thinking=True`，thinking mode 推荐 `temperature=0.6, top_p=0.95, top_k=20, min_p=0`。链接：{OFFICIAL_QWEN_REF}",
        f"- 结果文件数：{result_count}；sample 文件数：{sample_count}",
        "",
        "## 口径",
        "",
        "- AIME 主表：默认使用 `aime24_repeat8`、`aime25_repeat8`；如目录中存在 repeat64，也会一并汇总。`max_gen_toks=38912`，`temperature=0.6`，`top_p=0.95`，`top_k=20`，`min_p=0`，开启 chat template 和 thinking mode。",
        "- 核心 benchmark：`math500`、生成式 `gpqa_diamond`、`olympiadbench`，默认 `max_gen_toks=32768`，采样参数沿官方 thinking mode。",
        "- 诊断口径只在官方口径未达标时补跑：`temperature=1.0`、`top_p=0.95`、`top_k=0`、`max_gen_toks=38912`。",
        "",
    ]

    if rows:
        aime_benchmarks = [b for b in ("aime24_repeat8", "aime25_repeat8", "aime24_repeat64", "aime25_repeat64") if any(row["benchmark"] == b for row in rows)]
        if aime_benchmarks:
            metrics = ["exact_match", "pass_at_8"]
            if any("repeat64" in b for b in aime_benchmarks):
                metrics.append("pass_at_64")
            lines.extend(["## AIME", ""])
            lines.extend(table_for(rows, aime_benchmarks, metrics))
        present_benchmarks = sorted({row["benchmark"] for row in rows})
        core_order = ["math500", "gpqa_diamond", "olympiadbench"]
        core_benchmarks = [benchmark for benchmark in core_order if benchmark in present_benchmarks]
        supplemental_benchmarks = [
            benchmark
            for benchmark in present_benchmarks
            if benchmark not in set(aime_benchmarks) and benchmark not in set(core_benchmarks)
        ]
        if core_benchmarks:
            lines.extend(["", "## Core Benchmarks", ""])
            lines.extend(table_for(rows, core_benchmarks, ["exact_match", "acc_norm"]))
        if supplemental_benchmarks:
            lines.extend(["", "## Supplement Math Benchmarks", ""])
            lines.extend(table_for(rows, supplemental_benchmarks, ["exact_match"]))
        lines.extend(["", "## 结论", ""])
        for benchmark, target in (("aime24_repeat8", 0.50), ("aime25_repeat8", 0.40), ("aime24_repeat64", 0.50), ("aime25_repeat64", 0.40)):
            if not any(row["benchmark"] == benchmark for row in rows):
                continue
            best = best_by_family(rows, benchmark, "exact_match")
            mix = best.get("mix")
            base = best.get("base")
            sub = best.get("sub_only")
            ablation = best.get("ablation")
            status = "未达标"
            if mix and mix[1] >= target:
                status = "达标"
            lines.append(
                f"- `{benchmark}` mix best: {mix[0] if mix else '-'} = {fmt(mix[1] if mix else None)}，目标 {target:.2f}，状态：{status}；"
                f"base={fmt(base[1] if base else None)}，sub_only={fmt(sub[1] if sub else None)}，ablation={fmt(ablation[1] if ablation else None)}。"
            )
    else:
        lines.extend(["## 当前状态", "", "- 尚未发现 `results_*.json`，报告会在评测完成后由 watcher 重新生成。"])

    lines.extend(
        [
            "",
            "## 失败样例路径",
            "",
            f"- wrong-but-parseable：`{wrong_path}`",
            f"- unparseable：`{unparseable_path}`",
            f"- 原始 samples：`{run_root}` 下各模型/benchmark 子目录的 `samples_*.jsonl` 或 `samples_*.json`。",
            "",
        ]
    )

    output_doc.parent.mkdir(parents=True, exist_ok=True)
    output_doc.write_text("\n".join(lines), encoding="utf-8")
    print(output_doc)


if __name__ == "__main__":
    main()
