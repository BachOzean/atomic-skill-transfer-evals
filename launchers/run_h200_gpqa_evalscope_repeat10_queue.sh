#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_ROOT="${REPO_ROOT}/scripts"

ATOMIC_EVAL_RUN_ROOT="${ATOMIC_EVAL_RUN_ROOT:-/root/ast_eval_runtime}"
LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-${REPO_ROOT}/.venvs/lm_eval_overlay/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ATOMIC_EVAL_RUN_ROOT}/outputs/external_math}"
RUN_TAG="${RUN_TAG:-h200_mix_ablation_gpqa_evalscope_repeat10_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_ROOT="${LOG_ROOT:-${ATOMIC_EVAL_RUN_ROOT}/outputs/queued_eval_logs/${RUN_TAG}}"
MODEL_SPEC_FILE="${MODEL_SPEC_FILE:-${ATOMIC_EVAL_RUN_ROOT}/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/effective_model_specs.tsv}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
SWAP_SPACE="${SWAP_SPACE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
TASK_OVERRIDE="${TASK_OVERRIDE:-leaderboard_gpqa_diamond_evalscope_gen}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
REPORT_DOC="${REPORT_DOC:-/root/atomic-skill-transfer-evals/docs/2026-04-29_h200_mix_ablation_gpqa_evalscope_repeat10_report_zh.md}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${ATOMIC_EVAL_RUN_ROOT}/.cache/hf" "${ATOMIC_EVAL_RUN_ROOT}/tmp"

if [[ ! -x "${LM_EVAL_PYTHON}" ]]; then
    echo "LM_EVAL_PYTHON is not executable: ${LM_EVAL_PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_SPEC_FILE}" ]]; then
    echo "MODEL_SPEC_FILE does not exist: ${MODEL_SPEC_FILE}" >&2
    exit 1
fi

mapfile -t MODEL_LINES < <(awk -F '\t' 'NF >= 2 && $1 !~ /^#/ {print}' "${MODEL_SPEC_FILE}")
read -r -a GPU_ID_ARR <<<"${GPU_IDS}"
read -r -a SEED_ARR <<<"${SEEDS}"

if (( ${#MODEL_LINES[@]} == 0 )); then
    echo "No model specs found in ${MODEL_SPEC_FILE}" >&2
    exit 1
fi
if (( ${#GPU_ID_ARR[@]} == 0 )); then
    echo "No GPU ids configured." >&2
    exit 1
fi
if (( ${#SEED_ARR[@]} == 0 )); then
    echo "No seeds configured." >&2
    exit 1
fi

log() {
    echo "[$(date '+%F %T')] $*"
}

has_result() {
    local eval_tag="$1"
    find "${RUN_ROOT}/${eval_tag}/gpqa_diamond" -path '*results_*.json' -print -quit 2>/dev/null | grep -q .
}

run_one() {
    local job_line="$1"
    local gpu_id="$2"
    local seed tag path family step eval_tag log_file

    IFS=$'\t' read -r seed tag path family step _ <<<"${job_line}"
    eval_tag="$(printf '%s_s%02d' "${tag}" "${seed}")"
    log_file="${LOG_ROOT}/${RUN_TAG}__gpqa_evalscope__${eval_tag}.log"

    if [[ "${SKIP_EXISTING}" == "true" ]] && has_result "${eval_tag}"; then
        log "skip existing gpqa_evalscope:${eval_tag}:gpu${gpu_id}"
        return 0
    fi

    {
        log "benchmark=gpqa_evalscope tag=${eval_tag} base_tag=${tag} family=${family:-unknown} step=${step:-unknown} seed=${seed} gpu=${gpu_id} path=${path}"
        local cmd=(
            env
            "CUDA_VISIBLE_DEVICES=${gpu_id}"
            "VLLM_WORKER_MULTIPROC_METHOD=spawn"
            "RUN_ROOT=${ATOMIC_EVAL_RUN_ROOT}"
            "LM_EVAL_PYTHON=${LM_EVAL_PYTHON}"
            "GPQA_LOCAL_DATASET_DIR=${ATOMIC_EVAL_RUN_ROOT}/data/gpqa"
            "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
            "TENSOR_PARALLEL_SIZE=1"
            "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
            "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
            "SWAP_SPACE=${SWAP_SPACE}"
            "HF_HOME=${ATOMIC_EVAL_RUN_ROOT}/.cache/hf"
            "HF_DATASETS_CACHE=${ATOMIC_EVAL_RUN_ROOT}/.cache/hf/datasets"
            "XDG_CACHE_HOME=${ATOMIC_EVAL_RUN_ROOT}/.cache"
            "TMPDIR=${ATOMIC_EVAL_RUN_ROOT}/tmp"
            "${LM_EVAL_PYTHON}" "${SCRIPT_ROOT}/run_external_math_eval.py"
            --lm-eval-python "${LM_EVAL_PYTHON}"
            --output-root "${OUTPUT_ROOT}"
            --run-tag "${RUN_TAG}"
            --model-alias "${eval_tag}=${path}"
            --benchmarks gpqa_diamond
            --task-override "gpqa_diamond=${TASK_OVERRIDE}"
            --backend vllm
            --dtype bfloat16
            --batch-size auto
            --max-gen-toks "${MAX_GEN_TOKS}"
            --temperature "${TEMPERATURE}"
            --top-p "${TOP_P}"
            --top-k "${TOP_K}"
            --min-p "${MIN_P}"
            --seed "${seed}"
            --apply-chat-template
            --enable-thinking
            --no-system-instruction
            --gpqa-local-dataset-dir "${ATOMIC_EVAL_RUN_ROOT}/data/gpqa"
            --metadata-file-name "run_metadata.${eval_tag}.gpqa_evalscope.json"
        )
        if [[ -n "${LIMIT:-}" ]]; then
            cmd+=(--limit "${LIMIT}")
        fi
        if [[ "${DRY_RUN}" == "true" ]]; then
            cmd+=(--dry-run)
        fi
        "${cmd[@]}"
    } >>"${log_file}" 2>&1
}

build_job_lines() {
    local seed line
    for line in "${MODEL_LINES[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
            printf '%s\t%s\n' "${seed}" "${line}"
        done
    done
}

write_global_metadata() {
    MODEL_SPEC_FILE="${MODEL_SPEC_FILE}" RUN_ROOT="${RUN_ROOT}" RUN_TAG="${RUN_TAG}" SEEDS="${SEEDS}" REPORT_DOC="${REPORT_DOC}" \
    MAX_GEN_TOKS="${MAX_GEN_TOKS}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" TEMPERATURE="${TEMPERATURE}" TOP_P="${TOP_P}" TOP_K="${TOP_K}" MIN_P="${MIN_P}" TASK_OVERRIDE="${TASK_OVERRIDE}" \
    "${LM_EVAL_PYTHON}" - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

spec_file = Path(os.environ["MODEL_SPEC_FILE"])
models = []
for line in spec_file.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    parts = line.split("\t")
    models.append({"tag": parts[0], "path": parts[1], "family": parts[2] if len(parts) > 2 else "", "step": parts[3] if len(parts) > 3 else ""})
payload = {
    "run_tag": os.environ["RUN_TAG"],
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "model_spec_file": str(spec_file),
    "models": models,
    "benchmark": "gpqa_diamond",
    "task_override": os.environ["TASK_OVERRIDE"],
    "seeds": [int(x) for x in os.environ["SEEDS"].split()],
    "repeat_count": len(os.environ["SEEDS"].split()),
    "max_gen_toks": int(os.environ["MAX_GEN_TOKS"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "top_p": float(os.environ["TOP_P"]),
    "top_k": int(os.environ["TOP_K"]),
    "min_p": float(os.environ["MIN_P"]),
    "apply_chat_template": True,
    "enable_thinking": True,
    "system_instruction": "",
    "report_doc": os.environ["REPORT_DOC"],
}
(Path(os.environ["RUN_ROOT"]) / "run_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

aggregate_results() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        return 0
    fi
    "${LM_EVAL_PYTHON}" "${SCRIPT_ROOT}/summarize_external_math_eval.py" --run-root "${RUN_ROOT}" || true
    MODEL_SPEC_FILE="${MODEL_SPEC_FILE}" RUN_ROOT="${RUN_ROOT}" RUN_TAG="${RUN_TAG}" REPORT_DOC="${REPORT_DOC}" \
    "${LM_EVAL_PYTHON}" - <<'PY'
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
import os

run_root = Path(os.environ["RUN_ROOT"])
report_doc = Path(os.environ["REPORT_DOC"])
summary_dir = run_root / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

spec = {}
for line in Path(os.environ["MODEL_SPEC_FILE"]).read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    parts = line.split("\t")
    spec[parts[0]] = {
        "model_tag": parts[0],
        "path": parts[1],
        "family": parts[2] if len(parts) > 2 else "",
        "step": parts[3] if len(parts) > 3 else "",
    }

rows = []
pattern = re.compile(r"^(?P<tag>.+)_s(?P<seed>\d{2})$")
for path in sorted(run_root.glob("**/results_*.json")):
    rel = path.relative_to(run_root)
    if len(rel.parts) < 2 or rel.parts[1] != "gpqa_diamond":
        continue
    match = pattern.match(rel.parts[0])
    if not match:
        continue
    blob = json.loads(path.read_text(encoding="utf-8"))
    task_name, metrics = next(iter((blob.get("results") or {}).items()))
    value = metrics.get("exact_match,none")
    if not isinstance(value, (int, float)):
        continue
    base_tag = match.group("tag")
    seed = int(match.group("seed"))
    meta = spec.get(base_tag, {"model_tag": base_tag, "path": "", "family": "", "step": ""})
    rows.append({
        "model_tag": base_tag,
        "seed": seed,
        "family": meta["family"],
        "step": meta["step"],
        "task_name": task_name,
        "exact_match": float(value),
        "results_path": str(path),
    })

seed_csv = summary_dir / "gpqa_evalscope_seed_results.csv"
with seed_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["model_tag", "family", "step", "seed", "task_name", "exact_match", "results_path"])
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: (row["family"], row["model_tag"], row["seed"])))

grouped = {}
for row in rows:
    grouped.setdefault(row["model_tag"], []).append(row)

summary_rows = []
for tag, items in grouped.items():
    values = [item["exact_match"] for item in items]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    meta = spec.get(tag, {"family": "", "step": "", "path": ""})
    summary_rows.append({
        "model_tag": tag,
        "family": meta["family"],
        "step": meta["step"],
        "completed_seeds": len(values),
        "mean_exact_match": mean,
        "std_exact_match": math.sqrt(variance),
        "min_exact_match": min(values),
        "max_exact_match": max(values),
        "model_path": meta.get("path", ""),
    })

family_order = {"base": 0, "origin_only": 1, "sub_only": 2, "mix": 3, "ablation": 4}
summary_rows.sort(key=lambda row: (-row["mean_exact_match"], family_order.get(row["family"], 99), row["model_tag"]))

summary_csv = summary_dir / "gpqa_evalscope_repeat10_summary.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["model_tag", "family", "step", "completed_seeds", "mean_exact_match", "std_exact_match", "min_exact_match", "max_exact_match", "model_path"])
    writer.writeheader()
    writer.writerows(summary_rows)

summary_json = summary_dir / "gpqa_evalscope_repeat10_summary.json"
summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

report_doc.parent.mkdir(parents=True, exist_ok=True)
lines = [
    "# H200 GPQA-Diamond EvalScope-style Repeat10 重评报告",
    "",
    f"- run_tag: `{os.environ['RUN_TAG']}`",
    f"- run_root: `{run_root}`",
    f"- updated_at: `{datetime.now().isoformat(timespec='seconds')}`",
    "- 口径: EvalScope-style GPQA prompt, `ANSWER: [LETTER]` 解析, no system instruction, chat template, thinking mode, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_gen_toks=32768`, repeat10。",
    "",
    "## Top Results",
    "",
    "| rank | model | family | step | seeds | mean exact | std | min | max |",
    "|---:|---|---|---:|---:|---:|---:|---:|---:|",
]
for idx, row in enumerate(summary_rows[:20], start=1):
    lines.append(
        f"| {idx} | `{row['model_tag']}` | {row['family']} | {row['step']} | {row['completed_seeds']} | "
        f"{row['mean_exact_match']*100:.2f} | {row['std_exact_match']*100:.2f} | {row['min_exact_match']*100:.2f} | {row['max_exact_match']*100:.2f} |"
    )
lines.extend([
    "",
    "## Artifacts",
    "",
    f"- Summary CSV: `{summary_csv}`",
    f"- Seed-level CSV: `{seed_csv}`",
    f"- Summary JSON: `{summary_json}`",
])
report_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({
    "summary_csv": str(summary_csv),
    "seed_csv": str(seed_csv),
    "summary_json": str(summary_json),
    "report_doc": str(report_doc),
    "model_count": len(summary_rows),
    "seed_result_count": len(rows),
}, indent=2, ensure_ascii=False))
PY
}

run_queue() {
    mapfile -t JOB_LINES < <(build_job_lines)
    local total_jobs="${#JOB_LINES[@]}"
    local next_job_idx=0
    local running=0
    local gpu_id pid finished_pid status failures=0
    declare -A PID_TO_GPU=()
    declare -A PID_TO_LABEL=()

    launch_next_on_gpu() {
        local slot_gpu_id="$1"
        local job_idx="${next_job_idx}"
        local job_line seed tag label
        (( job_idx < total_jobs )) || return 1
        job_line="${JOB_LINES[$job_idx]}"
        IFS=$'\t' read -r seed tag _ <<<"${job_line}"
        label="$(printf 'gpqa_evalscope:%s_s%02d:gpu%s' "${tag}" "${seed}" "${slot_gpu_id}")"
        run_one "${job_line}" "${slot_gpu_id}" &
        pid="$!"
        PID_TO_GPU["${pid}"]="${slot_gpu_id}"
        PID_TO_LABEL["${pid}"]="${label}"
        next_job_idx=$((next_job_idx + 1))
        running=$((running + 1))
        log "launched ${label} pid=${pid} queue=${next_job_idx}/${total_jobs}"
    }

    log "starting gpqa_evalscope repeat queue jobs=${total_jobs} models=${#MODEL_LINES[@]} seeds=${SEEDS} scheduler=gpu_slot_pool"
    for gpu_id in "${GPU_ID_ARR[@]}"; do
        launch_next_on_gpu "${gpu_id}" || break
    done

    while (( running > 0 )); do
        finished_pid=""
        if wait -n -p finished_pid; then
            status=0
        else
            status=$?
        fi
        if [[ -z "${finished_pid}" ]]; then
            log "WARNING wait returned without a finished pid status=${status}"
            failures=$((failures + 1))
            break
        fi
        gpu_id="${PID_TO_GPU[${finished_pid}]:-}"
        label="${PID_TO_LABEL[${finished_pid}]:-gpqa_evalscope:pid${finished_pid}:gpu${gpu_id:-unknown}}"
        unset "PID_TO_GPU[${finished_pid}]" "PID_TO_LABEL[${finished_pid}]"
        running=$((running - 1))

        if (( status == 0 )); then
            log "completed ${label}"
        else
            failures=$((failures + 1))
            log "FAILED ${label} status=${status}"
        fi

        if [[ -n "${gpu_id}" ]]; then
            launch_next_on_gpu "${gpu_id}" || true
        fi
    done

    log "gpqa_evalscope queue finished failures=${failures}"
    return "${failures}"
}

main() {
    log "run_tag=${RUN_TAG}"
    log "run_root=${RUN_ROOT}"
    log "log_root=${LOG_ROOT}"
    log "model_spec_file=${MODEL_SPEC_FILE}"
    log "gpu_ids=${GPU_IDS}"
    log "seeds=${SEEDS}"
    log "task_override=${TASK_OVERRIDE}"
    log "decode=temp${TEMPERATURE}_top_p${TOP_P}_top_k${TOP_K}_min_p${MIN_P}_max_gen${MAX_GEN_TOKS}"
    cp "${MODEL_SPEC_FILE}" "${RUN_ROOT}/effective_model_specs.tsv"
    write_global_metadata
    run_queue
    local status="$?"
    aggregate_results
    log "finished run_tag=${RUN_TAG} status=${status}"
    return "${status}"
}

main "$@"
