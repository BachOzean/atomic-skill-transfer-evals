#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_ROOT="${REPO_ROOT}/scripts"

ATOMIC_EVAL_RUN_ROOT="${ATOMIC_EVAL_RUN_ROOT:-/root/ast_eval_runtime}"
LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-${REPO_ROOT}/.venvs/lm_eval_overlay/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ATOMIC_EVAL_RUN_ROOT}/outputs/external_math}"
RUN_TAG="${RUN_TAG:-h200_mix_ablation_official_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_ROOT="${LOG_ROOT:-${ATOMIC_EVAL_RUN_ROOT}/outputs/queued_eval_logs/${RUN_TAG}}"
MODEL_SPEC_FILE="${MODEL_SPEC_FILE:-}"
SUPPLEMENT_MODEL_SPEC_FILE="${SUPPLEMENT_MODEL_SPEC_FILE:-}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
AIME_BENCHMARKS="${AIME_BENCHMARKS:-aime24_repeat8 aime25_repeat8}"
CORE_BENCHMARKS="${CORE_BENCHMARKS:-math500 gpqa_diamond olympiadbench}"
MAX_GEN_TOKS_AIME="${MAX_GEN_TOKS_AIME:-38912}"
MAX_GEN_TOKS_CORE="${MAX_GEN_TOKS_CORE:-32768}"
MAX_MODEL_LEN_AIME="${MAX_MODEL_LEN_AIME:-40960}"
MAX_MODEL_LEN_CORE="${MAX_MODEL_LEN_CORE:-36864}"
TEMPERATURE_OFFICIAL="${TEMPERATURE_OFFICIAL:-0.6}"
TOP_P_OFFICIAL="${TOP_P_OFFICIAL:-0.95}"
TOP_K_OFFICIAL="${TOP_K_OFFICIAL:-20}"
MIN_P_OFFICIAL="${MIN_P_OFFICIAL:-0}"
RUN_DIAGNOSTIC="${RUN_DIAGNOSTIC:-auto}"
DIAGNOSTIC_MAX_MODELS="${DIAGNOSTIC_MAX_MODELS:-2}"
DRY_RUN="${DRY_RUN:-false}"
REPORT_DOC="${REPORT_DOC:-/root/atomic-skill-transfer-evals/docs/2026-04-29_h200_mix_ablation_eval_report_zh.md}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${ATOMIC_EVAL_RUN_ROOT}/.cache/hf" "${ATOMIC_EVAL_RUN_ROOT}/tmp"

if [[ ! -x "${LM_EVAL_PYTHON}" ]]; then
    echo "LM_EVAL_PYTHON is not executable: ${LM_EVAL_PYTHON}" >&2
    exit 1
fi

if [[ -z "${MODEL_SPEC_FILE}" || ! -f "${MODEL_SPEC_FILE}" ]]; then
    MODEL_SPEC_FILE="${RUN_ROOT}/model_specs.tsv"
    : >"${MODEL_SPEC_FILE}"
    [[ -d /root/models/Qwen3-1.7B ]] && printf 'base_qwen3_1_7b\t/root/models/Qwen3-1.7B\tbase\t-\n' >>"${MODEL_SPEC_FILE}"
    [[ -d /root/models/grpo_zero_only_origin_only_qwen3_1_7b_a800_flash_20260410_195340_global_step_310_huggingface ]] && printf 'origin_310\t/root/models/grpo_zero_only_origin_only_qwen3_1_7b_a800_flash_20260410_195340_global_step_310_huggingface\torigin_only\t310\n' >>"${MODEL_SPEC_FILE}"
    sub_root="/root/ast_eval_runtime/models/grpo_zero_only_sub_only_solq95_qwen3_1_7b_hf_h200_16k_rwdguard_r64_cpu256_u090_8gpu_20260424_023658"
    if [[ -d "${sub_root}" ]]; then
        for step in 230 460 690 920; do
            [[ -d "${sub_root}/global_step_${step}" ]] && printf 'sub_%s\t%s\tsub_only\t%s\n' "${step}" "${sub_root}/global_step_${step}" "${step}" >>"${MODEL_SPEC_FILE}"
        done
    fi
    mix_root="/root/ast_eval_runtime/models/grpo_zero_only_mixed_solq95_qwen3_1_7b_hf_h200_16k_rwdguard_r64_cpu256_u090_save92_8gpu_20260426_192005"
    if [[ -d "${mix_root}" ]]; then
        for step in 276 460 644 828 920; do
            [[ -d "${mix_root}/global_step_${step}" ]] && printf 'mix_%s\t%s\tmix\t%s\n' "${step}" "${mix_root}/global_step_${step}" "${step}" >>"${MODEL_SPEC_FILE}"
        done
    fi
fi

mapfile -t MODEL_LINES < <(awk -F '\t' 'NF >= 2 && $1 !~ /^#/ {print}' "${MODEL_SPEC_FILE}")
if (( ${#MODEL_LINES[@]} == 0 )); then
    echo "No model specs found in ${MODEL_SPEC_FILE}" >&2
    exit 1
fi

read -r -a GPU_ID_ARR <<<"${GPU_IDS}"
if (( ${#GPU_ID_ARR[@]} == 0 )); then
    echo "No GPU ids configured." >&2
    exit 1
fi

log() {
    echo "[$(date '+%F %T')] $*"
}

run_one() {
    local line="$1"
    local benchmark="$2"
    local gpu_id="$3"
    local max_gen_toks="$4"
    local max_model_len="$5"
    local temperature="$6"
    local top_k="$7"
    local suffix="${8:-}"
    local tag path family step eval_tag log_file

    IFS=$'\t' read -r tag path family step _ <<<"${line}"
    eval_tag="${tag}${suffix}"
    log_file="${LOG_ROOT}/${RUN_TAG}__${benchmark}__${eval_tag}.log"

    {
        log "benchmark=${benchmark} tag=${eval_tag} family=${family:-unknown} step=${step:-unknown} gpu=${gpu_id} path=${path}"
        local cmd=(
            env
            "CUDA_VISIBLE_DEVICES=${gpu_id}"
            "VLLM_WORKER_MULTIPROC_METHOD=spawn"
            "RUN_ROOT=${ATOMIC_EVAL_RUN_ROOT}"
            "LM_EVAL_PYTHON=${LM_EVAL_PYTHON}"
            "GPQA_LOCAL_DATASET_DIR=${ATOMIC_EVAL_RUN_ROOT}/data/gpqa"
            "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}"
            "TENSOR_PARALLEL_SIZE=1"
            "MAX_MODEL_LEN=${max_model_len}"
            "MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}"
            "SWAP_SPACE=${SWAP_SPACE:-8}"
            "HF_HOME=${ATOMIC_EVAL_RUN_ROOT}/.cache/hf"
            "HF_DATASETS_CACHE=${ATOMIC_EVAL_RUN_ROOT}/.cache/hf/datasets"
            "XDG_CACHE_HOME=${ATOMIC_EVAL_RUN_ROOT}/.cache"
            "TMPDIR=${ATOMIC_EVAL_RUN_ROOT}/tmp"
            "${LM_EVAL_PYTHON}" "${SCRIPT_ROOT}/run_external_math_eval.py"
            --lm-eval-python "${LM_EVAL_PYTHON}"
            --output-root "${OUTPUT_ROOT}"
            --run-tag "${RUN_TAG}"
            --model-alias "${eval_tag}=${path}"
            --benchmarks "${benchmark}"
            --backend vllm
            --dtype bfloat16
            --batch-size auto
            --max-gen-toks "${max_gen_toks}"
            --temperature "${temperature}"
            --top-p "${TOP_P_OFFICIAL}"
            --top-k "${top_k}"
            --min-p "${MIN_P_OFFICIAL}"
            --seed 7
            --apply-chat-template
            --enable-thinking
            --gpqa-local-dataset-dir "${ATOMIC_EVAL_RUN_ROOT}/data/gpqa"
            --metadata-file-name "run_metadata.${eval_tag}.${benchmark}.json"
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

run_benchmark_group() {
    local benchmark="$1"
    local max_gen_toks="$2"
    local max_model_len="$3"
    local temperature="$4"
    local top_k="$5"
    local suffix="${6:-}"
    local gpu_id pid finished_pid status failures=0
    local total_models="${#MODEL_LINES[@]}"
    local next_model_idx=0
    local running=0
    declare -A PID_TO_GPU=()
    declare -A PID_TO_LABEL=()

    launch_next_on_gpu() {
        local slot_gpu_id="$1"
        local model_idx="${next_model_idx}"
        local line label
        (( model_idx < total_models )) || return 1
        line="${MODEL_LINES[$model_idx]}"
        label="${benchmark}:${line%%$'\t'*}:gpu${slot_gpu_id}"
        run_one "${line}" "${benchmark}" "${slot_gpu_id}" "${max_gen_toks}" "${max_model_len}" "${temperature}" "${top_k}" "${suffix}" &
        pid="$!"
        PID_TO_GPU["${pid}"]="${slot_gpu_id}"
        PID_TO_LABEL["${pid}"]="${label}"
        next_model_idx=$((next_model_idx + 1))
        running=$((running + 1))
        log "launched ${label} pid=${pid} queue=${next_model_idx}/${total_models}"
    }

    log "starting benchmark=${benchmark} models=${total_models} temperature=${temperature} top_k=${top_k} scheduler=gpu_slot_pool"
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
            log "WARNING benchmark=${benchmark} wait returned without a finished pid status=${status}"
            failures=$((failures + 1))
            break
        fi

        gpu_id="${PID_TO_GPU[${finished_pid}]:-}"
        label="${PID_TO_LABEL[${finished_pid}]:-${benchmark}:pid${finished_pid}:gpu${gpu_id:-unknown}}"
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

    if (( failures > 0 )); then
        log "benchmark=${benchmark} finished with failures=${failures}"
        return 1
    fi
    log "benchmark=${benchmark} finished successfully"
}

write_global_metadata() {
    MODEL_SPEC_FILE="${MODEL_SPEC_FILE}" RUN_ROOT="${RUN_ROOT}" RUN_TAG="${RUN_TAG}" \
    AIME_BENCHMARKS="${AIME_BENCHMARKS}" CORE_BENCHMARKS="${CORE_BENCHMARKS}" \
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
    "aime_benchmarks": os.environ["AIME_BENCHMARKS"].split(),
    "core_benchmarks": os.environ["CORE_BENCHMARKS"].split(),
    "official_qwen_reference": "https://qwen.readthedocs.io/en/v3.0/getting_started/quickstart.html",
}
(Path(os.environ["RUN_ROOT"]) / "run_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

summarize_and_report() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        return 0
    fi
    "${LM_EVAL_PYTHON}" "${SCRIPT_ROOT}/summarize_external_math_eval.py" --run-root "${RUN_ROOT}" || true
    "${LM_EVAL_PYTHON}" "${SCRIPT_ROOT}/write_h200_mix_ablation_eval_report.py" --run-root "${RUN_ROOT}" --output-doc "${REPORT_DOC}" || true
}

official_targets_met() {
    RUN_ROOT="${RUN_ROOT}" AIME_BENCHMARKS="${AIME_BENCHMARKS}" "${LM_EVAL_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
aime_benchmarks = os.environ["AIME_BENCHMARKS"].split()
targets = {}
for benchmark in aime_benchmarks:
    if "aime24" in benchmark:
        targets[benchmark] = 0.50
    elif "aime25" in benchmark:
        targets[benchmark] = 0.40
best = {benchmark: 0.0 for benchmark in targets}
if not best:
    raise SystemExit(1)
for path in run_root.glob("**/results*.json"):
    rel = path.relative_to(run_root)
    if len(rel.parts) < 2:
        continue
    tag, benchmark = rel.parts[0], rel.parts[1]
    if not tag.startswith("mix_") or "_diag_" in tag or benchmark not in best:
        continue
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    for metrics in (blob.get("results") or {}).values():
        value = metrics.get("exact_match,none")
        if isinstance(value, (int, float)):
            best[benchmark] = max(best[benchmark], float(value))
raise SystemExit(0 if all(best[benchmark] >= target for benchmark, target in targets.items()) else 1)
PY
}

build_diagnostic_spec() {
    local output_file="$1"
    RUN_ROOT="${RUN_ROOT}" MODEL_SPEC_FILE="${MODEL_SPEC_FILE}" DIAGNOSTIC_MAX_MODELS="${DIAGNOSTIC_MAX_MODELS}" AIME_BENCHMARKS="${AIME_BENCHMARKS}" \
    "${LM_EVAL_PYTHON}" - "${output_file}" <<'PY'
import json
import os
import sys
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
spec_file = Path(os.environ["MODEL_SPEC_FILE"])
limit = int(os.environ["DIAGNOSTIC_MAX_MODELS"])
aime_benchmarks = set(os.environ["AIME_BENCHMARKS"].split())
targets = {}
for benchmark in aime_benchmarks:
    if "aime24" in benchmark:
        targets[benchmark] = 0.50
    elif "aime25" in benchmark:
        targets[benchmark] = 0.40
scores = {}
best_by_benchmark = {}
for path in run_root.glob("**/results*.json"):
    rel = path.relative_to(run_root)
    if len(rel.parts) < 2:
        continue
    tag, benchmark = rel.parts[0], rel.parts[1]
    if not tag.startswith("mix_") or benchmark not in aime_benchmarks:
        continue
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    for metrics in (blob.get("results") or {}).values():
        value = metrics.get("exact_match,none")
        if isinstance(value, (int, float)):
            scores.setdefault(tag, []).append(float(value))
            best_by_benchmark[benchmark] = max(best_by_benchmark.get(benchmark, 0.0), float(value))
if not scores:
    Path(sys.argv[1]).write_text("", encoding="utf-8")
    raise SystemExit
if targets and all(best_by_benchmark.get(benchmark, 0.0) >= target for benchmark, target in targets.items()):
    Path(sys.argv[1]).write_text("", encoding="utf-8")
    raise SystemExit
ranked = sorted(scores, key=lambda tag: sum(scores[tag]) / len(scores[tag]), reverse=True)[:limit]
lookup = {}
for line in spec_file.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    parts = line.split("\t")
    lookup[parts[0]] = line
Path(sys.argv[1]).write_text("\n".join(lookup[tag] for tag in ranked if tag in lookup) + "\n", encoding="utf-8")
PY
}

main() {
    log "run_tag=${RUN_TAG}"
    log "run_root=${RUN_ROOT}"
    log "model_spec_file=${MODEL_SPEC_FILE}"
    log "gpu_ids=${GPU_IDS}"
    cat "${MODEL_SPEC_FILE}"

    for benchmark in ${AIME_BENCHMARKS}; do
        run_benchmark_group "${benchmark}" "${MAX_GEN_TOKS_AIME}" "${MAX_MODEL_LEN_AIME}" "${TEMPERATURE_OFFICIAL}" "${TOP_K_OFFICIAL}"
    done

    write_global_metadata
    summarize_and_report

    if [[ -n "${SUPPLEMENT_MODEL_SPEC_FILE}" && -s "${SUPPLEMENT_MODEL_SPEC_FILE}" ]]; then
        if official_targets_met; then
            log "mix AIME targets met; supplemental mix checkpoints skipped"
        else
            log "mix AIME targets not met; evaluating supplemental mix checkpoints from ${SUPPLEMENT_MODEL_SPEC_FILE}"
            mapfile -t MODEL_LINES < <(awk -F '\t' 'NF >= 2 && $1 !~ /^#/ {print}' "${SUPPLEMENT_MODEL_SPEC_FILE}")
            for benchmark in ${AIME_BENCHMARKS}; do
                run_benchmark_group "${benchmark}" "${MAX_GEN_TOKS_AIME}" "${MAX_MODEL_LEN_AIME}" "${TEMPERATURE_OFFICIAL}" "${TOP_K_OFFICIAL}"
            done
            effective_spec="${RUN_ROOT}/effective_model_specs.tsv"
            awk 'NF && $1 !~ /^#/' "${MODEL_SPEC_FILE}" "${SUPPLEMENT_MODEL_SPEC_FILE}" >"${effective_spec}"
            MODEL_SPEC_FILE="${effective_spec}"
            mapfile -t MODEL_LINES < <(awk -F '\t' 'NF >= 2 && $1 !~ /^#/ {print}' "${MODEL_SPEC_FILE}")
            write_global_metadata
            summarize_and_report
        fi
    fi

    if [[ "${RUN_DIAGNOSTIC}" == "true" || "${RUN_DIAGNOSTIC}" == "auto" ]]; then
        diag_spec="${RUN_ROOT}/diagnostic_model_specs.tsv"
        build_diagnostic_spec "${diag_spec}"
        if [[ -s "${diag_spec}" ]]; then
            log "running diagnostic AIME pass with $(wc -l <"${diag_spec}") mix candidates"
            mapfile -t MODEL_LINES < <(awk -F '\t' 'NF >= 2 && $1 !~ /^#/ {print}' "${diag_spec}")
            for benchmark in ${AIME_BENCHMARKS}; do
                run_benchmark_group "${benchmark}" "${MAX_GEN_TOKS_AIME}" "${MAX_MODEL_LEN_AIME}" "1.0" "0" "_diag_t1_topk0"
            done
            mapfile -t MODEL_LINES < <(awk -F '\t' 'NF >= 2 && $1 !~ /^#/ {print}' "${MODEL_SPEC_FILE}")
        else
            log "diagnostic pass skipped; no mix AIME results found"
        fi
    fi

    for benchmark in ${CORE_BENCHMARKS}; do
        run_benchmark_group "${benchmark}" "${MAX_GEN_TOKS_CORE}" "${MAX_MODEL_LEN_CORE}" "${TEMPERATURE_OFFICIAL}" "${TOP_K_OFFICIAL}"
    done

    write_global_metadata
    summarize_and_report
    log "evaluation queue finished"
}

main "$@"
