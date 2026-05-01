#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_ROOT="${REPO_ROOT}/scripts"

ATOMIC_EVAL_RUN_ROOT="${ATOMIC_EVAL_RUN_ROOT:-/root/ast_eval_runtime}"
LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-${REPO_ROOT}/.venvs/lm_eval_overlay/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ATOMIC_EVAL_RUN_ROOT}/outputs/external_math}"
RUN_TAG="${RUN_TAG:-h200_mix_ablation_supplement_math_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_ROOT="${LOG_ROOT:-${ATOMIC_EVAL_RUN_ROOT}/outputs/queued_eval_logs/${RUN_TAG}}"
MODEL_SPEC_FILE="${MODEL_SPEC_FILE:-/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/effective_model_specs.tsv}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
BENCHMARKS="${BENCHMARKS:-matharena_hmmt_feb_2025 matharena_hmmt_nov_2025 matharena_brumo_2025 amc23 omni_math_500}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-38912}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
SEED="${SEED:-7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
SWAP_SPACE="${SWAP_SPACE:-8}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-true}"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-2000}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"
REPORT_DOC="${REPORT_DOC:-/root/atomic-skill-transfer-evals/docs/2026-04-30_h200_mix_ablation_supplement_math_zh.md}"

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

busy_gpu_count() {
    local output
    output="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
    awk -F, -v gpu_ids="${GPU_IDS}" -v max_mem="${GPU_IDLE_MEMORY_MB}" '
        BEGIN {
            split(gpu_ids, ids, " ")
            for (i in ids) {
                wanted[ids[i]] = 1
            }
        }
        {
            gsub(/ /, "", $1)
            gsub(/ /, "", $2)
            if (($1 in wanted) && ($2 + 0) > max_mem) {
                busy += 1
            }
        }
        END { print busy + 0 }
    ' <<<"${output}"
}

wait_for_idle_gpus() {
    if [[ "${WAIT_FOR_GPUS}" != "true" ]]; then
        return 0
    fi
    local busy
    while true; do
        busy="$(busy_gpu_count)"
        if [[ "${busy}" == "0" ]]; then
            log "all requested GPUs are idle enough; starting queue"
            return 0
        fi
        log "waiting for GPUs to free up: busy=${busy}/${#GPU_ID_ARR[@]} threshold=${GPU_IDLE_MEMORY_MB}MiB sleep=${WAIT_SECONDS}s"
        sleep "${WAIT_SECONDS}"
    done
}

has_existing_result() {
    local eval_tag="$1"
    local benchmark="$2"
    compgen -G "${RUN_ROOT}/${eval_tag}/${benchmark}/results*.json" >/dev/null
}

run_one() {
    local line="$1"
    local benchmark="$2"
    local gpu_id="$3"
    local tag path family step eval_tag log_file

    IFS=$'\t' read -r tag path family step _ <<<"${line}"
    eval_tag="${tag}"
    log_file="${LOG_ROOT}/${RUN_TAG}__${benchmark}__${eval_tag}.log"

    {
        log "benchmark=${benchmark} tag=${eval_tag} family=${family:-unknown} step=${step:-unknown} gpu=${gpu_id} path=${path}"
        if [[ "${SKIP_EXISTING}" == "true" ]] && has_existing_result "${eval_tag}" "${benchmark}"; then
            log "skip existing result for benchmark=${benchmark} tag=${eval_tag}"
            exit 0
        fi
        local cmd=(
            env
            "CUDA_VISIBLE_DEVICES=${gpu_id}"
            "VLLM_WORKER_MULTIPROC_METHOD=spawn"
            "RUN_ROOT=${ATOMIC_EVAL_RUN_ROOT}"
            "LM_EVAL_PYTHON=${LM_EVAL_PYTHON}"
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
            --benchmarks "${benchmark}"
            --backend vllm
            --dtype bfloat16
            --batch-size auto
            --max-gen-toks "${MAX_GEN_TOKS}"
            --temperature "${TEMPERATURE}"
            --top-p "${TOP_P}"
            --top-k "${TOP_K}"
            --min-p "${MIN_P}"
            --seed "${SEED}"
            --apply-chat-template
            --enable-thinking
            --no-system-instruction
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
        run_one "${line}" "${benchmark}" "${slot_gpu_id}" &
        pid="$!"
        PID_TO_GPU["${pid}"]="${slot_gpu_id}"
        PID_TO_LABEL["${pid}"]="${label}"
        next_model_idx=$((next_model_idx + 1))
        running=$((running + 1))
        log "launched ${label} pid=${pid} queue=${next_model_idx}/${total_models}"
    }

    log "starting benchmark=${benchmark} models=${total_models} scheduler=gpu_slot_pool max_gen_toks=${MAX_GEN_TOKS}"
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
    MODEL_SPEC_FILE="${MODEL_SPEC_FILE}" RUN_ROOT="${RUN_ROOT}" RUN_TAG="${RUN_TAG}" BENCHMARKS="${BENCHMARKS}" \
    MAX_GEN_TOKS="${MAX_GEN_TOKS}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" TEMPERATURE="${TEMPERATURE}" TOP_P="${TOP_P}" TOP_K="${TOP_K}" MIN_P="${MIN_P}" SEED="${SEED}" \
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
    "benchmarks": os.environ["BENCHMARKS"].split(),
    "generation": {
        "max_gen_toks": int(os.environ["MAX_GEN_TOKS"]),
        "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
        "temperature": float(os.environ["TEMPERATURE"]),
        "top_p": float(os.environ["TOP_P"]),
        "top_k": int(os.environ["TOP_K"]),
        "min_p": float(os.environ["MIN_P"]),
        "seed": int(os.environ["SEED"]),
        "enable_thinking": True,
        "apply_chat_template": True,
        "system_instruction": None,
    },
    "official_qwen_reference": "https://huggingface.co/Qwen/Qwen3-1.7B",
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

main() {
    log "run_tag=${RUN_TAG}"
    log "run_root=${RUN_ROOT}"
    log "model_spec_file=${MODEL_SPEC_FILE}"
    log "gpu_ids=${GPU_IDS}"
    log "benchmarks=${BENCHMARKS}"
    cat "${MODEL_SPEC_FILE}"

    wait_for_idle_gpus
    write_global_metadata

    for benchmark in ${BENCHMARKS}; do
        run_benchmark_group "${benchmark}"
        summarize_and_report
    done

    write_global_metadata
    summarize_and_report
    log "supplement math evaluation queue finished"
}

main "$@"
