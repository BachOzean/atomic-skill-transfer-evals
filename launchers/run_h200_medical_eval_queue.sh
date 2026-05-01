#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_ROOT="${REPO_ROOT}/scripts"

ATOMIC_EVAL_RUN_ROOT="${ATOMIC_EVAL_RUN_ROOT:-/root/ast_eval_runtime}"
LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-${REPO_ROOT}/.venvs/lm_eval_overlay/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ATOMIC_EVAL_RUN_ROOT}/outputs/medical}"
RUN_TAG="${RUN_TAG:-h200_medbullets_ood_medical_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_ROOT="${LOG_ROOT:-${ATOMIC_EVAL_RUN_ROOT}/outputs/queued_eval_logs/${RUN_TAG}}"
MODEL_SPEC_FILE="${MODEL_SPEC_FILE:-}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
MEDICAL_BENCHMARKS="${MEDICAL_BENCHMARKS:-medqa medmcqa pubmedqa mmlu_anatomy mmlu_clinical_knowledge mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MEDICAL_NUM_FEWSHOT="${MEDICAL_NUM_FEWSHOT:-0}"
MMLU_NUM_FEWSHOT="${MMLU_NUM_FEWSHOT:-5}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-true}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-false}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SWAP_SPACE="${SWAP_SPACE:-8}"
DRY_RUN="${DRY_RUN:-false}"
SUMMARIZE="${SUMMARIZE:-true}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${ATOMIC_EVAL_RUN_ROOT}/.cache/hf" "${ATOMIC_EVAL_RUN_ROOT}/tmp"

if [[ ! -x "${LM_EVAL_PYTHON}" ]]; then
    echo "LM_EVAL_PYTHON is not executable: ${LM_EVAL_PYTHON}" >&2
    exit 1
fi

append_model_spec() {
    local tag="$1"
    local path="$2"
    local family="${3:-unknown}"
    local step="${4:--}"
    if [[ -d "${path}" && -f "${path}/config.json" ]]; then
        printf '%s\t%s\t%s\t%s\n' "${tag}" "${path}" "${family}" "${step}" >>"${MODEL_SPEC_FILE}"
    fi
}

build_default_model_spec() {
    MODEL_SPEC_FILE="${RUN_ROOT}/model_specs.tsv"
    : >"${MODEL_SPEC_FILE}"

    append_model_spec "base_qwen3_1_7b" "/root/models/Qwen3-1.7B" "base" "-"
    [[ -n "${ROOT_ONLY_HF_MODEL_PATH:-}" ]] && append_model_spec "root_only" "${ROOT_ONLY_HF_MODEL_PATH}" "root_only" "${ROOT_ONLY_STEP:--}"
    [[ -n "${SUB_ONLY_HF_MODEL_PATH:-}" ]] && append_model_spec "sub_only" "${SUB_ONLY_HF_MODEL_PATH}" "sub_only" "${SUB_ONLY_STEP:--}"
    [[ -n "${MIX_HF_MODEL_PATH:-}" ]] && append_model_spec "mix" "${MIX_HF_MODEL_PATH}" "mix" "${MIX_STEP:--}"

    shopt -s nullglob
    local path tag family step
    for path in /root/ast_eval_runtime/models/grpo_medbullets_ood_* /root/models/grpo_medbullets_ood_*; do
        [[ -f "${path}/config.json" ]] || continue
        tag="$(basename "${path}")"
        family="medbullets_ood"
        step="-"
        [[ "${tag}" == *root_only* ]] && family="root_only"
        [[ "${tag}" == *sub_only* ]] && family="sub_only"
        [[ "${tag}" == *mix* || "${tag}" == *mixed* ]] && family="mix"
        append_model_spec "${tag}" "${path}" "${family}" "${step}"
    done
    for path in /root/ast_eval_runtime/models/grpo_medbullets_ood_*/global_step_* /root/models/grpo_medbullets_ood_*/global_step_*; do
        [[ -f "${path}/config.json" ]] || continue
        tag="$(basename "$(dirname "${path}")")_$(basename "${path}")"
        family="medbullets_ood"
        step="${path##*_}"
        [[ "${tag}" == *root_only* ]] && family="root_only"
        [[ "${tag}" == *sub_only* ]] && family="sub_only"
        [[ "${tag}" == *mix* || "${tag}" == *mixed* ]] && family="mix"
        append_model_spec "${tag}" "${path}" "${family}" "${step}"
    done
    shopt -u nullglob

    awk -F '\t' '!seen[$2]++' "${MODEL_SPEC_FILE}" >"${MODEL_SPEC_FILE}.tmp"
    mv "${MODEL_SPEC_FILE}.tmp" "${MODEL_SPEC_FILE}"
}

if [[ -z "${MODEL_SPEC_FILE}" || ! -f "${MODEL_SPEC_FILE}" ]]; then
    build_default_model_spec
fi

mapfile -t MODEL_LINES < <(awk -F '\t' 'NF >= 2 && $1 !~ /^#/ {print}' "${MODEL_SPEC_FILE}")
if (( ${#MODEL_LINES[@]} == 0 )); then
    echo "No HF-loadable model specs found in ${MODEL_SPEC_FILE}" >&2
    echo "Set MODEL_SPEC_FILE, or provide ROOT_ONLY_HF_MODEL_PATH/SUB_ONLY_HF_MODEL_PATH/MIX_HF_MODEL_PATH after checkpoint conversion." >&2
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

num_fewshot_for_benchmark() {
    local benchmark="$1"
    if [[ "${benchmark}" == mmlu_* ]]; then
        echo "${MMLU_NUM_FEWSHOT}"
    else
        echo "${MEDICAL_NUM_FEWSHOT}"
    fi
}

run_one() {
    local line="$1"
    local benchmark="$2"
    local gpu_id="$3"
    local num_fewshot="$4"
    local tag path family step log_file

    IFS=$'\t' read -r tag path family step _ <<<"${line}"
    log_file="${LOG_ROOT}/${RUN_TAG}__${benchmark}__${tag}.log"

    {
        log "benchmark=${benchmark} tag=${tag} family=${family:-unknown} step=${step:-unknown} gpu=${gpu_id} path=${path} fewshot=${num_fewshot}"
        local cmd=(
            env
            "CUDA_VISIBLE_DEVICES=${gpu_id}"
            "VLLM_WORKER_MULTIPROC_METHOD=spawn"
            "RUN_ROOT=${ATOMIC_EVAL_RUN_ROOT}"
            "LM_EVAL_PYTHON=${LM_EVAL_PYTHON}"
            "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
            "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
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
            --model-alias "${tag}=${path}"
            --benchmarks "${benchmark}"
            --backend vllm
            --dtype bfloat16
            --batch-size auto
            --max-gen-toks "${MAX_GEN_TOKS}"
            --temperature "${TEMPERATURE}"
            --top-p "${TOP_P}"
            --top-k "${TOP_K}"
            --min-p "${MIN_P}"
            --seed 7
            --num-fewshot "${num_fewshot}"
            --metadata-file-name "run_metadata.${tag}.${benchmark}.json"
        )
        if [[ "${APPLY_CHAT_TEMPLATE}" == "true" ]]; then
            cmd+=(--apply-chat-template)
        else
            cmd+=(--no-apply-chat-template)
        fi
        if [[ "${ENABLE_THINKING}" == "true" ]]; then
            cmd+=(--enable-thinking)
        else
            cmd+=(--disable-thinking)
        fi
        if [[ "${FEWSHOT_AS_MULTITURN}" == "true" ]]; then
            cmd+=(--fewshot-as-multiturn)
        fi
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
    local num_fewshot="$2"
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
        run_one "${line}" "${benchmark}" "${slot_gpu_id}" "${num_fewshot}" &
        pid="$!"
        PID_TO_GPU["${pid}"]="${slot_gpu_id}"
        PID_TO_LABEL["${pid}"]="${label}"
        next_model_idx=$((next_model_idx + 1))
        running=$((running + 1))
        log "launched ${label} pid=${pid} queue=${next_model_idx}/${total_models}"
    }

    log "starting benchmark=${benchmark} models=${total_models} fewshot=${num_fewshot} scheduler=gpu_slot_pool"
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
    MEDICAL_BENCHMARKS="${MEDICAL_BENCHMARKS}" MEDICAL_NUM_FEWSHOT="${MEDICAL_NUM_FEWSHOT}" MMLU_NUM_FEWSHOT="${MMLU_NUM_FEWSHOT}" \
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
    "benchmarks": os.environ["MEDICAL_BENCHMARKS"].split(),
    "medical_num_fewshot": int(os.environ["MEDICAL_NUM_FEWSHOT"]),
    "mmlu_num_fewshot": int(os.environ["MMLU_NUM_FEWSHOT"]),
}
(Path(os.environ["RUN_ROOT"]) / "run_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

main() {
    log "run_tag=${RUN_TAG}"
    log "run_root=${RUN_ROOT}"
    log "model_spec_file=${MODEL_SPEC_FILE}"
    log "gpu_ids=${GPU_IDS}"
    log "medical_benchmarks=${MEDICAL_BENCHMARKS}"
    cat "${MODEL_SPEC_FILE}"

    local benchmark num_fewshot
    for benchmark in ${MEDICAL_BENCHMARKS}; do
        num_fewshot="$(num_fewshot_for_benchmark "${benchmark}")"
        run_benchmark_group "${benchmark}" "${num_fewshot}"
    done

    write_global_metadata
    if [[ "${SUMMARIZE}" == "true" && "${DRY_RUN}" != "true" ]]; then
        "${LM_EVAL_PYTHON}" "${SCRIPT_ROOT}/summarize_external_math_eval.py" --run-root "${RUN_ROOT}" || true
    fi
    log "medical evaluation queue finished"
}

main "$@"
