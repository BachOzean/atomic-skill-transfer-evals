#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_ROOT="${REPO_ROOT}/scripts"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}}"

LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-${REPO_ROOT}/.venvs/lm_eval_overlay/bin/python}"
RUN_TAG="${RUN_TAG:-external_math_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/outputs/external_math}"
CACHE_ROOT="${CACHE_ROOT:-${RUN_ROOT}/.cache}"
TMP_ROOT="${TMP_ROOT:-${RUN_ROOT}/tmp}"
MODELS="${MODELS:-base}"
BENCHMARKS="${BENCHMARKS:-aime24,aime25,math500,olympiadbench,omni_math,gpqa_diamond}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${RUN_ROOT}/models/Qwen3-1.7B}"
BACKEND="${BACKEND:-vllm}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-38912}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
SEED="${SEED:-7}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-true}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-false}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
SYSTEM_INSTRUCTION="${SYSTEM_INSTRUCTION-__AUTO__}"
GPQA_LOCAL_DATASET_DIR="${GPQA_LOCAL_DATASET_DIR:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
SUMMARIZE="${SUMMARIZE:-true}"

export RUN_ROOT
export CACHE_ROOT
export TMPDIR="${TMPDIR:-${TMP_ROOT}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}"

CMD=(
  "${LM_EVAL_PYTHON}"
  "${SCRIPT_ROOT}/run_external_math_eval.py"
  --lm-eval-python "${LM_EVAL_PYTHON}"
  --output-root "${OUTPUT_ROOT}"
  --run-tag "${RUN_TAG}"
  --models "${MODELS}"
  --benchmarks "${BENCHMARKS}"
  --base-model-path "${BASE_MODEL_PATH}"
  --backend "${BACKEND}"
  --batch-size "${BATCH_SIZE}"
  --max-gen-toks "${MAX_GEN_TOKS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --min-p "${MIN_P}"
  --seed "${SEED}"
)

if [[ "${APPLY_CHAT_TEMPLATE}" == "true" ]]; then
  CMD+=(--apply-chat-template)
else
  CMD+=(--no-apply-chat-template)
fi

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  CMD+=(--chat-template "${CHAT_TEMPLATE}")
fi

if [[ "${FEWSHOT_AS_MULTITURN}" == "true" ]]; then
  CMD+=(--fewshot-as-multiturn)
fi

if [[ "${ENABLE_THINKING}" == "true" ]]; then
  CMD+=(--enable-thinking)
else
  CMD+=(--disable-thinking)
fi

if [[ "${SYSTEM_INSTRUCTION}" == "__AUTO__" ]]; then
  :
elif [[ -n "${SYSTEM_INSTRUCTION}" ]]; then
  CMD+=(--system-instruction "${SYSTEM_INSTRUCTION}")
else
  CMD+=(--no-system-instruction)
fi

if [[ -n "${ORIGIN_ONLY_MODEL_PATH:-}" ]]; then
  CMD+=(--origin-only-model-path "${ORIGIN_ONLY_MODEL_PATH}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ -n "${DEVICE:-}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

if [[ -n "${GPQA_LOCAL_DATASET_DIR}" ]]; then
  CMD+=(--gpqa-local-dataset-dir "${GPQA_LOCAL_DATASET_DIR}")
fi

if [[ -n "${NUM_FEWSHOT:-}" ]]; then
  CMD+=(--num-fewshot "${NUM_FEWSHOT}")
fi

if [[ -n "${EXTRA_MODEL_ARG:-}" ]]; then
  CMD+=(--extra-model-arg "${EXTRA_MODEL_ARG}")
fi

if [[ -n "${EXTRA_CLI_ARG:-}" ]]; then
  CMD+=(--extra-cli-arg "${EXTRA_CLI_ARG}")
fi

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  CMD+=(--dry-run)
fi

printf 'COMMAND='
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

if [[ "${SUMMARIZE}" == "true" && "${DRY_RUN:-false}" != "true" ]]; then
  SUMMARY_CMD=(
    "${LM_EVAL_PYTHON}"
    "${SCRIPT_ROOT}/summarize_external_math_eval.py"
    --run-root "${OUTPUT_ROOT}/${RUN_TAG}"
  )
  printf 'SUMMARY_COMMAND='
  printf '%q ' "${SUMMARY_CMD[@]}"
  printf '\n'
  "${SUMMARY_CMD[@]}"
fi
