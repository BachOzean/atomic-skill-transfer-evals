#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_ROOT="${REPO_ROOT}/scripts"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}}"

LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-${REPO_ROOT}/.venvs/lm_eval_overlay/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/outputs/musr}"
RUN_TAG="${RUN_TAG:-musr_$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
CACHE_ROOT="${CACHE_ROOT:-${RUN_ROOT}/.cache}"
TMP_ROOT="${TMP_ROOT:-${RUN_ROOT}/tmp}"

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${TMP_ROOT}"
export RUN_ROOT
export CACHE_ROOT
export TMPDIR="${TMPDIR:-${TMP_ROOT}}"

CMD=(
  "${LM_EVAL_PYTHON}"
  "${SCRIPT_ROOT}/run_musr_eval.py"
  --lm-eval-python "${LM_EVAL_PYTHON}"
  --output-root "${OUTPUT_ROOT}"
  --run-tag "${RUN_TAG}"
  --model-path "${MODEL_PATH}"
  --model-tag "${MODEL_TAG}"
)

if [[ -n "${TASK_NAME:-}" ]]; then
  CMD+=(--task-name "${TASK_NAME}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "${LIMIT}")
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
