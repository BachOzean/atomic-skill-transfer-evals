#!/usr/bin/env bash
#SBATCH --job-name=ast-ext-math
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/%u/run/_projects/atomic-skill-transfer-evals/runtime/slurm_logs/%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_CANDIDATE="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -x "${SLURM_SUBMIT_DIR}/launchers/run_external_math_benchmarks.sh" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="${REPO_ROOT_CANDIDATE}"
fi
LAUNCHER="${REPO_ROOT}/launchers/run_external_math_benchmarks.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl_cpython}"
CONDA_ROOT="${CONDA_ROOT:-/data/apps/miniforge3/25.11.0-1}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runtime}"
LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-${REPO_ROOT}/.venvs/lm_eval_overlay/bin/python}"
FLASH_ATTN_OVERLAY="${FLASH_ATTN_OVERLAY:-/data/home/scyb494/run/vendor/verl-main/.flash_attn_sm80_overlay}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/outputs/external_math}"
CACHE_ROOT="${CACHE_ROOT:-${RUN_ROOT}/.cache}"
TMP_ROOT="${TMP_ROOT:-${RUN_ROOT}/tmp}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
RUN_TAG="${RUN_TAG:-quick_external_math_$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}}"
BENCHMARKS="${BENCHMARKS:-aime24,aime25,gpqa_diamond}"
BACKEND="${BACKEND:-vllm}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-8192}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
MIN_P="${MIN_P:-0}"
SEED="${SEED:-7}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-true}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-false}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
SYSTEM_INSTRUCTION="${SYSTEM_INSTRUCTION-__AUTO__}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
SUMMARIZE="${SUMMARIZE:-true}"
DRY_RUN="${DRY_RUN:-false}"
LIMIT="${LIMIT:-}"
DEVICE="${DEVICE:-}"
EXTRA_MODEL_ARG="${EXTRA_MODEL_ARG:-}"
EXTRA_CLI_ARG="${EXTRA_CLI_ARG:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
SWAP_SPACE="${SWAP_SPACE:-8}"

if [[ -n "${GPQA_LOCAL_DATASET_DIR:-}" ]]; then
    GPQA_LOCAL_DATASET_DIR="${GPQA_LOCAL_DATASET_DIR}"
elif [[ -d "/data/home/scyb494/run/data/gpqa" ]]; then
    GPQA_LOCAL_DATASET_DIR="/data/home/scyb494/run/data/gpqa"
else
    GPQA_LOCAL_DATASET_DIR="${RUN_ROOT}/data/gpqa"
fi

if [[ ! -x "${LM_EVAL_PYTHON}" ]]; then
    echo "LM_EVAL_PYTHON is not executable: ${LM_EVAL_PYTHON}" >&2
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "MODEL_PATH is missing config.json: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ -n "${TOKENIZER_PATH}" ]]; then
    if [[ ! -d "${TOKENIZER_PATH}" ]]; then
        echo "TOKENIZER_PATH does not exist: ${TOKENIZER_PATH}" >&2
        exit 1
    fi
    if [[ ! -f "${TOKENIZER_PATH}/tokenizer_config.json" ]]; then
        echo "TOKENIZER_PATH is missing tokenizer_config.json: ${TOKENIZER_PATH}" >&2
        exit 1
    fi
fi

MODEL_CONFIG_MAX_LEN="$(
    python - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

value = cfg.get("max_position_embeddings")
if isinstance(value, (int, float)) and value > 0:
    print(int(value))
PY
)"

REQUESTED_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
REQUESTED_MAX_GEN_TOKS="${MAX_GEN_TOKS}"
CONTEXT_HEADROOM_TOKS=$(( REQUESTED_MAX_MODEL_LEN - REQUESTED_MAX_GEN_TOKS ))
if (( CONTEXT_HEADROOM_TOKS < 1 )); then
    CONTEXT_HEADROOM_TOKS=2048
fi

if [[ -n "${MODEL_CONFIG_MAX_LEN}" && "${MAX_MODEL_LEN}" -gt "${MODEL_CONFIG_MAX_LEN}" ]]; then
    echo "Clamping MAX_MODEL_LEN from ${MAX_MODEL_LEN} to ${MODEL_CONFIG_MAX_LEN} based on ${MODEL_PATH}/config.json"
    MAX_MODEL_LEN="${MODEL_CONFIG_MAX_LEN}"
fi

MAX_ALLOWED_GEN_TOKS=$(( MAX_MODEL_LEN - CONTEXT_HEADROOM_TOKS ))
if (( MAX_ALLOWED_GEN_TOKS < 1 )); then
    MAX_ALLOWED_GEN_TOKS=$(( MAX_MODEL_LEN - 1 ))
fi
if (( MAX_ALLOWED_GEN_TOKS < 1 )); then
    echo "Derived MAX_ALLOWED_GEN_TOKS=${MAX_ALLOWED_GEN_TOKS} is invalid for MODEL_PATH=${MODEL_PATH}" >&2
    exit 1
fi
if (( MAX_GEN_TOKS > MAX_ALLOWED_GEN_TOKS )); then
    echo "Clamping MAX_GEN_TOKS from ${MAX_GEN_TOKS} to ${MAX_ALLOWED_GEN_TOKS} to preserve ${CONTEXT_HEADROOM_TOKS} tokens of prompt headroom"
    MAX_GEN_TOKS="${MAX_ALLOWED_GEN_TOKS}"
fi

if [[ ! -x "${LAUNCHER}" ]]; then
    echo "Launcher not found or not executable: ${LAUNCHER}" >&2
    exit 1
fi

if [[ "${BACKEND}" == "vllm" ]]; then
    if [[ ! -e "${FLASH_ATTN_OVERLAY}/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so" ]]; then
        echo "Expected flash-attn overlay missing under ${FLASH_ATTN_OVERLAY}" >&2
        exit 1
    fi
    if [[ ! -f "${FLASH_ATTN_OVERLAY}/flash_attn/ops/triton/rotary.py" ]]; then
        echo "Expected flash-attn rotary source missing under ${FLASH_ATTN_OVERLAY}" >&2
        exit 1
    fi
fi

if command -v module >/dev/null 2>&1; then
    module load cuda/12.9 || true
fi

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    set +u
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda deactivate >/dev/null 2>&1 || true
    conda activate "${CONDA_ENV_NAME}"
    set -u
fi

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-${TMP_ROOT}}"

if [[ "${BACKEND}" == "vllm" ]]; then
    export PYTHONPATH="${FLASH_ATTN_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    if [[ -n "${CONDA_PREFIX:-}" && -e "${CONDA_PREFIX}/lib/libstdc++.so.6" && -e "${CONDA_PREFIX}/lib/libgcc_s.so.1" ]]; then
        export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${CONDA_PREFIX}/lib/libgcc_s.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
fi

mkdir -p "${RUN_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}" "${RUN_ROOT}/slurm_logs"

echo "Running portable external_math benchmarks on gpu_a800"
echo "REPO_ROOT=${REPO_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "MODEL_PATH=${MODEL_PATH}"
if [[ -n "${TOKENIZER_PATH}" ]]; then
    echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
fi
echo "RUN_TAG=${RUN_TAG}"
echo "BENCHMARKS=${BENCHMARKS}"
echo "BACKEND=${BACKEND}"
echo "MAX_GEN_TOKS=${MAX_GEN_TOKS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
if [[ "${BACKEND}" == "vllm" ]]; then
    echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
    echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
    echo "SWAP_SPACE=${SWAP_SPACE}"
fi

export RUN_ROOT
export LM_EVAL_PYTHON
export OUTPUT_ROOT
export CACHE_ROOT
export TMP_ROOT
export RUN_TAG
export MODELS="base"
export BASE_MODEL_PATH="${MODEL_PATH}"
export BENCHMARKS
export BACKEND
export BATCH_SIZE
export MAX_GEN_TOKS
export NUM_FEWSHOT
export TEMPERATURE
export TOP_P
export TOP_K
export MIN_P
export SEED
export APPLY_CHAT_TEMPLATE
export FEWSHOT_AS_MULTITURN
export ENABLE_THINKING
export SYSTEM_INSTRUCTION
export GPQA_LOCAL_DATASET_DIR
export CHAT_TEMPLATE
export SUMMARIZE
export DRY_RUN

if [[ "${BACKEND}" == "vllm" ]]; then
    VLLM_MODEL_ARGS="tensor_parallel_size=${TENSOR_PARALLEL_SIZE},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS},swap_space=${SWAP_SPACE}"
    if [[ -n "${TOKENIZER_PATH}" ]]; then
        VLLM_MODEL_ARGS="${VLLM_MODEL_ARGS},tokenizer=${TOKENIZER_PATH}"
    fi
    if [[ -n "${EXTRA_MODEL_ARG}" ]]; then
        export EXTRA_MODEL_ARG="${VLLM_MODEL_ARGS},${EXTRA_MODEL_ARG}"
    else
        export EXTRA_MODEL_ARG="${VLLM_MODEL_ARGS}"
    fi
elif [[ -n "${TOKENIZER_PATH}" ]]; then
    if [[ -n "${EXTRA_MODEL_ARG}" ]]; then
        export EXTRA_MODEL_ARG="tokenizer=${TOKENIZER_PATH},${EXTRA_MODEL_ARG}"
    else
        export EXTRA_MODEL_ARG="tokenizer=${TOKENIZER_PATH}"
    fi
fi

if [[ -n "${LIMIT}" ]]; then
    export LIMIT
fi

if [[ -n "${DEVICE}" ]]; then
    export DEVICE
fi

if [[ -n "${EXTRA_MODEL_ARG}" ]]; then
    export EXTRA_MODEL_ARG
fi

if [[ -n "${EXTRA_CLI_ARG}" ]]; then
    export EXTRA_CLI_ARG
fi

bash "${LAUNCHER}"
