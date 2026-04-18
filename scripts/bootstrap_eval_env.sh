#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}}"
OVERLAY_DIR="${OVERLAY_DIR:-${REPO_ROOT}/.venvs/lm_eval_overlay}"
OVERLAY_REQUIREMENTS="${REPO_ROOT}/env/requirements-overlay.txt"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl_cpython}"

activate_conda_if_needed() {
  if [[ -n "${BASE_PYTHON:-}" ]]; then
    return
  fi

  if [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
    return
  fi

  if [[ -n "${CONDA_ROOT:-}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    return
  fi

  echo "Could not activate conda env ${CONDA_ENV_NAME}. Set BASE_PYTHON or provide conda." >&2
  exit 1
}

activate_conda_if_needed

BASE_PYTHON="${BASE_PYTHON:-$(python -c 'import sys; print(sys.executable)')}"
if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "BASE_PYTHON is not executable: ${BASE_PYTHON}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/.venvs" "${RUN_ROOT}" "${RUN_ROOT}/outputs" "${RUN_ROOT}/.cache" "${RUN_ROOT}/tmp"

if [[ ! -x "${OVERLAY_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${OVERLAY_DIR}"
fi

"${OVERLAY_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
"${OVERLAY_DIR}/bin/python" -m pip install -r "${OVERLAY_REQUIREMENTS}"

cat <<EOF
Bootstrap complete.

Repo root: ${REPO_ROOT}
Run root: ${RUN_ROOT}
Base python: ${BASE_PYTHON}
Overlay python: ${OVERLAY_DIR}/bin/python

Recommended exports:
  export RUN_ROOT=${RUN_ROOT}
  export LM_EVAL_PYTHON=${OVERLAY_DIR}/bin/python
  export CACHE_ROOT=${RUN_ROOT}/.cache
  export TMPDIR=${RUN_ROOT}/tmp

Next:
  python ${REPO_ROOT}/scripts/verify_eval_stack.py
EOF
