#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="${REPO_ROOT}/env/lock"
LM_EVAL_PYTHON="${LM_EVAL_PYTHON:-${REPO_ROOT}/.venvs/lm_eval_overlay/bin/python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl_cpython}"

mkdir -p "${LOCK_DIR}"

if [[ ! -x "${LM_EVAL_PYTHON}" ]]; then
  echo "LM_EVAL_PYTHON is not executable: ${LM_EVAL_PYTHON}" >&2
  exit 1
fi

conda list -n "${CONDA_ENV_NAME}" --export > "${LOCK_DIR}/conda-${CONDA_ENV_NAME}-export.txt"
"${LM_EVAL_PYTHON}" -m pip freeze --local > "${LOCK_DIR}/lm_eval_overlay-local-freeze.txt"
"${LM_EVAL_PYTHON}" -m pip show lm_eval transformers vllm datasets flashinfer-python > "${LOCK_DIR}/core-package-show.txt"
"${LM_EVAL_PYTHON}" - <<'PY' > "${LOCK_DIR}/core-versions.json"
import importlib.metadata
import json

packages = ["lm_eval", "transformers", "vllm", "datasets", "flashinfer-python", "torch"]
payload = {}
for package in packages:
    try:
        payload[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        payload[package] = None
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Wrote snapshots under ${LOCK_DIR}"
