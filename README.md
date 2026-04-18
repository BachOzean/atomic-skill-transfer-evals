# Atomic Skill Transfer Evals

Portable evaluation stack for the atomic-skill-transfer experiments.

This repo extracts the reproducible evaluation pieces from the original research
workspace into a standalone tree that can be pushed to a private GitHub repo and
cloned on another machine.

Included:

- `external_math` evaluation runner and summarizer
- `musr` evaluation runner
- local `lm-eval-harness` task overrides under `lm_eval_tasks/`
- environment bootstrap, snapshot export, and verification tooling
- H200 non-Slurm migration notes under [docs/2026-04-18_h200_eval_migration_guide_zh.md](/data/home/scyb494/run/_projects/atomic-skill-transfer-evals/docs/2026-04-18_h200_eval_migration_guide_zh.md)

Not included:

- model weights
- private or large datasets
- historical result trees
- Slurm-only submission wrappers

## Layout

- `scripts/`: Python entrypoints and helper modules
- `launchers/`: shell entrypoints for single-machine runs
- `lm_eval_tasks/`: local task YAML and dataset helpers
- `env/`: overlay requirements, runtime examples, lock snapshots
- `docs/`: migration and operator notes

## Quick Start

1. Prepare a base Python stack that already contains the CUDA/vLLM/torch layer.
   The reference stack is documented in `env/lock/`.
2. Run:

```bash
bash scripts/bootstrap_eval_env.sh
python scripts/verify_eval_stack.py
```

3. Set runtime paths, for example:

```bash
export RUN_ROOT=/data/run01/$USER/ast_eval_runtime
export BASE_MODEL_PATH=/data/run01/$USER/models/Qwen3-1.7B
export GPQA_LOCAL_DATASET_DIR=/data/run01/$USER/data/gpqa
```

4. Smoke test external math:

```bash
BENCHMARKS=aime24 LIMIT=1 bash launchers/run_external_math_benchmarks.sh
```

5. Smoke test MuSR:

```bash
MODEL_PATH=/data/run01/$USER/models/Qwen3-1.7B LIMIT=30 bash launchers/run_musr_eval.sh
```

The default overlay interpreter lives at `.venvs/lm_eval_overlay/bin/python`.
