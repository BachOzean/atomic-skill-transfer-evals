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
- per-server runtime profiles under `servers/`
- migration and operator notes under `docs/`

Not included:

- model weights
- private or large datasets
- historical result trees
- runtime outputs, Slurm logs, or cache directories

## Layout

- `scripts/`: Python entrypoints and helper modules
- `launchers/`: shell entrypoints for single-machine runs
- `lm_eval_tasks/`: local task YAML and dataset helpers
- `env/`: overlay requirements, runtime examples, lock snapshots
- `servers/`: per-machine runtime profiles and scheduler notes
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
source servers/a800/runtime.env.example
export BASE_MODEL_PATH=/data/home/$USER/run/models/Qwen3-1.7B
export MODEL_PATH="$BASE_MODEL_PATH"  # only needed by the A800 Slurm wrapper
```

4. Smoke test external math:

```bash
BENCHMARKS=aime24 LIMIT=1 bash launchers/run_external_math_benchmarks.sh
```

5. Run AIME repeat-8 with one model load and eight generations per problem:

```bash
BENCHMARKS=aime24_repeat8,aime25_repeat8 MAX_GEN_TOKS=32768 MAX_MODEL_LEN=40960 \
  bash launchers/run_external_math_benchmarks.sh
```

6. Smoke test MuSR:

```bash
MODEL_PATH=/data/run01/$USER/models/Qwen3-1.7B LIMIT=30 bash launchers/run_musr_eval.sh
```

The default overlay interpreter lives at `.venvs/lm_eval_overlay/bin/python`.
