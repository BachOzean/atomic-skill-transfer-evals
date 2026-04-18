# H200 非 Slurm 评估链路迁移指南

这份文档面向另一台不使用 Slurm 的 H200 服务器，目标是让那台机器上的 Codex 能直接：

1. `git clone` 本仓库
2. 建好 `lm_eval_overlay`
3. 校验 `lm-evaluation-harness`、本地 task 和关键版本
4. 跑通 `external_math` 与 `musr` 的 smoke test

## 1. 迁移边界

本仓库只迁移可复现的评估链路：

- `external_math`
- `musr`
- `lm_eval_tasks/` 下的本地 task 扩展
- 环境 bootstrap、版本快照和 verify 工具

不迁移：

- 模型权重
- 历史 `outputs/` 结果
- 私有或大数据集
- Slurm 专用提交包装
- `PHYBench`

## 2. 推荐目录约定

建议在 H200 机器上把 repo 和运行时数据分开：

```bash
export REPO_DIR=/data/run01/$USER/_projects/atomic-skill-transfer-evals
export RUN_ROOT=/data/run01/$USER/ast_eval_runtime
```

然后把大文件放到：

- 模型：`$RUN_ROOT/models/...`
- GPQA 本地数据：`$RUN_ROOT/data/gpqa`
- 输出：`$RUN_ROOT/outputs`
- 缓存：`$RUN_ROOT/.cache`
- 临时目录：`$RUN_ROOT/tmp`

## 3. Clone 与环境准备

### 3.1 clone

```bash
git clone <your-private-github-url> "$REPO_DIR"
cd "$REPO_DIR"
```

### 3.2 基座环境

当前参考基线来自原机器的 `verl_cpython`：

- Python `3.10.19`
- `vllm==0.17.1`
- `datasets==4.6.0`
- `flashinfer-python==0.6.6`
- 其余参考见：
  - `env/lock/conda-verl_cpython-export.txt`
  - `env/lock/core-versions.json`

这里不强制你在 H200 上复刻完全相同的驱动/CUDA 版本，但要求：

- Python 大版本一致
- 上述核心 Python 包版本一致
- `lm_eval_overlay` 中的 overlay 包一致

### 3.3 创建 overlay

如果 H200 上已经有可用的基座 conda 环境：

```bash
CONDA_ENV_NAME=verl_cpython bash scripts/bootstrap_eval_env.sh
```

如果你不想走 conda 自动激活，也可以显式指定基座解释器：

```bash
BASE_PYTHON=/path/to/base/env/bin/python bash scripts/bootstrap_eval_env.sh
```

脚本会做的事：

- 创建 `.venvs/lm_eval_overlay`
- 打开 `--system-site-packages`
- 安装 `env/requirements-overlay.txt`
- 创建默认的 cache/tmp/output 目录

## 4. 严格校验

bootstrap 后先跑：

```bash
python scripts/verify_eval_stack.py
```

它会检查：

- `lm_eval` / `transformers` / `vllm` / `datasets` / `flashinfer-python` 版本
- `leaderboard_musr`
- `leaderboard_gpqa_diamond_local`
- `ext_math_olympiadbench`
- `ext_math_omni_math_rule`
- `hendrycks_math500`

如果你暂时只想看报告、不因为版本漂移失败：

```bash
python scripts/verify_eval_stack.py --allow-version-drift
```

## 5. 运行时环境变量

最少建议导出这些变量：

```bash
export RUN_ROOT=/data/run01/$USER/ast_eval_runtime
export LM_EVAL_PYTHON=$REPO_DIR/.venvs/lm_eval_overlay/bin/python
export BASE_MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B
export GPQA_LOCAL_DATASET_DIR=$RUN_ROOT/data/gpqa
```

如果跑 `vllm`，再加：

```bash
export GPU_MEMORY_UTILIZATION=0.90
export TENSOR_PARALLEL_SIZE=1
export MAX_MODEL_LEN=40960
export MAX_NUM_SEQS=16
export SWAP_SPACE=8
```

`musr` 建议单独覆盖：

```bash
export MAX_MODEL_LEN=8192
```

## 6. Smoke Test

### 6.1 external_math 最小 smoke

```bash
BENCHMARKS=aime24 \
LIMIT=1 \
bash launchers/run_external_math_benchmarks.sh
```

验收点：

- 生成 `outputs/external_math/<RUN_TAG>/run_metadata.json`
- 生成 `summary/summary_manifest.json`
- `run_metadata.json` 里能看到完整命令和参数

### 6.2 external_math 的 metric 型 benchmark smoke

MMLU-Pro：

```bash
BENCHMARKS=mmlu_pro \
LIMIT=1 \
bash launchers/run_external_math_benchmarks.sh
```

GPQA local task：

```bash
BENCHMARKS=gpqa_diamond \
LIMIT=1 \
GPQA_LOCAL_DATASET_DIR=$RUN_ROOT/data/gpqa \
bash launchers/run_external_math_benchmarks.sh
```

验收点：

- `resolved_tasks.gpqa_diamond == leaderboard_gpqa_diamond_local`
- `task_metadata.gpqa_local_dataset_dir` 被正确写入 `run_metadata.json`

### 6.3 musr smoke

```bash
MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B \
MODEL_TAG=base \
LIMIT=30 \
MAX_MODEL_LEN=8192 \
bash launchers/run_musr_eval.sh
```

当前参考参数：

- backend: `vllm`
- `dtype=bfloat16`
- `batch_size=auto`
- `temperature=0`
- `top_p=1.0`
- `top_k=1`
- `min_p=0`
- `max_gen_toks=256`
- `apply_chat_template=true`
- `enable_thinking=true`

## 7. Full Run 约束

### external_math

默认参数由 `launchers/run_external_math_benchmarks.sh` 固定：

- `backend=vllm`
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `seed=7`
- `max_gen_toks=38912`
- `apply_chat_template=true`
- `enable_thinking=true`
- `fewshot_as_multiturn=false`
- `system_instruction=__AUTO__`

### musr

请保持：

- 关闭数学专用 system instruction
- 保留 chat template
- 保留 `enable_thinking=true`
- `max_model_len=8192`
- `max_gen_toks=256`

## 8. 结果对齐方法

迁移完成后，至少比较这三类信息：

1. `run_metadata.json`
   核对 backend、sampling、chat template、thinking、长度参数
2. `verify_eval_stack.py` 输出
   核对核心版本
3. smoke 输出目录结构
   核对 `results*.json`、`samples*.jsonl`、`summary/*.csv`

如果这三项一致，再开始 full run。

## 9. 给另一台机器上的 Codex 的最短执行序列

```bash
git clone <your-private-github-url> /data/run01/$USER/_projects/atomic-skill-transfer-evals
cd /data/run01/$USER/_projects/atomic-skill-transfer-evals
CONDA_ENV_NAME=verl_cpython bash scripts/bootstrap_eval_env.sh
python scripts/verify_eval_stack.py
export RUN_ROOT=/data/run01/$USER/ast_eval_runtime
export BASE_MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B
export GPQA_LOCAL_DATASET_DIR=$RUN_ROOT/data/gpqa
BENCHMARKS=aime24 LIMIT=1 bash launchers/run_external_math_benchmarks.sh
MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B LIMIT=30 MAX_MODEL_LEN=8192 bash launchers/run_musr_eval.sh
```

如果这串命令失败，优先看：

- `python scripts/verify_eval_stack.py` 的失败项
- 生成的 `run_metadata.json`
- overlay Python 是否真的是 `.venvs/lm_eval_overlay/bin/python`
