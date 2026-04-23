# 8 卡 H100 评估链路最短落地步骤

这份文档面向一台不使用 Slurm 的 8 卡 H100 服务器，目标是用最短路径把本仓库落地过去，并尽快判断链路是否可运行。

这里强调两件事：

1. 先验证链路，再追求 8 卡利用率。
2. 8 卡 H100 是机器条件，不代表所有模型都应该直接 `tensor_parallel_size=8`。

如果你当前还是用仓库里的示例模型 `Qwen3-1.7B`，最短且最低风险的做法依然是先单卡 smoke；等链路确认无误，再按需要切到 8 卡。

## 1. 迁移边界

本仓库会迁移：

- `external_math`
- `musr`
- `lm_eval_tasks/` 下的本地 task 扩展
- 环境 bootstrap、版本快照和 verify 工具

本仓库不会迁移：

- 模型权重
- 历史 `outputs/`
- 私有或大数据集
- Slurm 专用提交包装
- `PHYBench`

## 2. 推荐目录

建议把 repo 和运行时目录分开：

```bash
export REPO_DIR=/data/run01/$USER/_projects/atomic-skill-transfer-evals
export RUN_ROOT=/data/run01/$USER/ast_eval_runtime
```

建议把大文件放到：

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

### 3.2 基座环境要求

仓库假设你已经有一层可用的基础 Python/CUDA/vLLM/torch 环境。参考基线见：

- `env/lock/conda-verl_cpython-export.txt`
- `env/lock/core-versions.json`

至少要尽量对齐这些核心版本：

- Python `3.10.x`
- `vllm==0.17.1`
- `datasets==4.6.0`
- `flashinfer-python==0.6.6`

### 3.3 创建 overlay

如果 H100 机器上已经有 `verl_cpython`：

```bash
CONDA_ENV_NAME=verl_cpython bash scripts/bootstrap_eval_env.sh
```

如果你要显式指定基础解释器：

```bash
BASE_PYTHON=/path/to/base/env/bin/python bash scripts/bootstrap_eval_env.sh
```

脚本会创建：

- `.venvs/lm_eval_overlay`
- `RUN_ROOT/outputs`
- `RUN_ROOT/.cache`
- `RUN_ROOT/tmp`

## 4. 严格校验

优先用 overlay 里的解释器执行：

```bash
LM_EVAL_PYTHON=$REPO_DIR/.venvs/lm_eval_overlay/bin/python
"$LM_EVAL_PYTHON" scripts/verify_eval_stack.py
```

如果你暂时只想看报告，不因为版本漂移直接失败：

```bash
"$LM_EVAL_PYTHON" scripts/verify_eval_stack.py --allow-version-drift
```

## 5. 最少环境变量

先导出最少必需项：

```bash
export RUN_ROOT=/data/run01/$USER/ast_eval_runtime
export LM_EVAL_PYTHON=$REPO_DIR/.venvs/lm_eval_overlay/bin/python
export BASE_MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B
export MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B
export GPQA_LOCAL_DATASET_DIR=$RUN_ROOT/data/gpqa
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_SEQS=16
export SWAP_SPACE=8
```

说明：

- `GPQA_LOCAL_DATASET_DIR` 不是所有 benchmark 都必需，但跑 `gpqa_diamond` 时最好提前准备好。
- 如果没有本地 GPQA 数据，就必须保证新机器能访问并认证 Hugging Face 上的 `Idavidrein/gpqa`。

## 6. 最短落地路径

### 6.1 先做单卡 smoke

即使你在 8 卡 H100 上，也建议先用最保守路径确认链路。

external_math：

```bash
BENCHMARKS=aime24 \
LIMIT=1 \
bash launchers/run_external_math_benchmarks.sh
```

musr：

```bash
MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B \
LIMIT=30 \
MAX_MODEL_LEN=8192 \
bash launchers/run_musr_eval.sh
```

验收点：

- `outputs/external_math/<RUN_TAG>/run_metadata.json` 已生成
- `outputs/external_math/<RUN_TAG>/summary/summary_manifest.json` 已生成
- `outputs/musr/<RUN_TAG>/run_metadata.json` 已生成
- `run_metadata.json` 里能看到完整命令和关键参数

### 6.2 再切到 8 卡

只有在这两条 smoke 都通过后，再切 8 卡执行 full run。

## 7. 8 卡 H100 的推荐起步参数

如果你要在 H100 上真的用到 8 张卡，先从这组保守参数开始：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_MEMORY_UTILIZATION=0.90
export TENSOR_PARALLEL_SIZE=8
export MAX_NUM_SEQS=16
export SWAP_SPACE=8
```

上下文长度建议分开处理：

- `external_math`：`MAX_MODEL_LEN=40960`
- `musr`：`MAX_MODEL_LEN=8192`

注意：

- 对很小的模型，比如 `Qwen3-1.7B`，8 卡张量并行不是必须条件，甚至不一定是最快路径。
- 8 卡更适合更大 checkpoint，或者你明确需要把单个 vLLM 实例摊到所有 GPU 上。

## 8. external_math 的 8 卡命令

这里有一个仓库当前实现上的细节：

- `musr` runner 会直接读取 `TENSOR_PARALLEL_SIZE/MAX_MODEL_LEN/MAX_NUM_SEQS/SWAP_SPACE`
- `external_math` runner 当前默认只自动读取 `GPU_MEMORY_UTILIZATION`

所以，`external_math` 如果要明确跑 8 卡，请把并行参数通过 `EXTRA_MODEL_ARG` 传进去：

```bash
export MAX_MODEL_LEN=40960
export EXTRA_MODEL_ARG="tensor_parallel_size=8,max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS},swap_space=${SWAP_SPACE}"

BENCHMARKS=aime24,aime25,math500,olympiadbench,omni_math,gpqa_diamond \
MODELS=base \
bash launchers/run_external_math_benchmarks.sh
```

如果你还要跑 `origin_only`：

```bash
export ORIGIN_ONLY_MODEL_PATH=$RUN_ROOT/models/qwen3-1.7B-origin-only
export EXTRA_MODEL_ARG="tensor_parallel_size=8,max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS},swap_space=${SWAP_SPACE}"

MODELS=base,origin_only \
bash launchers/run_external_math_benchmarks.sh
```

运行后建议检查 `run_metadata.json` 里的 `command` 字段，确认 `tensor_parallel_size=8` 等参数已经真实传入。

## 9. musr 的 8 卡命令

`musr` runner 会直接读取这些环境变量，所以命令更直接：

```bash
export TENSOR_PARALLEL_SIZE=8
export MAX_MODEL_LEN=8192

MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B \
MODEL_TAG=base \
LIMIT=30 \
bash launchers/run_musr_eval.sh
```

如果你是 full run，就去掉 `LIMIT=30`。

## 10. 最短执行序列

如果你想把命令压缩到最短，可以按这个顺序执行：

```bash
export REPO_DIR=/data/run01/$USER/_projects/atomic-skill-transfer-evals
export RUN_ROOT=/data/run01/$USER/ast_eval_runtime

git clone <your-private-github-url> "$REPO_DIR"
cd "$REPO_DIR"

CONDA_ENV_NAME=verl_cpython bash scripts/bootstrap_eval_env.sh

export LM_EVAL_PYTHON=$REPO_DIR/.venvs/lm_eval_overlay/bin/python
export BASE_MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B
export MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B
export GPQA_LOCAL_DATASET_DIR=$RUN_ROOT/data/gpqa
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_SEQS=16
export SWAP_SPACE=8

"$LM_EVAL_PYTHON" scripts/verify_eval_stack.py

BENCHMARKS=aime24 LIMIT=1 bash launchers/run_external_math_benchmarks.sh

MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B LIMIT=30 MAX_MODEL_LEN=8192 bash launchers/run_musr_eval.sh

export MAX_MODEL_LEN=40960
export EXTRA_MODEL_ARG="tensor_parallel_size=8,max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS},swap_space=${SWAP_SPACE}"
BENCHMARKS=aime24,aime25,math500,olympiadbench,omni_math,gpqa_diamond bash launchers/run_external_math_benchmarks.sh

export TENSOR_PARALLEL_SIZE=8
export MAX_MODEL_LEN=8192
MODEL_PATH=$RUN_ROOT/models/Qwen3-1.7B bash launchers/run_musr_eval.sh
```

## 11. 失败时先看什么

优先排查这几项：

1. `LM_EVAL_PYTHON` 是否真的是 `.venvs/lm_eval_overlay/bin/python`
2. `scripts/verify_eval_stack.py` 是否已经通过
3. `BASE_MODEL_PATH` 或 `MODEL_PATH` 是否存在，且目录下有 `config.json`
4. `GPQA_LOCAL_DATASET_DIR` 是否存在，或者 Hugging Face 访问是否已认证
5. `external_math` 的 `EXTRA_MODEL_ARG` 是否真的把 `tensor_parallel_size=8` 传入了 `command`
6. `musr` 的 `run_metadata.json` 里是否已经记录 `tensor_parallel_size=8` 和 `max_model_len=8192`
