# 2026-04-14 External Math Eval 运维指南

## 1. 文档目的

这份文档整理 2026-04-13 到 2026-04-14 这两次会话里 external math 评测链路的可复用经验，目标是让后续的人能够：

- 直接找到当前可用的脚本入口。
- 知道当前默认参数是什么，为什么这么设。
- 知道在这台集群上用 Slurm 跑这套评测时有哪些调度和环境坑。
- 知道当前三个模型为什么速度差异明显。
- 知道排障时应该先查什么，避免重复踩坑。

这份文档覆盖的是当前可工作的 `gpu_a800 + Slurm + vLLM + lm-eval-harness` 路径。

## 2. 当前评测对象

当前实际对比的三个模型路径是：

- `base`
  - `/data/home/scyb494/run/models/Qwen3-1.7B`
- `sub_only_step1800`
  - `/data/home/scyb494/run/models/qwen3-1.7B-sub-only-step1800`
- `zero_only_step310`
  - `/data/home/scyb494/run/verl_a800_checkpoints/grpo_zero_only_origin_only_qwen3_1_7b_a800_flash_20260410_195340/global_step_310/actor/huggingface`

当前 benchmark 套件是：

- `math500`
- `aime24`
- `aime25`
- `olympiadbench`
- `omni_math`

对应的 task 解析逻辑在：

- `/data/home/scyb494/run/verl/scripts/research/run_external_math_eval.py`
- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/lm_eval_tasks/`

## 3. 入口脚本和调用关系

### 3.1 单模型 Slurm launcher

- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/launchers/run_external_math_benchmarks_a800.sh`

职责：

- 申请 `gpu_a800` 单卡作业。
- 激活 `verl_cpython` 和 `lm_eval_overlay` 环境。
- 配好 `vllm` 运行需要的 `PYTHONPATH`、`LD_LIBRARY_PATH`、`LD_PRELOAD`。
- 把单个模型包装成一次 `MODELS=base` 的评测 run。
- 最终调用下层 shell launcher。

### 3.2 Shell launcher

- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/launchers/run_external_math_benchmarks.sh`

职责：

- 整理环境变量。
- 调用 Python runner。
- run 完成后自动调用 summarizer。

### 3.3 Python runner

- `/data/home/scyb494/run/verl/scripts/research/run_external_math_eval.py`

职责：

- 根据 benchmark slug 解析实际 `lm_eval` task 名称。
- 为每个 `model_tag + benchmark` 生成一条 `lm_eval` 命令。
- 按 benchmark 顺序串行执行。
- 在 run 根目录写 `run_metadata.json`。

### 3.4 单 run 汇总器

- `/data/home/scyb494/run/verl/scripts/research/summarize_external_math_eval.py`

职责：

- 读取单次 run 的 benchmark 输出。
- 生成：
  - `results_summary.csv`
  - `sample_summary.csv`
  - `wrong_but_parseable.csv`
  - `unparseable_outputs.csv`
  - `all_samples.csv`

### 3.5 三模型批量提交器

- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/launchers/submit_external_math_benchmarks_a800.sh`

职责：

- 一次性提交 3 个模型的 Slurm job。
- 为每个模型生成独立 `run_tag`。
- 在 compare 目录下写：
  - `submission_manifest.csv`
  - `aggregate_after_runs.sh`

### 3.6 跨 run 聚合脚本

- `/data/home/scyb494/run/verl/scripts/research/combine_external_math_eval_runs.py`

职责：

- 按 `submission_manifest.csv` 聚合多个 run 的 summary。
- 输出：
  - `combined_results_summary.csv`
  - `combined_sample_summary.csv`
  - `combined_manifest.json`

## 4. 当前默认参数

当前 `run_external_math_benchmarks_a800.sh` 的默认值：

- `BACKEND=vllm`
- `BENCHMARKS=math500,aime24,aime25,olympiadbench,omni_math`
- `BATCH_SIZE=auto`
- `MAX_GEN_TOKS=16384`
- `NUM_FEWSHOT=0`
- `TEMPERATURE=0`
- `TOP_P=1.0`
- `SUMMARIZE=true`

当前 vLLM 默认参数：

- `TENSOR_PARALLEL_SIZE=1`
- `GPU_MEMORY_UTILIZATION=0.90`
- `MAX_MODEL_LEN=24576`
- `MAX_NUM_SEQS=16`
- `SWAP_SPACE=8`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`

当前默认环境：

- Conda 主环境：`/data/home/scyb494/run/.conda/envs/verl_cpython`
- lm-eval overlay：`/data/home/scyb494/run/.venvs/lm_eval_overlay`
- flash-attn overlay：`/data/home/scyb494/run/vendor/verl-main/.flash_attn_sm80_overlay`
- cache：`/data/home/scyb494/run/.cache`
- tmp：`/data/home/scyb494/run/tmp`

### 为什么现在默认是 vLLM

原因很直接：

- 同样的 full suite，`hf` 后端在 `math500` 上极慢。
- `vllm` 至少能把吞吐提到可用水平。
- 当前这三个模型在 `vllm` 路径上已经被验证能越过初始化并进入生成。

### 为什么 `MAX_GEN_TOKS` 现在是 16384

这是本次会话里按当前需求改的默认值。它适合保守放宽长推理上限，但代价也非常明确：

- 更容易让模型在坏样本上长篇自言自语。
- 更容易出现重复输出，直接拖慢吞吐。
- 同样一套题，不同模型的速度差异会被放大。

如果目标是更快出完整结果，而不是尽可能给模型更长的发挥空间，优先考虑：

- `MAX_GEN_TOKS=8192`
- 或 `MAX_GEN_TOKS=4096`

## 5. 当前可用的使用方法

### 5.1 单模型 smoke test

先验证环境和模型能不能跑通：

```bash
sbatch \
  --job-name=ext-math-vllm-smoke-base \
  --partition=gpu_a800 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus=1 \
  --time=00:20:00 \
  --export=ALL,MODEL_PATH=/data/home/scyb494/run/models/Qwen3-1.7B,MODEL_TAG=base,RUN_TAG=external_math_smoke_base,BENCHMARKS=aime24,BACKEND=vllm,LIMIT=1,BATCH_SIZE=auto,MAX_GEN_TOKS=16384 \
  /data/home/scyb494/run/verl/research/atomic_skill_transfer/launchers/run_external_math_benchmarks_a800.sh
```

适用场景：

- 刚改完环境变量。
- 刚修完 tokenizer。
- 想确认 `vllm` 能否越过初始化。

### 5.2 三模型 full compare

```bash
COMPARE_TAG=external_math_compare_YYYYMMDD_HHMM_vllm \
BACKEND=vllm \
TIME_LIMIT=01:00:00 \
MAX_GEN_TOKS=16384 \
/data/home/scyb494/run/verl/research/atomic_skill_transfer/launchers/submit_external_math_benchmarks_a800.sh
```

输出：

- compare 目录：`/data/home/scyb494/run/evals/external_math/comparisons/<COMPARE_TAG>`
- manifest：`submission_manifest.csv`
- 聚合 helper：`aggregate_after_runs.sh`

### 5.3 结果聚合

批量提交器会自动写好 helper：

```bash
/data/home/scyb494/run/evals/external_math/comparisons/<COMPARE_TAG>/aggregate_after_runs.sh
```

也可以手动调用：

```bash
/data/home/scyb494/run/.venvs/lm_eval_overlay/bin/python \
  /data/home/scyb494/run/verl/scripts/research/combine_external_math_eval_runs.py \
  --manifest-csv /data/home/scyb494/run/evals/external_math/comparisons/<COMPARE_TAG>/submission_manifest.csv \
  --output-dir /data/home/scyb494/run/evals/external_math/comparisons/<COMPARE_TAG>
```

### 5.4 单 run 汇总

```bash
/data/home/scyb494/run/.venvs/lm_eval_overlay/bin/python \
  /data/home/scyb494/run/verl/scripts/research/summarize_external_math_eval.py \
  --run-root /data/home/scyb494/run/evals/external_math/<RUN_TAG>
```

### 5.5 AIME repeat8 单模型评估

2026-04-28 新增了 repeat8 AIME task，用于降低 AIME 单次采样随机性。它不是 8 个 seed job，而是在一次模型加载后让 lm-eval 对 30 道 AIME 题各采样 8 次，也就是每个 AIME benchmark 生成 `30 * 8 = 240` 个 response。

新增 task：

- `ext_math_aime24_repeat8`
- `ext_math_aime25_repeat8`

新增 benchmark slug：

- `aime24_repeat8`
- `aime25_repeat8`

单模型提交示例：

```bash
MODEL_PATH=/path/to/hf-checkpoint \
MODEL_TAG=<model_tag> \
RUN_TAG=aime_repeat8_<model_tag>_YYYYMMDD_HHMM \
BENCHMARKS=aime24_repeat8,aime25_repeat8 \
BACKEND=vllm \
MAX_GEN_TOKS=32768 \
TEMPERATURE=1.0 \
TOP_P=0.95 \
TOP_K=0 \
MIN_P=0 \
SEED=7 \
APPLY_CHAT_TEMPLATE=true \
ENABLE_THINKING=true \
MAX_MODEL_LEN=40960 \
MAX_NUM_SEQS=8 \
TENSOR_PARALLEL_SIZE=1 \
SWAP_SPACE=8 \
EVAL_SAMPLE_SHARD_SIZE=auto \
EVAL_REQUIRE_SAMPLE_FILES=true \
NUM_FEWSHOT=0 \
SUMMARIZE=true \
sbatch --parsable \
  --job-name=aime8-<model_tag> \
  --time=5:00:00 \
  /data/home/scyb494/run/verl/research/atomic_skill_transfer/launchers/run_external_math_benchmarks_a800.sh
```

输出指标：

- `exact_match`：8 次采样逐样本计分后的平均值。对 AIME24/AIME25 来说，是 240 个 response 的平均正确率。
- `pass_at_8`：每题 8 次采样里至少答对一次的比例。对 AIME24/AIME25 来说，仍按 30 题取平均。

实现文件：

- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/lm_eval_tasks/ext_math_aime24_repeat8.yaml`
- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/lm_eval_tasks/ext_math_aime25_repeat8.yaml`
- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/lm_eval_tasks/utils.py`
- `/data/home/scyb494/run/verl/scripts/research/external_math_eval_utils.py`

注意：不能只把原始 AIME task 的 `repeats` 改成 8。lm-eval 的默认 `generate_until` filter 是 `take_first`，会丢掉后 7 个 response。repeat8 task 必须配：

- `repeats: 8`
- `filter_list: take_first_k, k: 8`
- repeat-aware `process_results`

如果只想快速估计随机性，可以考虑降低 `MAX_GEN_TOKS`，例如 `16384`。但这会和当前 core 32k 设置不完全一致，不能直接和 32k core 分数混作严格横向对比。

## 6. 这台集群上的 Slurm 经验

### 6.1 `--time` 会明显影响能否被 backfill

这次实测结论：

- `UNLIMITED` 或一天级别的 `--time` 很容易长时间 `PENDING (Priority)`。
- `20m` 的 smoke job 很容易被调度进去。
- `1h` 比 `24h` 更容易拿到卡。

如果看起来有空卡但 job 一直不动，不要先怀疑脚本，先怀疑调度策略。

### 6.2 外部看起来“有空卡”不等于你的 job 能 backfill

这个分区上，Slurm 更在乎预计占用时长和 backfill 窗口，而不是你主观上看到的“现在像是有卡”。

### 6.3 `slurm_job_submit_lua` 会拦截某些参数组合

实测踩过的坑：

- 给 external math eval batch 脚本显式加 `-c 8` 会被站点的 `slurm_job_submit_lua` 拒绝。

因此：

- 对这类 `gpu_a800` eval job，先不要显式加 `-c` 或 `--cpus-per-task`。
- 除非你已经在真实提交里验证过该参数组合被这个分区接受。

### 6.4 查 Slurm 状态时用什么命令

优先用：

```bash
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -P
```

需要看 batch log 路径时用：

```bash
scontrol show job <jobid> -o
```

队列态用：

```bash
squeue -j <jobid> -o '%.18i %.9P %.30j %.8T %.10M %.6D %R'
```

### 6.5 输出目录在 run 中途可能还是空的

这套 `lm_eval` 流程在 benchmark 结束前不一定会把最终结果文件写出来，所以：

- 中途看到 run 目录空，不代表 job 卡死。
- 进度判断以 Slurm batch log 为准。

## 7. 当前 external math 链路的关键坑和修复

### 7.1 `hf` 后端能跑，但 full suite 太慢

实测 `hf` 路径在 `math500` 上吞吐非常差。当前 full suite 默认已经改成 `vllm`，不建议再回 `hf` 路径做整套对比，除非只是做兼容性验证。

### 7.2 新下载 HF checkpoint 必须先做 tokenizer/config 兼容预检

每次从 GDrive 或训练输出目录拿到新的 HF checkpoint 后，不要直接提交 full core eval。先检查并 patch 下面两类配置问题，否则会反复浪费 A800 排队窗口。

#### tokenizer_config.json 的 `extra_special_tokens` list 问题

已多次遇到的失败栈：

- `AttributeError: 'list' object has no attribute 'keys'`

典型位置：

- `transformers/tokenization_utils_base.py`
- `_set_model_specific_special_tokens(special_tokens=self.extra_special_tokens)`

原因是 checkpoint 的 `tokenizer_config.json` 里把 `extra_special_tokens` 写成了 list，例如：

```json
"extra_special_tokens": [
  "<|im_start|>",
  "<|im_end|>",
  "<|object_ref_start|>"
]
```

当前 `lm_eval_overlay + vLLM + transformers` 这条链路会把它当 dict 处理，所以 vLLM 初始化 tokenizer 时直接失败。失败发生在第一个 benchmark 之前，不会产生任何有效分数。

已确认案例：

- 早期 `sub_only_step1800` / `zero_only_step310`
- 2026-04-28 从 `gdrive:verl_hf_models/grpo_zero_only_mixed_solq95_qwen3_1_7b_hf_h200_16k_rwdguard_r64_cpu256_u090_save92_8gpu_20260426_192005/` 下载的 `global_step_276` 和 `global_step_460`
- 失败 job：`62225`、`62226`

修复方式：

1. 先备份 `tokenizer_config.json`，例如 `tokenizer_config.json.pre_compat_patch_YYYYMMDD`。
2. 如果 `extra_special_tokens` 是 list，直接从 `tokenizer_config.json` 删除这个字段。
3. 不要改 `tokenizer.json`，除非确认 tokenizer 文件本身也坏了。
4. patch 后先做 tokenizer smoke，再提交 Slurm：

```bash
/data/home/scyb494/run/.venvs/lm_eval_overlay/bin/python - <<'PY'
from transformers import AutoTokenizer
for path in [
    "/data/home/scyb494/run/models/<model-dir-or-symlink>",
]:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    print(path, type(tok).__name__, tok.eos_token, tok.pad_token)
PY
```

#### config.json 的 Qwen3/vLLM 兼容字段

训练导出的 HF checkpoint 还要检查 `config.json`。至少确认下面字段存在且值正确：

- top-level `rope_theta: 1000000`
- `bos_token_id: 151643`
- `torch_dtype: "bfloat16"`

如果缺失，先备份 `config.json`，例如 `config.json.pre_compat_patch_YYYYMMDD`，再补字段。

已确认案例：

- 2026-04-28 重新提交 `origin_only_step310` 同参数 32k core eval 前，旧 checkpoint 目录缺少 top-level `rope_theta`、`bos_token_id` 和 `torch_dtype`。已备份为 `config.json.pre_compat_patch_20260428` 并补齐字段。

#### 什么时候需要重做这一步

只要发生下面任一情况，都要重新做这两个预检：

- 从 GDrive 新下载模型。
- 从训练 checkpoint 新导出 HF 模型。
- 改了模型目录 symlink 指向。
- 重新同步了 `config.json` 或 `tokenizer_config.json`。
- 之前可跑的目录被覆盖。

如果 tokenizer 初始化失败，优先检查 `tokenizer_config.json` 里的 `extra_special_tokens`，不要先怀疑 Slurm、CUDA、flash-attn 或 benchmark task。

当前已修过的模型目录只是“当前副本”可用。如果未来重新导出 checkpoint，或者换回旧副本，这个坑可能会重新回来。

### 7.3 vLLM 最初会吃到错误的 flash-attn 实现

最初 vLLM 初始化报错：

- `ImportError: cannot import name 'apply_rotary' from 'flash_attn.ops.triton.rotary'`

原因：

- 运行时默认吃到的是系统 `flash_attn`，不是项目里准备好的 overlay。

当前修复方式：

- launcher 里显式检查并注入：
  - `FLASH_ATTN_OVERLAY=/data/home/scyb494/run/vendor/verl-main/.flash_attn_sm80_overlay`
- 通过 `PYTHONPATH` 前置 overlay。
- 启动前只做静态文件检查，不再在父进程里 import `flash_attn`。

### 7.4 父进程预检查会触发 CUDA fork 问题

踩过的坑：

- 曾经为了验证 overlay，在 launcher 里加过一段 Python 预检查，直接 import `flash_attn.ops.triton.rotary`。
- 这会让父进程提前触发 CUDA 相关初始化。
- 后果是 vLLM worker fork 后报：
  - `Cannot re-initialize CUDA in forked subprocess`

当前修复方式：

- 删除父进程 Python import 预检查。
- 改成纯静态文件检查。
- 同时显式设：
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn`

### 7.5 切到 `spawn` 后，又暴露出 `libstdc++` 版本冲突

切到 `spawn` 后，worker 子进程里导入 `sqlite3` 又报：

- `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found`

原因：

- 子进程没有优先吃到 conda 环境自己的 `libstdc++.so.6`。
- 系统 `/lib/x86_64-linux-gnu/libstdc++.so.6` 太老。

当前修复方式：

- 在激活 conda 后前置：
  - `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:...`
- 在 `vllm` 模式下再显式预加载：
  - `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1`

当前这条修复已经通过 smoke job 验证。

### 7.6 full compare 提交器的模型槽位设计

`run_external_math_eval.py` 原始模型槽位只支持：

- `base`
- `origin_only`

当前三模型对比没有去改这个核心设计，而是采用更稳的办法：

- 每个待评模型单独提交一个 job。
- 在该 job 里把这个模型当成一次 run 的 `base` 来评。

所以你会在 run metadata 里看到：

- `MODELS=base`

这不是 bug，是当前 submitter 的设计选择。

## 8. 为什么 `base` 和 `zero_only` 比 `sub_only` 慢很多

这个问题已经做了最小对照实验，不是猜的。

### 8.1 full run 日志结论

同样在跑 `hendrycks_math500`：

- `sub_only_step1800` 的推进明显更快。
- `base` 和 `zero_only` 在少数样本上耗时很长。

### 8.2 mini 对照实验

我额外跑了 3 个相同配置的 mini run：

- benchmark：`math500`
- `LIMIT=2`
- backend：`vllm`
- `MAX_GEN_TOKS=16384`

完成时间：

- `base`: `2m44s`
- `sub_only_step1800`: `1m46s`
- `zero_only_step310`: `2m43s`

样本汇总：

- `base avg_output_chars = 43120.5`
- `sub_only_step1800 avg_output_chars = 5408.0`
- `zero_only_step310 avg_output_chars = 26932.0`

对应文件：

- `base`
  - `/data/home/scyb494/run/evals/external_math/external_math_mini_20260414_base/summary/sample_summary.csv`
- `sub_only_step1800`
  - `/data/home/scyb494/run/evals/external_math/external_math_mini_20260414_sub/summary/sample_summary.csv`
- `zero_only_step310`
  - `/data/home/scyb494/run/evals/external_math/external_math_mini_20260414_zero/summary/sample_summary.csv`

### 8.3 生成内容特征

三个模型前两个样本都能 `parse_success=1.0`，但 `exact_match=0.0`。差异在于生成长度和退化程度：

- `base`
  - 第一题和第二题都输出了极长的自我修正和重复。
  - 尾部反复重复 `**Final Answer** \boxed{...}`。
- `zero_only_step310`
  - 也会进入很长的自我纠错。
  - 尾部出现大量重复的 `0` 或 `1.1.1...` 之类的退化文本。
- `sub_only_step1800`
  - 也有重复和格式错误。
  - 但整体长度短很多，所以吞吐高很多。

结论：

- `base` 和 `zero_only` 慢，不是因为初始化更慢。
- 主因是这两个模型更容易在当前 prompt / stop 条件下长篇重复生成。
- `MAX_GEN_TOKS=16384` 放大了这个问题。

## 9. 使用建议

### 9.1 如果目标是先把整套结果跑完

优先改这些，而不是继续盲目提更大的 job：

- 先把 `MAX_GEN_TOKS` 降到 `8192`。
- 如果还嫌慢，再降到 `4096`。
- 把 full suite 按 benchmark 拆开跑，而不是一个 job 串 5 个 benchmark。

### 9.2 如果目标是先验证链路

优先顺序：

1. `aime24` 或 `aime25`
2. `LIMIT=1`
3. `--time=00:20:00`
4. 单模型 smoke
5. 确认日志越过 `Initializing vllm model`
6. 再提 full compare

### 9.3 如果目标是看模型质量而不是吞吐

注意：

- `parse_success_rate` 高不代表答案对。
- 目前这三个模型在 mini 对照里都是 `parse_success=1.0`，但 `exact_match=0.0`。
- 所以看结果时至少同时看：
  - `pass_at_1`
  - `parse_success_rate`
  - `avg_output_chars`
  - `unparseable_outputs.csv`
  - `wrong_but_parseable.csv`

## 10. 推荐的排障顺序

如果 job 报错，按这个顺序排：

1. `sacct` 看最终状态和退出码。
2. `scontrol show job -o` 看 `StdOut` 实际路径。
3. 直接看 Slurm batch log，不要先猜。
4. 如果是 vLLM 初始化阶段失败，先检查：
   - `FLASH_ATTN_OVERLAY`
   - `VLLM_WORKER_MULTIPROC_METHOD`
   - `LD_LIBRARY_PATH`
   - `LD_PRELOAD`
5. 如果是 tokenizer 初始化失败，先检查 checkpoint 里的：
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `tokenizer_config.json` 是否包含 list 形式的 `extra_special_tokens`
   - `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` smoke 是否能通过
6. 如果 job 很慢但没报错，先看：
   - 当前 benchmark 是不是 `math500`
   - `Processed prompts` 的推进速度
   - 输出是否在长篇重复
   - `MAX_GEN_TOKS` 是否过大

## 11. 当前工作状态的简短总结

截至 2026-04-14，本地这条 external math 评测链路已经具备：

- 可用的 `gpu_a800` Slurm 单模型 launcher。
- 可用的三模型批量提交器。
- 可用的单 run 和跨 run 聚合脚本。
- 已验证可工作的 `vllm + flash-attn overlay + spawn + conda libstdc++` 环境修复。
- 已验证的速度差异解释：`base` 和 `zero_only` 更慢，主要因为生成更长、更重复。

如果未来再改这条链路，优先保证：

- 不要把父进程 CUDA 初始化问题重新引入。
- 不要把 checkpoint 的旧 tokenizer 文件重新带回来。
- 变更默认 `MAX_GEN_TOKS` 时同步更新这份文档。
