# 2026-04-27 GRPO HF 模型评测复盘与链路留痕改造

本文档记录 2026-04-27 这轮 `Qwen3-1.7B` GRPO HF checkpoint 下载、修复、评测、问题排查和评测链路改造。重点是把本次对话中的操作结论固化下来，避免后续只能从聊天记录或 Slurm 日志里恢复上下文。

## 1. 本轮需要记录的事项

本轮评测中有几类信息必须留档：

1. 三个 GRPO HF checkpoint 的本地路径、兼容性 patch 和 baseline 路径。
2. AIME 早期为什么输出异常、为什么评不对，以及最终如何修复。
3. 正式 32k external math 评测的参数、输出目录和核心结果。
4. MATH500 官方分数中的 false negative 是怎么发现和修正的。
5. 旧 GPQA 为什么四个模型分数完全一样，以及生成式 GPQA 的修复与结果。
6. Omni-MATH 为什么非常慢、被取消后为什么没有可分析分数。
7. 评估链路现在如何默认保存推理结果，以及为了防止中断丢结果做了什么改造。
8. 后续继续跑 Omni 或其他长 benchmark 时应该怎么操作。

## 2. 模型路径与兼容性修复

本轮评估对象：

- baseline: `/data/home/scyb494/run/models/Qwen3-1.7B`
- step230: `/data/home/scyb494/run/models/grpo_zero_only_sub_only_solq95_qwen3_1_7b_hf_h200_16k_rwdguard_r64_cpu256_u090_step230`
- step460: `/data/home/scyb494/run/models/grpo_zero_only_sub_only_solq95_qwen3_1_7b_hf_h200_16k_rwdguard_r64_cpu256_u090_step460`
- step690: `/data/home/scyb494/run/models/grpo_zero_only_sub_only_solq95_qwen3_1_7b_hf_h200_16k_rwdguard_r64_cpu256_u090_step690`

三个 step 模型下载后做过 HF config 兼容性 patch。每个目录里保留了备份：

- `config.json.pre_compat_patch_20260427`

patch 的关键字段：

- top-level `rope_theta: 1000000`
- `bos_token_id: 151643`
- `torch_dtype: "bfloat16"`

这一步是 AIME 修复的关键前置。未 patch 时，vLLM/transformers 读到的模型配置和 Qwen3 期望不完全一致，长上下文推理容易出现重复、乱码、模板串污染等异常输出。

### 2.1 2026-04-28 补充：新 HF checkpoint 的 tokenizer_config 兼容坑

2026-04-28 从 GDrive 下载的新模型：

- `gdrive:verl_hf_models/grpo_zero_only_mixed_solq95_qwen3_1_7b_hf_h200_16k_rwdguard_r64_cpu256_u090_save92_8gpu_20260426_192005/global_step_276`
- `gdrive:verl_hf_models/grpo_zero_only_mixed_solq95_qwen3_1_7b_hf_h200_16k_rwdguard_r64_cpu256_u090_save92_8gpu_20260426_192005/global_step_460`

首次提交 core 评测时，两个 Slurm job 都在 vLLM tokenizer 初始化阶段失败：

- `62225` / `mixed_save92_step276`
- `62226` / `mixed_save92_step460`

失败关键栈：

```text
AttributeError: 'list' object has no attribute 'keys'
```

触发位置是 `transformers/tokenization_utils_base.py` 中的 `_set_model_specific_special_tokens(...)`。根因是新导出的 `tokenizer_config.json` 含有 list 形式的 `extra_special_tokens`，而当前 `lm_eval_overlay + vLLM + transformers` 评估链路期望这里是 dict 或不存在。

修复方式：

1. 先备份：
   - `tokenizer_config.json.pre_compat_patch_20260428`
2. 从 `tokenizer_config.json` 删除 list 形式的 `extra_special_tokens` 字段。
3. 同时保留前文的 `config.json` 兼容 patch：
   - top-level `rope_theta: 1000000`
   - `bos_token_id: 151643`
   - `torch_dtype: "bfloat16"`
4. 重新提交前先做 `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` smoke。

后续凡是新下载或新导出的 HF checkpoint，都必须先检查这两类文件：

- `config.json`
- `tokenizer_config.json`

不要只 patch `config.json` 就直接提交 core eval；否则会再次在第一个 benchmark 前失败，且没有任何可用分数。

### 2.2 2026-04-28 补充：`origin_only_step310` 重新评估前的兼容 patch

为了和 2026-04-28 的 `mixed` 模型使用同一套 32k core 参数横向对比，重新提交了 `origin_only_step310` 的 core eval。旧目录为：

- `/data/home/scyb494/run/verl_a800_checkpoints/grpo_zero_only_origin_only_qwen3_1_7b_a800_flash_20260410_195340/global_step_310/actor/huggingface`

提交前检查发现该目录的 `config.json` 仍缺少 4/27 后固定下来的 Qwen3/vLLM 兼容字段：

- top-level `rope_theta`
- `bos_token_id`
- `torch_dtype`

处理方式：

1. 备份：
   - `config.json.pre_compat_patch_20260428`
2. 修改 `config.json`：
   - `bos_token_id: 151643`
   - `rope_theta: 1000000`
   - `torch_dtype: "bfloat16"`
3. `tokenizer_config.json` 已确认不存在 list 形式的 `extra_special_tokens`，无需删除字段。
4. 用 `AutoConfig.from_pretrained(...)` 和 `AutoTokenizer.from_pretrained(...)` 做 smoke，结果为：
   - `Qwen3Config`
   - `Qwen2TokenizerFast`
   - `eos=<|im_end|>`
   - `pad=<|endoftext|>`

## 3. AIME 早期为什么评不对

### 现象

刚开始 smoke AIME 时，部分输出出现不正常现象：

- 重复模板或循环输出
- 空输出或不可解析输出
- 出现类似 `WELCOME TO ASSISTANT` 的模板污染
- LaTeX/boxed 结构崩坏
- parse rate 不稳定

这些现象不是单纯的评分脚本问题，而是模型加载/生成配置和 prompt 包装共同导致的推理输出异常。

### 排查思路

当时按两条线排查：

1. 模型本身是否能正常加载和生成。
2. chat template / system instruction / vLLM generation kwargs 是否引入了包装问题。

因此做了两组 AIME smoke：

- chat template 配置：
  - `APPLY_CHAT_TEMPLATE=true`
  - `ENABLE_THINKING=true`
  - `SYSTEM_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'`
- raw prompt fallback：
  - `APPLY_CHAT_TEMPLATE=false`
  - `SYSTEM_INSTRUCTION=''`

采样侧最终使用了 vLLM 兼容的 `TOP_K=0`，而不是最初计划里的 `TOP_K=-1`。

### 修复点

最终有效修复包括：

1. patch 三个 HF checkpoint 的 `config.json`，补齐 Qwen3/vLLM 需要的关键字段。
2. 统一使用 `BACKEND=vllm`、`dtype=bfloat16`。
3. AIME smoke 先验证输出正常，再跑正式任务。
4. 正式任务使用 32k 输出长度：
   - `MAX_GEN_TOKS=32768`
   - `MAX_MODEL_LEN=40960`
   - `MAX_NUM_SEQS=8`

修复后 AIME 正式结果的 `parse_success_rate` 正常，输出不再是大面积重复/乱码。后续人工检查 AIME 原文输出时，结论是：AIME 错题基本是真答错，不是抽取错误。

### 3.1 2026-04-28 补充：AIME repeat8 评估链路

单次 AIME 是 30 题、每题采样 1 次，使用当前采样参数：

- `temperature=1.0`
- `top_p=0.95`
- `top_k=0`
- `seed=7`

因此单次分数有明显随机性。2026-04-28 新增了 AIME repeat8 评估链路，目标是在一次模型加载后，对同一套 AIME 题发出 `30 * 8 = 240` 个 generate 请求，而不是起 8 个 Slurm job 重复加载模型。

不能只把默认 AIME task 的 `repeats` 改成 8。lm-eval 的 `generate_until` 默认 filter 是 `take_first`，会只保留第一个 response，后 7 个 response 会被丢掉。正确做法是新增 repeat-aware task：

- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/lm_eval_tasks/ext_math_aime24_repeat8.yaml`
- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/lm_eval_tasks/ext_math_aime25_repeat8.yaml`

新增 benchmark slug：

- `aime24_repeat8`
- `aime25_repeat8`

对应代码改动：

- `/data/home/scyb494/run/verl/research/atomic_skill_transfer/lm_eval_tasks/utils.py`
  - 新增 `extract_math_answer(...)`
  - 新增 `math_target_from_doc(...)`
  - 新增 `flatten_repeat_results(...)`
  - 新增 `process_repeat_results(...)`
  - 原 `process_results(...)` 改为复用 `extract_math_answer(...)`
- `/data/home/scyb494/run/verl/scripts/research/external_math_eval_utils.py`
  - 新增非默认 benchmark spec：`aime24_repeat8`、`aime25_repeat8`

repeat8 task 的关键设置：

- `repeats: 8`
- `filter_list: take_first_k, k: 8`
- `process_results: utils.process_repeat_results`

输出指标含义：

- `exact_match`：8 次采样逐样本计分后取平均；AIME24/AIME25 各自是 240 个 sample 的 pass@1 估计。
- `pass_at_8`：每题 8 次采样里至少答对一次的比例；AIME24/AIME25 各自仍按 30 题取平均。

注意：这套 repeat8 仍沿用 core 32k 的 `MAX_GEN_TOKS=32768`。当前 `mixed` 模型 AIME 输出很长，单次 AIME24 的上一轮 samples 中位输出长度约 46k 到 48k 字符，p90 约 70k 字符。因此 repeat8 会很慢，`output toks/s` 在 500 到 700 左右并不一定表示 GPU 异常，而是长 thinking 输出导致总 token 数太大。

## 4. 正式 32k external math 核心评测

核心 32k 评估已完成，不含 Omni-MATH。任务为：

- `aime24`
- `aime25`
- `math500`
- `olympiadbench`
- old `gpqa_diamond` label-loglikelihood task

Slurm job：

- base core: `61538`, completed at `2026-04-27 14:31:13`
- step230 core: `61539`, completed at `2026-04-27 14:25:48`
- step460 core: `61540`, completed at `2026-04-27 14:29:20`
- step690 core: `61541`, completed at `2026-04-27 14:25:30`

输出目录：

- `/data/run01/scyb494/evals/external_math/full_math_ropefix32k_base_core_20260427_1050`
- `/data/run01/scyb494/evals/external_math/full_math_ropefix32k_step230_core_20260427_1050`
- `/data/run01/scyb494/evals/external_math/full_math_ropefix32k_step460_core_20260427_1050`
- `/data/run01/scyb494/evals/external_math/full_math_ropefix32k_step690_core_20260427_1050`

正式配置要点：

- `MAX_GEN_TOKS=32768`
- `MAX_MODEL_LEN=40960`
- `MAX_NUM_SEQS=8`
- `TEMPERATURE=1.0`
- `TOP_P=0.95`
- `TOP_K=0`
- `MIN_P=0`
- `APPLY_CHAT_TEMPLATE=true`
- `ENABLE_THINKING=true`

### 4.1 2026-04-28 补充：`mixed` 与 `origin_only` 的同参数 core eval

2026-04-28 从 GDrive 下载的 `mixed` 模型修复 tokenizer/config 后，重新提交了 core 32k eval：

| job | model | run tag | 记录时状态 |
|---:|---|---|---|
| `62317` | `mixed_step276` | `full_math_mixed_save92_step276_core_retry_20260428_2030` | RUNNING |
| `62318` | `mixed_step460` | `full_math_mixed_save92_step460_core_retry_20260428_2030` | RUNNING |

截至 2026-04-28 23:13，已经落盘的前三项结果：

| model | AIME24 | AIME25 | MATH500 |
|---|---:|---:|---:|
| `mixed_step276` | `0.3667` | `0.3667` | `0.8560` |
| `mixed_step460` | `0.4667` | `0.3333` | `0.8700` |

为了和旧 `origin_only_step310` 做同参数横向对比，也提交了：

| job | model | run tag | 记录时状态 |
|---:|---|---|---|
| `62371` | `origin_only_step310` | `full_math_origin_only_step310_core_20260428_2241` | RUNNING |

截至 2026-04-28 23:13，`origin_only_step310` 已落盘：

| model | AIME24 | AIME25 |
|---|---:|---:|
| `origin_only_step310` | `0.3667` | `0.4000` |

这些分数使用的参数和 `mixed` retry 保持一致：

- `BENCHMARKS=aime24,aime25,math500,olympiadbench,gpqa_diamond`
- `MAX_GEN_TOKS=32768`
- `MAX_MODEL_LEN=40960`
- `MAX_NUM_SEQS=8`
- `TEMPERATURE=1.0`
- `TOP_P=0.95`
- `TOP_K=0`
- `MIN_P=0`
- `SEED=7`
- `APPLY_CHAT_TEMPLATE=true`
- `ENABLE_THINKING=true`

`origin_only_step310` 这次不是旧的 4/15 `temperature=0.6, top_k=20` 结果；它是为了当前横向对比重新提交的同参数版本。

### 4.2 2026-04-28 补充：AIME repeat8 已提交作业

基于 3.1 的 repeat8 task，提交了 4 个单模型 AIME repeat8 job：

| job | model | run tag | 记录时状态 |
|---:|---|---|---|
| `62375` | `mixed_step276` | `aime_repeat8_mixed_step276_20260428_2255` | RUNNING |
| `62376` | `mixed_step460` | `aime_repeat8_mixed_step460_20260428_2255` | RUNNING |
| `62377` | `sub_step460` | `aime_repeat8_sub_step460_20260428_2255` | RUNNING |
| `62378` | `origin_step310` | `aime_repeat8_origin_step310_20260428_2255` | RUNNING |

提交参数：

- `BENCHMARKS=aime24_repeat8,aime25_repeat8`
- `MAX_GEN_TOKS=32768`
- `MAX_MODEL_LEN=40960`
- `MAX_NUM_SEQS=8`
- `TEMPERATURE=1.0`
- `TOP_P=0.95`
- `TOP_K=0`
- `MIN_P=0`
- `SEED=7`
- `APPLY_CHAT_TEMPLATE=true`
- `ENABLE_THINKING=true`

日志路径：

- `/data/home/scyb494/run/evals/external_math/slurm_logs/aime8-mixed_step276_62375.out`
- `/data/home/scyb494/run/evals/external_math/slurm_logs/aime8-mixed_step460_62376.out`
- `/data/home/scyb494/run/evals/external_math/slurm_logs/aime8-sub_step460_62377.out`
- `/data/home/scyb494/run/evals/external_math/slurm_logs/aime8-origin_step310_62378.out`

Slurm 分配说明：

- `62375` 与 `62376` 在同一台节点 `d1n41a13g02`，但分到不同 GPU：
  - `62375`: GPU `IDX:1`
  - `62376`: GPU `IDX:7`
- 不是两个 job 抢同一张 GPU。

## 5. Core 结果汇总

官方 strict metric 结果：

| model | AIME24 | AIME25 | MATH500 official | OlympiadBench | old GPQA label-loglikelihood |
|---|---:|---:|---:|---:|---:|
| base | 13/30 | 11/30 | 435/500 | 267/674 | 44/198 |
| step230 | 13/30 | 12/30 | 437/500 | 264/674 | 44/198 |
| step460 | 13/30 | 13/30 | 434/500 | 266/674 | 44/198 |
| step690 | 13/30 | 11/30 | 427/500 | 270/674 | 44/198 |

官方 weighted total：

- base: `770/1432`
- step230: `770/1432`
- step460: `770/1432`
- step690: `765/1432`

相关汇总文件：

- `/data/run01/scyb494/evals/external_math/core32k_official_summary_20260427.csv`

## 6. MATH500 false negative 修复

MATH500 官方 exact match 偏 strict，发现了明显 false negative。例如：

- 文本 wrapper 导致答案未匹配
- thousands comma 差异
- base suffix 省略
- 单位/格式差异
- 简单分数、rational expression 等价但字符串不同
- vector tuple / set order / interval prefix 等格式差异

做了 post-hoc 修复后，MATH500 分数为：

| model | official | posthoc fixed |
|---|---:|---:|
| base | 435/500 | 456/500 |
| step230 | 437/500 | 457/500 |
| step460 | 434/500 | 456/500 |
| step690 | 427/500 | 451/500 |

修复后 weighted total：

- base: `791/1432`
- step230: `790/1432`
- step460: `792/1432`
- step690: `789/1432`

相关文件：

- `/data/run01/scyb494/evals/external_math/math500_posthoc_summary_20260427.csv`
- `/data/run01/scyb494/evals/external_math/math500_posthoc_false_negative_fixes_v2_20260427.tsv`
- `/data/run01/scyb494/evals/external_math/math500_trained_wrong_analysis_20260427.tsv`

结论：修复 false negative 后，MATH500 仍没有显示 GRPO checkpoint 相对 base 的稳定提升。step230 只比 base 多 1 题，step460 与 base 持平，step690 低 5 题。

## 7. 旧 GPQA 为什么四个模型分数完全一样

旧 GPQA task：

- `/data/run01/scyb494/verl/research/atomic_skill_transfer/lm_eval_tasks/leaderboard_gpqa_diamond_local.yaml`

问题点：

- `output_type: multiple_choice`
- `doc_to_choice: ["(A)", "(B)", "(C)", "(D)"]`
- 这是 label-only loglikelihood，不会生成推理过程，也不会读完整选项内容后写 reasoning。

实际现象：

- 四个模型全部预测 `(A)`。
- GPQA Diamond 目标分布里 `(A)` 正好有 `44` 题。
- 所以四个模型分数完全相同：`44/198 = 22.22%`。

结论：旧 GPQA 结果不能作为有效科学推理评估，只能说明这个 label-only loglikelihood task 在当前 prompt/choice 设置下退化成了全选 A。

## 8. 生成式 GPQA 修复

为了解决旧 GPQA 全选 A 问题，新增了生成式 GPQA task 和 launcher。

新增/修改文件：

- `/data/run01/scyb494/verl/research/atomic_skill_transfer/lm_eval_tasks/leaderboard_gpqa_diamond_generative_local.yaml`
- `/data/run01/scyb494/verl/research/atomic_skill_transfer/lm_eval_tasks/utils.py`
- `/data/run01/scyb494/evals/external_math/gpqa_generative_task/run_gpqa_generative_a800.sh`

关键实现：

1. `process_gpqa_generative_docs`：
   - 使用 `Record ID` / `Question` 的 SHA256 做 deterministic shuffle。
   - 保证所有模型看到同一题的同一选项顺序。
2. `_extract_gpqa_choice`：
   - 优先抽取最后一个 boxed A-D。
   - 再匹配 final answer / answer / option 等模式。
   - 后来收紧 fallback，避免从未完成的 `<think>` 中误抽。
3. `process_gpqa_generative_results`：
   - 将抽取出的 `(A)`/`(B)`/`(C)`/`(D)` 与 doc target 对比。
4. GPQA system prompt 改短：
   - 避免 `\boxed{A}` 在 system instruction 中被 chat template 包坏。
   - task prompt 里仍要求最终输出 boxed choice。

Smoke 结果：

- step460 20 题 smoke：`10/20`
- parse：`18/20`
- 预测分布包含 A/B/C/D，不再全选 A。

正式 32k 生成式 GPQA Slurm job：

- base: `61708`
- step230: `61709`
- step460: `61710`
- step690: `61711`

正式结果：

| model | GPQA generative exact | vs base | parse |
|---|---:|---:|---:|
| base | 84/198 = 42.42% | 0 | 198/198 |
| step230 | 79/198 = 39.90% | -5 | 198/198 |
| step460 | 74/198 = 37.37% | -10 | 198/198 |
| step690 | 78/198 = 39.39% | -6 | 196/198 |

结果文件：

- `/data/run01/scyb494/evals/external_math/gpqa_gen32_summary_20260427.csv`
- `/data/run01/scyb494/evals/external_math/gpqa_gen32_per_doc_predictions_20260427.tsv`

结论：生成式 GPQA 修复了全选 A 的无效评估问题，但三个 GRPO checkpoint 均未超过 base。

## 9. Omni-MATH 任务状态与为什么没有分数

Omni-MATH task：

- `/data/run01/scyb494/verl/research/atomic_skill_transfer/lm_eval_tasks/ext_math_omni_math_rule.yaml`

规模：

- `4428` 题
- 是开放数学生成题，不是短 multiple-choice。

被取消的 32k Omni job：

- base: `61542`
- step230: `61543`
- step460: `61544`
- step690: `61545`

取消前进度：

| model | 进度 |
|---|---:|
| base | 708/4428 |
| step230 | 722/4428 |
| step460 | 708/4428 |
| step690 | 730/4428 |

耗时约 4.5 小时，只完成约 16%。按日志估算，单模型 full Omni 32k 大约需要 28 小时。

为什么慢：

1. Omni 有 `4428` 题，是 GPQA `198` 题的 22.4 倍。
2. `MAX_GEN_TOKS=32768`，很多数学题会生成很长推理。
3. `until` 只在遇到 `Question:` / EOS / chat end token 等停止，很多样本不会很快自然结束。
4. lm-eval 默认在整个 task 完成后才写完整 `samples_*.jsonl` 和 `results_*.json`。

为什么没有可分析分数：

- 被 `scancel` 时，lm-eval 尚未完成 task。
- 对应 run 目录存在，但文件数为 0。
- 没有 `samples_*omni*.jsonl`，也没有 `results_*.json`。
- Slurm 日志只保存了进度，不保存模型逐题输出，所以无法 post-hoc 算 exact match。

## 10. 评估链路现在如何保存推理数据

本轮最后对 external math 评估链路做了留痕改造。

相关文件：

- `/data/run01/scyb494/verl/scripts/research/external_math_eval_utils.py`
- `/data/run01/scyb494/verl/scripts/research/run_external_math_eval.py`
- `/data/run01/scyb494/verl/research/atomic_skill_transfer/launchers/run_external_math_benchmarks.sh`
- `/data/run01/scyb494/verl/research/atomic_skill_transfer/launchers/run_external_math_benchmarks_a800.sh`
- `/data/run01/scyb494/verl/research/atomic_skill_transfer/launchers/submit_external_math_benchmarks_a800.sh`
- `/data/run01/scyb494/verl/research/atomic_skill_transfer/launchers/submit_external_math_aime_seed_sweep_a800.sh`
- `/data/run01/scyb494/verl/research/atomic_skill_transfer/launchers/submit_external_math_aime8_non_aime1_a800.sh`

### 10.1 默认强制 `--log_samples`

`run_external_math_eval.py` 构造 lm-eval CLI 时硬编码带上：

```bash
--log_samples
```

因此正常完成的评估都会写出：

- `results_*.json`
- `samples_<task>_<timestamp>.jsonl`

这些 samples 文件包含：

- prompt arguments
- doc
- target
- raw responses
- filtered responses
- per-sample metric
- hashes

### 10.2 新增 sample 文件强校验

新增默认开关：

```bash
EVAL_REQUIRE_SAMPLE_FILES=true
```

作用：

- 每个 lm-eval call 正常返回后，runner 会检查对应 output dir 下是否有 `samples*.jsonl` 或 `samples*.json`。
- 如果没有 sample 文件，则直接报错失败。
- 这样可以避免“评估完成但没有留推理输出”的隐性问题。

如确实要关闭：

```bash
EVAL_REQUIRE_SAMPLE_FILES=false
```

### 10.3 大任务默认 shard 化，避免中断丢全部输出

新增默认开关：

```bash
EVAL_SAMPLE_SHARD_SIZE=auto
```

当前 benchmark spec 中为 Omni-MATH 设置：

- `sample_count=4428`
- `default_sample_shard_size=256`

所以 Omni-MATH 会自动拆成 18 个 shard：

- `shard_00000_00255`
- `shard_00256_00511`
- ...
- `shard_04352_04427`

每个 shard 都是一次独立的 lm-eval call，并有自己的 output dir。每个 shard 正常完成后都会写自己的 samples 文件。后续即使 Slurm job 被取消，也只损失当前正在跑的 shard；已经完成的 shard 可以直接分析或合并。

关闭 shard：

```bash
EVAL_SAMPLE_SHARD_SIZE=0
```

调小 shard：

```bash
EVAL_SAMPLE_SHARD_SIZE=128
```

强制所有已知 sample_count 的 benchmark shard：

```bash
EVAL_SAMPLE_SHARD_SIZE=256
```

注意：当显式传 `--limit` 时，shard 自动关闭，因为 lm-eval 不允许 `--limit` 和 `--samples` 同时使用。

### 10.4 `run_metadata.json` 改为运行中持续刷新

之前 `run_metadata.json` 只在整轮结束后写。长任务被中断时，这个 metadata 也可能不存在。

现在改为：

- run 开始时写一次 metadata。
- 每个 shard 开始时更新状态为 `running`。
- 每个 shard 完成后记录：
  - `status`
  - `started_at`
  - `finished_at`
  - `output_dir`
  - `sample_files`
  - command line
  - shard label 和样本范围
- 失败或中断时记录 `failed_or_interrupted`。

这样即使任务中断，也能从 `run_metadata.json` 里知道哪些 shard 已完成、样本文件在哪里。

## 11. 后续怎么跑 Omni

建议不要再单个 full Omni 32k 一把跑到底。更稳的策略：

### 快速近似

先跑 subset：

```bash
BENCHMARKS=omni_math LIMIT=500 ...
```

或：

```bash
BENCHMARKS=omni_math LIMIT=1000 ...
```

这类 subset 不会 shard，因为 `--limit` 和 `--samples` 不能同时使用。

### 正式可恢复 full run

使用默认 shard：

```bash
BENCHMARKS=omni_math EVAL_SAMPLE_SHARD_SIZE=auto EVAL_REQUIRE_SAMPLE_FILES=true ...
```

中断后：

1. 先看 `run_metadata.json`。
2. 检查已完成 shard 的 `samples_*.jsonl`。
3. 只重跑缺失 shard。

如果 shard 仍太大，改成：

```bash
EVAL_SAMPLE_SHARD_SIZE=128
```

## 12. 当前总体结论

这轮 GRPO checkpoint 在 AIME 上有小幅波动：

- step460 AIME combined 最好：`26/60`
- step230：`25/60`
- base 和 step690：`24/60`

但在更广泛评估上没有形成稳定超过 base 的结论：

- MATH500 posthoc 后 step230 仅 +1，step460 持平，step690 -5。
- OlympiadBench step690 略高，但整体差异不大。
- 生成式 GPQA 三个 checkpoint 都低于 base。

因此本轮最重要的产出不只是模型分数，而是修复了几个评估可信度问题：

1. AIME/vLLM 加载和生成配置问题。
2. MATH500 false negative 后处理。
3. GPQA label-only 全选 A 问题。
4. Omni 这类长 benchmark 中断后无样本的问题。
