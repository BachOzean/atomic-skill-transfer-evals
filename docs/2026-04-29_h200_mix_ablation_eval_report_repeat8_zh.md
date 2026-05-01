# H200 mix / sub_only / base / ablation 评测报告

- 生成时间：2026-04-29 23:19:45
- 评测目录：`/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127`
- 官方口径参考：Qwen Quickstart / thinking mode，`enable_thinking=True`，thinking mode 推荐 `temperature=0.6, top_p=0.95, top_k=20, min_p=0`。链接：https://qwen.readthedocs.io/en/v3.0/getting_started/quickstart.html
- 结果文件数：130；sample 文件数：130

## 口径

- AIME 主表：默认使用 `aime24_repeat8`、`aime25_repeat8`；如目录中存在 repeat64，也会一并汇总。`max_gen_toks=38912`，`temperature=0.6`，`top_p=0.95`，`top_k=20`，`min_p=0`，开启 chat template 和 thinking mode。
- 核心 benchmark：`math500`、生成式 `gpqa_diamond`、`olympiadbench`，默认 `max_gen_toks=32768`，采样参数沿官方 thinking mode。
- 诊断口径只在官方口径未达标时补跑：`temperature=1.0`、`top_p=0.95`、`top_k=0`、`max_gen_toks=38912`。

## AIME

| family | model | step | aime24_repeat8:exact_match | aime24_repeat8:pass_at_8 | aime25_repeat8:exact_match | aime25_repeat8:pass_at_8 |
|---|---|---|---|---|---|---|
| ablation | `ablation_184` | 184 | - | - | - | - |
| ablation | `ablation_276` | 276 | - | - | - | - |
| ablation | `ablation_368` | 368 | - | - | - | - |
| ablation | `ablation_460` | 460 | - | - | - | - |
| ablation | `ablation_552` | 552 | - | - | - | - |
| ablation | `ablation_644` | 644 | - | - | - | - |
| ablation | `ablation_736` | 736 | - | - | - | - |
| ablation | `ablation_828` | 828 | - | - | - | - |
| ablation | `ablation_92` | 92 | - | - | - | - |
| ablation | `ablation_920` | 920 | - | - | - | - |
| base | `base_qwen3_1_7b` | 1 | - | - | - | - |
| mix | `mix_184` | 184 | - | - | - | - |
| mix | `mix_276` | 276 | - | - | - | - |
| mix | `mix_368` | 368 | - | - | - | - |
| mix | `mix_460` | 460 | - | - | - | - |
| mix | `mix_552` | 552 | - | - | - | - |
| mix | `mix_644` | 644 | - | - | - | - |
| mix | `mix_736` | 736 | - | - | - | - |
| mix | `mix_828` | 828 | - | - | - | - |
| mix | `mix_92` | 92 | - | - | - | - |
| mix | `mix_920` | 920 | - | - | - | - |
| origin_only | `origin_310` | 310 | - | - | - | - |
| sub_only | `sub_230` | 230 | - | - | - | - |
| sub_only | `sub_460` | 460 | - | - | - | - |
| sub_only | `sub_690` | 690 | - | - | - | - |
| sub_only | `sub_920` | 920 | - | - | - | - |

## Core Benchmarks

| family | model | step | math500:exact_match | math500:acc_norm | gpqa_diamond:exact_match | gpqa_diamond:acc_norm | olympiadbench:exact_match | olympiadbench:acc_norm |
|---|---|---|---|---|---|---|---|---|
| ablation | `ablation_184` | 184 | 0.8580 | - | 0.3737 | - | 0.4050 | - |
| ablation | `ablation_276` | 276 | 0.8520 | - | 0.3485 | - | 0.3902 | - |
| ablation | `ablation_368` | 368 | 0.8560 | - | 0.3485 | - | 0.4006 | - |
| ablation | `ablation_460` | 460 | 0.8560 | - | 0.3283 | - | 0.3932 | - |
| ablation | `ablation_552` | 552 | 0.8700 | - | 0.3586 | - | 0.3991 | - |
| ablation | `ablation_644` | 644 | 0.8680 | - | 0.3586 | - | 0.3917 | - |
| ablation | `ablation_736` | 736 | 0.8540 | - | 0.3687 | - | 0.4021 | - |
| ablation | `ablation_828` | 828 | 0.8500 | - | 0.2980 | - | 0.3961 | - |
| ablation | `ablation_92` | 92 | 0.8500 | - | 0.3485 | - | 0.3872 | - |
| ablation | `ablation_920` | 920 | 0.8600 | - | 0.3434 | - | 0.3991 | - |
| base | `base_qwen3_1_7b` | 1 | 0.8520 | - | 0.2980 | - | 0.3828 | - |
| mix | `mix_184` | 184 | 0.8620 | - | 0.3081 | - | 0.3843 | - |
| mix | `mix_276` | 276 | 0.8580 | - | 0.3687 | - | 0.3932 | - |
| mix | `mix_368` | 368 | 0.8520 | - | 0.3485 | - | 0.3872 | - |
| mix | `mix_460` | 460 | 0.8560 | - | 0.3586 | - | 0.4021 | - |
| mix | `mix_552` | 552 | 0.8540 | - | 0.3434 | - | 0.4036 | - |
| mix | `mix_644` | 644 | 0.8520 | - | 0.3737 | - | 0.3917 | - |
| mix | `mix_736` | 736 | 0.8620 | - | 0.3434 | - | 0.3902 | - |
| mix | `mix_828` | 828 | 0.8600 | - | 0.3838 | - | 0.4036 | - |
| mix | `mix_92` | 92 | 0.8520 | - | 0.3434 | - | 0.3961 | - |
| mix | `mix_920` | 920 | 0.8360 | - | 0.3939 | - | 0.4184 | - |
| origin_only | `origin_310` | 310 | 0.8640 | - | 0.3384 | - | 0.3917 | - |
| sub_only | `sub_230` | 230 | 0.8660 | - | 0.3434 | - | 0.4036 | - |
| sub_only | `sub_460` | 460 | 0.8640 | - | 0.3485 | - | 0.3858 | - |
| sub_only | `sub_690` | 690 | 0.8660 | - | 0.3535 | - | 0.3887 | - |
| sub_only | `sub_920` | 920 | 0.8620 | - | 0.3434 | - | 0.4050 | - |

## 结论

- `aime24_repeat8` mix best: - = -，目标 0.50，状态：未达标；base=-，sub_only=-，ablation=-。
- `aime25_repeat8` mix best: - = -，目标 0.40，状态：未达标；base=-，sub_only=-，ablation=-。

## 失败样例路径

- wrong-but-parseable：`/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/summary/wrong_but_parseable.csv`
- unparseable：`/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/summary/unparseable_outputs.csv`
- 原始 samples：`/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127` 下各模型/benchmark 子目录的 `samples_*.jsonl` 或 `samples_*.json`。
