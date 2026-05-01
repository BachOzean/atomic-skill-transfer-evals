# H200 mix / sub_only / base / ablation 评测报告

- 生成时间：2026-04-30 12:26:32
- 评测目录：`/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_supplement_math_20260430_012030`
- 官方口径参考：Qwen Quickstart / thinking mode，`enable_thinking=True`，thinking mode 推荐 `temperature=0.6, top_p=0.95, top_k=20, min_p=0`。链接：https://qwen.readthedocs.io/en/v3.0/getting_started/quickstart.html
- 结果文件数：130；sample 文件数：130

## 口径

- AIME 主表：默认使用 `aime24_repeat8`、`aime25_repeat8`；如目录中存在 repeat64，也会一并汇总。`max_gen_toks=38912`，`temperature=0.6`，`top_p=0.95`，`top_k=20`，`min_p=0`，开启 chat template 和 thinking mode。
- 核心 benchmark：`math500`、生成式 `gpqa_diamond`、`olympiadbench`，默认 `max_gen_toks=32768`，采样参数沿官方 thinking mode。
- 诊断口径只在官方口径未达标时补跑：`temperature=1.0`、`top_p=0.95`、`top_k=0`、`max_gen_toks=38912`。


## Supplement Math Benchmarks

| family | model | step | amc23:exact_match | matharena_brumo_2025:exact_match | matharena_hmmt_feb_2025:exact_match | matharena_hmmt_nov_2025:exact_match | omni_math_500:exact_match |
|---|---|---|---|---|---|---|---|
| ablation | `ablation_184` | 184 | 0.7590 | 0.5333 | 0.2667 | 0.2667 | 0.1764 |
| ablation | `ablation_276` | 276 | 0.7349 | 0.4333 | 0.2000 | 0.2333 | 0.1683 |
| ablation | `ablation_368` | 368 | 0.7590 | 0.5667 | 0.1667 | 0.1667 | 0.1864 |
| ablation | `ablation_460` | 460 | 0.7229 | 0.5000 | 0.2667 | 0.2333 | 0.1764 |
| ablation | `ablation_552` | 552 | 0.7108 | 0.4667 | 0.2667 | 0.2333 | 0.1824 |
| ablation | `ablation_644` | 644 | 0.7831 | 0.5000 | 0.2333 | 0.2667 | 0.1764 |
| ablation | `ablation_736` | 736 | 0.7349 | 0.4000 | 0.2333 | 0.3000 | 0.1703 |
| ablation | `ablation_828` | 828 | 0.7470 | 0.5333 | 0.1667 | 0.3667 | 0.1784 |
| ablation | `ablation_92` | 92 | 0.7470 | 0.4667 | 0.2333 | 0.2000 | 0.1563 |
| ablation | `ablation_920` | 920 | 0.7711 | 0.5333 | 0.4000 | 0.3000 | 0.1844 |
| base | `base_qwen3_1_7b` | 1 | 0.7349 | 0.4000 | 0.1667 | 0.4000 | 0.1784 |
| mix | `mix_184` | 184 | 0.7590 | 0.5000 | 0.1667 | 0.3000 | 0.1743 |
| mix | `mix_276` | 276 | 0.7229 | 0.5000 | 0.2333 | 0.2333 | 0.1944 |
| mix | `mix_368` | 368 | 0.7831 | 0.5000 | 0.3000 | 0.2333 | 0.1603 |
| mix | `mix_460` | 460 | 0.6988 | 0.4000 | 0.2000 | 0.2667 | 0.1884 |
| mix | `mix_552` | 552 | 0.7349 | 0.4667 | 0.1667 | 0.2333 | 0.1723 |
| mix | `mix_644` | 644 | 0.7470 | 0.5333 | 0.2667 | 0.2667 | 0.1764 |
| mix | `mix_736` | 736 | 0.7711 | 0.5000 | 0.2333 | 0.2000 | 0.1884 |
| mix | `mix_828` | 828 | 0.7711 | 0.4000 | 0.2000 | 0.3000 | 0.1683 |
| mix | `mix_92` | 92 | 0.7470 | 0.5000 | 0.2667 | 0.2000 | 0.1764 |
| mix | `mix_920` | 920 | 0.7590 | 0.4667 | 0.2000 | 0.3000 | 0.1784 |
| origin_only | `origin_310` | 310 | 0.7711 | 0.5667 | 0.2667 | 0.3667 | 0.1824 |
| sub_only | `sub_230` | 230 | 0.8313 | 0.5667 | 0.2667 | 0.3333 | 0.1824 |
| sub_only | `sub_460` | 460 | 0.7711 | 0.4667 | 0.2000 | 0.2333 | 0.1683 |
| sub_only | `sub_690` | 690 | 0.7831 | 0.5000 | 0.2333 | 0.3000 | 0.1743 |
| sub_only | `sub_920` | 920 | 0.7952 | 0.5000 | 0.2333 | 0.2333 | 0.1743 |

## 结论


## 失败样例路径

- wrong-but-parseable：`/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_supplement_math_20260430_012030/summary/wrong_but_parseable.csv`
- unparseable：`/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_supplement_math_20260430_012030/summary/unparseable_outputs.csv`
- 原始 samples：`/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_supplement_math_20260430_012030` 下各模型/benchmark 子目录的 `samples_*.jsonl` 或 `samples_*.json`。
