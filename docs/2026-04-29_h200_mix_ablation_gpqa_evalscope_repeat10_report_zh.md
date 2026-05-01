# H200 GPQA-Diamond EvalScope-style Repeat10 重评报告

- run_tag: `h200_mix_ablation_gpqa_evalscope_repeat10_20260429_2332`
- run_root: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_gpqa_evalscope_repeat10_20260429_2332`
- updated_at: `2026-04-30T04:36:25`
- 口径: EvalScope-style GPQA prompt, `ANSWER: [LETTER]` 解析, no system instruction, chat template, thinking mode, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_gen_toks=32768`, repeat10。

## Top Results

| rank | model | family | step | seeds | mean exact | std | min | max |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `ablation_92` | ablation | 92 | 10 | 40.40 | 0.00 | 40.40 | 40.40 |
| 2 | `sub_230` | sub_only | 230 | 10 | 39.34 | 0.15 | 38.89 | 39.39 |
| 3 | `ablation_276` | ablation | 276 | 10 | 38.79 | 2.47 | 34.85 | 40.40 |
| 4 | `ablation_368` | ablation | 368 | 10 | 38.74 | 1.94 | 34.34 | 39.90 |
| 5 | `sub_920` | sub_only | 920 | 10 | 38.69 | 0.91 | 38.38 | 41.41 |
| 6 | `ablation_644` | ablation | 644 | 10 | 38.64 | 0.76 | 36.36 | 38.89 |
| 7 | `ablation_828` | ablation | 828 | 10 | 38.54 | 1.06 | 35.35 | 38.89 |
| 8 | `ablation_736` | ablation | 736 | 10 | 38.23 | 1.65 | 33.33 | 38.89 |
| 9 | `ablation_920` | ablation | 920 | 10 | 38.03 | 1.22 | 37.37 | 40.91 |
| 10 | `mix_552` | mix | 552 | 10 | 37.78 | 1.23 | 35.35 | 40.40 |
| 11 | `mix_184` | mix | 184 | 10 | 37.68 | 1.87 | 33.33 | 38.89 |
| 12 | `sub_690` | sub_only | 690 | 10 | 37.58 | 1.75 | 32.83 | 38.38 |
| 13 | `mix_736` | mix | 736 | 10 | 37.22 | 0.56 | 36.87 | 38.38 |
| 14 | `mix_368` | mix | 368 | 10 | 37.07 | 0.79 | 35.35 | 37.88 |
| 15 | `ablation_552` | ablation | 552 | 10 | 37.02 | 0.56 | 35.86 | 37.37 |
| 16 | `ablation_184` | ablation | 184 | 10 | 36.92 | 0.92 | 34.85 | 38.89 |
| 17 | `mix_644` | mix | 644 | 10 | 36.92 | 1.31 | 35.35 | 38.38 |
| 18 | `mix_920` | mix | 920 | 10 | 36.87 | 0.00 | 36.87 | 36.87 |
| 19 | `sub_460` | sub_only | 460 | 10 | 36.36 | 2.27 | 32.83 | 38.89 |
| 20 | `base_qwen3_1_7b` | base | - | 10 | 36.31 | 1.20 | 34.34 | 38.38 |

## Artifacts

- Summary CSV: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_gpqa_evalscope_repeat10_20260429_2332/summary/gpqa_evalscope_repeat10_summary.csv`
- Seed-level CSV: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_gpqa_evalscope_repeat10_20260429_2332/summary/gpqa_evalscope_seed_results.csv`
- Summary JSON: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_gpqa_evalscope_repeat10_20260429_2332/summary/gpqa_evalscope_repeat10_summary.json`
