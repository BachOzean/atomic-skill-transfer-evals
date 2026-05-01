# AIME25 Repeat8 Summary

- Run tag: `h200_mix_ablation_repeat8_20260429_113127`
- Generated at: `2026-04-29T15:56:58`
- Benchmark: `aime25_repeat8`, 30 AIME25 problems x 8 samples = 240 generations per model.
- Decode: thinking mode, chat template, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`, `max_gen_toks=38912`.
- Scope: primary model list only. Supplement mix checkpoints started after this primary AIME pass because AIME targets were not met.

## Main Takeaways

- Best `mix` by exact_match: `mix_460` at 35.83% exact_match, delta vs base +0.00 pp.
- Best `mix` by pass_at_8: `mix_276` at 56.67% pass_at_8, delta vs base +3.33 pp.
- Best `sub_only` by exact_match: `sub_230` at 37.92%; best mix exact is -2.08 pp vs that.
- `origin_310`: 36.67% exact_match and 63.33% pass_at_8.
- Best `ablation` by exact_match: `ablation_92` at 39.58% exact_match and 63.33% pass_at_8.

## Family Summary

| family | n | best exact model | best exact | best pass model | best pass_at_8 | avg exact | avg pass_at_8 |
|---|---:|---|---:|---|---:|---:|---:|
| ablation | 10 | `ablation_92` | 39.58% | `ablation_92` | 63.33% | 36.58% | 57.00% |
| base | 1 | `base_qwen3_1_7b` | 35.83% | `base_qwen3_1_7b` | 53.33% | 35.83% | 53.33% |
| mix | 5 | `mix_460` | 35.83% | `mix_276` | 56.67% | 34.00% | 52.00% |
| origin_only | 1 | `origin_310` | 36.67% | `origin_310` | 63.33% | 36.67% | 63.33% |
| sub_only | 4 | `sub_230` | 37.92% | `sub_230` | 66.67% | 36.15% | 55.83% |

## Ranked By Exact Match

| rank | model | family | step | exact_match | pass_at_8 |
|---:|---|---|---:|---:|---:|
| 1 | `ablation_92` | ablation | 92 | 39.58% | 63.33% |
| 2 | `ablation_184` | ablation | 184 | 38.33% | 60.00% |
| 3 | `sub_230` | sub_only | 230 | 37.92% | 66.67% |
| 4 | `ablation_460` | ablation | 460 | 37.92% | 63.33% |
| 5 | `ablation_276` | ablation | 276 | 37.50% | 53.33% |
| 6 | `ablation_828` | ablation | 828 | 37.08% | 56.67% |
| 7 | `origin_310` | origin_only | 310 | 36.67% | 63.33% |
| 8 | `sub_920` | sub_only | 920 | 36.67% | 56.67% |
| 9 | `ablation_644` | ablation | 644 | 36.25% | 56.67% |
| 10 | `base_qwen3_1_7b` | base | - | 35.83% | 53.33% |
| 11 | `mix_460` | mix | 460 | 35.83% | 53.33% |
| 12 | `mix_276` | mix | 276 | 35.42% | 56.67% |
| 13 | `ablation_552` | ablation | 552 | 35.42% | 53.33% |
| 14 | `ablation_920` | ablation | 920 | 35.42% | 53.33% |
| 15 | `sub_460` | sub_only | 460 | 35.42% | 50.00% |
| 16 | `mix_828` | mix | 828 | 35.00% | 53.33% |
| 17 | `sub_690` | sub_only | 690 | 34.58% | 50.00% |
| 18 | `ablation_368` | ablation | 368 | 34.17% | 60.00% |
| 19 | `ablation_736` | ablation | 736 | 34.17% | 50.00% |
| 20 | `mix_644` | mix | 644 | 32.92% | 50.00% |
| 21 | `mix_920` | mix | 920 | 30.83% | 46.67% |

## Ranked By Pass At 8

| rank | model | family | step | pass_at_8 | exact_match |
|---:|---|---|---:|---:|---:|
| 1 | `sub_230` | sub_only | 230 | 66.67% | 37.92% |
| 2 | `ablation_92` | ablation | 92 | 63.33% | 39.58% |
| 3 | `ablation_460` | ablation | 460 | 63.33% | 37.92% |
| 4 | `origin_310` | origin_only | 310 | 63.33% | 36.67% |
| 5 | `ablation_184` | ablation | 184 | 60.00% | 38.33% |
| 6 | `ablation_368` | ablation | 368 | 60.00% | 34.17% |
| 7 | `ablation_828` | ablation | 828 | 56.67% | 37.08% |
| 8 | `sub_920` | sub_only | 920 | 56.67% | 36.67% |
| 9 | `ablation_644` | ablation | 644 | 56.67% | 36.25% |
| 10 | `mix_276` | mix | 276 | 56.67% | 35.42% |
| 11 | `ablation_276` | ablation | 276 | 53.33% | 37.50% |
| 12 | `base_qwen3_1_7b` | base | - | 53.33% | 35.83% |
| 13 | `mix_460` | mix | 460 | 53.33% | 35.83% |
| 14 | `ablation_552` | ablation | 552 | 53.33% | 35.42% |
| 15 | `ablation_920` | ablation | 920 | 53.33% | 35.42% |
| 16 | `mix_828` | mix | 828 | 53.33% | 35.00% |
| 17 | `sub_460` | sub_only | 460 | 50.00% | 35.42% |
| 18 | `sub_690` | sub_only | 690 | 50.00% | 34.58% |
| 19 | `ablation_736` | ablation | 736 | 50.00% | 34.17% |
| 20 | `mix_644` | mix | 644 | 50.00% | 32.92% |
| 21 | `mix_920` | mix | 920 | 46.67% | 30.83% |

## Artifact Paths

- Run root: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127`
- TSV summary: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/aime25_repeat8_summary.tsv`
- Per-model results and samples are under `<run_root>/<model>/aime25_repeat8/`.
