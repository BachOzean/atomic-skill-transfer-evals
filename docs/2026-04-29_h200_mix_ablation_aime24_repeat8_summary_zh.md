# AIME24 Repeat8 Summary

- Run tag: `h200_mix_ablation_repeat8_20260429_113127`
- Generated at: `2026-04-29T13:59:02`
- Benchmark: `aime24_repeat8`, 30 AIME24 problems x 8 samples = 240 generations per model.
- Decode: thinking mode, chat template, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`, `max_gen_toks=38912`.
- Scope: primary model list only; supplement mix checkpoints have not been run yet in this queue phase.

## Main Takeaways

- Best `mix` by exact_match: `mix_644` at 48.33% exact_match, delta vs base +6.25 pp.
- Best `mix` by pass_at_8: `mix_276` at 76.67% pass_at_8, delta vs base +6.67 pp.
- Best `sub_only` by exact_match: `sub_690` at 44.17%; best mix exact is +4.17 pp vs that.
- `origin_310` is strong: 46.25% exact_match and 76.67% pass_at_8.
- Best `ablation` by exact_match: `ablation_92` at 48.33% exact_match and 80.00% pass_at_8.

## Family Summary

| family | n | best exact model | best exact | best pass model | best pass_at_8 | avg exact | avg pass_at_8 |
|---|---:|---|---:|---|---:|---:|---:|
| ablation | 10 | `ablation_92` | 48.33% | `ablation_92` | 80.00% | 44.79% | 75.67% |
| base | 1 | `base_qwen3_1_7b` | 42.08% | `base_qwen3_1_7b` | 70.00% | 42.08% | 70.00% |
| mix | 5 | `mix_644` | 48.33% | `mix_276` | 76.67% | 44.67% | 73.33% |
| origin_only | 1 | `origin_310` | 46.25% | `origin_310` | 76.67% | 46.25% | 76.67% |
| sub_only | 4 | `sub_690` | 44.17% | `sub_690` | 73.33% | 42.81% | 73.33% |

## Ranked By Exact Match

| rank | model | family | step | exact_match | pass_at_8 |
|---:|---|---|---:|---:|---:|
| 1 | `ablation_92` | ablation | 92 | 48.33% | 80.00% |
| 2 | `mix_644` | mix | 644 | 48.33% | 70.00% |
| 3 | `ablation_460` | ablation | 460 | 46.67% | 80.00% |
| 4 | `ablation_736` | ablation | 736 | 46.67% | 76.67% |
| 5 | `origin_310` | origin_only | 310 | 46.25% | 76.67% |
| 6 | `ablation_184` | ablation | 184 | 46.25% | 73.33% |
| 7 | `ablation_552` | ablation | 552 | 45.00% | 73.33% |
| 8 | `ablation_920` | ablation | 920 | 45.00% | 73.33% |
| 9 | `mix_276` | mix | 276 | 44.58% | 76.67% |
| 10 | `sub_690` | sub_only | 690 | 44.17% | 73.33% |
| 11 | `mix_460` | mix | 460 | 43.75% | 76.67% |
| 12 | `sub_230` | sub_only | 230 | 43.75% | 73.33% |
| 13 | `mix_828` | mix | 828 | 43.75% | 70.00% |
| 14 | `ablation_368` | ablation | 368 | 42.92% | 73.33% |
| 15 | `mix_920` | mix | 920 | 42.92% | 73.33% |
| 16 | `sub_920` | sub_only | 920 | 42.92% | 73.33% |
| 17 | `ablation_276` | ablation | 276 | 42.50% | 73.33% |
| 18 | `ablation_644` | ablation | 644 | 42.50% | 73.33% |
| 19 | `ablation_828` | ablation | 828 | 42.08% | 80.00% |
| 20 | `base_qwen3_1_7b` | base | - | 42.08% | 70.00% |
| 21 | `sub_460` | sub_only | 460 | 40.42% | 73.33% |

## Ranked By Pass At 8

| rank | model | family | step | pass_at_8 | exact_match |
|---:|---|---|---:|---:|---:|
| 1 | `ablation_92` | ablation | 92 | 80.00% | 48.33% |
| 2 | `ablation_460` | ablation | 460 | 80.00% | 46.67% |
| 3 | `ablation_828` | ablation | 828 | 80.00% | 42.08% |
| 4 | `ablation_736` | ablation | 736 | 76.67% | 46.67% |
| 5 | `origin_310` | origin_only | 310 | 76.67% | 46.25% |
| 6 | `mix_276` | mix | 276 | 76.67% | 44.58% |
| 7 | `mix_460` | mix | 460 | 76.67% | 43.75% |
| 8 | `ablation_184` | ablation | 184 | 73.33% | 46.25% |
| 9 | `ablation_552` | ablation | 552 | 73.33% | 45.00% |
| 10 | `ablation_920` | ablation | 920 | 73.33% | 45.00% |
| 11 | `sub_690` | sub_only | 690 | 73.33% | 44.17% |
| 12 | `sub_230` | sub_only | 230 | 73.33% | 43.75% |
| 13 | `ablation_368` | ablation | 368 | 73.33% | 42.92% |
| 14 | `mix_920` | mix | 920 | 73.33% | 42.92% |
| 15 | `sub_920` | sub_only | 920 | 73.33% | 42.92% |
| 16 | `ablation_276` | ablation | 276 | 73.33% | 42.50% |
| 17 | `ablation_644` | ablation | 644 | 73.33% | 42.50% |
| 18 | `sub_460` | sub_only | 460 | 73.33% | 40.42% |
| 19 | `mix_644` | mix | 644 | 70.00% | 48.33% |
| 20 | `mix_828` | mix | 828 | 70.00% | 43.75% |
| 21 | `base_qwen3_1_7b` | base | - | 70.00% | 42.08% |

## Artifact Paths

- Run root: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127`
- TSV summary: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/aime24_repeat8_summary.tsv`
- Per-model results and samples are under `<run_root>/<model>/aime24_repeat8/`.
