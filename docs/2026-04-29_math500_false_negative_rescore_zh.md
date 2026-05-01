# MATH500 Saved-Generation False Negative Rescore

- run_root: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127`
- updated_at: `2026-04-30T00:04:34`
- method: reuse saved `samples_*.jsonl`; no model inference. Added normalization for LaTeX text wrappers, omitted base subscripts, units/ordinals, thousands separators, vector/matrix forms, plus conservative SymPy equivalence for short algebraic/radical expressions.
- validation: rescued `426` old false negatives across `26` sample files; old-correct regressions after the fix: `0`.

## Family Best

| family | best model | old exact | rescored exact | delta |
|---|---|---:|---:|---:|
| `ablation` | `ablation_552` | 87.00 | 90.20 | 3.20 |
| `base` | `base_qwen3_1_7b` | 85.20 | 88.80 | 3.60 |
| `mix` | `mix_736` | 86.20 | 89.20 | 3.00 |
| `origin` | `origin_310` | 86.40 | 89.80 | 3.40 |
| `sub` | `sub_230` | 86.60 | 90.20 | 3.60 |

## All Checkpoints

| model | old exact | rescored exact | delta | rescued | regressions |
|---|---:|---:|---:|---:|---:|
| `ablation_552` | 87.00 | 90.20 | 3.20 | 16 | 0 |
| `sub_230` | 86.60 | 90.20 | 3.60 | 18 | 0 |
| `sub_460` | 86.40 | 90.20 | 3.80 | 19 | 0 |
| `sub_690` | 86.60 | 90.20 | 3.60 | 18 | 0 |
| `origin_310` | 86.40 | 89.80 | 3.40 | 17 | 0 |
| `ablation_644` | 86.80 | 89.60 | 2.80 | 14 | 0 |
| `sub_920` | 86.20 | 89.40 | 3.20 | 16 | 0 |
| `ablation_184` | 85.80 | 89.20 | 3.40 | 17 | 0 |
| `mix_736` | 86.20 | 89.20 | 3.00 | 15 | 0 |
| `mix_828` | 86.00 | 89.20 | 3.20 | 16 | 0 |
| `ablation_460` | 85.60 | 89.00 | 3.40 | 17 | 0 |
| `mix_276` | 85.80 | 89.00 | 3.20 | 16 | 0 |
| `mix_552` | 85.40 | 89.00 | 3.60 | 18 | 0 |
| `ablation_368` | 85.60 | 88.80 | 3.20 | 16 | 0 |
| `base_qwen3_1_7b` | 85.20 | 88.80 | 3.60 | 18 | 0 |
| `mix_184` | 86.20 | 88.80 | 2.60 | 13 | 0 |
| `mix_460` | 85.60 | 88.80 | 3.20 | 16 | 0 |
| `ablation_736` | 85.40 | 88.60 | 3.20 | 16 | 0 |
| `ablation_828` | 85.00 | 88.60 | 3.60 | 18 | 0 |
| `ablation_920` | 86.00 | 88.60 | 2.60 | 13 | 0 |
| `ablation_92` | 85.00 | 88.40 | 3.40 | 17 | 0 |
| `mix_368` | 85.20 | 88.40 | 3.20 | 16 | 0 |
| `mix_644` | 85.20 | 88.40 | 3.20 | 16 | 0 |
| `ablation_276` | 85.20 | 88.20 | 3.00 | 15 | 0 |
| `mix_92` | 85.20 | 88.20 | 3.00 | 15 | 0 |
| `mix_920` | 83.60 | 87.60 | 4.00 | 20 | 0 |

## Artifacts

- Summary CSV: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/summary/math500_rescore/math500_rescore_summary.csv`
- Rescued false negatives: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/summary/math500_rescore/math500_rescued_false_negatives.csv`
- All rows: `/root/ast_eval_runtime/outputs/external_math/h200_mix_ablation_repeat8_20260429_113127/summary/math500_rescore/math500_rescore_rows.csv`
