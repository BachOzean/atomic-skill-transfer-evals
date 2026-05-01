# H200 mix / sub_only / base / ablation 评测报告

- 生成时间：2026-04-29 11:02:44
- 评测目录：`/root/ast_eval_runtime/outputs/external_math/h200_eval_spawn_smoke_20260429_105824`
- 官方口径参考：Qwen Quickstart / thinking mode，`enable_thinking=True`，thinking mode 推荐 `temperature=0.6, top_p=0.95, top_k=20, min_p=0`。链接：https://qwen.readthedocs.io/en/v3.0/getting_started/quickstart.html
- 结果文件数：0；sample 文件数：0

## 口径

- AIME 主表：`aime24_repeat64`、`aime25_repeat64`，`max_gen_toks=38912`，`temperature=0.6`，`top_p=0.95`，`top_k=20`，`min_p=0`，开启 chat template 和 thinking mode。
- 核心 benchmark：`math500`、生成式 `gpqa_diamond`、`olympiadbench`，默认 `max_gen_toks=32768`，采样参数沿官方 thinking mode。
- 诊断口径只在官方口径未达标时补跑：`temperature=1.0`、`top_p=0.95`、`top_k=0`、`max_gen_toks=38912`。

## 当前状态

- 尚未发现 `results_*.json`，报告会在评测完成后由 watcher 重新生成。

## 失败样例路径

- wrong-but-parseable：`-`
- unparseable：`-`
- 原始 samples：`/root/ast_eval_runtime/outputs/external_math/h200_eval_spawn_smoke_20260429_105824` 下各模型/benchmark 子目录的 `samples_*.jsonl` 或 `samples_*.json`。
