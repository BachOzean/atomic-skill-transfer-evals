# A800 Profile

This profile matches the current A800 Slurm environment used for the
atomic-skill-transfer eval jobs.

Usage:

```bash
source servers/a800/runtime.env.example
sbatch --export=ALL,MODEL_PATH=/data/home/$USER/run/models/Qwen3-1.7B \
  servers/a800/launchers/run_external_math_benchmarks_a800.sh
```

Notes:

- Keep Slurm output and all benchmark outputs under `/data/home/$USER/run` or
  this repo's ignored `runtime/` directory.
- The site policy has rejected `--cpus-per-task` for these eval jobs before; do
  not add an explicit CPU request unless the partition policy changes.
- Short `--time` values backfill more reliably on `gpu_a800`; split long evals
  by benchmark when queue latency matters.
- For long-thinking AIME runs, the current stable profile is
  `MAX_GEN_TOKS=32768`, `MAX_MODEL_LEN=40960`, and `MAX_NUM_SEQS=8`.
