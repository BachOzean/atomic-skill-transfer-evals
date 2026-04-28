# H200 Profile

Use this folder for H200-specific runtime paths and scheduler notes. Keep
evaluation logic shared with the rest of the repository.

Start from `runtime.env.example`, adjust the model and data paths for the host,
then run the common launchers:

```bash
source servers/h200/runtime.env.example
BENCHMARKS=aime24,aime25,math500 bash launchers/run_external_math_benchmarks.sh
```

If the H200 host needs a scheduler wrapper, keep it thin and pass behavior
through environment variables instead of copying Python or task code.
