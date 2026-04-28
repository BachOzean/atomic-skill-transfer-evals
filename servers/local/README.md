# Local Profile

This profile is for quick smoke tests on a single interactive machine. It avoids
Slurm assumptions and defaults to the common launcher.

```bash
source servers/local/runtime.env.example
LIMIT=1 BENCHMARKS=aime24 bash launchers/run_external_math_benchmarks.sh
```

Use `BACKEND=hf` only for tiny smoke tests; production results should use the
same vLLM settings as the target server.
