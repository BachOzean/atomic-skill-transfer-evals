# Server Profiles

Keep shared evaluation code in `scripts/`, `lm_eval_tasks/`, and `launchers/`.
Put machine-specific defaults here so one repository can be cloned on multiple
servers without creating branch drift.

Each server directory may contain:

- `runtime.env.example`: exportable paths and eval defaults for that machine.
- `README.md`: scheduler, CUDA, filesystem, and known-site notes.
- thin wrapper scripts only when environment variables cannot express the local
  scheduler requirement.

Do not commit model weights, private datasets, cache trees, run outputs, Slurm
logs, or credentials. Keep those under the server's runtime root and rely on
environment variables to point the shared launchers at them.
