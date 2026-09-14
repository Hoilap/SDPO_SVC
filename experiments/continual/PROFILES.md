# Continual training profiles

Submit one thin profile entry point from the repository root.  All three call
the shared `run_sdpo_svc_cl.sh` engine, so dataset preparation, validation,
checkpoint export, and SVC logic remain in one place.

| Script | Per-task training data | Training schedule | Output root |
| --- | --- | --- | --- |
| `smoke.sh` | At most 128 candidates | 2 optimizer updates | `outputs/sdpo_svc_cl_smoke/<job-id>` |
| `try.sh` | At most 3,000 candidates | Full one-epoch settings | `outputs/sdpo_svc_cl_try/<job-id>` |
| `full.sh` | Manifest-defined full data | Full one-epoch settings | `outputs/sdpo_svc_cl` |

The try and full profiles use an actor PPO micro batch size of 4 per GPU.
Smoke keeps 2 per GPU so its global mini batch of 8 remains divisible across
the four training GPUs.

```bash
sbatch experiments/continual/smoke.sh
sbatch experiments/continual/try.sh
sbatch experiments/continual/full.sh
```

The 3,000-row try cap is applied independently to math, science, tool, and
code before prompt-length filtering.  It never enlarges a smaller manifest
dataset.  Environment overrides such as `START_TASK`, `END_TASK`,
`INITIAL_CONTINUAL_MODEL`, and `SVC_DEVICES` continue to work normally.

Use `bash experiments/continual/<profile>.sh --dry-run` to inspect resolved
commands or `--preflight` to validate data without loading a model.
