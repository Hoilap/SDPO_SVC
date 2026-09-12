# No-gradient-checkpointing smoke test

From the repository root, with the normal `sdpo` environment activated:

```bash
RUN_PROFILE=smoke bash experiments/continual/run_sdpo_svc_cl.sh --dry-run
RUN_PROFILE=smoke sbatch experiments/continual/run_sdpo_svc_cl.sh
```

Submission is a separate, explicit action. The profile does not submit itself.
Use a clean shell or unset previous experiment overrides: explicit environment
settings take precedence over the profile's shell defaults.

Defaults: all four tasks, 2 updates per task, 8 prompts per global batch,
4 responses per prompt, micro batch 1 per GPU on 4 GPUs (8 accumulation steps),
learning rate 1e-5, no warmup, Top-100 + tail distillation with alpha 0.5,
EMA rate 0.05. Gradient checkpointing is disabled. Student prompt/response
limits remain 2048/8192; teacher reprompt limit remains 10240 and max_model_len
remains 18944. All length limits and derived token budgets are inherited from
the production configuration without overrides. SVC is unchanged
(rank 64, strength 0.5, CPU by default), as are model export and model handoff.
Outputs default to `outputs/sdpo_svc_cl_smoke/<Slurm job ID>`.

The loader caps each task at 128 candidate training rows before prompt-length
filtering. Math retains its ordered unique-prefix policy; other tasks sample
with seed 42. At least 16 eligible rows per task are needed for both updates
in the single epoch. This cap does not avoid reading source files or the
existing full-file preflight checks, and never regenerates train/test splits.

Validation runs only at each task's final step, with at most 32 candidate rows
from the combined cumulative pool and one response each. This is not stratified
per benchmark and does not guarantee coverage of every reward backend. Overlong
prompt filtering may further reduce both train and validation pools.

Check that each task reaches global_step_2, losses/gradient norms are finite,
checkpoints export, SVC completes, and the next task loads the calibrated model.
Also inspect distilled-token fraction: an empty feedback mask can run without
exercising a meaningful distillation update. These tiny-sample scores are not
scientific results.
Task-final validation is before SVC, not an evaluation of the calibrated output.

GPU memory safety is not guaranteed by configuration alone. Inspect all GPUs
through a complete update and validation; teacher offload, FSDP communication,
long logits, and model initialization can still cause peaks. In particular,
disabling gradient checkpointing while retaining production length limits can
cause OOM even with micro batch 1. A short-sample pass does not establish safety
at the length limits. If this smoke test passes, increase steps and sample counts
before a full experiment; keep length limits unchanged.
