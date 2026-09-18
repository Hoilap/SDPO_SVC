# Continual smoke test

From the repository root, with the normal `sdpo` environment activated:

```bash
bash experiments/continual/smoke.sh --dry-run
sbatch experiments/continual/smoke.sh
```

To opt into four-GPU shard-parallel SVC, without changing training settings:

```bash
SVC_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
  sbatch experiments/continual/smoke.sh
```

`SVC_DEVICES` overrides `SVC_DEVICE`. These are logical device indices inside
the Slurm allocation. Each spawned worker calibrates and writes distinct raw
checkpoint shards. Shards are greedily balanced by file size, not exact compute
cost; fewer than four shards cannot keep four GPUs busy. Single-file checkpoints
use only one worker. CPU memory and filesystem bandwidth demand increase with
parallelism; no fourfold end-to-end speedup is guaranteed. Check that training
workers have released GPU memory before SVC begins.

All workers write into a private sibling staging directory. The parent publishes
the model directory only after every worker and metadata write succeeds. Normal
exceptions remove the private staging directory and do not publish partial output;
hard kills can leave hidden staging directories but no new final model directory.
The report records requested devices and shard assignments. The matrix RNG seed
is unchanged from single-device SVC and independent of worker scheduling; compare
results with numerical tolerances, not bitwise equality across CPU and CUDA.

Submission is a separate, explicit action. The profile does not submit itself.
Use a clean shell or unset previous experiment overrides: explicit environment
settings take precedence over the profile's shell defaults.

Defaults: all four tasks, 2 updates per task, 8 prompts per global batch,
4 responses per prompt, micro batch 2 per GPU on 4 GPUs (4 accumulation steps),
24 CPU cores per job (Ray follows the Slurm CPU allocation),
learning rate 1e-5, no warmup, Top-100 + tail distillation with alpha 0.5,
EMA rate 0.05. Gradient checkpointing is enabled. Student prompt/response
limits remain 2048/8192; teacher reprompt limit remains 10240 and max_model_len
remains 18944. All length limits and derived token budgets are inherited from
the production configuration without overrides. SVC is unchanged
(rank 64, strength 0.5, CPU by default), as are model export and model handoff.
Outputs default to `outputs/sdpo_svc_cl_smoke/<Slurm job ID>`.
Tasks other than code also write validation generations and per-sample scores to
`evaluation/<experiment-name>/<step>.jsonl`, plus aggregate metrics and the
evaluated sample count to the adjacent `<step>.metrics.json`.  The JSONL records
include the validation data source and sample UID for mixed-domain analysis.
The code stage saves only `<step>.metrics.json` with aggregate metrics and the
evaluated sample count; it does not write the full per-sample JSONL.
The SVC report records its wall-clock runtime in `elapsed_seconds`.

The loader caps each task at 128 candidate training rows before prompt-length
filtering. Math retains its ordered unique-prefix policy; other tasks sample
with seed 42. At least 16 eligible rows per task are needed for both updates
in the single epoch. This cap does not avoid reading source files or the
existing full-file preflight checks, and never regenerates train/test splits.

Code data is normalized before loader sampling. Dolci is a mixed-domain dataset:
only `dataset=["code"]` and `["code_stdio"]` rows enter the code stage. Assertions
and stdin/stdout test pairs become the existing verifier's `code` and `stdin`
formats; non-code rows are counted and excluded. Raw LiveCodeBench is converted
separately for evaluation, retaining public and private tests and its original
problem set. Outputs and count reports live under this run's `train_data/` and
`eval_data/`; source files are never overwritten. Invalid code test payloads fail
conversion rather than receiving fabricated labels or being silently dropped.
Conversion does not execute test code. Actual scoring still uses the existing
local Python verifier, whose resource guards are NOT a security sandbox.
Syntactically valid assertion suites without an observable correctness oracle
are excluded and reported separately with their fraction among code candidates.
Recognized oracles include Python `assert`, unittest-style `assert*` calls,
`pytest.raises`/`assertRaises`, and explicit `raise AssertionError` checks.
Dolci stdin tests are accepted in both JSON and its compressed plain-data
pickle representation; pickle globals and persistent references are rejected.
Decoded pickle payloads remain capped at 128 MiB.
Compressed `code_stdio` rows containing list-valued input/output are excluded
and reported separately because the source mixes stdin lines with functional
arguments without retaining a reliable test-mode or function-name field.
Dependent assertion snippets (shared definitions, setup or helper calls) run as
one ordered suite with an aggregate one-second-per-source-snippet time budget.
All source snippets are retained, with `original_test_count` in the test payload.
Simple independent assertions remain separate. For grouped suites, accuracy is
suite-level (all-or-nothing), while the configured sparse reward remains 1 only
when all checks pass; feedback stops at the first failure within that suite.

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
gradient checkpointing saves student activations, but micro batch 2 increases
teacher/logits memory relative to micro batch 1 and can still cause OOM.
A short-sample pass does not establish safety
at the length limits. If this smoke test passes, increase steps and sample counts
before a full experiment; keep length limits unchanged.
