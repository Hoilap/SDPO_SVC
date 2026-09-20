#!/usr/bin/env bash

# Submit the remaining tail-causal stages after an interrupted SDPO-Math run.
# The stages are linked with afterok dependencies, so a failure stops the chain.

set -euo pipefail

if [[ $# -ne 1 || "$1" != /* ]]; then
    echo "Usage: $0 /absolute/path/to/existing/output-root" >&2
    exit 2
fi

OUTPUT_ROOT="$1"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this submission helper on the manager host, outside a Slurm job." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$script_dir/../.." && pwd)"
cd "$PROJECT_ROOT"

[[ -f "$OUTPUT_ROOT/data/dataset_manifest.json" ]] || {
    echo "Missing dataset manifest below $OUTPUT_ROOT" >&2
    exit 1
}
[[ -s "$OUTPUT_ROOT/math_hf/grpo_full/model.safetensors.index.json" ]] || {
    echo "The completed GRPO-Math checkpoint is missing below $OUTPUT_ROOT" >&2
    exit 1
}
[[ ! -e "$OUTPUT_ROOT/math_hf/sdpo_full" ]] || {
    echo "Refusing to restart because SDPO-Math output already exists below $OUTPUT_ROOT" >&2
    exit 1
}

if squeue -h -u "${USER:?}" -n tail-causal | grep -q .; then
    echo "A tail-causal job is already queued or running for $USER; refusing a duplicate chain." >&2
    squeue -u "$USER" -n tail-causal -o '%.18i %.20j %.10T %.24R %.10M'
    exit 1
fi

export_arg="ALL,OUTPUT_ROOT=$OUTPUT_ROOT"
sdpo_job="$(sbatch --parsable --export="$export_arg" experiments/causal/run.sh math-sdpo)"
surgery_job="$(sbatch --parsable --dependency="afterok:$sdpo_job" --export="$export_arg" experiments/causal/run.sh surgery)"
eval_math_job="$(sbatch --parsable --dependency="afterok:$surgery_job" --export="$export_arg" experiments/causal/run.sh eval-math)"
science_job="$(sbatch --parsable --dependency="afterok:$eval_math_job" --export="$export_arg" experiments/causal/run.sh science)"
eval_science_job="$(sbatch --parsable --dependency="afterok:$science_job" --export="$export_arg" experiments/causal/run.sh eval-science)"

printf 'SDPO-Math:    %s\n' "$sdpo_job"
printf 'Surgery:      %s\n' "$surgery_job"
printf 'Eval-Math:    %s\n' "$eval_math_job"
printf 'Science:      %s\n' "$science_job"
printf 'Eval-Science: %s\n' "$eval_science_job"
printf '\nAfter job %s completes, build the summary with:\n' "$eval_science_job"
printf 'OUTPUT_ROOT=%q bash experiments/causal/run.sh summary\n' "$OUTPUT_ROOT"
