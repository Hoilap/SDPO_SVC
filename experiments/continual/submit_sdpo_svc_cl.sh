#!/usr/bin/env bash

# Submit one 4-GPU sbatch job per continual-learning task.  Jobs are connected
# with afterok dependencies, so each task consumes the calibrated checkpoint
# produced by its predecessor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKER_SCRIPT="$SCRIPT_DIR/run_sdpo_svc_cl.sh"
source "$SCRIPT_DIR/dataset_manifest.sh"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/sdpo_svc_cl}"
HF_ROOT="${HF_ROOT:-$OUTPUT_ROOT/hf_models}"
START_TASK="${START_TASK:-0}"
END_TASK="${END_TASK:-$((${#CL_TRAIN_DATASETS[@]} - 1))}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-a156}"
SLURM_PARTITION="${SLURM_PARTITION:-normal}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_ENVIRONMENT="${SLURM_ENVIRONMENT:-sdpo}"
SLURM_MEM="${SLURM_MEM:-460000}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-288}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$OUTPUT_ROOT/slurm_logs}"

if (( START_TASK < 0 || END_TASK >= ${#CL_TRAIN_DATASETS[@]} || START_TASK > END_TASK )); then
    echo "Invalid task range START_TASK=$START_TASK END_TASK=$END_TASK" >&2
    exit 2
fi
if [[ ! -x "$WORKER_SCRIPT" ]]; then
    echo "Worker script is not executable: $WORKER_SCRIPT" >&2
    exit 1
fi
if [[ "$DRY_RUN" != true ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; run this launcher from a Slurm login node." >&2
    exit 1
fi
if [[ "${WANDB_MODE:-online}" == "disabled" ]]; then
    echo "WANDB_MODE=disabled is not allowed; W&B logging is required." >&2
    exit 2
fi

mkdir -p "$SLURM_LOG_DIR"

calibrated_path_for_task() {
    local index="$1"
    local number name
    number="$(printf '%02d' "$((index + 1))")"
    name="${CL_TASK_NAMES[$index]}"
    printf '%s/%s-%s-svc' "$HF_ROOT" "$number" "$name"
}

previous_job_id=""
for ((task_index = START_TASK; task_index <= END_TASK; task_index++)); do
    task_number="$(printf '%02d' "$((task_index + 1))")"
    if (( task_index == 0 )); then
        previous_model="$BASE_MODEL"
    elif (( task_index == START_TASK )) && [[ -n "${INITIAL_CONTINUAL_MODEL:-}" ]]; then
        previous_model="$INITIAL_CONTINUAL_MODEL"
    else
        previous_model="$(calibrated_path_for_task "$((task_index - 1))")"
    fi

    export_values="ALL,START_TASK=$task_index,END_TASK=$task_index"
    export_values+=",BASE_MODEL=$BASE_MODEL,INITIAL_CONTINUAL_MODEL=$previous_model"
    export_values+=",OUTPUT_ROOT=$OUTPUT_ROOT,HF_ROOT=$HF_ROOT,WANDB_MODE=${WANDB_MODE:-online}"

    submit_cmd=(
        sbatch
        --parsable
        --job-name="svc-cl-$task_number"
        --account="$SLURM_ACCOUNT"
        --nodes=1
        --partition="$SLURM_PARTITION"
        --time="$SLURM_TIME"
        --ntasks-per-node=1
        --gpus-per-node=4
        --mem="$SLURM_MEM"
        --cpus-per-task="$SLURM_CPUS_PER_TASK"
        --output="$SLURM_LOG_DIR/%x-%j.out"
        --error="$SLURM_LOG_DIR/%x-%j.err"
        --export="$export_values"
    )
    if [[ -n "$SLURM_ENVIRONMENT" ]]; then
        submit_cmd+=(--environment="$SLURM_ENVIRONMENT")
    fi
    if [[ -n "$previous_job_id" ]]; then
        submit_cmd+=(--dependency="afterok:$previous_job_id")
    fi
    submit_cmd+=("$WORKER_SCRIPT")

    if [[ "$DRY_RUN" == true ]]; then
        printf 'DRY-RUN:'
        printf ' %q' "${submit_cmd[@]}"
        printf '\n'
        previous_job_id="DRYRUN_${task_number}"
    else
        submission="$("${submit_cmd[@]}")"
        previous_job_id="${submission%%;*}"
        echo "Submitted task $task_number as job $previous_job_id (previous model: $previous_model)"
    fi
done

if [[ "$DRY_RUN" != true ]]; then
    echo "Submitted tasks $START_TASK..$END_TASK with afterok dependencies."
    echo "Final job id: $previous_job_id"
fi
