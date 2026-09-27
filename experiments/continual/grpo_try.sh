#!/usr/bin/env bash

#SBATCH --job-name=grpo-try
#SBATCH --nodes=1
#SBATCH --partition=gpu_chen
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=460000
#SBATCH --cpus-per-task=24
#SBATCH --time=0
#SBATCH --output=logs/grpo-try-%j.out
#SBATCH --error=logs/grpo-try-%j.err

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script from the repository root}}"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$script_dir/../.." && pwd)"
fi

export PROJECT_ROOT RUN_PROFILE=try CONTINUAL_METHOD=grpo
# RLHFDataset applies train_max_samples before prompt-length filtering. With
# shuffle disabled, all four stages select indices [0, 3000) in file order.
export TRAIN_SAMPLE_LIMIT=3000 TRAIN_SHUFFLE_OVERRIDE=false
exec bash "$PROJECT_ROOT/experiments/continual/run_sdpo_svc_cl.sh" "$@"
