#!/usr/bin/env bash

#SBATCH --job-name=sdpo-svc-full
#SBATCH --nodes=1
#SBATCH --partition=gpu_chen
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=460000
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/sdpo-svc-full-%j.out
#SBATCH --error=logs/sdpo-svc-full-%j.err

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script from the repository root}}"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$script_dir/../.." && pwd)"
fi

export PROJECT_ROOT RUN_PROFILE=full
exec bash "$PROJECT_ROOT/experiments/continual/run_sdpo_svc_cl.sh" "$@"
