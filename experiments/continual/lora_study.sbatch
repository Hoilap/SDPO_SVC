#!/usr/bin/env bash
#SBATCH --partition=gpu_chen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=0
set -euo pipefail
ulimit -S -u 16384
ulimit -c 0
: "${LORA_SOURCE_ROOT:?Set isolated source root}"
: "${LORA_STUDY_ROOT:?Set isolated output root}"
source /share/home/dengkn/miniconda3/etc/profile.d/conda.sh
conda activate sdpo
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module load gcc/9.1.0
export CC="$(command -v gcc)" CXX="$(command -v g++)"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export https_proxy=http://192.168.10.22:3128
export PYTHONUNBUFFERED=1 VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CODE_REWARD_MAX_CONCURRENCY=2 HYDRA_FULL_ERROR=1 WANDB_MODE=offline
export PYTHONPATH="$LORA_SOURCE_ROOT"
cd "$LORA_SOURCE_ROOT"
exec python -m experiments.continual.lora_study --study "$LORA_STUDY_ROOT" "$@"
