#!/usr/bin/env bash

#SBATCH --job-name=tail-causal
#SBATCH --nodes=1
#SBATCH --partition=gpu_chen
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=460000
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output=logs/tail-causal-%j.out
#SBATCH --error=logs/tail-causal-%j.err

# One-seed Math -> Science causal experiment from PLAN.md.
# Usage:
#   bash experiments/causal/run.sh <prepare|math|math-sdpo|surgery|eval-math|science|eval-science|summary|all> [--dry-run]

set -euo pipefail

STAGE="${1:-}"
DRY_RUN=false
if [[ "${2:-}" == "--dry-run" ]]; then DRY_RUN=true; fi
if [[ -z "$STAGE" || $# -gt 2 || ( $# -eq 2 && "${2:-}" != "--dry-run" ) ]]; then
    echo "Usage: $0 <prepare|math|math-sdpo|surgery|eval-math|science|eval-science|summary|all> [--dry-run]" >&2
    exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$script_dir/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

SDPO_CONDA_ROOT="${SDPO_CONDA_ROOT:-$HOME/miniconda3}"
if [[ ! -f "$SDPO_CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Cannot find Conda initialization below $SDPO_CONDA_ROOT" >&2
    exit 2
fi
source "$SDPO_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate sdpo

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/causal_tail_seed1/${SLURM_JOB_ID:-manual}}"
BASE_MODEL="${BASE_MODEL:-../model/Qwen3-4B-Instruct-2507}"
SEED="${SEED:-42}"
TRAIN_SAMPLE_LIMIT="${TRAIN_SAMPLE_LIMIT:-3000}"
if [[ ! "$TRAIN_SAMPLE_LIMIT" =~ ^[1-9][0-9]*$ ]] || (( TRAIN_SAMPLE_LIMIT > 3000 )); then
    echo "TRAIN_SAMPLE_LIMIT must be an integer in [1, 3000]; got $TRAIN_SAMPLE_LIMIT" >&2
    exit 2
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
VAL_ROLLOUT_BATCH_SIZE="${VAL_ROLLOUT_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TAIL_DEVICE="${TAIL_DEVICE:-cuda:0}"
SCIENCE_EVAL_FREQ="${SCIENCE_EVAL_FREQ:-$(( (TRAIN_SAMPLE_LIMIT + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE / 4 ))}"
(( SCIENCE_EVAL_FREQ > 0 )) || SCIENCE_EVAL_FREQ=1

DATA_DIR="$OUTPUT_ROOT/data"
DATA_MANIFEST="$DATA_DIR/dataset_manifest.json"
MATH_HF_ROOT="$OUTPUT_ROOT/math_hf"
TAIL_ROOT="$OUTPUT_ROOT/tail_arms"
SCIENCE_HF_ROOT="$OUTPUT_ROOT/science_hf"
EVAL_ROOT="$OUTPUT_ROOT/evaluation"
SUMMARY_ROOT="$OUTPUT_ROOT/summary"
CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints"
HF_CACHE_DIR="$OUTPUT_ROOT/hf_cache"
mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT" "$MATH_HF_ROOT" "$SCIENCE_HF_ROOT" "$EVAL_ROOT" "$HF_CACHE_DIR"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONBUFFERED=1
# Bound scoring fan-out and native threads even when sbatch inherits larger values.
export CODE_REWARD_MAX_CONCURRENCY=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-SDPO}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT/wandb}"
mkdir -p "$WANDB_DIR"

if command -v module >/dev/null 2>&1; then module load gcc/9.1.0; fi
if command -v gcc >/dev/null 2>&1; then export CC="$(command -v gcc)"; fi
if command -v g++ >/dev/null 2>&1; then export CXX="$(command -v g++)"; fi
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

run_command() {
    if [[ "$DRY_RUN" == true ]]; then
        printf 'DRY-RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

hydra_list() {
    local result="[" separator="" item
    for item in "$@"; do result+="${separator}'${item}'"; separator=","; done
    result+="]"
    printf '%s' "$result"
}

find_latest_actor_checkpoint() {
    find "$1" -mindepth 2 -maxdepth 2 -type d -name actor -print | sort -V | tail -n 1
}

prepare_data() {
    local args=(python3 experiments/causal/prepare_manifest.py --project-root "$PROJECT_ROOT" --output-dir "$DATA_DIR")
    if [[ "$DRY_RUN" == true ]]; then args+=(--dry-run); fi
    run_command "${args[@]}"
}

load_data() {
    if [[ ! -f "$DATA_MANIFEST" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            MATH_TRAIN_FILES=("$PROJECT_ROOT/datasets/cl/math/DAPO-Math-17k/data/dapo-math-17k.parquet")
            SCIENCE_TRAIN_FILES=(
                "$PROJECT_ROOT/datasets/sciknoweval/biology/train.parquet"
                "$PROJECT_ROOT/datasets/sciknoweval/chemistry/train.parquet"
                "$PROJECT_ROOT/datasets/sciknoweval/material/train.parquet"
                "$PROJECT_ROOT/datasets/sciknoweval/physics/train.parquet"
            )
            EVAL_FILES=(
                "$DATA_DIR/normalized/aime24.parquet"
                "$DATA_DIR/normalized/aime25.parquet"
                "$DATA_DIR/normalized/math500.parquet"
                "$PROJECT_ROOT/datasets/sciknoweval/biology/test.parquet"
                "$PROJECT_ROOT/datasets/sciknoweval/chemistry/test.parquet"
                "$PROJECT_ROOT/datasets/sciknoweval/material/test.parquet"
                "$PROJECT_ROOT/datasets/sciknoweval/physics/test.parquet"
                "$DATA_DIR/normalized/gpqa_diamond.parquet"
                "$PROJECT_ROOT/datasets/tooluse/test.parquet"
                "$DATA_DIR/normalized/livecodebench-v6.parquet"
            )
            return
        fi
        echo "Missing $DATA_MANIFEST; run the prepare stage first." >&2
        exit 1
    fi
    mapfile -t MATH_TRAIN_FILES < <(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))["math_train"], sep="\n")' "$DATA_MANIFEST")
    mapfile -t SCIENCE_TRAIN_FILES < <(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))["science_train"], sep="\n")' "$DATA_MANIFEST")
    mapfile -t EVAL_FILES < <(
        python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["evaluation"]; print(*(x for v in d.values() for x in (v if isinstance(v,list) else [v])), sep="\n")' \
            "$DATA_MANIFEST"
    )
}

train_model() {
    local method="$1" task="$2" input_model="$3" run_name="$4" output_hf="$5"
    shift 5
    local train_files=("$@") config_name checkpoint_dir actor_checkpoint train_override val_override shuffle
    local val_before test_freq validation_dir
    [[ "$method" == "grpo" ]] && config_name="baseline_grpo" || config_name="sdpo"
    if [[ "$task" == "math" ]]; then
        shuffle=false
        val_before=False
        test_freq=1000000000
    else
        shuffle=true
        val_before=True
        test_freq="$SCIENCE_EVAL_FREQ"
    fi
    checkpoint_dir="$CHECKPOINT_ROOT/$run_name"
    validation_dir="$EVAL_ROOT/during_$task/$run_name"
    export EXPERIMENT="$run_name"
    export TASK="$task"
    train_override="$(hydra_list "${train_files[@]}")"
    val_override="$(hydra_list "${EVAL_FILES[@]}")"
    if [[ "$DRY_RUN" != true && ( -e "$output_hf" || -e "$checkpoint_dir" ) ]]; then
        echo "Refusing to overwrite $output_hf or $checkpoint_dir" >&2
        exit 1
    fi
    local cmd=(
        python3 -m verl.trainer.main_ppo --config-name "$config_name"
        "ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-24}"
        "data.train_files=$train_override" "data.val_files=$val_override"
        "data.train_max_samples=$TRAIN_SAMPLE_LIMIT" "data.train_batch_size=$TRAIN_BATCH_SIZE"
        "data.shuffle=$shuffle" "data.seed=$SEED" "data.val_batch_size=$VAL_BATCH_SIZE"
        "actor_rollout_ref.model.path=$input_model"
        "actor_rollout_ref.actor.data_loader_seed=$SEED"
        "actor_rollout_ref.actor.fsdp_config.seed=$SEED"
        "actor_rollout_ref.ref.fsdp_config.seed=$SEED"
        "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE"
        "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
        "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_PARALLEL_SIZE"
        "actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_BATCH_SIZE"
        "algorithm.rollout_correction.rollout_is=token"
        "trainer.project_name=$WANDB_PROJECT" "trainer.group_name=tail-causal-seed1"
        "trainer.experiment_name=$run_name" "trainer.logger=['console','wandb']"
        "trainer.default_local_dir=$checkpoint_dir" "trainer.resume_mode=disable"
        "trainer.total_epochs=$TOTAL_EPOCHS" "trainer.val_before_train=$val_before"
        "trainer.test_freq=$test_freq" "trainer.validation_data_dir=$validation_dir"
        "trainer.validation_dump_generations=False"
        "trainer.save_freq=1000000000"
        "trainer.max_actor_ckpt_to_keep=1" "trainer.n_gpus_per_node=4" "trainer.nnodes=1"
        "actor_rollout_ref.actor.checkpoint.save_contents=['model','extra']"
        "actor_rollout_ref.actor.checkpoint.async_save=False"
        "actor_rollout_ref.actor.checkpoint.load_contents=['model','extra']"
        "custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"
    )
    if [[ "$method" == "sdpo" ]]; then
        cmd+=("actor_rollout_ref.actor.self_distillation.teacher_path=$input_model" "actor_rollout_ref.actor.self_distillation.teacher_init_alpha=1.0")
    fi
    run_command "${cmd[@]}"
    if [[ "$DRY_RUN" == true ]]; then
        actor_checkpoint="$checkpoint_dir/<latest-global-step>/actor"
    else
        actor_checkpoint="$(find_latest_actor_checkpoint "$checkpoint_dir")"
        [[ -n "$actor_checkpoint" ]] || { echo "No actor checkpoint below $checkpoint_dir" >&2; exit 1; }
    fi
    run_command python3 -m verl.model_merger merge --backend fsdp --local_dir "$actor_checkpoint" --target_dir "$output_hf"
}

run_math() {
    load_data
    train_model grpo math "$BASE_MODEL" "tail-seed1-GRPO-Math" "$MATH_HF_ROOT/grpo_full" "${MATH_TRAIN_FILES[@]}"
    train_model sdpo math "$BASE_MODEL" "tail-seed1-SDPO-Math" "$MATH_HF_ROOT/sdpo_full" "${MATH_TRAIN_FILES[@]}"
}

run_math_sdpo() {
    local run_name="tail-seed1-SDPO-Math"
    local checkpoint_dir="$CHECKPOINT_ROOT/$run_name"
    local validation_dir="$EVAL_ROOT/during_math/$run_name"
    local output_hf="$MATH_HF_ROOT/sdpo_full"
    local archive_root="$OUTPUT_ROOT/incomplete_attempts/${run_name}-${SLURM_JOB_ID:-manual}"
    load_data
    if [[ "$DRY_RUN" != true ]]; then
        [[ ! -e "$output_hf" ]] || { echo "Refusing to overwrite $output_hf" >&2; exit 1; }
        if [[ -e "$checkpoint_dir" || -e "$validation_dir" ]]; then
            [[ ! -e "$archive_root" ]] || { echo "Archive already exists: $archive_root" >&2; exit 1; }
            mkdir -p "$archive_root"
            [[ ! -e "$checkpoint_dir" ]] || mv "$checkpoint_dir" "$archive_root/checkpoint"
            [[ ! -e "$validation_dir" ]] || mv "$validation_dir" "$archive_root/validation"
            echo "Archived the previous SDPO-Math attempt under $archive_root"
        fi
    fi
    train_model sdpo math "$BASE_MODEL" "$run_name" "$output_hf" "${MATH_TRAIN_FILES[@]}"
}

run_surgery() {
    run_command python3 experiments/causal/tail_surgery.py \
        --base-model "$BASE_MODEL" --grpo-model "$MATH_HF_ROOT/grpo_full" \
        --sdpo-model "$MATH_HF_ROOT/sdpo_full" --output-dir "$TAIL_ROOT" \
        --rank-fraction 0.1 --device "$TAIL_DEVICE" --seed "$SEED" --cache-dir "$HF_CACHE_DIR"
}

arm_path() {
    local arm="$1"
    if [[ "$DRY_RUN" == true && ! -f "$TAIL_ROOT/manifest.json" ]]; then
        case "$arm" in
            GRPO-full) echo "$MATH_HF_ROOT/grpo_full" ;;
            SDPO-full) echo "$MATH_HF_ROOT/sdpo_full" ;;
            *) echo "$TAIL_ROOT/$(echo "$arm" | tr '[:upper:] +' '[:lower:]__')" ;;
        esac
    else
        python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["arms"][sys.argv[2]])' "$TAIL_ROOT/manifest.json" "$arm"
    fi
}

evaluate_model() {
    local model="$1" arm="$2" phase="$3" slug val_override train_override
    slug="$(echo "$arm" | tr '[:upper:] +' '[:lower:]__')"
    export EXPERIMENT="eval-$phase-$slug"
    export TASK="evaluation"
    val_override="$(hydra_list "${EVAL_FILES[@]}")"
    train_override="$(hydra_list "${MATH_TRAIN_FILES[@]}")"
    run_command python3 -m verl.trainer.main_ppo --config-name baseline_grpo \
        "ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-24}" \
        "data.train_files=$train_override" "data.val_files=$val_override" \
        "data.train_max_samples=$TRAIN_SAMPLE_LIMIT" "data.val_batch_size=$VAL_BATCH_SIZE" \
        "actor_rollout_ref.model.path=$model" \
        "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_PARALLEL_SIZE" \
        "actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_BATCH_SIZE" \
        "trainer.project_name=$WANDB_PROJECT" "trainer.experiment_name=eval-$phase-$slug" \
        "trainer.logger=['console']" "trainer.val_before_train=True" "trainer.val_only=True" \
        "trainer.validation_data_dir=$EVAL_ROOT/$phase/$arm" \
        "trainer.validation_dump_generations=False" \
        "trainer.n_gpus_per_node=4" "trainer.nnodes=1" \
        "custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"
}

run_eval_math() {
    load_data
    local arms=(
        "GRPO-full" "GRPO-no-tail" "GRPO-random-tail"
        "SDPO-full" "SDPO-no-tail" "SDPO-random-tail"
        "GRPO-principal+SDPO-tail" "SDPO-principal+GRPO-tail"
    ) arm
    for arm in "${arms[@]}"; do evaluate_model "$(arm_path "$arm")" "$arm" after_math; done
}

run_science() {
    load_data
    local arm method input output
    for arm in "GRPO-full" "GRPO-no-tail" "SDPO-full" "SDPO-no-tail"; do
        [[ "$arm" == GRPO-* ]] && method=grpo || method=sdpo
        input="$(arm_path "$arm")"
        output="$SCIENCE_HF_ROOT/$arm"
        train_model "$method" science "$input" "tail-seed1-$arm-Science" "$output" "${SCIENCE_TRAIN_FILES[@]}"
    done
}

run_eval_science() {
    load_data
    local arm
    for arm in "GRPO-full" "GRPO-no-tail" "SDPO-full" "SDPO-no-tail"; do
        evaluate_model "$SCIENCE_HF_ROOT/$arm" "$arm" after_science
    done
}

run_summary() {
    run_command python3 experiments/causal/summarize_results.py --evaluation-root "$EVAL_ROOT" --output-dir "$SUMMARY_ROOT"
}

if [[ "$DRY_RUN" != true && -z "${SLURM_JOB_ID:-}" && "$STAGE" != prepare && "$STAGE" != summary ]]; then
    echo "GPU stages must run inside Slurm; submit this script with sbatch." >&2
    exit 2
fi

case "$STAGE" in
    prepare) prepare_data ;;
    math) run_math ;;
    math-sdpo) run_math_sdpo ;;
    surgery) run_surgery ;;
    eval-math) run_eval_math ;;
    science) run_science ;;
    eval-science) run_eval_science ;;
    summary) run_summary ;;
    all)
        prepare_data
        run_math
        run_surgery
        run_eval_math
        run_science
        run_eval_science
        run_summary
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
