#!/usr/bin/env bash

#SBATCH --job-name=sdpo-svc-cl
#SBATCH --nodes=1
#SBATCH --partition=gpu_chen
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=460000
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/sdpo-svc-cl-%j.out
#SBATCH --error=logs/sdpo-svc-cl-%j.err

# Sequential SDPO + task-boundary Singular Value Calibration (SVC).
#
# Run this script directly on an allocated compute node.  It executes all task
# boundaries sequentially in the current 4-GPU allocation:
#   bash experiments/continual/run_sdpo_svc_cl.sh
#
# Useful overrides:
#   BASE_MODEL=../model/Qwen3-4B-Instruct TOTAL_EPOCHS=1 \
#   SVC_DEVICE=cuda:0 sbatch experiments/continual/run_sdpo_svc_cl.sh
#
# Use --dry-run to validate the task order and print commands without training.

set -euo pipefail

# sbatch executes a copied script from /var/spool/slurmd/job*/slurm_script, so
# BASH_SOURCE does not point into the repository inside a batch job.  Slurm's
# submission directory is the project root for the documented launch command.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
    SCRIPT_DIR="$PROJECT_ROOT/experiments/continual"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
if [[ ! -f "$SCRIPT_DIR/dataset_manifest.sh" ]]; then
    echo "Cannot find dataset manifest: $SCRIPT_DIR/dataset_manifest.sh" >&2
    echo "Submit this script from the repository root, or export PROJECT_ROOT explicitly." >&2
    exit 2
fi
source "$SCRIPT_DIR/dataset_manifest.sh"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONBUFFERED=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export USER="${USER:-$(whoami)}"
export WANDB_API_KEY="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"
export WANDB_ENTITY="20040817dkn-facebook"
export WANDB_PROJECT="SDPO"
export WANDB_MODE="${WANDB_MODE:-online}"
ulimit -c 0

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

# Training configuration.  BASE_MODEL remains the common SVC anchor throughout
# the curriculum; CURRENT_MODEL advances after every calibrated task boundary.
BASE_MODEL="${BASE_MODEL:-../model/Qwen3-4B-Instruct-2507}"
CURRENT_MODEL="${INITIAL_CONTINUAL_MODEL:-$BASE_MODEL}"
CONFIG_NAME="${CONFIG_NAME:-sdpo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/sdpo_svc_cl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"
HF_ROOT="${HF_ROOT:-$OUTPUT_ROOT/hf_models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$OUTPUT_ROOT/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT/wandb}"

START_TASK="${START_TASK:-0}"
END_TASK="${END_TASK:-$((${#CL_TRAIN_DATASETS[@]} - 1))}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
# This experiment is intentionally fixed to one 4-GPU Slurm node.
N_GPUS_PER_NODE=4
NNODES=1
TEST_FREQ="${TEST_FREQ:-5}"
DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-0.5}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
AUTO_PREPROCESS="${AUTO_PREPROCESS:-1}"

# SVC defaults are conservative: top-64 directions, suppression only, and a
# half-strength interpolation between the raw and fully calibrated spectra.
SVC_RANK="${SVC_RANK:-64}"
SVC_OVERSAMPLE="${SVC_OVERSAMPLE:-8}"
SVC_NITER="${SVC_NITER:-2}"
SVC_ALPHA="${SVC_ALPHA:-1.0}"
SVC_STRENGTH="${SVC_STRENGTH:-0.5}"
SVC_DEVICE="${SVC_DEVICE:-cpu}"
SVC_SEED="${SVC_SEED:-0}"

if (( START_TASK < 0 || END_TASK >= ${#CL_TRAIN_DATASETS[@]} || START_TASK > END_TASK )); then
    echo "Invalid task range START_TASK=$START_TASK END_TASK=$END_TASK" >&2
    exit 2
fi
if (( START_TASK > 0 )) && [[ -z "${INITIAL_CONTINUAL_MODEL:-}" ]]; then
    echo "START_TASK > 0 requires INITIAL_CONTINUAL_MODEL=<previous calibrated HF checkpoint>." >&2
    exit 2
fi
if [[ "$DRY_RUN" != true && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted through Slurm." >&2
    echo "Run: sbatch experiments/continual/run_sdpo_svc_cl.sh" >&2
    exit 2
fi
if [[ "$DRY_RUN" != true && -n "${SLURM_GPUS_ON_NODE:-}" \
      && "${SLURM_GPUS_ON_NODE}" =~ ([0-9]+)$ \
      && "${BASH_REMATCH[1]}" -ne 4 ]]; then
    echo "This experiment requires exactly 4 GPUs, but Slurm reports SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE." >&2
    exit 2
fi
if [[ "$DRY_RUN" != true && "$WANDB_MODE" == "disabled" ]]; then
    echo "WANDB_MODE=disabled is not allowed for this experiment." >&2
    exit 2
fi

mkdir -p "$CHECKPOINT_ROOT" "$HF_ROOT" "$HF_CACHE_DIR" "$WANDB_DIR"

run_command() {
    if [[ "$DRY_RUN" == true ]]; then
        printf 'DRY-RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# Appends evaluation parquet files to EXTERNAL_EVAL_FILES.  External benchmarks
# are never used as train files.  Parquet, JSON, and JSONL eval files are all
# accepted directly by RLHFDataset.
resolve_external_eval_files() {
    local eval_path="$1"
    local eval_file candidate diamond_parquet initial_count
    initial_count="${#EXTERNAL_EVAL_FILES[@]}"

    if [[ -f "$eval_path" ]]; then
        case "$eval_path" in
            *.parquet|*.json|*.jsonl)
                EXTERNAL_EVAL_FILES+=("$(realpath "$eval_path")")
                return
                ;;
            */gpqa_diamond.csv)
                diamond_parquet="${eval_path%.csv}.parquet"
                if [[ ! -f "$diamond_parquet" ]]; then
                    echo "Preprocessing GPQA Diamond evaluation set: $eval_path"
                    run_command python3 "$PROJECT_ROOT/data/preprocess_gpqa.py" \
                        --csv-file "$eval_path" \
                        --output-file "$diamond_parquet"
                fi
                if [[ "$DRY_RUN" == true || -f "$diamond_parquet" ]]; then
                    EXTERNAL_EVAL_FILES+=("$(realpath -m "$diamond_parquet")")
                    return
                fi
                ;;
        esac
    fi

    if [[ "$DRY_RUN" == true && ! -e "$eval_path" ]]; then
        if [[ "$eval_path" == */gpqa_diamond.csv ]]; then
            EXTERNAL_EVAL_FILES+=("$(realpath -m "${eval_path%.csv}.parquet")")
        elif [[ "$eval_path" == *.parquet || "$eval_path" == *.json || "$eval_path" == *.jsonl ]]; then
            EXTERNAL_EVAL_FILES+=("$(realpath -m "$eval_path")")
        else
            EXTERNAL_EVAL_FILES+=("$(realpath -m "$eval_path/test.parquet")")
        fi
        return
    fi
    if [[ ! -d "$eval_path" ]]; then
        echo "External evaluation path does not exist: $eval_path" >&2
        exit 1
    fi

    # Prefer explicit eval/validation/test files and deliberately ignore train
    # splits even if a benchmark directory happens to contain one.
    for candidate in \
        "$eval_path/eval.parquet" \
        "$eval_path/validation.parquet" \
        "$eval_path/test.parquet" \
        "$eval_path/eval.jsonl" \
        "$eval_path/validation.jsonl" \
        "$eval_path/test.jsonl"; do
        if [[ -f "$candidate" ]]; then
            EXTERNAL_EVAL_FILES+=("$(realpath "$candidate")")
            return
        fi
    done
    while IFS= read -r -d '' eval_file; do
        EXTERNAL_EVAL_FILES+=("$(realpath "$eval_file")")
    done < <(find "$eval_path" -maxdepth 2 -type f \
        \( -iname '*eval*.parquet' -o -iname '*validation*.parquet' -o -iname '*test*.parquet' \
        -o -iname '*eval*.jsonl' -o -iname '*validation*.jsonl' -o -iname '*test*.jsonl' \) \
        -print0 | sort -z)

    if (( ${#EXTERNAL_EVAL_FILES[@]} == initial_count )); then
        echo "No eval/validation/test Parquet or JSONL file found under $eval_path" >&2
        exit 1
    fi
}

# Populates the global TRAIN_FILES and VAL_FILES arrays.  A dataset may be a
# normal train/test directory or an aggregate directory such as sciknoweval,
# whose immediate children each contain a split.
resolve_dataset_files() {
    local dataset_dir="$1"
    local train_file_hint="${2:-}"
    local prepartitioned=false prepartitioned_root hinted_train_file hinted_pattern
    local hint_found=false hint_count=0
    TRAIN_FILES=()
    VAL_FILES=()
    MISSING_SPLITS=()

    if [[ -n "$train_file_hint" ]]; then
        hinted_pattern="$dataset_dir/$train_file_hint"
        while IFS= read -r hinted_train_file; do
            [[ -z "$hinted_train_file" ]] && continue
            TRAIN_FILES+=("$(realpath "$hinted_train_file")")
            hint_found=true
            hint_count=$((hint_count + 1))
        done < <(compgen -G "$hinted_pattern" | sort)

        if [[ "$hint_found" == true ]]; then
            if [[ "$train_file_hint" == "data/train-*.parquet" && "$hint_count" -ne 9 ]]; then
                echo "Expected 9 Dolci training shards, found $hint_count matching $hinted_pattern" >&2
                exit 1
            fi
        elif [[ "$DRY_RUN" == true ]]; then
            TRAIN_FILES=("$(realpath -m "$hinted_pattern")")
        else
            echo "Configured training parquet pattern has no matches: $hinted_pattern" >&2
            exit 1
        fi
        # Train-only datasets obtain validation from CL_EXTERNAL_EVAL_GROUPS.
        return
    fi

    for prepartitioned_root in "${CL_PREPARTITIONED_DATASETS[@]}"; do
        if [[ "$dataset_dir" == "$prepartitioned_root" ]]; then
            prepartitioned=true
            break
        fi
    done

    if [[ "$DRY_RUN" == true && ! -d "$dataset_dir" ]]; then
        TRAIN_FILES=("$PROJECT_ROOT/$dataset_dir/train.parquet")
        VAL_FILES=("$PROJECT_ROOT/$dataset_dir/test.parquet")
        return
    fi
    if [[ ! -d "$dataset_dir" ]]; then
        echo "Dataset directory does not exist: $dataset_dir" >&2
        exit 1
    fi

    local candidates=("$dataset_dir")
    local child
    while IFS= read -r -d '' child; do
        if [[ -f "$child/train.parquet" || -f "$child/train.json" ]]; then
            candidates+=("$child")
        fi
    done < <(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    local candidate train_file val_file
    for candidate in "${candidates[@]}"; do
        train_file="$candidate/train.parquet"
        val_file="$candidate/test.parquet"

        # Aggregate roots such as datasets/sciknoweval contain domain
        # subdirectories rather than their own train/test pair.
        if [[ ! -f "$train_file" && ! -f "$val_file" \
              && ! -f "$candidate/train.json" && ! -f "$candidate/test.json" ]]; then
            continue
        fi

        if [[ "$prepartitioned" == true && (! -f "$train_file" || ! -f "$val_file") ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                TRAIN_FILES+=("$(realpath -m "$train_file")")
                VAL_FILES+=("$(realpath -m "$val_file")")
            else
                MISSING_SPLITS+=("$train_file or $val_file")
            fi
            continue
        fi

        if [[ (! -f "$train_file" || ! -f "$val_file") \
              && "$prepartitioned" != true \
              && "$AUTO_PREPROCESS" == "1" \
              && -f "$candidate/train.json" \
              && -f "$candidate/test.json" ]]; then
            echo "Preprocessing JSON dataset: $candidate"
            run_command python3 "$PROJECT_ROOT/data/preprocess.py" --data_source "$candidate"
            if [[ "$DRY_RUN" == true ]]; then
                TRAIN_FILES+=("$(realpath -m "$train_file")")
                VAL_FILES+=("$(realpath -m "$val_file")")
                continue
            fi
        fi

        if [[ -f "$train_file" && -f "$val_file" ]]; then
            TRAIN_FILES+=("$(realpath "$train_file")")
            VAL_FILES+=("$(realpath "$val_file")")
        fi
    done

    if (( ${#MISSING_SPLITS[@]} > 0 )); then
        echo "Pre-partitioned dataset is missing existing parquet splits:" >&2
        printf '  %s\n' "${MISSING_SPLITS[@]}" >&2
        echo "Automatic JSON preprocessing is disabled for $dataset_dir." >&2
        exit 1
    fi

    # Some prepared datasets use descriptive parquet filenames instead of
    # train.parquet/test.parquet.  Accept those as a fallback.
    if (( ${#TRAIN_FILES[@]} == 0 )); then
        while IFS= read -r -d '' train_file; do TRAIN_FILES+=("$(realpath "$train_file")"); done \
            < <(find "$dataset_dir" -maxdepth 2 -type f -iname '*train*.parquet' -print0 | sort -z)
        while IFS= read -r -d '' val_file; do VAL_FILES+=("$(realpath "$val_file")"); done \
            < <(find "$dataset_dir" -maxdepth 2 -type f \
                \( -iname '*test*.parquet' -o -iname '*val*.parquet' \) -print0 | sort -z)
    fi

    if (( ${#TRAIN_FILES[@]} == 0 || ${#VAL_FILES[@]} == 0 )); then
        echo "Could not find train/test parquet files under $dataset_dir." >&2
        echo "Expected train.parquet and test.parquet, or train.json/test.json with AUTO_PREPROCESS=1." >&2
        exit 1
    fi
}

hydra_list() {
    local result="["
    local item separator=""
    for item in "$@"; do
        result+="${separator}'${item}'"
        separator=","
    done
    result+="]"
    printf '%s' "$result"
}

find_latest_actor_checkpoint() {
    local checkpoint_dir="$1"
    find "$checkpoint_dir" -mindepth 2 -maxdepth 2 -type d -name actor -print \
        | sort -V | tail -n 1
}

echo "============================================================"
echo "Sequential SDPO + SVC"
echo "Base anchor:       $BASE_MODEL"
echo "Starting model:    $CURRENT_MODEL"
echo "Task range:        $START_TASK..$END_TASK"
echo "Output root:       $OUTPUT_ROOT"
echo "SVC:               rank=$SVC_RANK alpha=$SVC_ALPHA strength=$SVC_STRENGTH device=$SVC_DEVICE"
echo "============================================================"

for ((task_index = START_TASK; task_index <= END_TASK; task_index++)); do
    dataset_path="${CL_TRAIN_DATASETS[$task_index]}"
    dataset_name="${CL_TASK_NAMES[$task_index]}"
    task_number="$(printf '%02d' "$((task_index + 1))")"
    experiment_name="SDPO-SVC-CL-${task_number}-${dataset_name}"
    task_checkpoint_dir="$CHECKPOINT_ROOT/$experiment_name"
    raw_hf_dir="$HF_ROOT/${task_number}-${dataset_name}-raw"
    calibrated_hf_dir="$HF_ROOT/${task_number}-${dataset_name}-svc"

    # Same-distribution held-out tests and external benchmarks are cumulative,
    # providing a task-by-task forgetting matrix in W&B.
    current_train_files=()
    cumulative_val_files=()
    for ((eval_task = 0; eval_task <= task_index; eval_task++)); do
        resolve_dataset_files \
            "${CL_TRAIN_DATASETS[$eval_task]}" \
            "${CL_TRAIN_FILE_HINTS[$eval_task]}"
        # A non-empty train-file hint denotes a train-only dataset (DAPO), so
        # there is deliberately no VAL_FILES array to expand.  This explicit
        # guard is required for older Bash versions when `set -u` is active.
        if [[ -z "${CL_TRAIN_FILE_HINTS[$eval_task]}" ]]; then
            cumulative_val_files+=("${VAL_FILES[@]}")
        fi
        if (( eval_task == task_index )); then
            current_train_files=("${TRAIN_FILES[@]}")
        fi
    done
    EXTERNAL_EVAL_FILES=()
    for ((eval_task = 0; eval_task <= task_index; eval_task++)); do
        eval_group_spec="${CL_EXTERNAL_EVAL_GROUPS[$eval_task]}"
        # ToolUse has no additional OOD benchmark in this setup.  Skip its
        # empty group before creating/expanding an array, which is required for
        # older Bash versions under `set -u`.
        [[ -z "$eval_group_spec" ]] && continue
        IFS='|' read -r -a eval_group <<< "$eval_group_spec"
        for eval_path in "${eval_group[@]}"; do
            resolve_external_eval_files "$eval_path"
        done
    done
    cumulative_val_files+=("${EXTERNAL_EVAL_FILES[@]}")
    TRAIN_FILES=("${current_train_files[@]}")
    VAL_FILES=("${cumulative_val_files[@]}")

    train_files_override="$(hydra_list "${TRAIN_FILES[@]}")"
    val_files_override="$(hydra_list "${VAL_FILES[@]}")"

    if [[ "$DRY_RUN" != true && ( -e "$raw_hf_dir" || -e "$calibrated_hf_dir" ) ]]; then
        echo "Refusing to overwrite an existing HF output for task $task_number:" >&2
        echo "  $raw_hf_dir" >&2
        echo "  $calibrated_hf_dir" >&2
        exit 1
    fi

    export EXPERIMENT="$experiment_name"
    export TASK="$dataset_path"

    echo
    echo "[$task_number/${#CL_TRAIN_DATASETS[@]}] Training $dataset_path"
    echo "  input model: $CURRENT_MODEL"
    echo "  train files: ${TRAIN_FILES[*]}"
    echo "  val files:   ${VAL_FILES[*]}"

    train_cmd=(
        python3 -m verl.trainer.main_ppo
        --config-name "$CONFIG_NAME"
        "data.train_files=$train_files_override"
        "data.val_files=$val_files_override"
        "data.train_batch_size=$TRAIN_BATCH_SIZE"
        "actor_rollout_ref.model.path=$CURRENT_MODEL"
        "actor_rollout_ref.actor.self_distillation.teacher_path=$CURRENT_MODEL"
        "actor_rollout_ref.actor.self_distillation.teacher_init_alpha=1.0"
        "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE"
        "actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK"
        "actor_rollout_ref.actor.self_distillation.alpha=$DISTILLATION_ALPHA"
        "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
        "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE"
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
        "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
        "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
        "actor_rollout_ref.rollout.val_kwargs.n=16"
        "algorithm.rollout_correction.rollout_is=token"
        "trainer.project_name=$WANDB_PROJECT"
        "trainer.group_name=SDPO-SVC-CL"
        "trainer.experiment_name=$experiment_name"
        "trainer.logger=['console','wandb']"
        "trainer.default_local_dir=$task_checkpoint_dir"
        "trainer.resume_mode=disable"
        "trainer.total_epochs=$TOTAL_EPOCHS"
        "trainer.save_freq=1000000000"
        "trainer.max_actor_ckpt_to_keep=1"
        "trainer.test_freq=$TEST_FREQ"
        "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
        "trainer.nnodes=$NNODES"
        "actor_rollout_ref.actor.checkpoint.save_contents=['model','extra']"
        "actor_rollout_ref.actor.checkpoint.load_contents=['model','extra']"
        "custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"
    )
    run_command "${train_cmd[@]}"

    if [[ "$DRY_RUN" == true ]]; then
        actor_checkpoint="$task_checkpoint_dir/<latest-global-step>/actor"
    else
        actor_checkpoint="$(find_latest_actor_checkpoint "$task_checkpoint_dir")"
        if [[ -z "$actor_checkpoint" ]]; then
            echo "No actor checkpoint found under $task_checkpoint_dir" >&2
            exit 1
        fi
    fi

    echo "[$task_number/${#CL_TRAIN_DATASETS[@]}] Exporting FSDP checkpoint to Hugging Face format"
    run_command python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_checkpoint" \
        --target_dir "$raw_hf_dir"

    echo "[$task_number/${#CL_TRAIN_DATASETS[@]}] Applying task-boundary SVC"
    run_command python3 -m verl.model_merger.svc \
        --base-model "$BASE_MODEL" \
        --previous-model "$CURRENT_MODEL" \
        --raw-model "$raw_hf_dir" \
        --output-dir "$calibrated_hf_dir" \
        --rank "$SVC_RANK" \
        --oversample "$SVC_OVERSAMPLE" \
        --niter "$SVC_NITER" \
        --alpha "$SVC_ALPHA" \
        --strength "$SVC_STRENGTH" \
        --device "$SVC_DEVICE" \
        --seed "$SVC_SEED" \
        --cache-dir "$HF_CACHE_DIR"

    CURRENT_MODEL="$calibrated_hf_dir"
    echo "[$task_number/${#CL_TRAIN_DATASETS[@]}] Boundary complete: $CURRENT_MODEL"
done

echo
echo "Continual-learning curriculum complete."
echo "Final calibrated model: $CURRENT_MODEL"
