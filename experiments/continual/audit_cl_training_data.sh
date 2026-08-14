#!/usr/bin/env bash

# Audit every continual-learning training dataset before allocating GPUs.
# Activate the same Python environment used for training, then run:
#   bash experiments/continual/audit_cl_training_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$SCRIPT_DIR/dataset_manifest.sh"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/sdpo_svc_cl}"
DATA_AUDIT_ROOT="${DATA_AUDIT_ROOT:-$OUTPUT_ROOT/audited_training_data}"
mkdir -p "$DATA_AUDIT_ROOT"
AUDIT_SUMMARIES=()

collect_training_files() {
    local dataset_dir="$1"
    local train_file_hint="$2"
    local hinted_file
    TRAIN_FILES=()

    if [[ -n "$train_file_hint" ]]; then
        while IFS= read -r hinted_file; do
            [[ -n "$hinted_file" ]] && TRAIN_FILES+=("$(realpath "$hinted_file")")
        done < <(compgen -G "$dataset_dir/$train_file_hint" | sort)
    else
        while IFS= read -r -d '' hinted_file; do
            TRAIN_FILES+=("$(realpath "$hinted_file")")
        done < <(find "$dataset_dir" -maxdepth 2 -type f -name train.parquet -print0 | sort -z)
    fi

    if (( ${#TRAIN_FILES[@]} == 0 )); then
        echo "No training parquet found for $dataset_dir" >&2
        exit 1
    fi
}

echo "Auditing ${#CL_TASK_NAMES[@]} continual-training datasets"
echo "Audit cache: $DATA_AUDIT_ROOT"

for ((task_index = 0; task_index < ${#CL_TASK_NAMES[@]}; task_index++)); do
    task_name="${CL_TASK_NAMES[$task_index]}"
    dataset_dir="${CL_TRAIN_DATASETS[$task_index]}"
    train_file_hint="${CL_TRAIN_FILE_HINTS[$task_index]}"
    collect_training_files "$dataset_dir" "$train_file_hint"

    echo
    echo "[$((task_index + 1))/${#CL_TASK_NAMES[@]}] $task_name: ${#TRAIN_FILES[@]} parquet file(s)"
    audit_report="$DATA_AUDIT_ROOT/${task_name}-audit.json"
    python3 "$PROJECT_ROOT/data/audit_training_parquet.py" \
        --check-only \
        --report "$audit_report" \
        "${TRAIN_FILES[@]}"

    audit_summary="$(
        python3 -c 'import json, sys; r=json.load(open(sys.argv[1])); print("{}|{}|{}|{:.2%}".format(r["total_rows"], r["unique_rows"], r["duplicate_rows"], r["duplicate_fraction"]))' "$audit_report"
    )"
    AUDIT_SUMMARIES+=("$task_name|$audit_summary")
done

echo
echo "Continual-training data audit summary"
printf '%-10s %12s %12s %12s %10s\n' "task" "total" "unique" "duplicates" "ratio"
printf '%-10s %12s %12s %12s %10s\n' "----------" "------------" "------------" "------------" "----------"
for audit_summary in "${AUDIT_SUMMARIES[@]}"; do
    IFS='|' read -r task_name total_rows unique_rows duplicate_rows duplicate_fraction <<< "$audit_summary"
    printf '%-10s %12s %12s %12s %10s\n' \
        "$task_name" "$total_rows" "$unique_rows" "$duplicate_rows" "$duplicate_fraction"
done

echo
echo "Audit complete. No training data or loader configuration was changed."
echo "Reports: $DATA_AUDIT_ROOT/*-audit.json"
