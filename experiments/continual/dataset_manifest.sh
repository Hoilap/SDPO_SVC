#!/usr/bin/env bash

# Continual-training domains.  Only these datasets contribute optimizer steps.
CL_TASK_NAMES=(
    "math"
    "science"
    "tool"
    "code"
)

CL_TRAIN_DATASETS=(
    "datasets/cl/math/DAPO-Math-17k"
    "datasets/sciknoweval"
    "datasets/tooluse"
    "datasets/cl/code/Dolci-Think-RL-7B"
)

# Optional path or glob, relative to CL_TRAIN_DATASETS, for train-only datasets
# whose parquet is not named train.parquet. Empty entries use split discovery.
CL_TRAIN_FILE_HINTS=(
    "data/dapo-math-17k.parquet"
    ""
    ""
    "data/train-*.parquet"
)

# Per-task loader policy. The distributed DAPO parquet consists of 100
# consecutive copies of one 17,917-row source block, so math consumes only the
# first complete block. The other datasets are loaded in full and shuffled.
CL_TRAIN_MAX_SAMPLES=(
    "17917"
    "-1"
    "-1"
    "-1"
)

CL_TRAIN_SHUFFLE=(
    "false"
    "true"
    "true"
    "true"
)

# These repository datasets are already split.  The runner must consume their
# existing train.parquet/test.parquet files and must never regenerate splits.
CL_PREPARTITIONED_DATASETS=(
    "datasets/sciknoweval"
    "datasets/tooluse"
)

# External evaluation benchmarks introduced at each task boundary.  Entries in
# one group are separated by '|'.  Evaluation is cumulative: after the science
# task, for example, all three math benchmarks and GPQA are run.
#
# These datasets must never be included in data.train_files:
#   code:    LiveCodeBench-v6
#   math in-domain: AIME24, AIME25
#   math additional generalization: MATH-500
#   science: GPQA Diamond (only; do not concatenate the four CSV variants)
#   tool:    no extra benchmark; use datasets/tooluse/test.parquet
CL_EXTERNAL_EVAL_GROUPS=(
    "datasets/cl/math/aime24|datasets/cl/math/aime25/test.jsonl|datasets/cl/math/math500"
    "datasets/cl/science/gpqa/gpqa_diamond.csv"
    ""
    "datasets/cl/code/LiveCodeBench-v6"
)

if (( ${#CL_TASK_NAMES[@]} != ${#CL_TRAIN_DATASETS[@]} \
      || ${#CL_TASK_NAMES[@]} != ${#CL_TRAIN_FILE_HINTS[@]} \
      || ${#CL_TASK_NAMES[@]} != ${#CL_TRAIN_MAX_SAMPLES[@]} \
      || ${#CL_TASK_NAMES[@]} != ${#CL_TRAIN_SHUFFLE[@]} \
      || ${#CL_TASK_NAMES[@]} != ${#CL_EXTERNAL_EVAL_GROUPS[@]} )); then
    echo "Invalid continual dataset manifest: array lengths differ." >&2
    return 1 2>/dev/null || exit 1
fi
