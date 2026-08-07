#!/usr/bin/env bash

# Continual-training domains.  Only these datasets contribute optimizer steps.
CL_TASK_NAMES=(
    "code"
    "math"
    "science"
    "tool"
)

CL_TRAIN_DATASETS=(
    "datasets/cl/code/Dolci-Think-RL-7B"
    "datasets/cl/math/DAPO-Math-17k"
    "datasets/sciknoweval"
    "datasets/tooluse"
)

# External evaluation benchmarks introduced at each task boundary.  Entries in
# one group are separated by '|'.  Evaluation is cumulative: after the math
# task, for example, both LiveCodeBench and all three math benchmarks are run.
#
# These datasets must never be included in data.train_files:
#   code:    LiveCodeBench-v6
#   math:    AIME24, AIME25, MATH-500
#   science: GPQA
#   tool:    BFCL v4 multiple-call
CL_EXTERNAL_EVAL_GROUPS=(
    "datasets/cl/code/LiveCodeBench-v6"
    "datasets/cl/math/aime24|datasets/cl/math/aime25|datasets/cl/math/math500"
    "datasets/cl/science/gpqa"
    "datasets/cl/tool/bfcl_v4/data/BFCL_v4_multiple.json"
)

if (( ${#CL_TASK_NAMES[@]} != ${#CL_TRAIN_DATASETS[@]} \
      || ${#CL_TASK_NAMES[@]} != ${#CL_EXTERNAL_EVAL_GROUPS[@]} )); then
    echo "Invalid continual dataset manifest: array lengths differ." >&2
    return 1 2>/dev/null || exit 1
fi
