#!/usr/bin/env bash
#SBATCH --job-name=code-from-114597
#SBATCH --partition=gpu_chen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mem=460000M
#SBATCH --time=0
#SBATCH --output=logs/code-from-114597-%j.out
#SBATCH --error=logs/code-from-114597-%j.err
# Resume only Code from 114597's completed Tool-SVC boundary. Submit from the
# repository root; outputs are isolated by the new job ID. Resource samples
# live in /tmp/code-from-114597-<jobid>/monitor/resources.jsonl. A bounded
# recent window is mirrored to outputs/code_from_114597/<jobid>/diagnostics/
# every minute; SIGKILL or a failed shared filesystem may leave the complete
# history only in the node-local copy.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PROJECT_ROOT="$PWD"
source /share/home/dengkn/miniconda3/etc/profile.d/conda.sh
conda activate sdpo
export START_TASK=3 END_TASK=3 RUN_PROFILE=try CONFIG_NAME=sdpo
export INITIAL_CONTINUAL_MODEL="$PWD/outputs/sdpo_svc_cl_try/114597/hf_models/03-tool-svc"
export OUTPUT_ROOT="$PWD/outputs/code_from_114597/$SLURM_JOB_ID"
export CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints" HF_ROOT="$OUTPUT_ROOT/hf_models"
export EVAL_RESULT_ROOT="$OUTPUT_ROOT/evaluation" HF_CACHE_DIR="$OUTPUT_ROOT/hf_cache"
export WANDB_DIR="$OUTPUT_ROOT/wandb"
export TEST_FREQ=1000000000 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
export N_GPUS_PER_NODE=4 NNODES=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export RAY_TMPDIR="/tmp/code-from-114597-$SLURM_JOB_ID"
monitor_dir="$RAY_TMPDIR/monitor"
archive="$OUTPUT_ROOT/diagnostics"
archive_lines="${MONITOR_ARCHIVE_LINES:-1000}"
if [[ ! "$archive_lines" =~ ^[1-9][0-9]*$ ]]; then
    echo "MONITOR_ARCHIVE_LINES must be a positive integer" >&2
    exit 2
fi
mkdir -p "$monitor_dir" "$archive"
python - "$INITIAL_CONTINUAL_MODEL" <<'PY'
import json
import sys
from pathlib import Path
import yaml
p = Path(sys.argv[1])
json.loads((p / 'config.json').read_text())
shards = set(json.loads((p / 'model.safetensors.index.json').read_text())['weight_map'].values())
assert shards and all((p / name).is_file() and (p / name).stat().st_size > 0 for name in shards)
assert (p / 'tokenizer.json').is_file()
assert yaml.safe_load(Path('verl/trainer/config/sdpo.yaml').read_text())['trainer']['val_before_train'] is False
assert yaml.safe_load(Path('verl/trainer/config/actor/actor.yaml').read_text())['checkpoint']['async_save'] is False
source = Path('verl/trainer/ppo/ray_trainer.py').read_text()
start = source.index('if self.config.trainer.save_freq > 0')
assert source.index('self._save_checkpoint()', start) < source.index('val_metrics: dict = self._validate()', start)
print('PREFLIGHT: Tool-SVC shards present; synchronous checkpoint precedes final validation', flush=True)
PY
git rev-parse HEAD
hostname
python experiments/continual/monitor_resources.py --interval 5 --output "$monitor_dir/resources.jsonl" &
monitor_pid=$!
# Archive a bounded recent window independently; a slow shared filesystem must
# not block the sampler, and the shared filesystem must not receive the entire
# growing history again every minute.
archive_resources() {
    if [[ -f "$monitor_dir/resources.jsonl" ]]; then
        timeout 15 tail -n "$archive_lines" "$monitor_dir/resources.jsonl" \
            > "$archive/resources.recent.jsonl.tmp" \
            && mv "$archive/resources.recent.jsonl.tmp" "$archive/resources.recent.jsonl" \
            || true
    fi
}
(
    while true; do
        archive_resources
        sleep 60
    done
) &
archive_pid=$!
train_pid=""
cleanup() {
    status=$?
    trap - EXIT TERM INT
    if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
        kill -TERM "$train_pid" 2>/dev/null || true
    fi
    kill "$archive_pid" "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    archive_resources
    for session in "$RAY_TMPDIR"/ray/session_*; do
        [[ -d "$session/logs" && ! -L "$session" ]] || continue
        timeout 60 cp -r "$session/logs" "$archive/$(basename "$session")-logs" || true
    done
    echo "DIAGNOSTICS: $archive ; node-local fallback: $monitor_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash experiments/continual/try.sh &
train_pid=$!
wait "$train_pid"
