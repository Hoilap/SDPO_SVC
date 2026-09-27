#!/usr/bin/env bash
#SBATCH --job-name=causal-merge-hf
#SBATCH --partition=gpu_chen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/causal-merge-hf-%j.out
#SBATCH --error=logs/causal-merge-hf-%j.err

# Submit from the repository root:
# sbatch experiments/causal/merge_saved_checkpoint.sh /path/global_step_93/actor /path/math_hf/sdpo_full
# Does not train, evaluate, or change downstream job dependencies.
set -euo pipefail
if [[ $# -ne 2 || "$1" != /* || "$2" != /* ]]; then
    echo "Usage: $0 /absolute/checkpoint/actor /absolute/output_hf" >&2
    exit 2
fi
actor_checkpoint="$1"
output_hf="${2%/}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
else
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
fi
source "${SDPO_CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate sdpo
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$output_hf")"
exec 9>"${output_hf}.merge.lock"
flock -n 9 || { echo "Another merge holds ${output_hf}.merge.lock" >&2; exit 1; }
[[ ! -e "$output_hf" && ! -L "$output_hf" ]] || {
    echo "Refusing to overwrite $output_hf" >&2; exit 1;
}
python - "$actor_checkpoint" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
config = json.loads((p / 'fsdp_config.json').read_text())
world_size = int(config['world_size'])
for rank in range(world_size):
    shard = p / f'model_world_size_{world_size}_rank_{rank}.pt'
    if not shard.is_file() or not shard.stat().st_size:
        raise RuntimeError(f'Missing/empty checkpoint shard: {shard}')
assert (p / 'huggingface/config.json').is_file(), 'Missing Hugging Face config'
print(f'Preflight: {world_size} model shards present at {p}', flush=True)
PY

# Keep partial results separate. Retain them on failure for diagnosis.
staging="$(mktemp -d "${output_hf}.merging-${SLURM_JOB_ID:-manual}-XXXXXX")"
echo "Merging $actor_checkpoint into $staging on $(hostname)"
python -m verl.model_merger merge --backend fsdp \
    --local_dir "$actor_checkpoint" --target_dir "$staging"
python - "$staging" "$output_hf" "$actor_checkpoint" <<'PY'
import json
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

staging, target, source = map(Path, sys.argv[1:])
tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    staging, local_files_only=True, torch_dtype='auto', low_cpu_mem_usage=True,
)
print(f'Load check passed: {type(model).__name__}, {type(tokenizer).__name__}', flush=True)
del model
(staging / 'merge_source.json').write_text(json.dumps({
    'actor_checkpoint': str(source), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    'validated': 'AutoModelForCausalLM and AutoTokenizer loaded locally on CPU',
}, indent=2) + '\n')
if target.exists() or target.is_symlink():
    raise RuntimeError(f'Refusing to overwrite {target}')
staging.rename(target)
print(f'MERGE_COMPLETE: {target}', flush=True)
PY
