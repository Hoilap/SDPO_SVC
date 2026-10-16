#!/usr/bin/env bash
#SBATCH --job-name=math-GRPO-lenient-2gpu
#SBATCH --partition=gpu_chen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=230000
#SBATCH --time=0
#SBATCH --output=/dev/shm/math-GRPO-lenient-%j.log
#SBATCH --error=/dev/shm/math-GRPO-lenient-%j.log

# Results and logs stay on the compute node's tmpfs, never in the repository.
set -euo pipefail
ulimit -S -u 16384
cd "${SLURM_SUBMIT_DIR:-/share/home/dengkn/SDPO_SVC}"
source /share/home/dengkn/miniconda3/etc/profile.d/conda.sh
conda activate sdpo
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module load gcc/9.1.0
export CC="$(command -v gcc)" CXX="$(command -v g++)"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export https_proxy=http://192.168.10.22:3128
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export SOURCE_ROOT="${SOURCE_ROOT:-$PWD/outputs/causal_tail_seed1/118343}"

python3 - "$@" <<'PY'
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--preflight-only', action='store_true')
args = parser.parse_args()
source = Path(os.environ['SOURCE_ROOT'])
data = json.loads((source / 'data/dataset_manifest.json').read_text())
tails = json.loads((source / 'tail_arms/manifest.json').read_text())
files = []
for key in ['aime24', 'aime25', 'math500']:
    value = data['evaluation'][key]
    files.extend(value if isinstance(value, list) else [value])
for file in files:
    assert Path(file).is_file(), file
models = {arm: Path(tails['arms'][arm]) for arm in ['GRPO-full', 'GRPO-no-tail']}
for model in models.values():
    assert (model / 'config.json').is_file(), model
    assert list(model.glob('*.safetensors')), model
assert subprocess.check_output(['stat', '-f', '-c', '%T', '/dev/shm'], text=True).strip() == 'tmpfs'
evaluator = Path('experiments/continual/evaluate_stage.py')
code = evaluator.read_text()
assert code.count('trainer.n_gpus_per_node=4') == 1
code = code.replace('trainer.n_gpus_per_node=4', 'trainer.n_gpus_per_node=2')
assert Path('verl/utils/reward_score/feedback_lenient.py').is_file()
print('PREFLIGHT: ' + json.dumps(dict(models={k: str(v) for k, v in models.items()},
      val_files=files, gpus=2, tensor_parallel=2, responses_per_question=8,
      dump_generations=False, results_storage='compute-node /dev/shm (tmpfs)')), flush=True)
if args.preflight_only:
    compile(code, str(evaluator), 'exec')
    print('PREFLIGHT COMPLETE', flush=True)
    sys.exit(0)
job = os.environ['SLURM_JOB_ID']
root = Path('/dev/shm') / ('mgl-' + job)
root.mkdir(exist_ok=False)
tmp = root / 'tmp'
tmp.mkdir()
os.environ['TMPDIR'] = str(tmp)
os.environ['RAY_TMPDIR'] = str(tmp)
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['TRITON_CACHE_DIR'] = str(root / 'triton')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(root / 'torchinductor')
original_run = subprocess.run

def run_in_memory(command, *a, **kw):
    if isinstance(command, list) and 'verl.trainer.main_ppo' in command:
        assert 'trainer.n_gpus_per_node=2' in command
        command = [*command, 'hydra.run.dir=' + str(root / 'hydra'), 'hydra.output_subdir=null']
    return original_run(command, *a, **kw)

subprocess.run = run_in_memory
summary = {}
for arm, model in models.items():
    name = 'after-math-' + arm
    out = root / arm
    sys.argv = [str(evaluator), '--model', str(model), '--name', name,
                '--output-root', str(out), '--agent-loop-workers', '2',
                '--scorer', 'verl/utils/reward_score/feedback_lenient.py',
                '--val-files', *files]
    exec(compile(code, str(evaluator), 'exec'), {'__name__': '__main__', '__file__': str(evaluator)})
    result = json.loads((out / name / '0.metrics.json').read_text())
    assert result['num_questions'] == 560, result['questions_by_source']
    assert result['num_samples'] == 4480
    metrics = result['metrics']
    summary[arm] = {k: v for k, v in metrics.items() if k.endswith('/mean@8') and
                    any('/' + field + '/' in k for field in
                        ['acc', 'strict_acc', 'lenient_rescued', 'truncated', 'candidate_count'])}
    print('RESULT ' + arm + ' ' + json.dumps(summary[arm]), flush=True)
print('COMPLETE: after-Math full/no-tail, 2 GPUs, strict and lenient scores; all results in tmpfs', flush=True)
print('SUMMARY ' + json.dumps(summary), flush=True)
PY
