#!/usr/bin/env bash
#SBATCH --job-name=grpo-lenient-eval
#SBATCH --partition=gpu_chen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=460000
#SBATCH --time=0
#SBATCH --output=logs/grpo-lenient-eval-%j.out
#SBATCH --error=logs/grpo-lenient-eval-%j.err
set -euo pipefail
ulimit -S -u 16384
source /share/home/dengkn/miniconda3/etc/profile.d/conda.sh
conda activate sdpo
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export https_proxy=http://192.168.10.22:3128
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module load gcc/9.1.0
export CC="$(command -v gcc)" CXX="$(command -v g++)"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CODE_REWARD_MAX_CONCURRENCY=2 HYDRA_FULL_ERROR=1
export SOURCE_ROOT="${SOURCE_ROOT:-$PWD/outputs/causal_tail_seed1/science_restart_20261009_ulimit16384}"
export OUTPUT_ROOT="$PWD/outputs/grpo_lenient_science/$SLURM_JOB_ID"
python3 - <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
snapshot=Path('experiments/causal/scorer_snapshots/strict_20261014')
manifest=json.loads((snapshot/'manifest.json').read_text())
for file,digest in manifest['files'].items():
    for root in [snapshot/'feedback',Path('verl/utils/reward_score/feedback')]:
        assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest, (root,file)
source=Path(os.environ['SOURCE_ROOT'])
plans=json.loads((source/'evaluation/test/after_science/plan.json').read_text())
selected=[]
for arm in ['GRPO-full','GRPO-no-tail']:
    name=f'tail-seed1-{arm}-Science'
    plan=next((p for p in plans if p['name']==name), None)
    if plan is None:
        # The shared evaluator may leave only the most recent stage's plan.
        plan=dict(name=name,model=str(source/'science_hf'/arm),val_files=plans[0]['val_files'])
    model=Path(plan['model'])
    assert (model/'config.json').is_file() and list(model.glob('*.safetensors')), model
    selected.append(plan)
root=Path(os.environ['OUTPUT_ROOT']);root.mkdir(parents=True,exist_ok=False)
provenance=dict(scoring_policy='any_explicit_correct_answer_v1', source_root=str(source), stages=selected,
                strict_scorer_manifest=manifest,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                lenient_scorer_sha256=hashlib.sha256(Path('verl/utils/reward_score/feedback_lenient.py').read_bytes()).hexdigest())
(root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('PREFLIGHT: frozen strict scorer hashes verified; fresh Science full/no-tail evaluation',flush=True)
for plan in selected:
    subprocess.run([sys.executable,'experiments/continual/evaluate_stage.py',
                    '--model',plan['model'],'--name',plan['name'],
                    '--output-root',str(root/plan['name']), '--agent-loop-workers','2',
                    '--scorer','verl/utils/reward_score/feedback_lenient.py','--dump-generations',
                    '--val-files',*plan['val_files']],check=True)
print('COMPLETE: both GRPO Science arms re-generated and lenient/strict scored',flush=True)
PY
