"""Isolated LoRA variants of the frozen A4 Science50 and sequence-Math studies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

STEPS = dict(math=93, science=50, tool=93, code=93)
LORA = {
    'actor_rollout_ref.model.lora_rank': '64',
    'actor_rollout_ref.rollout.load_format': 'safetensors',
    'actor_rollout_ref.model.lora_alpha': '128',
    'actor_rollout_ref.model.target_modules': 'all-linear',
    'actor_rollout_ref.model.enable_gradient_checkpointing': 'False',
    '+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook':
        'experiments.continual.lora_sdpo_train.worker_setup',
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def replace(command, changes):
    pending = dict(changes)
    result = []
    for arg in command:
        key = arg.split('=', 1)[0]
        result.append(key + '=' + pending.pop(key) if key in pending else arg)
    return result + [k + '=' + v for k, v in pending.items()]


def lora_command(command):
    command = list(command)
    assert command[1:3] == ['-m', 'experiments.continual.warmup_ab_train']
    command[2] = 'experiments.continual.lora_sdpo_train'
    return replace(command, LORA)


def a4_command(source, root, stage, model, manifest):
    from experiments.continual.a_group_four_stage_w0 import command
    original = command(source, root, stage, model, manifest['train'][stage], manifest['validation'][stage])
    original = replace(original, {
        'trainer.group_name': 'a4-science50-lora',
        'trainer.total_training_steps': str(STEPS[stage]),
        '+trainer.warmup_checkpoint_steps': json.dumps([STEPS[stage]]),
        'trainer.max_actor_ckpt_to_keep': '1',
    })
    return lora_command(original)


def run_cmd(cmd, source):
    print('COMMAND ' + json.dumps(list(map(str, cmd))), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=source, check=True)


def export(source, base, checkpoint, target):
    run_cmd([sys.executable, '-m', 'experiments.continual.lora_export', '--base', base,
             '--checkpoint', checkpoint, '--target', target], source)
    from experiments.continual.a_group_four_stage_restart import validate_weights
    validate_weights(target)


def train(source, root, stage, current, manifest):
    os.environ.update(TASK=stage, EXPERIMENT='a4-lora-' + root.name + '-' + stage)
    cmd = a4_command(source, root, stage, current, manifest)
    write(root / (stage + '_train_command.json'), cmd)
    run_cmd(cmd, source)
    checkpoint = root / 'checkpoints' / stage / f'global_step_{STEPS[stage]}'
    target = root / 'hf_models' / f'{list(STEPS).index(stage)+1:02d}-{stage}-raw'
    export(source, current, checkpoint / 'actor', target)
    write(root / (stage + '_export_complete.json'), dict(model=str(target), step=STEPS[stage]))
    shutil.rmtree(checkpoint)
    return target


def boundary(source, root, stage, current, manifest):
    from experiments.continual.a_group_four_stage_restart import evaluate
    number = list(STEPS).index(stage) + 1
    metrics = evaluate(source, root, f'{number:02d}-{stage}', current, manifest, stage)
    write(root / (stage + '_complete.json'), dict(model=str(current), metrics=metrics, step=STEPS[stage]))


def a4(source, study, mode):
    from experiments.continual.a_group_four_stage_w0 import calibrate
    manifest = json.loads((study / 'parent.json').read_text())
    root = study / mode
    root.mkdir(exist_ok=False)
    (root / 'wandb').mkdir()
    os.environ.update(WANDB_MODE='offline', WANDB_DIR=str(root / 'wandb'))
    base = Path(manifest['base_model'])
    if mode == 'shared':
        current = base
        for stage in ['math', 'science']:
            current = train(source, root, stage, current, manifest)
            boundary(source, root, stage, current, manifest)
    else:
        shared = study / 'shared'
        assert (shared / 'complete.json').is_file()
        current = shared / 'hf_models/02-science-raw'
        if mode == 'sdpo_svc':
            target = root / 'hf_models/02-science-svc'
            calibrate(base, shared / 'hf_models/01-math-raw', current, target, root, 'science')
            current = target
            boundary(source, root, 'science', current, manifest)
        else:
            write(root / 'science_complete.json', json.loads((shared / 'science_complete.json').read_text()))
        for stage in ['tool', 'code']:
            previous = current
            current = train(source, root, stage, current, manifest)
            if mode == 'sdpo_svc':
                target = root / 'hf_models' / f'{list(STEPS).index(stage)+1:02d}-{stage}-svc'
                calibrate(base, previous, current, target, root, stage)
                current = target
            boundary(source, root, stage, current, manifest)
    write(root / 'complete.json', dict(model=str(current), job_id=os.environ['SLURM_JOB_ID']))


def sequence_commands(source, root):
    project = Path('/share/home/dengkn/SDPO_SVC')
    model = Path('/share/home/dengkn/model/Qwen3-4B-Instruct-2507')
    original = [sys.executable, '-m', 'experiments.continual.sdpo_reward_granularity',
        '--arm', 'sequence_js', '--stage', 'math', '--initial-model', str(model),
        '--train-files', str(project / 'datasets/cl/math/DAPO-Math-17k/data/dapo-math-17k.parquet'),
        '--val-files', *[str(project / 'outputs/sdpo_cl_try/124450/eval_data' / (n + '.parquet'))
                         for n in ['aime24', 'aime25', 'math500']],
        '--output-root', str(root), '--seed', '42', '--warmup', '10', '--steps', '93', '--dry-run']
    payload = json.loads(subprocess.check_output(original, cwd=source, text=True))
    commands = payload['commands']
    commands[0] = lora_command(commands[0])
    commands[1] = [sys.executable, '-m', 'experiments.continual.lora_export', '--base', str(model),
        '--checkpoint', str(root / 'checkpoints/math/global_step_93/actor'),
        '--target', str(root / 'hf_models/step_93')]
    return commands


def compose(source, command):
    from hydra import compose as hydra_compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(source / 'verl/trainer/config')):
        return OmegaConf.to_container(hydra_compose(config_name='sdpo', overrides=command[5:]), resolve=True)


def preflight(source, study, mode):
    from data.preflight_rl_dataset import validate_group
    from experiments.continual.a_group_four_stage_w0 import validate_model
    if mode == 'sequence':
        configs = {'math': sequence_commands(source, study / 'sequence')[0]}
    else:
        m = json.loads((study / 'parent.json').read_text())
        validate_model(Path(m['base_model']))
        configs = {s: a4_command(source, study / 'audit', s, Path(m['base_model']), m) for s in STEPS}
    audited = {}
    for stage, command in configs.items():
        os.environ.update(TASK=stage, EXPERIMENT='lora-preflight')
        c = compose(source, command)
        model = c['actor_rollout_ref']['model']
        assert model['lora_rank'] == 64 and model['lora_alpha'] == 128
        assert model['enable_gradient_checkpointing'] is False
        assert c['actor_rollout_ref']['actor']['self_distillation']['teacher_update_rate'] == 0.05
        assert c['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu'] == 1
        assert c['trainer']['total_training_steps'] == (93 if mode == 'sequence' else STEPS[stage])
        for kind in ['train', 'val']:
            validate_group(stage + ' ' + kind, [Path(p) for p in c['data'][kind + '_files']])
        audited[stage] = dict(command=command, config=c)
    write(study / ('preflight-' + mode + '.json'), audited)
    print('PREFLIGHT PASSED ' + mode, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--mode', choices=['shared', 'sdpo', 'sdpo_svc', 'sequence'], required=True)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    source = Path.cwd()
    if a.preflight:
        preflight(source, a.study, a.mode)
    elif a.mode == 'sequence':
        os.environ.update(TASK='math', EXPERIMENT='sdpo-sequence-math-lora')
        root = a.study / 'sequence'
        commands = sequence_commands(source, root)
        root.mkdir(exist_ok=False)
        write(root / 'commands.json', commands)
        for cmd in commands:
            run_cmd(cmd, source)
        write(root / 'complete.json', dict(job_id=os.environ['SLURM_JOB_ID']))
    else:
        a4(source, a.study, a.mode)
