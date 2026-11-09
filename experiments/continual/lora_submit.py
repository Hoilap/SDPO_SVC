"""Replace the two pending full-parameter jobs only after all LoRA jobs are held."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path('/share/home/dengkn/SDPO_SVC')
STUDY = PROJECT / 'outputs/lora_science50/study-20261109'
SOURCES = {kind: PROJECT / '.runtime/lora-20261109' / kind for kind in ['a4', 'sequence']}


def output(args):
    return subprocess.check_output(args, text=True).strip()


def job_info(job):
    return json.loads(output(['scontrol', 'show', 'job', str(job), '--json']))['jobs'][0]


def state(job):
    return output(['squeue', '-h', '-j', str(job), '-o', '%T'])


def submit(name, mode, kind, dependencies=None):
    source = SOURCES[kind]
    logs = STUDY / 'logs'
    logs.mkdir(exist_ok=True)
    cmd = ['sbatch', '--parsable', '--hold', '--job-name=' + name,
           '--output=' + str(logs / (name + '-%j.out')),
           '--error=' + str(logs / (name + '-%j.err')),
           '--export=ALL,LORA_SOURCE_ROOT=' + str(source) + ',LORA_STUDY_ROOT=' + str(STUDY)]
    if kind == 'sequence':
        cmd.append('--cpus-per-task=24')
    if dependencies:
        cmd.append('--dependency=' + dependencies)
    cmd += [str(source / 'experiments/continual/lora_study.sbatch'), '--mode', mode]
    job = output(cmd).split(';')[0]
    assert job.isdigit(), job
    return dict(job_id=job, command=cmd, name=name)


if __name__ == '__main__':
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'sdpo'
    assert (STUDY / 'preflight-shared.json').is_file()
    assert (STUDY / 'preflight-sequence.json').is_file()
    assert (STUDY / 'validation-passed.json').is_file()
    assert not (STUDY / 'submission.json').exists(), 'Already submitted'
    for job in [592878, 592900]:
        assert state(job) == 'PENDING', (job, state(job))
    existing = output(['squeue', '-h', '-u', 'dengkn', '-o', '%j'])
    assert 'a4-s50-sdpo-lora' not in existing
    assert 'sdpo-sequence-math-lora' not in existing
    jobs = {}
    try:
        jobs['shared'] = submit('a4-s50-prefix-lora', 'shared', 'a4', 'afterok:592530')
        prefix = jobs['shared']['job_id']
        jobs['svc'] = submit('a4-s50-svc-lora', 'sdpo_svc', 'a4', 'afterok:' + prefix)
        jobs['sdpo'] = submit('a4-s50-sdpo-lora', 'sdpo', 'a4',
                              'afterok:' + prefix + ',afterany:' + jobs['svc']['job_id'])
        jobs['sequence'] = submit('sdpo-sequence-math-lora', 'sequence', 'sequence')
    except BaseException:
        for job in jobs.values():
            subprocess.run(['scancel', job['job_id']], check=True)
        raise
    for job in [592878, 592900]:
        assert state(job) == 'PENDING', (job, state(job))
    record = dict(jobs=jobs, replaced=[592878, 592900], preserved=[592879],
                  source_roots={k: str(v) for k, v in SOURCES.items()},
                  sources={k: {str(p.relative_to(v)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(v.rglob('*.py')) if '__pycache__' not in p.parts}
                      for k, v in SOURCES.items()})
    (STUDY / 'submission.json').write_text(json.dumps(record, indent=2) + '\n')
    subprocess.run(['scancel', '592878', '592900'], check=True)
    for job in jobs.values():
        subprocess.run(['scontrol', 'release', job['job_id']], check=True)
    print(json.dumps(jobs, indent=2), flush=True)
