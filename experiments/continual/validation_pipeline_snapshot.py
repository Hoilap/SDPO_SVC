"""Upgrade a frozen evaluation runtime while preserving its training implementation.

Run locally against a downloaded snapshot, then deploy reviewed files and refresh
its launcher's snapshot checksums. Shared historical snapshots must be copied
before using this tool. The pipeline source must be a reviewed repository tree.
"""

import argparse
import ast
from pathlib import Path


TRAINER = "verl/trainer/ppo/ray_trainer.py"
REWARD = "verl/trainer/ppo/reward.py"
PIPELINE = "verl/trainer/ppo/validation_pipeline.py"
EVALUATOR = "experiments/continual/warmup_ab_evaluate.py"
METHODS = {"_generate_validation_batches", "_validate"}
HELPERS = {"compute_validation_reward_timed", "score_validation_batch"}


def functions(text, class_name=None):
    body = ast.parse(text).body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def segment(text, node):
    first = min([node.lineno] + [n.lineno for n in node.decorator_list])
    return "".join(text.splitlines(keepends=True)[first - 1:node.end_lineno])


def upgrade(source, pipeline_source):
    """Return changed files; retain every training function exactly as parsed."""
    target = (source / TRAINER).read_text()
    baseline = (pipeline_source / TRAINER).read_text()
    before = functions(target, "RayPPOTrainer")
    available = functions(baseline, "RayPPOTrainer")
    assert METHODS <= available.keys()
    lines = target.splitlines(keepends=True)
    old = before["_validate"]
    replacement = "\n\n".join(segment(baseline, available[name]) for name in
                                 ("_generate_validation_batches", "_validate")) + "\n"
    assert "_generate_validation_batches" not in before, "Snapshot already upgraded"
    lines[old.lineno - 1:old.end_lineno] = [replacement]
    updated = "".join(lines)
    anchor = "from verl.trainer.ppo.reward import "
    pos = updated.index(anchor)
    updated = (updated[:pos] + "from verl.trainer.ppo.validation_pipeline import ValidationPipeline\n"
               "from verl.trainer.ppo.reward import score_validation_batch\n" + updated[pos:])
    after = functions(updated, "RayPPOTrainer")
    for name, node in before.items():
        if name not in METHODS:
            assert ast.dump(node) == ast.dump(after[name]), "Unexpected training change: " + name
    reward = (source / REWARD).read_text()
    reward_baseline = (pipeline_source / REWARD).read_text()
    available = functions(reward_baseline)
    assert not HELPERS & functions(reward).keys(), "Reward helper already present"
    reward = reward.replace("import inspect\n", "import inspect\nimport time\n", 1)
    if "from collections import defaultdict" not in reward:
        reward = reward.replace("import inspect\n", "from collections import defaultdict\nimport inspect\n", 1)
    reward += "\n\n" + "\n\n".join(segment(reward_baseline, available[name]) for name in
                                      ("compute_validation_reward_timed", "score_validation_batch")) + "\n"
    evaluator = (source / EVALUATOR).read_text()
    assert "trainer.validation_pipeline" not in evaluator
    evaluator = evaluator.replace('"trainer.val_only=True", "trainer.val_before_train=True",',
        '"trainer.val_only=True", "trainer.val_before_train=True",\n'
        '            "++trainer.validation_pipeline=True", "++trainer.validation_reward_num_workers=2",\n'
        '            "reward_model.enable=False", "reward_model.use_reward_loop=False",', 1)
    evaluator = evaluator.replace('"ray_kwargs.ray_init.num_cpus=24",',
        '"ray_kwargs.ray_init.num_cpus=" + str(int(os.environ.get("SLURM_CPUS_PER_TASK", "24"))),', 1)
    evaluator = evaluator.replace('train_file = max(',
        'env["CODE_REWARD_MAX_CONCURRENCY"] = "2"\n        train_file = max(', 1)
    changed = {TRAINER: updated, REWARD: reward, PIPELINE: (pipeline_source / PIPELINE).read_text(),
               EVALUATOR: evaluator}
    for name, value in changed.items():
        compile(value, name, "exec")
        path = source / name
        # Do not follow snapshot overlay links into the historical source.
        if path.is_symlink():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pipeline-source", type=Path, required=True)
    args = parser.parse_args()
    for name in upgrade(args.source, args.pipeline_source):
        print(name)


if __name__ == "__main__":
    main()
