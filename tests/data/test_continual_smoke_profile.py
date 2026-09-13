"""CPU-only checks for the continual runner's opt-in smoke profile."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/continual/run_sdpo_svc_cl.sh"


class SmokeProfileTest(unittest.TestCase):
    def commands(self, module="verl.trainer.main_ppo", **overrides):
        with tempfile.TemporaryDirectory(prefix="sdpo-profile-test-") as output:
            env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
            env.update(OUTPUT_ROOT=output, **overrides)
            result = subprocess.run(
                ["bash", str(RUNNER), "--dry-run"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            return [
                shlex.split(line.removeprefix("DRY-RUN: "))
                for line in result.stdout.splitlines()
                if line.startswith(f"DRY-RUN: python3 -m {module} ")
            ]

    def test_four_gpu_svc_overrides_single_device(self):
        devices = "cuda:0,cuda:1,cuda:2,cuda:3"
        commands = self.commands(module="verl.model_merger.svc", RUN_PROFILE="smoke", SVC_DEVICES=devices)
        self.assertEqual(len(commands), 4)
        for command in commands:
            self.assertEqual(command[command.index("--devices") + 1], devices)
            self.assertNotIn("--device", command)
        for command in self.commands(module="verl.model_merger.svc"):
            self.assertEqual(command[command.index("--device") + 1], "cpu")
            self.assertNotIn("--devices", command)

    def test_smoke_all_four_tasks(self):
        commands = self.commands(RUN_PROFILE="smoke")
        self.assertEqual(len(commands), 4)
        self.assertIn("#SBATCH --cpus-per-task=24", RUNNER.read_text())
        for command in commands:
            self.assertIn("ray_kwargs.ray_init.num_cpus=24", command)
            for value in (
                "sdpo_smoke",
                "data.train_batch_size=8",
                "data.train_max_samples=128",
                "trainer.total_training_steps=2",
                "actor_rollout_ref.rollout.n=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                "actor_rollout_ref.rollout.val_kwargs.n=1",
            ):
                self.assertIn(value, command)

    def test_production_defaults_unchanged(self):
        commands = self.commands()
        for command in commands:
            self.assertIn("sdpo", command)
            self.assertIn("data.train_batch_size=32", command)
            self.assertIn("trainer.total_training_steps=null", command)
            self.assertIn("actor_rollout_ref.actor.optim.lr_warmup_steps=10", command)
        self.assertIn("data.train_max_samples=17917", commands[0])
        self.assertIn("data.train_max_samples=-1", commands[1])

    def test_sample_limit_does_not_enlarge_math_unique_pool(self):
        commands = self.commands(TRAIN_SAMPLE_LIMIT="20000")
        self.assertIn("data.train_max_samples=17917", commands[0])
        self.assertIn("data.train_max_samples=20000", commands[1])

    def test_smoke_yaml_memory_settings(self):
        with (ROOT / "verl/trainer/config/sdpo_smoke.yaml").open() as source:
            config = yaml.safe_load(source)
        self.assertEqual(config["defaults"], ["sdpo", "_self_"])
        self.assertFalse(config["actor_rollout_ref"]["model"]["enable_gradient_checkpointing"])
        self.assertEqual(config["data"]["val_max_samples"], 32)

        # Lengths and token budgets must remain inherited from production.
        def check_no_length_overrides(mapping):
            for key, value in mapping.items():
                self.assertNotIn("len", key)
                self.assertNotIn("token", key)
                if isinstance(value, dict):
                    check_no_length_overrides(value)

        check_no_length_overrides(config)


if __name__ == "__main__":
    unittest.main()
