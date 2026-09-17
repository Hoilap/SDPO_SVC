import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/causal/run.sh"


def run_dry(output_root: Path, *, limit: int = 3000) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["TRAIN_SAMPLE_LIMIT"] = str(limit)
    environment["OUTPUT_ROOT"] = str(output_root)
    return subprocess.run(
        ["bash", str(RUNNER), "all", "--dry-run"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )


def test_dry_run_caps_every_training_command_at_3000(tmp_path):
    result = run_dry(tmp_path / "output")
    assert result.returncode == 0, result.stderr
    training_lines = [
        line
        for line in result.stdout.splitlines()
        if "verl.trainer.main_ppo" in line and "trainer.val_only=True" not in line
    ]
    assert len(training_lines) == 6  # 2 Math + 4 Science continuations
    assert all("data.train_max_samples=3000" in line for line in training_lines)


def test_runner_rejects_sample_limit_above_3000(tmp_path):
    result = run_dry(tmp_path / "output", limit=3001)
    assert result.returncode == 2
    assert "[1, 3000]" in result.stderr
