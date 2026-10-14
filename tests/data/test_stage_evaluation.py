"""Verify full tests retain all source rows and record actual question counts."""
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def test_stage_test_keeps_long_prompts_and_all_sources(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock())))
    spec = importlib.util.spec_from_file_location("stage_test", ROOT / "experiments/continual/evaluate_stage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").touch()
    data = tmp_path / "evaluation.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"prompt": [{"role": "user", "content": "short"}], "data_source": "math"},
        {"prompt": [{"role": "user", "content": "x" * 3000}], "data_source": "gpqa"},
    ]), data)
    tokenizer = Mock()
    tokenizer.backend_tokenizer.to_str.return_value = "tokenizer"
    tokenizer.apply_chat_template.side_effect = lambda prompt, **kwargs: prompt[0]["content"]
    tokenizer.side_effect = lambda text, **kwargs: {"input_ids": list(text)}
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        path = next(x.split("=", 1)[1] for x in command if x.startswith("trainer.validation_data_dir="))
        (Path(path) / "0.metrics.json").write_text(json.dumps({"num_samples": 16, "metrics": {}}))
    monkeypatch.setattr(module.subprocess, "run", run)
    out = tmp_path / "test"
    monkeypatch.setattr(module.sys, "argv", ["evaluate_stage.py", "--model", str(model),
        "--name", "boundary", "--output-root", str(out), "--val-files", str(data)])
    module.main()
    command = commands[0]
    for setting in ["trainer.val_only=True", "data.training_val_fraction=1.0",
                    "data.val_max_samples=-1", "data.filter_overlong_prompts=False",
                    "actor_rollout_ref.rollout.val_kwargs.n=8"]:
        assert setting in command
    plan = json.loads((out / "plan.json").read_text())[0]
    assert plan["max_prompt_length"] >= 3000
    result = json.loads((out / "boundary/0.metrics.json").read_text())
    assert result["num_questions"] == 2
    assert result["questions_by_source"] == {"math": 1, "gpqa": 1}
    assert result["evaluation_kind"] == "test"
