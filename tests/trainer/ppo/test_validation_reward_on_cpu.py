"""Real Ray/code-sandbox parity and validation aggregation checks."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn, score_validation_batch
from verl.workers.reward_manager.naive import NaiveRewardManager


class CodeTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        value = int(ids[0])
        if value == -1:
            return "missing code block"
        code = "while True: pass" if value == 44 else f"print({value})"
        return f"```python\n{code}\n```"


def batch_and_reward():
    config = OmegaConf.create({"custom_reward_function": {
        "path": str(Path(__file__).resolve().parents[3] / "verl/utils/reward_score/feedback/__init__.py"),
        "name": "compute_score",
    }})
    reward = NaiveRewardManager(CodeTokenizer(), 0, get_custom_reward_fn(config))
    truth = json.dumps({"testtype": "stdin", "inputs": ["", ""], "outputs": ["42", "42"], "time_limit": 0.1})
    data = DataProto.from_dict(tensors={
        "prompts": torch.ones(5, 1, dtype=torch.long),
        "responses": torch.tensor([[42, 0], [43, 0], [-1, 0], [44, 0], [42, 0]]),
        "attention_mask": torch.ones(5, 3, dtype=torch.long),
    }, non_tensors={
        "data_source": np.array(["code"] * 5, dtype=object),
        "uid": np.array([str(i) for i in range(5)], dtype=object),
        "reward_model": np.array([{"ground_truth": truth} for _ in range(5)], dtype=object),
        "extra_info": np.array([{"split": "test"} for _ in range(5)], dtype=object),
    })
    return data, reward


class ValidationRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {"CODE_REWARD_MAX_CONCURRENCY": "2"})
        cls.environment.start()
        ray.init(num_cpus=6, include_dashboard=False)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()
        cls.environment.stop()

    def test_real_cpu_score_matches_serial_for_correct_wrong_format_and_timeout(self):
        data, reward = batch_and_reward()
        expected = reward(data, return_dict=True)
        actual, intervals = score_validation_batch(data, reward, 2)
        torch.testing.assert_close(actual["reward_tensor"], expected["reward_tensor"])
        assert actual["reward_extra_info"] == dict(expected["reward_extra_info"])
        assert actual["reward_tensor"].sum(-1).tolist() == [1, 0, 0, 0, 1]
        assert len(intervals) == 2
        assert all(end > start for start, end in intervals)


    def test_validate_keeps_scores_metadata_and_timing_in_output(self):
        with tempfile.TemporaryDirectory() as directory:
            self._check_validation(Path(directory))

    def _check_validation(self, tmp_path):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        data, reward = batch_and_reward()
        trainer = object.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create({
            "actor_rollout_ref": {"rollout": {"agent": {"num_workers": 2}, "val_kwargs": {"n": 1}}},
            "trainer": {"val_only": True, "validation_pipeline": True, "validation_reward_num_workers": 2,
                        "validation_dump_generations": False, "validation_data_dir": str(tmp_path)},
            "reward_model": {"enable": False, "use_reward_loop": False}, "data": {},
        })
        trainer.tokenizer = CodeTokenizer()
        trainer.val_reward_fn = reward
        captured = {}

        def generate(repeats, pipeline):
            import time
            for start in (0, 3):
                stamp = time.time()
                pipeline.record_rollout(stamp, time.time())
                yield data[start:start + 3]

        def aggregate(sources, uids, extras, turns):
            captured.update(sources=sources.tolist(), uids=uids, scores=extras["reward"])
            return {"test/accuracy": sum(extras["reward"]) / len(extras["reward"])}

        trainer._generate_validation_batches = generate
        trainer._maybe_log_val_generations = lambda **kwargs: None
        trainer._val_metrics_update = aggregate
        trainer._dump_validation_metrics = lambda metrics, count, directory, extra: captured.update(extra=extra, count=count)
        metrics = trainer._validate()
        assert captured["uids"] == [str(i) for i in range(5)]
        assert captured["scores"] == [1, 0, 0, 0, 1]
        assert captured["sources"] == ["code"] * 5
        assert captured["count"] == 5
        assert captured["extra"]["pipeline_timing"]["num_batches"] == 2
        assert metrics["test/accuracy"] == 0.4
        assert metrics["val-timing/cpu_eval_active_wall_s"] > 0



if __name__ == "__main__":
    unittest.main()
