"""Context-limit regression tests without importing the GPU rollout package."""

import runpy
import unittest
from pathlib import Path


resolve_max_model_len = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout/utils.py")
)["resolve_max_model_len"]


class TestContextLimit(unittest.TestCase):
    def test_context_limit(self):
        for configured, expected in [(None, 262144), (18944, 18944), (262144, 262144), (1, 1)]:
            with self.subTest(configured=configured):
                self.assertEqual(resolve_max_model_len(configured, 262144), expected)


    def test_invalid_context_limit(self):
        for configured in [0, -1, 262145]:
            with self.subTest(configured=configured):
                with self.assertRaisesRegex(ValueError, "rollout.max_model_len"):
                    resolve_max_model_len(configured, 262144)


if __name__ == "__main__":
    unittest.main()
