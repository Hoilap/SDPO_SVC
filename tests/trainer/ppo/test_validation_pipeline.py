"""Deterministic checks for overlap, queue bounds, ordering, and wall accounting."""

import importlib.util
from pathlib import Path
import threading
import time
import unittest

MODULE = Path(__file__).resolve().parents[3] / "verl/trainer/ppo/validation_pipeline.py"
spec = importlib.util.spec_from_file_location("validation_pipeline_under_test", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ValidationPipeline = module.ValidationPipeline


class ValidationPipelineTest(unittest.TestCase):
    def test_cpu_eval_overlaps_next_gpu_rollout_and_stays_bounded(self):
        pipeline = ValidationPipeline(enabled=True, max_pending=2)
        scoring = threading.Event()
        generated_next = threading.Event()
        completed = []

        def batches():
            for index in range(5):
                self.assertLessEqual(index + 1 - len(completed), 2)
                if index == 1:
                    self.assertTrue(scoring.wait(5), "CPU scorer never started")
                    started = time.time()
                    generated_next.set()
                    pipeline.record_rollout(started, time.time())
                yield index

        def score(index):
            started = time.time()
            if index == 0:
                scoring.set()
                self.assertTrue(generated_next.wait(5), "Generation blocked on CPU eval")
            completed.append(index)
            return index * 10, [(started, time.time())]

        result = list(pipeline.run(batches(), score))
        self.assertEqual(result, [(i, i * 10) for i in range(5)])
        self.assertEqual(completed, list(range(5)))
        self.assertGreater(pipeline.summary()["overlap_wall_s"], 0)

    def test_serial_and_parallel_results_match_including_partial_last_batch(self):
        def score(batch):
            started = time.time()
            return [x * x for x in batch], [(started, time.time())]

        batches = [[0, 1], [2, 3], [4]]
        serial = list(ValidationPipeline().run(iter(batches), score))
        parallel = list(ValidationPipeline(True).run(iter(batches), score))
        self.assertEqual(serial, parallel)

    def test_intervals_do_not_double_count_workers_or_overlap(self):
        pipeline = ValidationPipeline(True)
        pipeline.started, pipeline.ended = 0, 10
        pipeline.rollout_intervals = [(0, 4), (5, 7)]
        pipeline.eval_intervals = [(2, 6), (3, 5), (8, 10)]
        result = pipeline.summary()
        self.assertEqual(result["gpu_rollout_wall_s"], 6)
        self.assertEqual(result["cpu_eval_active_wall_s"], 6)
        self.assertEqual(result["cpu_eval_worker_wall_s"], 8)
        self.assertEqual(result["overlap_wall_s"], 3)
        self.assertEqual(result["other_wall_s"], 1)
        self.assertEqual(result["gpu_rollout_wall_pct"], 60)
        self.assertEqual(result["cpu_eval_active_wall_pct"], 60)

    def test_scoring_failure_propagates_and_closes_producer(self):
        closed = threading.Event()

        def batches():
            try:
                yield 0
                yield 1
                self.fail("Prefetch exceeded bound before raising scoring error")
            finally:
                closed.set()

        def score(batch):
            raise RuntimeError("scoring failure")

        pipeline = ValidationPipeline(True)
        with self.assertRaisesRegex(RuntimeError, "scoring failure"):
            list(pipeline.run(batches(), score))
        self.assertTrue(closed.is_set())
        self.assertIsNotNone(pipeline.ended)

    def test_generation_failure_propagates(self):
        def batches():
            yield 0
            raise RuntimeError("generation failure")

        def score(batch):
            started = time.time()
            return batch, [(started, time.time())]

        with self.assertRaisesRegex(RuntimeError, "generation failure"):
            list(ValidationPipeline(True).run(batches(), score))


if __name__ == "__main__":
    unittest.main()
