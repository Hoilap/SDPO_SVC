"""Bounded validation prefetch and wall-clock accounting (no model dependencies)."""

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def merged_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Interval ends before it starts")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_seconds(intervals):
    return sum(end - start for start, end in merged_intervals(intervals))


def overlap_seconds(left, right):
    left, right = merged_intervals(left), merged_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


class ValidationPipeline:
    """One scoring coordinator, a bounded batch queue, and ordered collection.

    The coordinator scores only one batch at a time; parallel scoring workers
    are owned by the scorer, so prefetch never multiplies CPU concurrency.
    Timestamps use time.time across processes on the evaluation node.
    """

    def __init__(self, enabled=False, max_pending=2):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.enabled = enabled
        self.max_pending = max_pending
        self.rollout_intervals = []
        self.eval_intervals = []
        self.wait_seconds = 0.0
        self.started = None
        self.ended = None
        self.completed = 0

    def record_rollout(self, start, end):
        self.rollout_intervals.append((start, end))
        print("EVAL_ROLLOUT " + json.dumps({
            "batch": len(self.rollout_intervals), "gpu_rollout_wall_s": end - start,
        }), flush=True)

    def _collect(self, batch, result, submitted, wait_s):
        reward, intervals = result
        self.eval_intervals.extend(intervals)
        self.wait_seconds += wait_s
        self.completed += 1
        print("EVAL_CPU " + json.dumps({
            "batch": self.completed,
            "cpu_eval_active_wall_s": interval_seconds(intervals),
            "cpu_eval_worker_wall_s": sum(end - start for start, end in intervals),
            "score_queue_and_completion_s": time.time() - submitted,
            "driver_wait_s": wait_s,
        }), flush=True)
        progress = self.summary()
        progress["cpu_accounting"] = "completed_batches_only"
        print("EVAL_PIPELINE_PROGRESS " + json.dumps(progress), flush=True)
        return batch, reward

    def run(self, batches, score):
        self.started = time.time()
        try:
            if not self.enabled:
                for batch in batches:
                    submitted = time.time()
                    result = score(batch)
                    yield self._collect(batch, result, submitted, time.time() - submitted)
                return
            # Only this thread coordinates Ray CPU tasks. The caller stays free
            # to generate the next batch on the GPU.
            pending = deque()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="validation-score") as executor:
                try:
                    for batch in batches:
                        submitted = time.time()
                        pending.append((batch, executor.submit(score, batch), submitted))
                        if len(pending) >= self.max_pending:
                            batch, future, submitted = pending.popleft()
                            waited = time.time()
                            result = future.result()
                            yield self._collect(batch, result, submitted, time.time() - waited)
                    while pending:
                        batch, future, submitted = pending.popleft()
                        waited = time.time()
                        result = future.result()
                        yield self._collect(batch, result, submitted, time.time() - waited)
                finally:
                    for _, future, _ in pending:
                        future.cancel()
        finally:
            self.ended = time.time()
            if hasattr(batches, "close"):
                batches.close()

    def summary(self):
        wall = (self.ended or time.time()) - self.started
        rollout = interval_seconds(self.rollout_intervals)
        cpu = interval_seconds(self.eval_intervals)
        overlap = overlap_seconds(self.rollout_intervals, self.eval_intervals)
        return {
            "wall_s": wall,
            "gpu_rollout_wall_s": rollout,
            "cpu_eval_active_wall_s": cpu,
            "cpu_eval_worker_wall_s": sum(end - start for start, end in self.eval_intervals),
            "overlap_wall_s": overlap,
            "driver_wait_s": self.wait_seconds,
            "gpu_rollout_wall_pct": 100 * rollout / wall if wall else 0.0,
            "cpu_eval_active_wall_pct": 100 * cpu / wall if wall else 0.0,
            "overlap_wall_pct": 100 * overlap / wall if wall else 0.0,
            "other_wall_s": max(0.0, wall - rollout - cpu + overlap),
            "num_batches": self.completed,
        }
