"""Deterministic, source-stratified sampling for in-training validation."""

import math
import random
from collections import defaultdict


def should_validate(step, total_steps, frequency, skip_last_interval=False):
    """Always keep final validation; optionally omit nearby periodic checks."""
    if frequency <= 0:
        return False
    if step >= total_steps:
        return True
    return step > 0 and step % frequency == 0 and (
        not skip_last_interval or total_steps - step >= frequency
    )


def validation_indices(sources, fraction=0.1, seed=42, val_only=False):
    """Keep every source represented; round each source's sample count up.

    Sample once before constructing the loader, independently of training RNG.
    Standalone evaluation retains all rows in their original order.
    """
    if not 0 < fraction <= 1:
        raise ValueError("training_val_fraction must be in (0, 1]")
    if val_only or fraction == 1:
        return list(range(len(sources)))
    groups = defaultdict(list)
    for index, source in enumerate(sources):
        groups[source].append(index)
    selected = []
    for indices in groups.values():
        selected.extend(random.Random(seed).sample(indices, math.ceil(len(indices) * fraction)))
    return sorted(selected)
