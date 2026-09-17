#!/usr/bin/env python3
"""Summarize causal-run ``*.metrics.json`` files into tidy CSV and contrasts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PRIMARY_SCORE = re.compile(r"^val-aux/(?P<dataset>[^/]+)/score/mean@(?P<n>\d+)$")


def load_scores(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/*.metrics.json")):
        arm = path.parent.name
        payload = json.loads(path.read_text())
        for key, value in payload.get("metrics", {}).items():
            match = PRIMARY_SCORE.match(key)
            if match:
                rows.append(
                    {
                        "phase": root.name,
                        "arm": arm,
                        "dataset": match.group("dataset"),
                        "rollouts": int(match.group("n")),
                        "score": float(value),
                        "source_file": str(path),
                    }
                )
    if not rows:
        raise ValueError(f"No primary score metrics found below {root}")
    max_rollouts = max(row["rollouts"] for row in rows)
    return [row for row in rows if row["rollouts"] == max_rollouts]


def contrasts(rows: list[dict]) -> list[dict]:
    by_key = {(row["phase"], row["dataset"], row["arm"]): row["score"] for row in rows}
    results = []
    comparisons = {
        "grpo_tail_effect": ("GRPO-full", "GRPO-no-tail"),
        "sdpo_tail_effect": ("SDPO-full", "SDPO-no-tail"),
        "natural_method_gap": ("GRPO-full", "SDPO-full"),
        "no_tail_method_gap": ("GRPO-no-tail", "SDPO-no-tail"),
        "grpo_direction_specificity": ("GRPO-full", "GRPO-random-tail"),
        "grpo_to_sdpo_transplant": ("SDPO-principal+GRPO-tail", "SDPO-full"),
    }
    phase_datasets = sorted({(row["phase"], row["dataset"]) for row in rows})
    for phase, dataset in phase_datasets:
        values = {}
        for name, (left, right) in comparisons.items():
            left_value = by_key.get((phase, dataset, left))
            right_value = by_key.get((phase, dataset, right))
            if left_value is not None and right_value is not None:
                values[name] = left_value - right_value
        if "grpo_tail_effect" in values and "sdpo_tail_effect" in values:
            values["method_by_tail_interaction"] = values["grpo_tail_effect"] - values["sdpo_tail_effect"]
        if values:
            results.append({"phase": phase, "dataset": dataset, **values})

    # Cross-phase forgetting uses each checkpoint's own pre-Science score.
    datasets = sorted({row["dataset"] for row in rows})
    core_arms = ("GRPO-full", "GRPO-no-tail", "SDPO-full", "SDPO-no-tail")
    for dataset in datasets:
        forgetting = {}
        for arm in core_arms:
            before = by_key.get(("after_math", dataset, arm))
            after = by_key.get(("after_science", dataset, arm))
            if before is not None and after is not None:
                forgetting[arm] = before - after
        if len(forgetting) == len(core_arms):
            grpo_effect = forgetting["GRPO-no-tail"] - forgetting["GRPO-full"]
            sdpo_effect = forgetting["SDPO-no-tail"] - forgetting["SDPO-full"]
            results.append(
                {
                    "phase": "forgetting",
                    "dataset": dataset,
                    **{f"forgetting_{arm}": value for arm, value in forgetting.items()},
                    "grpo_tail_retention_effect": grpo_effect,
                    "sdpo_tail_retention_effect": sdpo_effect,
                    "retention_method_by_tail_interaction": grpo_effect - sdpo_effect,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for phase in ("after_math", "after_science"):
        phase_root = args.evaluation_root / phase
        if phase_root.is_dir():
            rows.extend(load_scores(phase_root))
    if not rows:
        raise ValueError(f"No after_math/after_science evaluations found under {args.evaluation_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contrast_rows = contrasts(rows)
    (args.output_dir / "contrasts.json").write_text(json.dumps(contrast_rows, indent=2) + "\n")
    print(f"Wrote {args.output_dir / 'scores.csv'}")
    print(f"Wrote {args.output_dir / 'contrasts.json'}")


if __name__ == "__main__":
    main()
