#!/usr/bin/env python3
"""Resolve and normalize the fixed Math -> Science causal experiment data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCIENCE_DOMAINS = ("biology", "chemistry", "material", "physics")
TRAIN_SAMPLE_CAP = 3000


def _find_eval_file(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    candidates = [
        path / "eval.parquet",
        path / "validation.parquet",
        path / "test.parquet",
        path / "eval.jsonl",
        path / "validation.jsonl",
        path / "test.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(
        file
        for file in path.glob("**/*")
        if file.is_file() and file.suffix in {".parquet", ".jsonl"} and "train" not in file.name.lower()
    )
    if not matches:
        raise FileNotFoundError(f"No evaluation file found under {path}")
    return matches[0].resolve()


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _normalize_math(root: Path, source: Path, target: Path, data_source: str, dry_run: bool) -> Path:
    if dry_run or not target.is_file() or source.stat().st_mtime > target.stat().st_mtime:
        _run(
            [
                sys.executable,
                str(root / "data/preprocess_math_eval.py"),
                "--input-file",
                str(source),
                "--output-file",
                str(target),
                "--data-source",
                data_source,
            ],
            dry_run,
        )
    return target.resolve() if not dry_run else target


def prepare(root: Path, output_dir: Path, dry_run: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = output_dir / "normalized"
    normalized.mkdir(exist_ok=True)

    math_train = root / "datasets/cl/math/DAPO-Math-17k/data/dapo-math-17k.parquet"
    science_train = [root / f"datasets/sciknoweval/{domain}/train.parquet" for domain in SCIENCE_DOMAINS]
    science_test = [root / f"datasets/sciknoweval/{domain}/test.parquet" for domain in SCIENCE_DOMAINS]
    tool_test = root / "datasets/tooluse/test.parquet"

    if dry_run:
        aime24_source = root / "datasets/cl/math/aime24/test.parquet"
        aime25_source = root / "datasets/cl/math/aime25/test.jsonl"
        math500_source = root / "datasets/cl/math/math500/test.parquet"
        gpqa_source = root / "datasets/cl/science/gpqa/gpqa_diamond.csv"
        lcb_source = root / "datasets/cl/code/LiveCodeBench-v6/test.parquet"
    else:
        required = [math_train, *science_train, *science_test, tool_test]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required prepared train/test parquet files are missing:\n  "
                + "\n  ".join(missing)
                + "\nPrepare the repository datasets before launching the causal experiment."
            )
        aime24_source = _find_eval_file(root / "datasets/cl/math/aime24")
        aime25_source = _find_eval_file(root / "datasets/cl/math/aime25")
        math500_source = _find_eval_file(root / "datasets/cl/math/math500")
        gpqa_source = root / "datasets/cl/science/gpqa/gpqa_diamond.csv"
        lcb_source = _find_eval_file(root / "datasets/cl/code/LiveCodeBench-v6")
        for source in (gpqa_source, lcb_source):
            if not source.is_file():
                raise FileNotFoundError(source)

    aime24 = _normalize_math(root, aime24_source, normalized / "aime24.parquet", "math", dry_run)
    aime25 = _normalize_math(root, aime25_source, normalized / "aime25.parquet", "math", dry_run)
    math500 = _normalize_math(root, math500_source, normalized / "math500.parquet", "math500", dry_run)
    gpqa = normalized / "gpqa_diamond.parquet"
    if dry_run or not gpqa.is_file() or gpqa_source.stat().st_mtime > gpqa.stat().st_mtime:
        _run(
            [
                sys.executable,
                str(root / "data/preprocess_gpqa.py"),
                "--csv-file",
                str(gpqa_source),
                "--output-file",
                str(gpqa),
            ],
            dry_run,
        )
    lcb = normalized / "livecodebench-v6.parquet"
    if dry_run or not lcb.is_file() or lcb_source.stat().st_mtime > lcb.stat().st_mtime:
        _run(
            [
                sys.executable,
                str(root / "data/preprocess_cl_code.py"),
                "--kind",
                "lcb",
                "--input-files",
                str(lcb_source),
                "--output-file",
                str(lcb),
            ],
            dry_run,
        )

    manifest = {
        "train_sample_cap": TRAIN_SAMPLE_CAP,
        "math_train": [str(math_train.resolve() if not dry_run else math_train)],
        "science_train": [str(path.resolve() if not dry_run else path) for path in science_train],
        "evaluation": {
            "aime24": str(aime24),
            "aime25": str(aime25),
            "math500": str(math500),
            "sciknoweval": [str(path.resolve() if not dry_run else path) for path in science_test],
            "gpqa": str(gpqa.resolve() if not dry_run else gpqa),
            "tooluse": str(tool_test.resolve() if not dry_run else tool_test),
            "livecodebench": str(lcb.resolve() if not dry_run else lcb),
        },
    }
    manifest_path = output_dir / "dataset_manifest.json"
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        from data.preflight_rl_dataset import validate_group

        validate_group("causal Math train", [Path(path) for path in manifest["math_train"]])
        validate_group("causal Science train", [Path(path) for path in manifest["science_train"]])
        eval_paths = []
        for value in manifest["evaluation"].values():
            eval_paths.extend(value if isinstance(value, list) else [value])
        validate_group("causal evaluation", [Path(path) for path in eval_paths])
        print(f"Wrote {manifest_path}")
    else:
        print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepare(args.project_root.resolve(), args.output_dir.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
