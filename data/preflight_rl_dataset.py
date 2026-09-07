"""Fast, model-free validation of files consumed by ``RLHFDataset``.

The production loader reads each file independently with Hugging Face
``datasets`` and then concatenates the resulting datasets.  This command does
the same thing without loading a tokenizer, starting Ray, or allocating a GPU,
so incompatible features are found before a long training curriculum starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets


def load_file(path: Path) -> datasets.Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    if path.suffix == ".parquet":
        loader = "parquet"
    elif path.suffix in {".json", ".jsonl"}:
        loader = "json"
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    return datasets.load_dataset(loader, data_files=str(path), split="train")


def validate_group(name: str, paths: list[Path]) -> int:
    if not paths:
        raise ValueError(f"{name} has no input files")

    loaded: list[datasets.Dataset] = []
    print(f"Checking {name}: {len(paths)} file(s)")
    for path in paths:
        dataset = load_file(path)
        loaded.append(dataset)
        print(f"  {path}: {len(dataset)} rows")

    try:
        combined = datasets.concatenate_datasets(loaded)
    except ValueError as error:
        print(f"\n{name} schemas are incompatible:")
        for path, dataset in zip(paths, loaded, strict=True):
            print(f"  {path}: {dataset.features}")
        raise ValueError(
            f"RLHFDataset cannot concatenate the files in {name}. "
            "Normalize conflicting columns to one type before training."
        ) from error

    required = {"prompt", "data_source", "reward_model"}
    missing = required.difference(combined.column_names)
    if missing:
        raise ValueError(f"{name} is missing required verl column(s): {', '.join(sorted(missing))}")
    if len(combined) == 0:
        raise ValueError(f"{name} is empty")
    print(f"  OK: {len(combined)} combined rows; features align")
    return len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Label shown in diagnostics")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    validate_group(args.name, args.files)


if __name__ == "__main__":
    main()
