"""Convert a local math benchmark file to the verl evaluation schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import datasets

from data.format.prompts import PROMPT
from data.preprocess import make_map_fn, write_rowgrouped_large


def _first_present(example: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = example.get(key)
        if value is not None:
            return value
    return None


def _normalize_answer(answer: Any) -> str:
    answer = str(answer).strip()
    if answer.startswith(r"\boxed{") and answer.endswith("}"):
        return answer[len(r"\boxed{") : -1]
    return answer


def format_math_eval(example: dict[str, Any], index: int, data_source: str) -> dict[str, Any]:
    problem = _first_present(example, ("problem", "question"))
    answer = _first_present(example, ("answer", "solution"))
    sample_id = _first_present(example, ("id", "idx"))

    if problem is None:
        raise ValueError(f"Sample {index} has neither a 'problem' nor a 'question' field")
    if answer is None:
        raise ValueError(f"Sample {index} has neither an 'answer' nor a 'solution' field")
    if sample_id is None:
        sample_id = index

    problem = str(problem)
    return {
        "prompt": PROMPT.format(problem=problem),
        "answer": _normalize_answer(answer),
        "idx": str(sample_id),
        "tests": None,
        "description": problem,
        "kind": "math",
        "dataset": data_source,
        "elo": "-",
        "system": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--data-source", choices=("math", "math500"), required=True)
    args = parser.parse_args()

    if args.input_file.suffix == ".parquet":
        loader = "parquet"
    elif args.input_file.suffix in {".json", ".jsonl"}:
        loader = "json"
    else:
        raise ValueError(f"Unsupported math evaluation format: {args.input_file}")

    source = datasets.load_dataset(loader, data_files=str(args.input_file), split="train")
    formatted = source.map(
        lambda example, index: format_math_eval(example, index, args.data_source),
        with_indices=True,
        remove_columns=source.column_names,
    )
    processed = formatted.map(make_map_fn("test"), with_indices=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output_file.with_suffix(args.output_file.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        write_rowgrouped_large(processed, str(temporary_output))
        temporary_output.replace(args.output_file)
    finally:
        temporary_output.unlink(missing_ok=True)
    print(f"Wrote {len(processed)} {args.data_source} samples to {args.output_file}")


if __name__ == "__main__":
    main()
