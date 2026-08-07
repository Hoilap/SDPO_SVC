"""Convert the local GPQA Diamond CSV to the existing verl evaluation schema."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import datasets

from data.format.gpqa import PROMPT
from data.preprocess import make_map_fn, write_rowgrouped_large


def format_gpqa(example: dict, index: int, seed: int) -> dict:
    answers = [
        example["Correct Answer"],
        example["Incorrect Answer 1"],
        example["Incorrect Answer 2"],
        example["Incorrect Answer 3"],
    ]
    random.Random(f"{seed}:{index}").shuffle(answers)
    letters = ["A", "B", "C", "D"]
    options = ", ".join(f"{letter}) {answer}" for letter, answer in zip(letters, answers, strict=True))
    correct_letter = letters[answers.index(example["Correct Answer"])]
    question = example["Question"]
    return {
        "prompt": PROMPT.format(problem=question, options=options),
        "answer": correct_letter,
        "idx": index,
        "tests": None,
        "description": question,
        "kind": "gpqa",
        "dataset": "gpqa",
        "elo": "-",
        "system": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = datasets.load_dataset("csv", data_files=str(args.csv_file), split="train")
    formatted = source.map(
        lambda example, index: format_gpqa(example, index, args.seed),
        with_indices=True,
        remove_columns=source.column_names,
    )
    processed = formatted.map(make_map_fn("test"), with_indices=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    write_rowgrouped_large(processed, str(args.output_file))
    print(f"Wrote {len(processed)} GPQA Diamond samples to {args.output_file}")


if __name__ == "__main__":
    main()
