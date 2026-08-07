"""Convert paired BFCL data/possible_answer JSONL files to verl parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from data.preprocess import write_rowgrouped_large
from verl.utils.reward_score.feedback.bfcl import normalize_ground_truth


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            records = value
        elif isinstance(value, dict):
            records = [value]
        else:
            raise ValueError(f"Top-level JSON must be an object or list in {path}")
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every BFCL record must be an object in {path}")
    return records


def index_by_id(records: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    indexed = {}
    for record in records:
        sample_id = record.get("id")
        if not isinstance(sample_id, str):
            raise ValueError(f"Missing string id in {source}: {record}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate BFCL id {sample_id!r} in {source}")
        indexed[sample_id] = record
    return indexed


def render_question(question: Any) -> str:
    if isinstance(question, str):
        return question
    if not isinstance(question, list):
        raise ValueError(f"Unsupported BFCL question: {question}")
    lines = []
    for turn in question:
        messages = turn if isinstance(turn, list) else [turn]
        for message in messages:
            if not isinstance(message, dict) or "content" not in message:
                raise ValueError(f"Invalid BFCL message: {message}")
            role = str(message.get("role", "user")).capitalize()
            lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def render_prompt(question: Any, functions: Any) -> str:
    if not isinstance(functions, list) or not functions:
        raise ValueError("BFCL sample has no function definitions.")
    tool_docs = json.dumps(functions, ensure_ascii=False, indent=2)
    return (
        "Your task is to answer the user's question using the available functions.\n"
        "Available function definitions:\n"
        f"{tool_docs}\n\n"
        "For every required call, use this format:\n"
        "Action: <function name>\n"
        "Action Input: <JSON object>\n"
        "Repeat Action/Action Input for multiple calls.\n\n"
        f"{render_question(question)}"
    )


def convert_records(
    data_records: list[dict[str, Any]], answer_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    answers = index_by_id(answer_records, Path("possible_answer"))
    data_ids = index_by_id(data_records, Path("data"))
    missing = sorted(set(data_ids) - set(answers))
    extra = sorted(set(answers) - set(data_ids))
    if missing or extra:
        raise ValueError(f"BFCL data/answer id mismatch; missing answers={missing[:10]}, extra answers={extra[:10]}")

    converted = []
    for sample_id, sample in data_ids.items():
        ground_truth = normalize_ground_truth(answers[sample_id])
        prompt = render_prompt(sample.get("question"), sample.get("function"))
        converted.append(
            {
                "data_source": "bfcl",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "bfcl",
                "reward_model": {
                    "style": "bfcl",
                    "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
                },
                "extra_info": {
                    "split": "test",
                    "index": sample_id,
                    "category": sample_id.split("_", 1)[0],
                    "functions": sample.get("function"),
                },
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--answer-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    data_records = read_json_records(args.data_file)
    answer_records = read_json_records(args.answer_file)
    converted = convert_records(data_records, answer_records)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    write_rowgrouped_large(Dataset.from_list(converted), str(args.output_file))
    print(f"Wrote {len(converted)} BFCL evaluation samples to {args.output_file}")


if __name__ == "__main__":
    main()
