"""Normalize Dolci's code subset and raw LiveCodeBench without executing tests."""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import pickle
import tempfile
import zlib
from collections import Counter
from pathlib import Path


class UnverifiableTestSuite(ValueError):
    """A syntactically valid suite with no observable correctness oracle."""


class UnsupportedTestSuite(ValueError):
    """A suite whose execution mode cannot be represented without guessing."""


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Executable pickle globals are forbidden in test payloads")

    def persistent_load(self, pid):
        raise ValueError("Persistent pickle references are forbidden")


def _decode_pickled_data(value):
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(base64.b64decode(value, validate=True), 64 * 1024 * 1024)
    if not decoder.eof or decoder.unconsumed_tail:
        raise ValueError("Oversized or incomplete private-test payload")
    return DataOnlyUnpickler(io.BytesIO(decoded)).load()


def decode_private_tests(value):
    # LCB stores a pickled JSON string, not executable Python test objects.
    payload = _decode_pickled_data(value)
    if not isinstance(payload, str):
        raise ValueError("Expected a JSON string inside private-test pickle")
    return json.loads(payload)


def decode_dolci_tests(value):
    try:
        return json.loads(value), False
    except json.JSONDecodeError:
        # Dolci's stdin subset also contains zlib/base64 encoded plain lists.
        payload = _decode_pickled_data(value)
        if not isinstance(payload, list):
            raise ValueError("Expected a test list inside Dolci test pickle")
        return payload, True


def _row(prompt, tests, source, split, index):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing prompt")
    if not tests["inputs"] or len(tests["inputs"]) != len(tests["outputs"]):
        raise ValueError("Empty or unaligned tests")
    return {
        "data_source": source,
        "ability": "code",
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"style": "code", "ground_truth": json.dumps(tests, ensure_ascii=False)},
        "extra_info": {
            "split": split,
            "index": str(index),
            "description": prompt,
            "problem": prompt,
            "elo": None,
            "achievement_prior": 0,
        },
    }


def _stdio_text(value, location):
    """Unwrap a single text payload without guessing multi-item list semantics."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise ValueError(f"{location}: expected text or a singleton text list, got {type(value).__name__}")


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _verification_signal(tree):
    """Recognize executable Python checks without running dataset code."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            return "assert"
        if isinstance(node, ast.Raise) and node.exc is not None:
            exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if _call_name(exception).endswith("AssertionError"):
                return "raise AssertionError"
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            leaf = name.rsplit(".", 1)[-1]
            if leaf.startswith("assert") or leaf == "raises":
                return name
    return None


def _assertion_suite(tests, index):
    """Keep dependent snippets in one namespace instead of inventing setup."""
    trees = [ast.parse(test) for test in tests]
    signals = [signal for tree in trees if (signal := _verification_signal(tree))]
    if not signals:
        raise UnverifiableTestSuite(f"{index}: test suite has no correctness oracle")
    independent = all(tree.body and all(isinstance(node, ast.Assert) for node in tree.body) for tree in trees)
    if independent:
        return tests, None
    # Definitions, imports, assignments and helper calls may carry shared state.
    suite = "\n\n".join(tests)
    ast.parse(suite)
    # Retain the original aggregate budget of one second per source test.
    return [suite], len(tests)


def format_dolci(row, index):
    labels = row.get("dataset")
    if labels not in (["code"], ["code_stdio"]):
        return None
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, list) or len(ground_truth) != 1:
        raise ValueError(f"{index}: expected exactly one ground_truth payload")
    tests, compressed = decode_dolci_tests(ground_truth[0])
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"{index}: expected nonempty test list")
    original_test_count = len(tests)
    time_limit = None
    if labels == ["code_stdio"]:
        if not all(isinstance(t, dict) for t in tests):
            raise ValueError(f"{index}: malformed stdin tests")
        if compressed and any(
            not isinstance(t.get(field), str) for t in tests for field in ("input", "output")
        ):
            raise UnsupportedTestSuite(
                f"{index}: compressed code_stdio suite has ambiguous stdin/functional list values"
            )
        inputs = [_stdio_text(t.get("input"), f"{index}: test {i} input") for i, t in enumerate(tests)]
        outputs = [_stdio_text(t.get("output"), f"{index}: test {i} output") for i, t in enumerate(tests)]
        testtype = "stdin"
    else:
        if not all(isinstance(t, str) and t.strip() for t in tests):
            raise ValueError(f"{index}: malformed assertion tests")
        inputs, time_limit = _assertion_suite(tests, index)
        outputs, testtype = [""] * len(inputs), "code"
    prompt = row["prompt"]
    if prompt.startswith("user: "):
        prompt = prompt[len("user: ") :]
    prompt += "\n\nReturn the complete Python solution in a ```python ... ``` code block."
    return _row(
        prompt,
        dict(
            inputs=inputs,
            outputs=outputs,
            testtype=testtype,
            fn_name="",
            time_limit=time_limit,
            original_test_count=original_test_count,
        ),
        "code",
        "train",
        index,
    )


def format_lcb(row, index):
    public = json.loads(row["public_test_cases"])
    private = decode_private_tests(row["private_test_cases"])
    tests = public + private
    if not tests or not all(isinstance(t, dict) for t in tests):
        raise ValueError(f"{index}: missing LCB tests")
    kinds = {t["testtype"] for t in tests}
    metadata = json.loads(row.get("metadata") or "{}")
    starter = row.get("starter_code") or ""
    prompt = row["question_content"]
    if starter.strip():
        prompt += "\n\nUse this interface:\n```python\n" + starter + "\n```"
    prompt += "\n\nReturn the complete Python solution in a ```python ... ``` code block."
    if kinds == {"stdin"}:
        inputs = [t["input"] for t in tests]
        outputs = [t["output"] for t in tests]
        if not all(isinstance(x, str) for x in inputs + outputs):
            raise ValueError(f"{index}: non-string stdin test")
        testtype = "stdin"
    elif kinds == {"functional"}:
        fn_name = metadata.get("func_name")
        if not isinstance(fn_name, str) or not fn_name.isidentifier():
            raise ValueError(f"{index}: missing functional test function name")
        target = f"Solution().{fn_name}" if "class Solution" in starter else fn_name
        inputs, outputs = [], []
        for test in tests:
            # LCB functional inputs encode one JSON argument per line.
            args = [json.loads(line) for line in test["input"].splitlines() if line.strip()]
            expected = json.loads(test["output"])
            code = f"import json\nassert json.loads(json.dumps({target}(*{args!r}))) == {expected!r}"
            ast.parse(code)
            inputs.append(code)
            outputs.append("")
        testtype = "code"
    else:
        raise ValueError(f"{index}: unsupported LCB test types {kinds}")
    return _row(
        prompt,
        dict(inputs=inputs, outputs=outputs, testtype=testtype, fn_name="", time_limit=6),
        "livecodebench",
        "test",
        row.get("question_id", index),
    )


def convert(input_files, output_file, kind):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output_file)
    sources = [Path(p) for p in input_files]
    if output.resolve() in [p.resolve() for p in sources]:
        raise ValueError("Refusing to overwrite source data")
    schema = pa.schema(
        [
            ("data_source", pa.string()),
            ("ability", pa.string()),
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [
                        ("split", pa.string()),
                        ("index", pa.string()),
                        ("description", pa.string()),
                        ("problem", pa.string()),
                        ("elo", pa.null()),
                        ("achievement_prior", pa.int64()),
                    ]
                ),
            ),
        ]
    )
    counts = Counter()
    max_tests = 0
    filtered_examples = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".code-convert-", dir=output.parent) as staging:
        temp = Path(staging) / "data.parquet"
        with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            buffer = []
            for source in sources:
                print(f"Converting {source}", flush=True)
                if kind == "dolci":
                    parquet = pq.ParquetFile(source)

                    # Skip all-non-code row groups before reading large test payloads.
                    def rows(parquet=parquet):
                        for group in range(parquet.num_row_groups):
                            labels = parquet.read_row_group(group, columns=["dataset"]).to_pylist()
                            if not any(r["dataset"] in (["code"], ["code_stdio"]) for r in labels):
                                counts["skipped_non_code"] += len(labels)
                                continue
                            for batch in parquet.iter_batches(
                                batch_size=32, row_groups=[group], columns=["dataset", "ground_truth", "prompt"]
                            ):
                                yield from batch.to_pylist()

                    iterator = rows()
                else:

                    def rows(source=source):
                        with source.open() as stream:
                            for line in stream:
                                if line.strip():
                                    yield json.loads(line)

                    iterator = rows()
                for index, row in enumerate(iterator):
                    try:
                        sample = (
                            format_dolci(row, f"{source.name}:{index}")
                            if kind == "dolci"
                            else format_lcb(row, index)
                        )
                    except (UnverifiableTestSuite, UnsupportedTestSuite) as error:
                        key = (
                            "skipped_unverifiable"
                            if isinstance(error, UnverifiableTestSuite)
                            else "skipped_unsupported_stdio"
                        )
                        counts[key] += 1
                        if len(filtered_examples) < 20:
                            filtered_examples.append(str(error))
                        continue
                    if sample is None:
                        counts["skipped_non_code"] += 1
                        continue
                    tests = json.loads(sample["reward_model"]["ground_truth"])
                    max_tests = max(max_tests, len(tests["inputs"]))
                    counts[tests["testtype"]] += 1
                    buffer.append(sample)
                    if len(buffer) >= 32:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                        buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        retained = counts["code"] + counts["stdin"]
        if retained == 0:
            raise ValueError("No usable code rows; no output published")
        temp.replace(output)
    report = dict(
        kind=kind,
        input_files=[str(p) for p in sources],
        output_file=str(output),
        counts=dict(counts),
        retained=retained,
        max_tests=max_tests,
    )
    if kind == "dolci":
        filtered_total = counts["skipped_unverifiable"] + counts["skipped_unsupported_stdio"]
        code_candidates = retained + filtered_total
        report.update(
            code_candidates=code_candidates,
            filtered_unverifiable=counts["skipped_unverifiable"],
            filtered_unverifiable_fraction=(
                counts["skipped_unverifiable"] / code_candidates if code_candidates else 0.0
            ),
            filtered_unsupported_stdio=counts["skipped_unsupported_stdio"],
            filtered_total=filtered_total,
            filtered_fraction=(filtered_total / code_candidates if code_candidates else 0.0),
            filtered_examples=filtered_examples,
        )
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("dolci", "lcb"), required=True)
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    convert(args.input_files, args.output_file, args.kind)


if __name__ == "__main__":
    main()
