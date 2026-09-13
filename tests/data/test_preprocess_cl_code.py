import base64
import json
import pickle
import zlib

import pytest

from data.preprocess_cl_code import (
    UnsupportedTestSuite,
    UnverifiableTestSuite,
    convert,
    decode_private_tests,
    format_dolci,
    format_lcb,
)


def encoded(tests):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


def encoded_object(value):
    return base64.b64encode(zlib.compress(pickle.dumps(value))).decode()


def test_dolci_filters_noncode_and_preserves_tests():
    assert format_dolci({"dataset": ["math"]}, 0) is None
    for label, payload, expected in (
        ("code_stdio", [{"input": "1 2\n", "output": "3\n"}], "stdin"),
        ("code", ["assert add(1, 2) == 3"], "code"),
    ):
        row = format_dolci(dict(dataset=[label], ground_truth=[json.dumps(payload)], prompt="user: Add"), 1)
        tests = json.loads(row["reward_model"]["ground_truth"])
        assert tests["testtype"] == expected
        assert len(tests["inputs"]) == 1
        assert row["extra_info"]["split"] == "train"
        assert row["data_source"] == "code"
        assert row["prompt"][0]["content"].startswith("Add")
        assert "assert add" not in row["prompt"][0]["content"]


def test_dolci_rejects_missing_tests():
    with pytest.raises(ValueError):
        format_dolci(dict(dataset=["code"], ground_truth=["[]"], prompt="q"), 0)
    with pytest.raises(UnverifiableTestSuite):
        format_dolci(dict(dataset=["code"], ground_truth=['["pass"]'], prompt="q"), 0)


@pytest.mark.parametrize(
    "check",
    [
        "self.assertEqual(candidate(), 1)",
        "with pytest.raises(ValueError):\n    candidate()",
        "if candidate() != 1:\n    raise AssertionError('wrong')",
    ],
)
def test_dolci_preserves_supported_non_assert_oracles(check):
    row = format_dolci(dict(dataset=["code"], ground_truth=[json.dumps([check])], prompt="q"), 0)
    tests = json.loads(row["reward_model"]["ground_truth"])
    assert tests["inputs"] == [check]


def test_dolci_mixed_singleton_lists_preserve_all_102_tests():
    payload = [{"input": ["123"], "output": ["123"], "type": "code"}]
    payload += [{"input": str(i), "output": f"{i}\n", "type": "code"} for i in range(101)]
    row = format_dolci(dict(dataset=["code_stdio"], ground_truth=[json.dumps(payload)], prompt="user: Echo"), 981)
    tests = json.loads(row["reward_model"]["ground_truth"])
    assert len(tests["inputs"]) == len(tests["outputs"]) == 102
    assert tests["inputs"] == ["123"] + [str(i) for i in range(101)]
    assert tests["outputs"] == ["123"] + [f"{i}\n" for i in range(101)]


def test_dolci_decodes_compressed_stdio_test_lists():
    payload = [{"input": "1 2\n", "output": "3\n"}]
    row = format_dolci(
        dict(dataset=["code_stdio"], ground_truth=[encoded_object(payload)], prompt="user: Add"), 1
    )
    tests = json.loads(row["reward_model"]["ground_truth"])
    assert tests["inputs"] == ["1 2\n"]
    assert tests["outputs"] == ["3\n"]


def test_dolci_rejects_ambiguous_compressed_stdio_lists():
    payload = [{"input": [[1, 2], [1]], "output": [[2]]}]
    with pytest.raises(UnsupportedTestSuite, match="ambiguous stdin/functional"):
        format_dolci(
            dict(dataset=["code_stdio"], ground_truth=[encoded_object(payload)], prompt="user: Difference"), 1
        )


@pytest.mark.parametrize("value", [[], ["one", "two"], [123], [["123"]], None, 123])
@pytest.mark.parametrize("field", ["input", "output"])
def test_dolci_rejects_ambiguous_stdio_payload(value, field):
    test = {"input": "123", "output": "123", field: value}
    with pytest.raises(ValueError, match=f"test 0 {field}"):
        format_dolci(dict(dataset=["code_stdio"], ground_truth=[json.dumps([test])], prompt="q"), 0)


def test_shared_helper_calls_form_one_ordered_suite():
    snippets = [
        "def run_case(x):\n    assert candidate(x) == x\nrun_case(1)",
        "run_case(2)",
        "run_case(3)",
    ]
    row = format_dolci(dict(dataset=["code"], ground_truth=[json.dumps(snippets)], prompt="q"), 2461)
    tests = json.loads(row["reward_model"]["ground_truth"])
    assert tests["inputs"] == ["\n\n".join(snippets)]
    assert tests["time_limit"] == 3
    assert tests["original_test_count"] == 3
    # Only authored synthetic code is executed in this test, never dataset code.
    exec(tests["inputs"][0], {"candidate": lambda x: x})
    with pytest.raises(AssertionError):
        exec(tests["inputs"][0], {"candidate": lambda x: x if x != 3 else -1})


def test_shared_state_and_imports_preserve_order():
    snippets = ["import math\nvalues = []", "values.append(candidate(4))", "assert values == [math.sqrt(4)]"]
    row = format_dolci(dict(dataset=["code"], ground_truth=[json.dumps(snippets)], prompt="q"), 0)
    tests = json.loads(row["reward_model"]["ground_truth"])
    assert len(tests["inputs"]) == 1
    exec(tests["inputs"][0], {"candidate": lambda x: 2})


def test_independent_assertions_remain_separate():
    snippets = ["assert f(1) == 1", "assert f(2) == 2"]
    row = format_dolci(dict(dataset=["code"], ground_truth=[json.dumps(snippets)], prompt="q"), 0)
    tests = json.loads(row["reward_model"]["ground_truth"])
    assert tests["inputs"] == snippets
    assert tests["time_limit"] is None


def lcb_row(test):
    return dict(
        question_content="q",
        question_id="q1",
        starter_code="",
        metadata="{}",
        public_test_cases=json.dumps([test]),
        private_test_cases=encoded([test]),
    )


def test_lcb_keeps_public_and_private_stdin_tests():
    row = format_lcb(lcb_row(dict(input="1\n", output="2\n", testtype="stdin")), 0)
    tests = json.loads(row["reward_model"]["ground_truth"])
    assert tests["inputs"] == ["1\n", "1\n"]
    assert tests["outputs"] == ["2\n", "2\n"]
    assert row["extra_info"]["split"] == "test"


def test_lcb_functional_preserves_spaces_in_json_arguments():
    raw = lcb_row(dict(input='"hello world"\n[1, 2]\n', output='["hello world", 3]', testtype="functional"))
    raw.update(
        starter_code="class Solution:\n    def solve(self, s, xs):\n        pass", metadata='{"func_name":"solve"}'
    )
    row = format_lcb(raw, 0)
    tests = json.loads(row["reward_model"]["ground_truth"])
    assert tests["testtype"] == "code"
    assert "Solution().solve" in tests["inputs"][0]
    assert "hello world" in tests["inputs"][0]


def test_private_pickle_cannot_import_or_execute_globals():
    value = base64.b64encode(zlib.compress(pickle.dumps(eval))).decode()
    with pytest.raises(ValueError, match="forbidden"):
        decode_private_tests(value)


def test_lcb_private_decode_error_includes_row_index():
    raw = lcb_row(dict(input="1\n", output="2\n", testtype="stdin"))
    raw["private_test_cases"] = "not-base64"
    with pytest.raises(ValueError, match=r"^17: invalid LCB private tests"):
        format_lcb(raw, 17)


def test_streaming_conversion_filters_and_reports(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "raw.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(dataset=["math"], ground_truth=["42"], prompt="user: math"),
                dict(dataset=["code"], ground_truth=['["assert f() == 1"]'], prompt="user: code"),
                dict(dataset=["code"], ground_truth=['["def f(): pass", "f()"]'], prompt="user: no oracle"),
                dict(
                    dataset=["code_stdio"],
                    ground_truth=[encoded_object([{"input": [[1]], "output": [[1]]}])],
                    prompt="user: ambiguous mode",
                ),
            ]
        ),
        source,
        row_group_size=1,
    )
    target = tmp_path / "converted.parquet"
    convert([source], target, "dolci")
    assert pq.ParquetFile(source).metadata.num_rows == 4
    assert pq.ParquetFile(target).metadata.num_rows == 1
    report = json.loads(target.with_suffix(".report.json").read_text())
    assert report["counts"] == {
        "skipped_non_code": 1,
        "code": 1,
        "skipped_unverifiable": 1,
        "skipped_unsupported_stdio": 1,
    }
    assert report["retained"] == 1
    assert report["code_candidates"] == 3
    assert report["filtered_unverifiable"] == 1
    assert report["filtered_unverifiable_fraction"] == pytest.approx(1 / 3)
    assert report["filtered_unsupported_stdio"] == 1
    assert report["filtered_total"] == 2
    assert report["filtered_fraction"] == pytest.approx(2 / 3)
    assert report["filtered_examples"] == [
        "raw.parquet:1: test suite has no correctness oracle",
        "raw.parquet:2: compressed code_stdio suite has ambiguous stdin/functional list values",
    ]


def test_bad_conversion_does_not_publish(tmp_path):
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(lcb_row(dict(input="", output="", testtype="unknown"))) + "\n")
    target = tmp_path / "output.parquet"
    with pytest.raises(ValueError):
        convert([source], target, "lcb")
    assert not target.exists()
