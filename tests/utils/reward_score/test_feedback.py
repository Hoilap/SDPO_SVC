from unittest.mock import patch

import pytest

from verl.utils.reward_score.feedback import compute_score


@pytest.mark.parametrize("data_source", ["math_dapo", "math_dapo_reasoning"])
def test_math_dapo_aliases_use_feedback_math_score(data_source):
    expected = {"score": 1.0, "feedback": ""}

    with patch("verl.utils.reward_score.feedback.math.compute_score", return_value=expected) as mock_compute_score:
        result = compute_score(
            data_source=data_source,
            solution_str="Answer: 42",
            ground_truth="42",
            extra_info={"split": "train"},
        )

    assert result == {
        **expected,
        "error_in_test_cases": 0,
        "timed_out": 0,
        "truncated": 0,
        "truncated_and_missing_answer": 0,
    }
    mock_compute_score.assert_called_once_with("Answer: 42", "42", {"split": "train"})


@pytest.mark.parametrize("data_source", ["gpqa", "sciknoweval"])
def test_non_math_scores_include_truncation_metadata(data_source):
    scorer = "gpqa" if data_source == "gpqa" else "mcq"
    expected = {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "incorrect_format": 1,
        "feedback": "",
    }

    with patch(f"verl.utils.reward_score.feedback.{scorer}.compute_score", return_value=expected):
        result = compute_score(
            data_source=data_source,
            solution_str="",
            ground_truth="A",
            extra_info={"truncated": True},
        )

    assert result["truncated"] == 1
    assert result["truncated_and_missing_answer"] == 1
    assert result["error_in_test_cases"] == 0
    assert result["timed_out"] == 0


@pytest.mark.parametrize(
    ("data_source", "scorer"),
    [
        ("math", "math"),
        ("gpqa", "gpqa"),
        ("sciknoweval", "mcq"),
        ("tooluse", "tooluse"),
    ],
)
def test_non_code_scores_fill_code_only_metrics(data_source, scorer):
    with patch(
        f"verl.utils.reward_score.feedback.{scorer}.compute_score",
        return_value={"score": 0.0},
    ):
        result = compute_score(data_source, "response", "answer", {})

    assert result["error_in_test_cases"] == 0
    assert result["timed_out"] == 0


def test_code_scores_preserve_real_test_error_metrics():
    expected = {"score": 0.0, "error_in_test_cases": 1, "timed_out": 1}
    with patch("verl.utils.reward_score.feedback.code.compute_score", return_value=expected):
        result = compute_score("code", "response", "tests", {})

    assert result["error_in_test_cases"] == 1
    assert result["timed_out"] == 1


def test_mcq_incorrect_format_flag():
    valid = compute_score(
        data_source="sciknoweval",
        solution_str="<answer>A</answer>",
        ground_truth="A",
        extra_info={"truncated": False},
    )
    invalid = compute_score(
        data_source="sciknoweval",
        solution_str="A",
        ground_truth="A",
        extra_info={"truncated": False},
    )

    assert valid["incorrect_format"] == 0
    assert invalid["incorrect_format"] == 1
