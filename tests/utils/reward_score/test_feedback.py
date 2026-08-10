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

    assert result == expected
    mock_compute_score.assert_called_once_with("Answer: 42", "42", {"split": "train"})
