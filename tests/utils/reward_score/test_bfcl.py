import json

from verl.utils.reward_score.feedback.bfcl import compute_score, normalize_ground_truth


def test_standard_bfcl_ground_truth_and_action_output() -> None:
    ground_truth = {
        "id": "multiple_0",
        "ground_truth": [
            {
                "triangle_properties.get": {
                    "side1": 5,
                    "side2": 4,
                    "side3": 3,
                }
            }
        ],
    }
    solution = (
        "Action: triangle_properties.get\n"
        'Action Input: {"side1": 5, "side2": 4, "side3": 3}'
    )

    result = compute_score(solution, json.dumps(ground_truth))

    assert result["acc"] == 1.0
    assert result["incorrect_format"] == 0


def test_multiple_calls_preserve_call_argument_binding() -> None:
    ground_truth = [
        {"weather.get": {"city": "Paris"}},
        {"weather.get": {"city": "Tokyo"}},
    ]
    correct = (
        "Action: weather.get\n"
        'Action Input: {"city": "Tokyo"}\n'
        "Action: weather.get\n"
        'Action Input: {"city": "Paris"}'
    )
    wrong = (
        "Action: weather.get\n"
        'Action Input: {"city": "Paris"}'
    )

    assert compute_score(correct, ground_truth)["acc"] == 1.0
    assert compute_score(wrong, ground_truth)["acc"] == 0.0


def test_normalize_string_function_call() -> None:
    calls = normalize_ground_truth(["triangle_properties.get(side1=5, side2=4, side3=3)"])

    assert calls == [
        {
            "name": "triangle_properties.get",
            "arguments": {"side1": 5, "side2": 4, "side3": 3},
        }
    ]
