from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import tooluse


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "math_dapo", "math_dapo_reasoning", "gsm8k"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Reward style {data_source} not found.")

    # Validation can mix datasets whose scorers expose different auxiliary
    # fields. Keep truncation metadata present for every sample so the trainer
    # can align each metric with the corresponding reward.
    was_truncated = bool((extra_info or {}).get("truncated", False))
    results.setdefault("truncated", int(was_truncated))
    results.setdefault(
        "truncated_and_missing_answer",
        int(was_truncated and bool(results.get("incorrect_format", 0))),
    )
    return results
