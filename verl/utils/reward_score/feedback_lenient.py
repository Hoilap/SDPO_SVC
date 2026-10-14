"""Opt-in evaluation scorer: any explicit correct answer counts, despite repetition.

Original feedback scorers remain untouched. Never enable this for training.
Multiple conflicting explicit answers can pass; candidate_count records this
optimistic diagnostic policy rather than a single-final-answer benchmark.
"""
import re
from verl.utils.reward_score.feedback import compute_score as strict_score
from verl.utils.reward_score.feedback import math as math_score


def boxed_candidates(text):
    """Read every complete box, ignoring a trailing unfinished box."""
    found = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                value = text[start:end].strip()
                if value and value not in found:
                    found.append(value)
                break
    return found


def choice_candidates(text):
    tagged = re.findall(r"<answer>\s*([A-D])\s*(?:</answer>|$)", text, re.I)
    labelled = re.findall(r"(?:final\s+answer|the\s+answer\s+is|answer)\s*[:=]?\s*\(?([A-D])\)?(?=[\s.,<]|$)", text, re.I)
    # A bare answer is explicit too; letters appearing in reasoning are not.
    bare = re.fullmatch(r"\s*([A-D])[.]?\s*", text, re.I)
    return list(dict.fromkeys(x.upper() for x in tagged + labelled + ([bare[1]] if bare else [])))


def equivalent(candidate, ground_truth):
    if candidate.strip() == str(ground_truth).strip():
        return True
    try:
        with math_score.timeout(seconds=5):
            gold = math_score.mv_parse(str(ground_truth))
            pred = math_score.mv_parse(r"\boxed{" + candidate + "}")
            return bool(gold and pred and math_score.mv_verify(gold, pred))
    except Exception:
        return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    extra_info = extra_info or {}
    if extra_info.get("split") == "train":
        raise ValueError("Lenient scoring is evaluation-only")
    result = dict(strict_score(data_source, solution_str, ground_truth, extra_info))
    result["strict_acc"] = result.get("acc", 0)
    result["strict_score"] = result.get("score", 0)
    result["strict_incorrect_format"] = result.get("incorrect_format", 0)
    candidates = []
    matched = None
    if data_source in ["math", "math500", "dapo_math", "math_dapo", "math_dapo_reasoning", "gsm8k"]:
        candidates = boxed_candidates(solution_str)
        matched = next((x for x in candidates if equivalent(x, ground_truth)), None)
    elif data_source in ["sciknoweval", "gpqa"]:
        candidates = choice_candidates(solution_str)
        matched = next((x for x in candidates if x == str(ground_truth).strip().upper()), None)
    if matched is not None:
        result.update(score=1.0, acc=1.0, pred=matched)
    # Keep original format/truncation flags: forgiving accuracy does not repair text.
    result["lenient_rescued"] = int(matched is not None and not result["strict_acc"])
    result["candidate_count"] = len(candidates)
    return result
