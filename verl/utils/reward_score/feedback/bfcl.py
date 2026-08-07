"""Scoring helpers for BFCL function-calling evaluation data."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any


def _function_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_function_name(node.value)}.{node.attr}"
    raise ValueError(f"Unsupported function expression: {ast.dump(node)}")


def _literal_call(text: str) -> dict[str, Any]:
    """Parse ``tool.name(arg=value)`` without evaluating arbitrary code."""

    expression = ast.parse(text.strip(), mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError(f"Not a function call: {text}")
    if expression.args:
        raise ValueError("BFCL string answers with positional arguments are unsupported.")
    arguments = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("BFCL **kwargs answers are unsupported.")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return {"name": _function_name(expression.func), "arguments": arguments}


def normalize_call(call: Any) -> dict[str, Any]:
    """Normalize common BFCL and OpenAI function-call representations."""

    if isinstance(call, str):
        stripped = call.strip()
        try:
            return normalize_call(json.loads(stripped))
        except json.JSONDecodeError:
            return _literal_call(stripped)

    if not isinstance(call, dict):
        raise ValueError(f"Function call must be a mapping or string, got {type(call).__name__}")

    if "name" in call or "function" in call:
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", call.get("arguments", {}))
        else:
            name = call.get("name", function)
            arguments = call.get("arguments", call.get("parameters", {}))
    elif "Action" in call:
        name = call["Action"]
        arguments = call.get("Action_Input", {})
    elif len(call) == 1:
        name, arguments = next(iter(call.items()))
    else:
        raise ValueError(f"Unrecognized BFCL call representation: {call}")

    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError(f"Invalid BFCL function name or arguments: {call}")
    return {"name": name, "arguments": arguments}


def normalize_ground_truth(value: Any) -> list[dict[str, Any]]:
    """Extract the expected call list from a BFCL answer record."""

    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        extracted = False
        for key in ("ground_truth", "possible_answer", "answer", "answers"):
            if key in value:
                value = value[key]
                extracted = True
                break
        if not extracted and "id" in value:
            value = {key: item for key, item in value.items() if key != "id"}
    if not isinstance(value, list):
        value = [value]
    return [normalize_call(call) for call in value]


def _decode_json_after(text: str, start: int) -> tuple[Any, int]:
    decoder = json.JSONDecoder()
    whitespace = len(text[start:]) - len(text[start:].lstrip())
    value, consumed = decoder.raw_decode(text[start + whitespace :])
    return value, start + whitespace + consumed


def extract_predicted_calls(text: str) -> list[dict[str, Any]]:
    """Extract Action/Action Input, Qwen tool_call, or JSON call outputs."""

    calls = []
    action_pattern = re.compile(r"Action:\s*([^\n]+)\s*\n\s*Action Input:\s*", re.IGNORECASE)
    for match in action_pattern.finditer(text):
        try:
            arguments, _ = _decode_json_after(text, match.end())
            calls.append(normalize_call({"name": match.group(1).strip(), "arguments": arguments}))
        except (json.JSONDecodeError, ValueError):
            continue
    if calls:
        return calls

    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL):
        try:
            calls.append(normalize_call(json.loads(block)))
        except (json.JSONDecodeError, ValueError):
            continue
    if calls:
        return calls

    # Accept a bare JSON object/list for models configured with JSON tool calls.
    try:
        decoded = json.loads(text.strip())
        if not isinstance(decoded, list):
            decoded = [decoded]
        return [normalize_call(call) for call in decoded]
    except (json.JSONDecodeError, ValueError):
        return []


def _canonical_call(call: dict[str, Any]) -> str:
    return json.dumps(call, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_score(solution: str, ground_truth: str | list | dict) -> dict[str, Any]:
    """Compare calls as a multiset, preserving duplicates and argument binding."""

    try:
        expected = normalize_ground_truth(ground_truth)
    except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "incorrect_format": 1,
            "feedback": f"Invalid BFCL ground truth: {exc}",
        }

    predicted = extract_predicted_calls(solution)
    expected_counter = Counter(_canonical_call(call) for call in expected)
    predicted_counter = Counter(_canonical_call(call) for call in predicted)
    correct = predicted_counter == expected_counter
    feedback = "" if correct else f"Predicted calls {predicted}; expected {expected}"
    return {
        "score": float(correct),
        "acc": float(correct),
        "pred": json.dumps(predicted, ensure_ascii=False),
        "incorrect_format": int(not predicted),
        "feedback": feedback,
    }
