"""Shared calculator semantics for GRPO rewards and held-out evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Literal

Number = int | float
NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"

ANSWER_PATTERNS = (
    re.compile(
        rf"(?:the\s+)?answer\s+is\s*[:=]?\s*\$?\s*({NUMBER_PATTERN})\s*\$?",
        re.I,
    ),
    re.compile(
        r"(?:the\s+)?(?:final\s+)?result(?:\s+of.*?)?\s+is\s*[:=]?\s*"
        rf"\$?\s*({NUMBER_PATTERN})\s*\$?",
        re.I | re.DOTALL,
    ),
    re.compile(rf"\\boxed\{{\s*({NUMBER_PATTERN})\s*\}}"),
    re.compile(
        rf"final(?:\s+answer)?\s*[:=]\s*\$?\s*({NUMBER_PATTERN})\s*\$?",
        re.I,
    ),
    re.compile(rf"^\s*\$?\s*({NUMBER_PATTERN})\s*\$?\s*\.?\s*$"),
)


def remove_null_fields(value: Any) -> Any:
    """Restore sparse nested dictionaries after Arrow materializes null fields."""
    if isinstance(value, dict):
        return {
            key: remove_null_fields(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, list):
        return [remove_null_fields(child) for child in value]
    return value


def calculate(op: str, a: Number, b: Number) -> Number:
    """Apply one supported numeric operation."""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"Unsupported calculator operation: {op}")


def parse_final_answer(text: str) -> Number | None:
    """Extract the last supported final-answer form from assistant text."""
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            value = float(matches[-1])
            return int(value) if value.is_integer() else value
    return None


def _operands_match(
    op: str, left: Number, right: Number, a: Number, b: Number
) -> bool:
    """Match ordered operands, allowing reversal for commutative operations."""
    return (left == a and right == b) or (
        op in {"+", "*"} and left == b and right == a
    )


def _classify_illegal_call(
    call: dict[str, Any],
    possible_states: set[tuple[Number | str, ...]],
    known_values: set[Number],
) -> str:
    """Classify why a call cannot reduce any currently possible state."""
    if any(len(tokens) == 1 for tokens in possible_states):
        return "post_completion_call"
    if call["op"] in {"+", "-"} and any("*" in tokens for tokens in possible_states):
        return "precedence_violation"

    required = Counter((call["a"], call["b"]))
    for tokens in possible_states:
        available = Counter(
            value for value in tokens[::2] if isinstance(value, (int, float))
        )
        if all(available[value] >= count for value, count in required.items()):
            return "branch_mismatch"

    if call["a"] in known_values and call["b"] in known_values:
        return "reused_operand"
    return "ungrounded_operand"


def score_trajectory(
    expression: str,
    calls: list[dict[str, Any]],
    stated_answer: Number | None,
    expected_call_count: int,
    correct_answer: Number,
) -> dict[str, Any]:
    """Score a precedence-respecting reduction trajectory."""
    pieces = re.findall(rf"{NUMBER_PATTERN}|[+*-]", expression)
    possible_states: set[tuple[Number | str, ...]] = {
        tuple(
            float(piece)
            if "." in piece
            else int(piece)
            if re.fullmatch(NUMBER_PATTERN, piece)
            else piece
            for piece in pieces
        )
    }
    known_values = {
        value
        for value in next(iter(possible_states))[::2]
        if isinstance(value, (int, float))
    }
    legal_calls = 0
    all_calls_legal = True
    first_illegal_call_index = None
    first_illegal_reason = None
    post_completion_call_count = 0

    for call_index, call in enumerate(calls):
        next_states: set[tuple[int | str, ...]] = set()
        for tokens in possible_states:
            multiplication_remains = "*" in tokens
            for index in range(1, len(tokens), 2):
                if tokens[index] != call["op"]:
                    continue
                if call["op"] in {"+", "-"} and (multiplication_remains or index != 1):
                    continue
                if not _operands_match(
                    call["op"],
                    tokens[index - 1],
                    tokens[index + 1],
                    call["a"],
                    call["b"],
                ):
                    continue
                value = calculate(call["op"], call["a"], call["b"])
                next_states.add(tokens[: index - 1] + (value,) + tokens[index + 2 :])
        if not next_states:
            all_calls_legal = False
            first_illegal_call_index = call_index + 1
            first_illegal_reason = _classify_illegal_call(
                call, possible_states, known_values
            )
            if first_illegal_reason == "post_completion_call":
                post_completion_call_count = len(calls) - call_index
            break
        legal_calls += 1
        known_values.add(calculate(call["op"], call["a"], call["b"]))
        possible_states = next_states

    fully_reduced = all_calls_legal and any(
        len(tokens) == 1 and tokens[0] == correct_answer for tokens in possible_states
    )
    call_count_correct = len(calls) == expected_call_count
    overcall_count = max(0, len(calls) - expected_call_count)
    valid_trajectory = all_calls_legal and call_count_correct and fully_reduced
    answer_correct = stated_answer == correct_answer if stated_answer is not None else False

    if stated_answer is None:
        reward, outcome = -0.1, "no_answer"
    elif valid_trajectory and answer_correct:
        reward, outcome = 2.0, "valid_correct"
    elif not valid_trajectory and not answer_correct:
        reward, outcome = -1.0, "invalid_incorrect"
    elif valid_trajectory:
        reward, outcome = 0.0, "valid_incorrect"
    else:
        reward, outcome = 0.0, "invalid_correct"

    return {
        "reward": reward,
        "outcome": outcome,
        "all_calls_legal": all_calls_legal,
        "legal_call_count": legal_calls,
        "fully_reduced": fully_reduced,
        "call_count_correct": call_count_correct,
        "overcall_count": overcall_count,
        "first_illegal_call_index": first_illegal_call_index,
        "first_illegal_reason": first_illegal_reason,
        "post_completion_call_count": post_completion_call_count,
        "valid_trajectory": valid_trajectory,
        "answer_correct": answer_correct,
        "stated_answer": stated_answer,
    }


def validate_semantic_trace(
    expression: str,
    calls: list[dict[str, Any]],
    final_answer: Number,
    expected_call_count: int,
) -> tuple[bool, bool]:
    """Return whether observed calls are legal and whether reduction is complete."""
    score = score_trajectory(
        expression=expression,
        calls=calls,
        stated_answer=final_answer,
        expected_call_count=expected_call_count,
        correct_answer=final_answer,
    )
    return bool(calls) and score["all_calls_legal"], score["fully_reduced"]


class CalculatorEnv:
    """Stateful TRL tool environment backed by the shared calculator semantics."""

    def reset(self, **kwargs: Any) -> None:
        self.id = kwargs["id"]
        self.expression = kwargs["expression"]
        self.correct_answer = kwargs["final_answer"]
        self.expected_call_count = int(kwargs["expected_call_count"])
        self.tier = kwargs["tier"]
        self.calls: list[dict[str, Any]] = []

    def calculator(self, op: Literal["+", "-", "*"], a: float, b: float) -> str:
        """Apply one arithmetic operation and record the call.

        Args:
            op: The arithmetic operation to apply.
            a: The numeric left operand.
            b: The numeric right operand.

        Returns:
            A JSON object containing the numeric result.
        """
        if type(a) not in {int, float} or type(b) not in {int, float}:
            raise TypeError("Calculator operands must be numeric.")
        value = calculate(op, a, b)
        self.calls.append({"op": op, "a": a, "b": b})
        return json.dumps({"result": value})
