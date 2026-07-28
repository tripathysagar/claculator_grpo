"""Validate GRPO environment/evaluator parity on fixed reference trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from calculator_semantics import (
    CalculatorEnv,
    parse_final_answer,
    remove_null_fields,
    score_trajectory,
    validate_semantic_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/calculator_qwen3")
    )
    parser.add_argument("--split", choices=("train", "eval", "test"), default="train")
    parser.add_argument(
        "--grpo-ids", type=Path, default=Path("data/calculator_qwen3_grpo600_ids.txt")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluations/grpo-environment-parity.json"),
    )
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message["tool_calls"][0]["function"]["arguments"]
        for message in record["messages"]
        if message["role"] == "assistant" and "tool_calls" in message
    ]


def expected_observations(record: dict[str, Any]) -> list[str]:
    return [
        message["content"]
        for message in record["messages"]
        if message["role"] == "tool"
    ]


def final_assistant_text(record: dict[str, Any]) -> str:
    return next(
        message["content"]
        for message in reversed(record["messages"])
        if message["role"] == "assistant" and "tool_calls" not in message
    )


def main() -> None:
    args = parse_args()
    from datasets import load_from_disk

    dataset = load_from_disk(str(args.dataset))[args.split]
    records = [remove_null_fields(dict(record)) for record in dataset]
    grpo_ids = {
        line.strip() for line in args.grpo_ids.read_text().splitlines() if line.strip()
    }
    candidates = sorted(
        (record for record in records if record["id"] in grpo_ids),
        key=lambda record: record["id"],
    )
    assert len(candidates) == 600
    selected = random.Random(args.seed).sample(candidates, args.rows)

    results = []
    for record in selected:
        calls = expected_calls(record)
        observations = expected_observations(record)
        final_answer = int(record["metadata"]["final_answer"])
        expected_count = int(record["metadata"]["operation_count"])

        env = CalculatorEnv()
        env.reset(
            id=record["id"],
            expression=record["expression"],
            final_answer=final_answer,
            expected_call_count=expected_count,
            tier=record["metadata"]["tier"],
        )
        actual_observations = [
            env.calculator(call["op"], call["a"], call["b"]) for call in calls
        ]
        stated_answer = parse_final_answer(final_assistant_text(record))
        reward_score = score_trajectory(
            expression=record["expression"],
            calls=env.calls,
            stated_answer=stated_answer,
            expected_call_count=expected_count,
            correct_answer=final_answer,
        )
        semantic_valid, semantic_complete = validate_semantic_trace(
            record["expression"], env.calls, final_answer, expected_count
        )

        assert actual_observations == observations
        assert stated_answer == final_answer
        assert semantic_valid and semantic_complete
        assert reward_score["outcome"] == "valid_correct"
        assert reward_score["reward"] == 2.0

        incomplete_calls = calls[:-1]
        _, incomplete_complete = validate_semantic_trace(
            record["expression"],
            incomplete_calls,
            final_answer,
            expected_count,
        )
        incomplete_score = score_trajectory(
            expression=record["expression"],
            calls=incomplete_calls,
            stated_answer=final_answer,
            expected_call_count=expected_count,
            correct_answer=final_answer,
        )
        assert not incomplete_complete
        assert incomplete_score["outcome"] == "invalid_correct"

        results.append(
            {
                "id": record["id"],
                "tier": record["metadata"]["tier"],
                "call_count": len(calls),
                "observations_match": True,
                "answer_parser_match": True,
                "semantic_valid": semantic_valid,
                "semantic_complete": semantic_complete,
                "reward_outcome": reward_score["outcome"],
                "omitted_final_call_rejected": True,
            }
        )

    payload = {
        "status": "passed",
        "seed": args.seed,
        "num_rows": len(results),
        "dataset": str(args.dataset),
        "dataset_split": args.split,
        "dataset_fingerprint": dataset._fingerprint,
        "grpo_ids_sha256": sha256(args.grpo_ids),
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": "passed", "rows": len(results)}))


if __name__ == "__main__":
    main()
