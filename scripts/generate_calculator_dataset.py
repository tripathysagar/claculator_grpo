from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from datasets import DatasetDict

Operator = Literal["+", "-", "*"]
Number = int | float

SYSTEM_PROMPT = """You are a calculator agent. You cannot do arithmetic yourself — you must
call the `calculator` tool for every operation. Respect standard operator
precedence (multiplication before addition/subtraction)."""

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Apply one arithmetic operation to two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["+", "-", "*"],
                    "description": "The arithmetic operation to apply.",
                },
                "a": {"type": "integer", "description": "The left operand."},
                "b": {"type": "integer", "description": "The right operand."},
            },
            "required": ["op", "a", "b"],
            "additionalProperties": False,
        },
    },
}
FLOAT_CALCULATOR_TOOL = copy.deepcopy(CALCULATOR_TOOL)
FLOAT_CALCULATOR_TOOL["function"]["parameters"]["properties"]["a"]["type"] = "number"
FLOAT_CALCULATOR_TOOL["function"]["parameters"]["properties"]["b"]["type"] = "number"


@dataclass(frozen=True)
class Expression:
    operands: tuple[Number, ...]
    operators: tuple[Operator, ...]

    def __post_init__(self) -> None:
        if len(self.operands) != len(self.operators) + 1:
            raise ValueError("an expression needs one more operand than operator")

    def render(self) -> str:
        text = str(self.operands[0])
        for op, operand in zip(self.operators, self.operands[1:]):
            text += f"*{operand}" if op == "*" else f" {op} {operand}"
        return text


@dataclass(frozen=True)
class CalculationStep:
    op: Operator
    a: Number
    b: Number
    result: Number
    stage: Literal["precedence", "combine"]


def calculate(op: Operator, a: Number, b: Number) -> Number:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"unsupported operator: {op}")


def naive_left_to_right_answer(expression: Expression) -> Number:
    result = expression.operands[0]
    for op, operand in zip(expression.operators, expression.operands[1:]):
        result = calculate(op, result, operand)
    return result


def precedence_trace(expression: Expression) -> tuple[list[CalculationStep], Number]:
    """Evaluate multiplication terms first, then addition/subtraction."""
    steps: list[CalculationStep] = []
    term_values: list[Number] = []
    combine_operators: list[Operator] = []
    current = expression.operands[0]

    for op, right in zip(expression.operators, expression.operands[1:]):
        if op == "*":
            result = calculate(op, current, right)
            steps.append(CalculationStep(op, current, right, result, "precedence"))
            current = result
        else:
            term_values.append(current)
            combine_operators.append(op)
            current = right
    term_values.append(current)

    result = term_values[0]
    for op, right in zip(combine_operators, term_values[1:]):
        combined = calculate(op, result, right)
        steps.append(CalculationStep(op, result, right, combined, "combine"))
        result = combined

    return steps, result


MULTIPLY_THOUGHTS = (
    "Multiplication has priority, so I'll resolve {a}*{b} first.",
    "Before addition or subtraction, I need the product {a}*{b}.",
    "The next precedence operation is {a}*{b}.",
)
COMBINE_THOUGHTS = {
    "+": (
        "The higher-precedence products are resolved. Now add {a} and {b}.",
        "Next, combine the resolved values with {a}+{b}.",
    ),
    "-": (
        "The higher-precedence products are resolved. Now subtract {b} from {a}.",
        "Next, combine the resolved values with {a}-{b}.",
    ),
}


def thought_for_step(step: CalculationStep, rng: random.Random) -> str:
    templates = (
        MULTIPLY_THOUGHTS
        if step.stage == "precedence"
        else COMBINE_THOUGHTS[step.op]
    )
    return rng.choice(templates).format(a=step.a, b=step.b)


def build_messages(
    expression: Expression,
    trace: Sequence[CalculationStep],
    final_answer: Number,
    rng: random.Random,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve: {expression.render()}"},
    ]

    for index, step in enumerate(trace, start=1):
        call_id = f"call_{index:02d}"
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>\n{thought_for_step(step, rng)}\n</think>",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": {"op": step.op, "a": step.a, "b": step.b},
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": "calculator",
                "tool_call_id": call_id,
                "content": json.dumps({"result": step.result}),
            }
        )

    messages.append(
        {
            "role": "assistant",
            "content": (
                "<think>\nAll operations are resolved in precedence order.\n</think>\n"
                f"The answer is {final_answer}."
            ),
        }
    )
    return messages


def random_candidate(rng: random.Random) -> Expression:
    operation_count = rng.randint(2, 5)
    operands = tuple(rng.randint(2, 20) for _ in range(operation_count + 1))
    operators = tuple(
        rng.choices(("+", "-", "*"), weights=(3, 2, 4), k=operation_count)
    )
    return Expression(operands, operators)


def random_length_ood_candidate(rng: random.Random) -> Expression:
    """Sample expressions longer than the two-to-five-operation ID range."""
    operation_count = rng.randint(6, 8)
    operands = tuple(rng.randint(2, 20) for _ in range(operation_count + 1))
    operators = tuple(
        rng.choices(("+", "-", "*"), weights=(3, 2, 4), k=operation_count)
    )
    return Expression(operands, operators)


def random_negative_ood_candidate(rng: random.Random) -> Expression:
    """Sample ID-length expressions containing at least one negative operand."""
    operation_count = rng.randint(2, 5)
    operands = [rng.choice((*range(-20, -1), *range(2, 21))) for _ in range(operation_count + 1)]
    operands[rng.randrange(len(operands))] = -rng.randint(2, 20)
    operators = tuple(
        rng.choices(("+", "-", "*"), weights=(3, 2, 4), k=operation_count)
    )
    return Expression(tuple(operands), operators)


def random_float_ood_candidate(rng: random.Random) -> Expression:
    """Sample ID-length expressions with positive half-integer operands."""
    operation_count = rng.randint(2, 5)
    numerators = [rng.randint(4, 40) for _ in range(operation_count + 1)]
    numerators[rng.randrange(len(numerators))] |= 1
    operands = tuple(numerator / 2 for numerator in numerators)
    operators = tuple(
        rng.choices(("+", "-", "*"), weights=(3, 2, 4), k=operation_count)
    )
    return Expression(operands, operators)


def expression_structure(expression: Expression) -> dict[str, int | str]:
    factor_counts = [1]
    for op in expression.operators:
        if op == "*":
            factor_counts[-1] += 1
        else:
            factor_counts.append(1)

    operation_count = len(expression.operators)
    if operation_count == 2:
        tier = "easy"
    elif operation_count <= 4:
        tier = "medium"
    else:
        tier = "hard"

    return {
        "tier": tier,
        "num_terms": len(factor_counts),
        "max_factors_in_a_term": max(factor_counts),
    }


def is_useful_candidate(expression: Expression) -> bool:
    # A precedence-sensitive example needs both precedence levels.
    if "*" not in expression.operators:
        return False
    if not ({"+", "-"} & set(expression.operators)):
        return False

    trace, final_answer = precedence_trace(expression)
    naive_answer = naive_left_to_right_answer(expression)
    return (
        naive_answer != final_answer
        and abs(final_answer) <= 100_000
        and all(abs(step.result) <= 100_000 for step in trace)
    )


def validate_record(record: dict[str, Any]) -> None:
    metadata = record["metadata"]
    if metadata["naive_left_to_right_answer"] == metadata["final_answer"]:
        raise AssertionError("record is not sensitive to operator precedence")
    if "(" in record["expression"] or ")" in record["expression"]:
        raise AssertionError("parentheses are not allowed")

    rendered_terms = record["expression"].replace(" - ", " + ").split(" + ")
    factor_counts = [len(term.split("*")) for term in rendered_terms]
    if metadata["num_terms"] != len(rendered_terms):
        raise AssertionError("num_terms metadata is incorrect")
    if metadata["max_factors_in_a_term"] != max(factor_counts):
        raise AssertionError("max_factors_in_a_term metadata is incorrect")
    expected_tier = (
        "easy"
        if metadata["operation_count"] == 2
        else "medium"
        if metadata["operation_count"] <= 4
        else "hard"
    )
    if metadata["tier"] != expected_tier:
        raise AssertionError("tier metadata is incorrect")

    messages = record["messages"]
    tool_assistants = [
        message
        for message in messages
        if message["role"] == "assistant" and "tool_calls" in message
    ]
    tool_results = [message for message in messages if message["role"] == "tool"]
    if len(tool_assistants) != metadata["operation_count"]:
        raise AssertionError("every operation must have one tool call")
    if len(tool_results) != len(tool_assistants):
        raise AssertionError("every tool call must have one result")

    for assistant, tool_result in zip(tool_assistants, tool_results):
        tool_call = assistant["tool_calls"][0]
        arguments = tool_call["function"]["arguments"]
        expected = calculate(arguments["op"], arguments["a"], arguments["b"])
        actual = json.loads(tool_result["content"])["result"]
        if expected != actual:
            raise AssertionError("tool result does not match its arguments")
        if tool_result["tool_call_id"] != tool_call["id"]:
            raise AssertionError("tool result references the wrong call")

    answer_text = messages[-1]["content"]
    if not answer_text.endswith(f"The answer is {metadata['final_answer']}."):
        raise AssertionError("final assistant answer is incorrect")


def build_record(
    expression: Expression,
    *,
    record_id: str,
    rng: random.Random,
    split: str | None = None,
    ood_type: str | None = None,
) -> dict[str, Any]:
    trace, final_answer = precedence_trace(expression)
    metadata: dict[str, Any] = {
        "final_answer": final_answer,
        "naive_left_to_right_answer": naive_left_to_right_answer(expression),
        "operation_count": len(expression.operators),
        "tool_call_count": len(trace),
        **expression_structure(expression),
    }
    if split is not None:
        metadata["split"] = split
    if ood_type is not None:
        metadata["ood_type"] = ood_type
        metadata["source_operands"] = list(expression.operands)

    record = {
        "id": record_id,
        "expression": expression.render(),
        "messages": build_messages(expression, trace, final_answer, rng),
        "tools": [
            FLOAT_CALCULATOR_TOOL if ood_type == "float" else CALCULATOR_TOOL
        ],
        "metadata": metadata,
    }
    validate_record(record)
    return record


def generate_records(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0

    while len(records) < count:
        attempts += 1
        if attempts > count * 1_000:
            raise RuntimeError("unable to generate enough unique expressions")

        expression = random_candidate(rng)
        rendered = expression.render()
        if rendered in seen or not is_useful_candidate(expression):
            continue

        record = build_record(
            expression,
            record_id=f"calculator_{len(records):04d}",
            rng=rng,
        )
        seen.add(rendered)
        records.append(record)

    return records


def generate_ood_splits(
    count: int,
    seed: int,
    *,
    seen_expressions: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Generate isolated length, negative-number, and floating-point OOD sets."""
    candidate_factories = {
        "length": random_length_ood_candidate,
        "negative": random_negative_ood_candidate,
        "float": random_float_ood_candidate,
    }
    splits: dict[str, list[dict[str, Any]]] = {}
    for offset, (ood_type, candidate_factory) in enumerate(
        candidate_factories.items()
    ):
        rng = random.Random(seed + offset)
        rows: list[dict[str, Any]] = []
        attempts = 0
        while len(rows) < count:
            attempts += 1
            if attempts > count * 2_000:
                raise RuntimeError(f"unable to generate enough {ood_type} OOD rows")
            expression = candidate_factory(rng)
            rendered = expression.render()
            if rendered in seen_expressions or not is_useful_candidate(expression):
                continue
            split_name = f"ood_{ood_type}"
            record = build_record(
                expression,
                record_id=f"calculator_{split_name}_{len(rows):04d}",
                rng=rng,
                split=split_name,
                ood_type=ood_type,
            )
            seen_expressions.add(rendered)
            rows.append(record)
        splits[f"ood_{ood_type}"] = rows
    return splits


def validate_ood_splits(
    splits: dict[str, list[dict[str, Any]]], expected_count: int
) -> None:
    expected_names = {"ood_length", "ood_negative", "ood_float"}
    if set(splits) != expected_names:
        raise AssertionError(f"unexpected OOD splits: {set(splits)}")
    if any(len(rows) != expected_count for rows in splits.values()):
        raise AssertionError("OOD split sizes do not match --ood-count")

    for split_name, rows in splits.items():
        if not all(row["metadata"]["split"] == split_name for row in rows):
            raise AssertionError(f"{split_name} contains incorrect split metadata")
        if not all(
            row["metadata"]["ood_type"] == split_name.removeprefix("ood_")
            for row in rows
        ):
            raise AssertionError(f"{split_name} contains incorrect OOD metadata")

    if not all(
        6 <= row["metadata"]["operation_count"] <= 8
        for row in splits["ood_length"]
    ):
        raise AssertionError("length OOD must contain six-to-eight operations")
    if not all(
        any(operand < 0 for operand in row["metadata"]["source_operands"])
        for row in splits["ood_negative"]
    ):
        raise AssertionError("negative OOD must contain a negative operand")
    if not all(
        any(
            isinstance(operand, float) and not operand.is_integer()
            for operand in row["metadata"]["source_operands"]
        )
        for row in splits["ood_float"]
    ):
        raise AssertionError("float OOD must contain a non-integer operand")


def verify_qwen3_chat_template(
    records: Sequence[dict[str, Any]], model_name: str
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for record in records:
        rendered = tokenizer.apply_chat_template(
            record["messages"],
            tools=record["tools"],
            tokenize=False,
        )
        expected_calls = record["metadata"]["tool_call_count"]
        # Qwen3's system instructions contain a sample <tool_call> pair, so
        # count only calls whose payload names our concrete calculator tool.
        concrete_call = '<tool_call>\n{"name": "calculator"'
        if rendered.count(concrete_call) != expected_calls:
            raise AssertionError("Qwen3 template omitted a tool call")
        if rendered.count("<tool_response>") != expected_calls:
            raise AssertionError("Qwen3 template omitted a tool response")
        if f"The answer is {record['metadata']['final_answer']}." not in rendered:
            raise AssertionError("Qwen3 template omitted the final answer")


def write_jsonl(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_hf_dataset(
    splits: dict[str, list[dict[str, Any]]],
) -> DatasetDict:
    """Convert generated splits to a native Hugging Face DatasetDict."""
    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            split_name: Dataset.from_list(split_rows)
            for split_name, split_rows in splits.items()
        }
    )


def _proportional_quotas(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if not 0 <= target <= total:
        raise ValueError("split target must fit within the available records")
    if total == 0:
        return {name: 0 for name in group_sizes}

    exact = {name: size * target / total for name, size in group_sizes.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = target - sum(quotas.values())
    priority = sorted(
        group_sizes,
        key=lambda name: (exact[name] - quotas[name], group_sizes[name], name),
        reverse=True,
    )
    for name in priority[:remainder]:
        quotas[name] += 1
    return quotas


def split_records(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float = 0.8,
    eval_fraction: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic, tier-stratified train/eval/test splits."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    if train_fraction + eval_fraction >= 1:
        raise ValueError("train and eval fractions must leave room for test")

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {
        "easy": [],
        "medium": [],
        "hard": [],
    }
    for record in records:
        grouped[record["metadata"]["tier"]].append(record)
    for group in grouped.values():
        rng.shuffle(group)

    total = len(records)
    train_target = round(total * train_fraction)
    eval_target = round(total * eval_fraction)
    train_quotas = _proportional_quotas(
        {tier: len(group) for tier, group in grouped.items()}, train_target
    )
    remaining_sizes = {
        tier: len(group) - train_quotas[tier] for tier, group in grouped.items()
    }
    eval_quotas = _proportional_quotas(remaining_sizes, eval_target)

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "eval": [],
        "test": [],
    }
    for tier, group in grouped.items():
        train_end = train_quotas[tier]
        eval_end = train_end + eval_quotas[tier]
        splits["train"].extend(group[:train_end])
        splits["eval"].extend(group[train_end:eval_end])
        splits["test"].extend(group[eval_end:])

    for split_name, split_rows in splits.items():
        rng.shuffle(split_rows)
        for row in split_rows:
            row["metadata"]["split"] = split_name

    expected_sizes = {
        "train": train_target,
        "eval": eval_target,
        "test": total - train_target - eval_target,
    }
    if {name: len(rows) for name, rows in splits.items()} != expected_sizes:
        raise AssertionError("split sizes do not match their targets")
    split_ids = [row["id"] for rows in splits.values() for row in rows]
    if len(split_ids) != len(set(split_ids)) or len(split_ids) != total:
        raise AssertionError("splits overlap or omit records")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate precedence-sensitive calculator trajectories."
    )
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--ood-seed", type=int, default=20260721)
    parser.add_argument(
        "--ood-count",
        type=int,
        default=100,
        help="Number of rows in each isolated OOD split.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/calculator_qwen3"),
        help="Directory for the Hugging Face DatasetDict produced by save_to_disk.",
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("data"),
        help="Also export backward-compatible JSONL files to this directory.",
    )
    parser.add_argument(
        "--export-jsonl",
        action="store_true",
        help="Also write legacy JSONL files; disabled by default.",
    )
    parser.add_argument(
        "--hub-repo",
        help="Optionally push the DatasetDict to this Hugging Face Hub repository.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update a private Hub dataset when --hub-repo is set.",
    )
    parser.add_argument(
        "--verify-model",
        default="Qwen/Qwen3-0.6B",
        help="Qwen3 tokenizer whose real chat template validates every record.",
    )
    parser.add_argument(
        "--skip-template-verification",
        action="store_true",
        help="Skip loading the tokenizer (useful only in offline environments).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.ood_count < 1:
        raise ValueError("--ood-count must be positive")

    records = generate_records(args.count, args.seed)
    splits = split_records(
        records,
        seed=args.split_seed,
        train_fraction=args.train_fraction,
        eval_fraction=args.eval_fraction,
    )
    ood_splits = generate_ood_splits(
        args.ood_count,
        args.ood_seed,
        seen_expressions={record["expression"] for record in records},
    )
    validate_ood_splits(ood_splits, args.ood_count)
    all_splits = {**splits, **ood_splits}
    if not args.skip_template_verification:
        verify_qwen3_chat_template(
            [*records, *(row for rows in ood_splits.values() for row in rows)],
            args.verify_model,
        )

    dataset = build_hf_dataset(all_splits)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    if args.hub_repo:
        dataset.push_to_hub(args.hub_repo, private=args.private)

    if args.export_jsonl:
        write_jsonl(records, args.jsonl_dir / "calculator_qwen3_1000.jsonl")
        for split_name, split_rows in all_splits.items():
            write_jsonl(
                split_rows,
                args.jsonl_dir / f"calculator_qwen3_{split_name}.jsonl",
            )

    operation_counts = {
        count: sum(row["metadata"]["operation_count"] == count for row in records)
        for count in range(2, 6)
    }
    tier_counts = {
        tier: sum(row["metadata"]["tier"] == tier for row in records)
        for tier in ("easy", "medium", "hard")
    }
    print(
        f"Saved Hugging Face DatasetDict with {len(records) + 3 * args.ood_count:,} "
        "validated records "
        f"to {args.output_dir}"
    )
    print(dataset)
    if args.export_jsonl:
        print(f"Exported legacy JSONL files to {args.jsonl_dir}")
    if args.hub_repo:
        visibility = "private" if args.private else "public"
        print(f"Pushed {visibility} dataset to https://huggingface.co/datasets/{args.hub_repo}")
    print(f"Operation-count distribution: {operation_counts}")
    print(f"Tier distribution: {tier_counts}")
    for split_name, split_rows in splits.items():
        split_tiers = {
            tier: sum(row["metadata"]["tier"] == tier for row in split_rows)
            for tier in ("easy", "medium", "hard")
        }
        print(f"{split_name}: {len(split_rows):,} rows, tiers={split_tiers}")
    for split_name, split_rows in ood_splits.items():
        operation_range = (
            min(row["metadata"]["operation_count"] for row in split_rows),
            max(row["metadata"]["operation_count"] for row in split_rows),
        )
        print(f"{split_name}: {len(split_rows):,} rows, operations={operation_range}")
    if not args.skip_template_verification:
        print(f"Qwen3 chat-template validation passed with {args.verify_model}")


if __name__ == "__main__":
    main()
