"""Evaluate a pinned calculator adapter after merging it into Qwen3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import transformers
import calculator_semantics
from calculator_semantics import (
    calculate,
    parse_final_answer,
    remove_null_fields,
    validate_semantic_trace,
)
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
METRIC_FIELDS = (
    "answer_correct",
    "used_tool",
    "valid_tool_use",
    "parseable_completion",
    "tool_call_count_correct",
    "semantic_trace_valid",
    "semantic_trace_complete",
    "exact_trace",
    "task_success",
    "thinking_budget_exhausted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--base-revision", default="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    parser.add_argument(
        "--adapter", default="tripathysagar/qwen3-0.6b-calc-sft200"
    )
    parser.add_argument(
        "--adapter-revision", default="86db06c14d91acc734e89a45bb5ca3ec4e1ee8f3"
    )
    parser.add_argument(
        "--tokenizer", default="tripathysagar/qwen3-0.6b-calc-sft200"
    )
    parser.add_argument(
        "--tokenizer-revision",
        default="86db06c14d91acc734e89a45bb5ca3ec4e1ee8f3",
    )
    parser.add_argument(
        "--policy-name",
        default="sft",
        help="Policy label recorded in artifacts and progress output.",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the pinned base model with the identical protocol.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling for the hardware-controlled primary benchmark.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("/content/data/calculator_qwen3")
    )
    parser.add_argument(
        "--split",
        choices=(
            "eval",
            "test",
            "ood_length",
            "ood_negative",
            "ood_float",
        ),
        default="eval",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/experiments/qwen3-calc-sft200-eval"),
    )
    parser.add_argument("--seeds", default="20260719,20260720,20260721")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_tool_call(text: str) -> tuple[dict[str, Any] | None, str | None]:
    matches = TOOL_TAG_RE.findall(text)
    if not matches:
        return (None, "unclosed_tool_call") if "<tool_call>" in text else (None, None)
    if len(matches) != 1:
        return None, "multiple_tool_calls_in_one_turn"
    try:
        payload = json.loads(matches[0])
        arguments = payload["arguments"]
        if payload["name"] != "calculator":
            return None, "wrong_tool_name"
        if set(arguments) != {"op", "a", "b"}:
            return None, "wrong_argument_schema"
        if arguments["op"] not in {"+", "-", "*"}:
            return None, "invalid_operator"
        if (
            type(arguments["a"]) not in {int, float}
            or type(arguments["b"]) not in {int, float}
        ):
            return None, "non_numeric_operand"
        return {"name": "calculator", "arguments": arguments}, None
    except (KeyError, TypeError, json.JSONDecodeError):
        return None, "malformed_tool_call"


def expected_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message["tool_calls"][0]["function"]["arguments"]
        for message in record["messages"]
        if message["role"] == "assistant" and "tool_calls" in message
    ]


def rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        name: round(mean(float(row[name]) for row in rows), 4)
        for name in METRIC_FIELDS
    }


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    args = parse_args()
    seeds = tuple(int(seed) for seed in args.seeds.split(","))
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be injected before loading the private adapter.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")
    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)

    from datasets import load_from_disk

    dataset = load_from_disk(str(args.dataset))[args.split]
    records = [remove_null_fields(dict(record)) for record in dataset]
    assert records and all(
        row["metadata"]["split"] == args.split for row in records
    ), f"Dataset rows do not all belong to requested split {args.split!r}"
    if args.max_examples is not None:
        records = records[: args.max_examples]
    tool_budget = max(
        6, max(int(record["metadata"]["operation_count"]) for record in records)
    )
    selected_sha = sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    )

    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision, token=token
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        token=token,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if args.no_adapter:
        model = base_model
    else:
        peft_model = PeftModel.from_pretrained(
            base_model, args.adapter, revision=args.adapter_revision, token=token
        )
        model = peft_model.merge_and_unload(safe_merge=True)
        del peft_model
    model.eval()
    torch.cuda.empty_cache()
    assert not isinstance(model, PeftModel), "PEFT wrapper remains during inference."

    device_name = torch.cuda.get_device_name(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    initial_prompts = [
        tokenizer.apply_chat_template(
            record["messages"][:2],
            tools=record["tools"],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
        for record in records
    ]
    rendered_prompts_sha256 = sha256_bytes(
        json.dumps(
            initial_prompts, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    )
    config = {
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "policy": "base" if args.no_adapter else args.policy_name,
        "adapter": None if args.no_adapter else args.adapter,
        "adapter_revision": None if args.no_adapter else args.adapter_revision,
        "lora_merged_for_inference": not args.no_adapter,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "rendered_initial_prompts_sha256": rendered_prompts_sha256,
        "semantic_module_sha256": sha256_bytes(
            Path(calculator_semantics.__file__).read_bytes()
        ),
        "dataset": str(args.dataset),
        "dataset_fingerprint": dataset._fingerprint,
        "selected_records_sha256": selected_sha,
        "split": args.split,
        "num_examples": len(records),
        "seeds": list(seeds),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_tool_rounds": tool_budget,
        "max_tool_calls": tool_budget,
        "enable_thinking": True,
        "do_sample": not args.greedy,
        "temperature": None if args.greedy else 0.6,
        "top_p": None if args.greedy else 0.95,
        "top_k": None if args.greedy else 20,
    }
    write_json_atomic(args.output_dir / "config.json", config)

    def render_prompt(state: dict[str, Any]) -> str:
        return tokenizer.apply_chat_template(
            state["messages"],
            tools=state["tools"],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

    def generate_turns(prompts: list[str]) -> list[dict[str, Any]]:
        inputs = tokenizer(
            prompts,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_width = inputs["input_ids"].shape[1]
        generation_config: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": not args.greedy,
            "use_cache": True,
            "stop_strings": ["</tool_call>"],
            "tokenizer": tokenizer,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if not args.greedy:
            generation_config.update(temperature=0.6, top_p=0.95, top_k=20)
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_config)
        generated = output[:, input_width:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        turn_outputs = []
        for token_row, text in zip(generated, texts):
            token_ids = token_row.tolist()
            complete_tool_call = "</tool_call>" in text
            try:
                first_eos = token_ids.index(tokenizer.eos_token_id)
                count = first_eos if complete_tool_call else first_eos + 1
            except ValueError:
                count = len(token_ids)
            turn_outputs.append(
                {
                    "text": text,
                    "generated_token_count": count,
                    "thinking_budget_exhausted": (
                        count >= args.max_new_tokens and not complete_tool_call
                    ),
                }
            )
        return turn_outputs

    def new_state(record: dict[str, Any], seed: int) -> dict[str, Any]:
        return {
            "record": record,
            "seed": seed,
            "messages": [dict(message) for message in record["messages"][:2]],
            "tools": record["tools"],
            "expected_calls": expected_trace(record),
            "model_calls": [],
            "generated_turns": [],
            "generated_token_counts": [],
            "thinking_budget_exhausted": False,
            "predicted_answer": None,
            "error": None,
            "tool_calls_valid": True,
            "done": False,
        }

    def process_turn(
        state: dict[str, Any], turn_output: dict[str, Any], turn_index: int
    ) -> None:
        text = turn_output["text"].strip()
        first_call_end = text.find("</tool_call>")
        if first_call_end >= 0:
            text = text[: first_call_end + len("</tool_call>")]
        state["generated_turns"].append(text)
        state["generated_token_counts"].append(
            turn_output["generated_token_count"]
        )
        if turn_output["thinking_budget_exhausted"] and first_call_end < 0:
            state["thinking_budget_exhausted"] = True
            state["error"] = "thinking_budget_exhausted"
            state["done"] = True
            return

        parsed_call, parse_error = parse_tool_call(text)
        if parse_error:
            state["error"] = parse_error
            state["tool_calls_valid"] = False
            state["done"] = True
            return
        if parsed_call is None:
            state["predicted_answer"] = parse_final_answer(text)
            if state["predicted_answer"] is None:
                state["error"] = "missing_final_answer"
            state["done"] = True
            return
        if len(state["model_calls"]) >= tool_budget:
            state["error"] = "max_tool_calls_exceeded"
            state["done"] = True
            return

        arguments = parsed_call["arguments"]
        result = calculate(arguments["op"], arguments["a"], arguments["b"])
        call_id = f"eval_call_{turn_index:02d}"
        state["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": text.split("<tool_call>", 1)[0].strip(),
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "calculator",
                    "tool_call_id": call_id,
                    "content": json.dumps({"result": result}),
                },
            ]
        )
        state["model_calls"].append(arguments)

    def finalize(state: dict[str, Any]) -> dict[str, Any]:
        record = state["record"]
        model_calls = state["model_calls"]
        final_answer = record["metadata"]["final_answer"]
        semantic_valid, semantic_complete = validate_semantic_trace(
            record["expression"],
            model_calls,
            final_answer,
            len(state["expected_calls"]),
        )
        answer_correct = state["predicted_answer"] == final_answer
        valid_tool_use = bool(model_calls) and state["tool_calls_valid"]
        return {
            "id": record["id"],
            "seed": state["seed"],
            "expression": record["expression"],
            "tier": record["metadata"]["tier"],
            "final_answer": final_answer,
            "predicted_answer": state["predicted_answer"],
            "answer_correct": answer_correct,
            "used_tool": bool(model_calls),
            "valid_tool_use": valid_tool_use,
            "parseable_completion": state["predicted_answer"] is not None,
            "tool_call_count": len(model_calls),
            "expected_tool_call_count": len(state["expected_calls"]),
            "tool_call_count_correct": len(model_calls) == len(state["expected_calls"]),
            "semantic_trace_valid": semantic_valid,
            "semantic_trace_complete": semantic_complete,
            "exact_trace": model_calls == state["expected_calls"],
            "task_success": answer_correct and valid_tool_use and semantic_complete,
            "error": state["error"],
            "model_calls": model_calls,
            "expected_calls": state["expected_calls"],
            "generated_turns": state["generated_turns"],
            "generated_token_counts": state["generated_token_counts"],
            "total_generated_tokens": sum(state["generated_token_counts"]),
            "thinking_budget_exhausted": state["thinking_budget_exhausted"],
        }

    seed_summaries = []
    all_seed_rows: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        set_seed(seed)
        seed_dir = args.output_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = seed_dir / "predictions.jsonl"
        temporary_path = predictions_path.with_suffix(".jsonl.tmp")
        states = [new_state(record, seed) for record in records]
        rows = []
        seed_started = time.perf_counter()

        with temporary_path.open("w", encoding="utf-8") as handle, tqdm(
            total=len(states),
            desc=f"{config['policy']} eval seed {seed}",
        ) as progress:
            for turn_index in range(tool_budget + 1):
                active = [state for state in states if not state["done"]]
                if not active:
                    break
                prompt_states = [
                    (len(tokenizer(render_prompt(state))["input_ids"]), state)
                    for state in active
                ]
                prompt_states.sort(key=lambda item: item[0])
                for start in range(0, len(prompt_states), args.batch_size):
                    batch_states = [
                        state
                        for _, state in prompt_states[start : start + args.batch_size]
                    ]
                    outputs = generate_turns([render_prompt(state) for state in batch_states])
                    for state, output in zip(batch_states, outputs):
                        process_turn(state, output, turn_index)
                        if state["done"]:
                            row = finalize(state)
                            rows.append(row)
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            handle.flush()
                            progress.update(1)

            for state in (state for state in states if not state["done"]):
                state["error"] = "agent_loop_exhausted"
                state["done"] = True
                row = finalize(state)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress.update(1)

        temporary_path.replace(predictions_path)
        elapsed = time.perf_counter() - seed_started
        rows.sort(key=lambda row: row["id"])
        all_seed_rows[seed] = rows
        token_counts = [
            count for row in rows for count in row["generated_token_counts"]
        ]
        seed_metrics = {
            "seed": seed,
            "num_examples": len(rows),
            "overall": rates(rows),
            "by_tier": {
                tier: {
                    "num_examples": len(tier_rows),
                    **rates(tier_rows),
                }
                for tier in ("easy", "medium", "hard")
                if (tier_rows := [row for row in rows if row["tier"] == tier])
            },
            "errors": dict(Counter(row["error"] or "none" for row in rows)),
            "token_usage": {
                "total_generated_tokens": sum(
                    row["total_generated_tokens"] for row in rows
                ),
                "mean_generated_tokens_per_example": round(
                    mean(row["total_generated_tokens"] for row in rows), 3
                ),
                "mean_generated_tokens_per_turn": round(mean(token_counts), 3),
                "p95_generated_tokens_per_turn": percentile(token_counts, 0.95),
            },
            "wall_time_seconds": round(elapsed, 3),
        }
        write_json_atomic(seed_dir / "metrics.json", seed_metrics)
        seed_summaries.append(seed_metrics)

    aggregate = {
        **config,
        "mean_over_seeds": {
            name: round(
                mean(summary["overall"][name] for summary in seed_summaries), 4
            )
            for name in METRIC_FIELDS
        },
        "per_seed": [
            {
                "seed": summary["seed"],
                "overall": summary["overall"],
                "by_tier": summary["by_tier"],
                "errors": summary["errors"],
                "token_usage": summary["token_usage"],
                "wall_time_seconds": summary["wall_time_seconds"],
            }
            for summary in seed_summaries
        ],
    }
    write_json_atomic(args.output_dir / "aggregate_metrics.json", aggregate)
    source_path = Path(
        globals().get("__file__", "/content/evaluate_sft_qwen3_calculator.py")
    )
    runtime = {
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_name,
        "dtype": str(next(model.parameters()).dtype),
        "python_source_sha256": (
            sha256_bytes(source_path.read_bytes()) if source_path.is_file() else None
        ),
        "semantic_module_sha256": config["semantic_module_sha256"],
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": version("peft"),
            "accelerate": version("accelerate"),
        },
    }
    write_json_atomic(args.output_dir / "runtime.json", runtime)
    print(json.dumps(aggregate["mean_over_seeds"], indent=2))
    print(f"Saved evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
