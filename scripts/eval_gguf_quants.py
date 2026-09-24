"""Evaluate GGUF-quantized calculator models via llama-server.

Runs the held-out eval set against each quantized variant, logging
task_success and tokens/sec per bit-width.  Designed to run on WSL
with CPU-only llama.cpp.

Usage (on WSL):
    ~/miniconda3/bin/python ~/scripts/eval_gguf_quants.py
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

sys.path.insert(0, str(Path.home() / "scripts"))
import calculator_semantics
from calculator_semantics import (
    calculate,
    parse_final_answer,
    remove_null_fields,
    validate_semantic_trace,
)

SERVER_PORT = 8123
QUANTS = ["Q8_0", "Q4_K_M", "Q3_K_M", "Q2_K"]
MODEL_DIR = Path.home() / "models"
LLAMA_SERVER = Path.home() / "git/llama.cpp/build/bin/llama-server"
TOKENIZER_PATH = MODEL_DIR / "Qwen3-0.6B"
DATASET_PATH = Path.home() / "data/calculator_qwen3"
OUTPUT_DIR = Path.home() / "experiments/gguf-quant-eval"

MAX_NEW_TOKENS = 256
MAX_TOOL_ROUNDS = 6
TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

METRIC_FIELDS = (
    "answer_correct",
    "used_tool",
    "valid_tool_use",
    "parseable_completion",
    "tool_call_count_correct",
    "semantic_trace_valid",
    "semantic_trace_complete",
    "task_success",
)


def start_server(gguf_path: Path, port: int = SERVER_PORT) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            str(LLAMA_SERVER),
            "-m", str(gguf_path),
            "--port", str(port),
            "-t", "4",
            "-c", "4096",
            "--log-disable",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    for attempt in range(90):
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return proc
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1)
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    raise RuntimeError(f"Server failed to start for {gguf_path}")


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def generate(prompt: str, port: int = SERVER_PORT) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"http://localhost:{port}/completion",
        json={
            "prompt": prompt,
            "n_predict": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "stop": ["</tool_call>"],
            "cache_prompt": True,
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - t0
    data = r.json()
    text = data.get("content", "")
    if data.get("stopped_word", False) or data.get("stopping_word") == "</tool_call>":
        if "</tool_call>" not in text:
            text += "</tool_call>"
    tokens = data.get("tokens_predicted", 0)
    return text, tokens, elapsed


def parse_tool_call(
    text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    matches = TOOL_TAG_RE.findall(text)
    if not matches:
        if "<tool_call>" in text:
            return None, "unclosed_tool_call"
        return None, None
    if len(matches) != 1:
        return None, "multiple_tool_calls"
    try:
        payload = json.loads(matches[0])
        args = payload["arguments"]
        if payload["name"] != "calculator":
            return None, "wrong_tool_name"
        if set(args) != {"op", "a", "b"}:
            return None, "wrong_schema"
        if args["op"] not in {"+", "-", "*"}:
            return None, "invalid_operator"
        if type(args["a"]) not in {int, float} or type(args["b"]) not in {int, float}:
            return None, "non_numeric"
        return {"name": "calculator", "arguments": args}, None
    except (KeyError, TypeError, json.JSONDecodeError):
        return None, "malformed_tool_call"


def expected_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        msg["tool_calls"][0]["function"]["arguments"]
        for msg in record["messages"]
        if msg["role"] == "assistant" and "tool_calls" in msg
    ]


def eval_one(
    record: dict[str, Any],
    tokenizer: Any,
    port: int = SERVER_PORT,
) -> dict[str, Any]:
    messages = [dict(m) for m in record["messages"][:2]]
    tools = record["tools"]
    exp_calls = expected_trace(record)
    model_calls: list[dict[str, Any]] = []
    gen_tokens_total = 0
    gen_time_total = 0.0
    error: str | None = None
    predicted: int | float | None = None
    tool_calls_valid = True
    generated_turns: list[str] = []

    for turn in range(MAX_TOOL_ROUNDS + 1):
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
        text, n_tokens, elapsed = generate(prompt, port)
        gen_tokens_total += n_tokens
        gen_time_total += elapsed

        first_end = text.find("</tool_call>")
        if first_end >= 0:
            text = text[: first_end + len("</tool_call>")]
        generated_turns.append(text.strip())

        if n_tokens >= MAX_NEW_TOKENS and first_end < 0:
            error = "thinking_budget_exhausted"
            break

        parsed, parse_err = parse_tool_call(text)
        if parse_err:
            error = parse_err
            tool_calls_valid = False
            break
        if parsed is None:
            predicted = parse_final_answer(text)
            if predicted is None:
                error = "missing_final_answer"
            break
        if len(model_calls) >= MAX_TOOL_ROUNDS:
            error = "max_tool_calls_exceeded"
            break

        args = parsed["arguments"]
        result = calculate(args["op"], args["a"], args["b"])
        call_id = f"eval_call_{turn:02d}"
        messages.extend(
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
                                "arguments": args,
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
        model_calls.append(args)

    final_answer = record["metadata"]["final_answer"]
    sem_valid, sem_complete = validate_semantic_trace(
        record["expression"],
        model_calls,
        final_answer,
        len(exp_calls),
    )
    answer_correct = predicted == final_answer
    valid_tool_use = bool(model_calls) and tool_calls_valid

    return {
        "id": record["id"],
        "expression": record["expression"],
        "tier": record["metadata"]["tier"],
        "final_answer": final_answer,
        "predicted_answer": predicted,
        "answer_correct": answer_correct,
        "used_tool": bool(model_calls),
        "valid_tool_use": valid_tool_use,
        "parseable_completion": predicted is not None,
        "tool_call_count": len(model_calls),
        "expected_tool_call_count": len(exp_calls),
        "tool_call_count_correct": len(model_calls) == len(exp_calls),
        "semantic_trace_valid": sem_valid,
        "semantic_trace_complete": sem_complete,
        "exact_trace": model_calls == exp_calls,
        "task_success": answer_correct and valid_tool_use and sem_complete,
        "error": error,
        "model_calls": model_calls,
        "generated_turns": generated_turns,
        "total_generated_tokens": gen_tokens_total,
        "total_generation_time": round(gen_time_total, 4),
    }


def rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        name: round(mean(float(row[name]) for row in rows), 4)
        for name in METRIC_FIELDS
    }


def main() -> None:
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    print("Loading dataset and tokenizer...")
    dataset = load_from_disk(str(DATASET_PATH))["eval"]
    records = [remove_null_fields(dict(r)) for r in dataset]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    print(f"Eval set: {len(records)} examples")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    for quant in QUANTS:
        gguf_path = MODEL_DIR / f"qwen3-0.6b-calc-grpo600.{quant}.gguf"
        if not gguf_path.exists():
            print(f"SKIP {quant}: {gguf_path} not found")
            continue

        size_mb = gguf_path.stat().st_size / 1e6
        print(f"\n{'=' * 60}")
        print(f"Evaluating {quant} ({size_mb:.0f} MB)")
        print(f"{'=' * 60}")

        proc = start_server(gguf_path)
        try:
            rows: list[dict[str, Any]] = []
            t_start = time.perf_counter()
            for i, record in enumerate(records):
                row = eval_one(record, tokenizer)
                rows.append(row)
                if (i + 1) % 10 == 0 or i == len(records) - 1:
                    succ = mean(float(r["task_success"]) for r in rows)
                    ans = mean(float(r["answer_correct"]) for r in rows)
                    print(
                        f"  [{i+1:>3}/{len(records)}] "
                        f"task_success={succ:.3f}  "
                        f"answer_correct={ans:.3f}"
                    )
            wall_time = time.perf_counter() - t_start

            overall = rates(rows)
            by_tier: dict[str, Any] = {}
            for tier in ("easy", "medium", "hard"):
                tier_rows = [r for r in rows if r["tier"] == tier]
                if tier_rows:
                    by_tier[tier] = {
                        "n": len(tier_rows),
                        **rates(tier_rows),
                    }

            total_tokens = sum(r["total_generated_tokens"] for r in rows)
            total_time = sum(r["total_generation_time"] for r in rows)

            result = {
                "quant": quant,
                "model_size_mb": round(size_mb, 1),
                "num_examples": len(rows),
                "overall": overall,
                "by_tier": by_tier,
                "errors": dict(Counter(r["error"] or "none" for r in rows)),
                "tokens_per_second": round(total_tokens / total_time, 1)
                if total_time > 0
                else 0,
                "total_generated_tokens": total_tokens,
                "total_generation_time_s": round(total_time, 2),
                "wall_time_s": round(wall_time, 2),
            }
            all_results[quant] = result

            quant_dir = OUTPUT_DIR / quant
            quant_dir.mkdir(parents=True, exist_ok=True)
            (quant_dir / "predictions.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
            (quant_dir / "metrics.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )

            print(f"\n{quant} summary:")
            print(json.dumps(overall, indent=2))
            print(f"  tokens/sec: {result['tokens_per_second']}")
            print(f"  wall time:  {wall_time:.1f}s")
        finally:
            stop_server(proc)

    (OUTPUT_DIR / "aggregate.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(
        f"{'Quant':<10} {'Size':>8} {'task_success':>13} "
        f"{'answer_correct':>15} {'t/s':>6}"
    )
    print("-" * 60)
    for q, r in all_results.items():
        print(
            f"{q:<10} {r['model_size_mb']:>6.0f}MB "
            f"{r['overall']['task_success']:>13.4f} "
            f"{r['overall']['answer_correct']:>15.4f} "
            f"{r['tokens_per_second']:>6.1f}"
        )
    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
