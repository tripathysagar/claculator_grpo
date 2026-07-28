"""Build the pinned gen-4/8/16 GRPO quality-efficiency comparison."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
NOTEBOOK_OUTPUTS = ROOT / "outputs" / "notebooks"
EVAL_OUTPUTS = ROOT / "outputs" / "evaluations"
OUTPUT_PATH = EVAL_OUTPUTS / "qwen3-calc-grpo-generation-ablation-comparison.json"

ARMS = {
    "gen4": {
        "generations": 4,
        "revision": "13ae8ff6e3c74f3cc62a4f17ce388f37c97e7825",
        "zip": NOTEBOOK_OUTPUTS
        / "qwen3-calc-grpo600-full-vllm-colocate-seed42-a100b16g4mem25full20260723-artifacts.zip",
        "eval": EVAL_OUTPUTS
        / "qwen3-calc-grpo-ablation-g4-eval-aggregate_metrics.json",
    },
    "gen8": {
        "generations": 8,
        "revision": "3348e4da1765e7a544fd7540bb8d4d1695f9d872",
        "metrics": NOTEBOOK_OUTPUTS
        / "qwen3-calc-grpo600-full-vllm-colocate-seed42-a100b16g8full20260723-metrics.json",
        "config": NOTEBOOK_OUTPUTS
        / "qwen3-calc-grpo600-full-vllm-colocate-seed42-a100b16g8full20260723-run_config.json",
        "trainer_state": NOTEBOOK_OUTPUTS
        / "qwen3-calc-grpo600-full-vllm-colocate-seed42-a100b16g8full20260723-trainer_state.json",
        "eval": EVAL_OUTPUTS
        / "qwen3-calc-grpo-ablation-g8-eval-aggregate_metrics.json",
    },
    "gen16": {
        "generations": 16,
        "revision": "b446308cb6397645d68ae792b6a012c39c7e63b1",
        "zip": NOTEBOOK_OUTPUTS
        / "qwen3-calc-grpo600-full-vllm-colocate-seed42-a100b16g16fullretry20260723-artifacts.zip",
        "eval": EVAL_OUTPUTS
        / "qwen3-calc-grpo-ablation-g16-eval-aggregate_metrics.json",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_training(arm: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "zip" in arm:
        with zipfile.ZipFile(arm["zip"]) as archive:
            return (
                json.loads(archive.read("metrics.json")),
                json.loads(archive.read("grpo_run_config.json")),
                json.loads(archive.read("trainer_state.json")),
            )
    return (
        load_json(arm["metrics"]),
        load_json(arm["config"]),
        load_json(arm["trainer_state"]),
    )


def main() -> None:
    comparison: dict[str, Any] = {}
    prompt_hash = None
    for name, arm in ARMS.items():
        metrics, config, state = load_training(arm)
        evaluation = load_json(arm["eval"])
        generations = arm["generations"]
        assert metrics["status"] == "training_complete"
        assert config["seed"] == 42 and config["num_rows"] == 600
        assert config["grpo_config"]["num_generations"] == generations
        assert config["grpo_config"]["per_device_train_batch_size"] == 16
        assert metrics["last_log"]["step"] == 75
        assert evaluation["seeds"] == [20260719]
        if prompt_hash is None:
            prompt_hash = evaluation["selected_records_sha256"]
        assert evaluation["selected_records_sha256"] == prompt_hash

        step_logs = [
            entry
            for entry in state["log_history"]
            if entry.get("step") == 75 and "reward/task_success" in entry
        ]
        assert step_logs
        final_step = step_logs[-1]
        gpu = metrics["gpu_telemetry_summary"]
        overall = evaluation["mean_over_seeds"]
        tiers = evaluation["per_seed"][0]["by_tier"]
        runtime = metrics["train_metrics"]["train_runtime"]
        comparison[name] = {
            "generations": generations,
            "adapter_revision": arm["revision"],
            "gradient_accumulation_steps": config["grpo_config"][
                "gradient_accumulation_steps"
            ],
            "steps_per_generation": config["grpo_config"]["steps_per_generation"],
            "optimizer_steps": 75,
            "unique_prompts": 600,
            "rollout_completions": 600 * generations,
            "training_runtime_seconds": runtime,
            "relative_runtime_vs_gen4": None,
            "training_tokens": int(final_step["num_tokens"]),
            "mean_gpu_utilization_percent": gpu["mean_gpu_utilization_percent"],
            "p95_gpu_utilization_percent": gpu["p95_gpu_utilization_percent"],
            "peak_memory_used_mib": gpu["peak_memory_used_mib"],
            "minimum_memory_free_mib": gpu["minimum_memory_free_mib"],
            "final_training_task_success": final_step["reward/task_success"],
            "final_training_reward_std": final_step["reward_std"],
            "final_training_kl": final_step["kl"],
            "eval_task_success": overall["task_success"],
            "eval_answer_correct": overall["answer_correct"],
            "eval_tool_call_count_correct": overall["tool_call_count_correct"],
            "eval_easy_task_success": tiers["easy"]["task_success"],
            "eval_medium_task_success": tiers["medium"]["task_success"],
            "eval_hard_task_success": tiers["hard"]["task_success"],
        }

    baseline_runtime = comparison["gen4"]["training_runtime_seconds"]
    baseline_success = comparison["gen4"]["eval_task_success"]
    for arm in comparison.values():
        arm["relative_runtime_vs_gen4"] = (
            arm["training_runtime_seconds"] / baseline_runtime
        )
        arm["eval_task_success_delta_vs_gen4"] = (
            arm["eval_task_success"] - baseline_success
        )

    payload = {
        "seed": 42,
        "eval_seed": 20260719,
        "selected_records_sha256": prompt_hash,
        "controls": {
            "unique_prompts_per_optimizer_step": 8,
            "optimizer_steps": 75,
            "per_device_train_batch_size": 16,
            "generation_batch_size": 32,
            "vllm_gpu_memory_utilization": 0.25,
            "eval_protocol": "100 rows, greedy, 128 tokens/turn, six calls, A100",
        },
        "arms": comparison,
        "winner": "gen4",
        "conclusion": (
            "At seed 42, four generations dominates: 0.96 eval task success in "
            "433.4 seconds versus 0.95 in 820.5 seconds for eight generations "
            "and 0.93 in 1526.2 seconds for sixteen. Extra generations consumed "
            "more compute and GPU memory without improving held-out eval quality."
        ),
        "limitations": [
            "Single training seed and deterministic eval seed.",
            "Prompt exposure and optimizer-step count are matched; rollout compute is not.",
            "The held-out test split was not touched.",
        ],
    }
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(OUTPUT_PATH)
    print(json.dumps({"winner": payload["winner"], "arms": comparison}))


if __name__ == "__main__":
    main()
