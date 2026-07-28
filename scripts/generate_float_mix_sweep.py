"""Build nested 20%, 30%, and 40% floating-point mixture datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset, DatasetDict, load_from_disk

from generate_calculator_dataset import (
    build_record,
    expression_structure,
    is_useful_candidate,
    random_float_ood_candidate,
    validate_record,
)

UNCHANGED_SPLITS = ("eval", "test", "ood_length", "ood_negative", "ood_float")
MIX_PERCENTAGES = (20, 30, 40)
SFT_FLOAT_COUNTS = {20: 40, 30: 60, 40: 80}
GRPO_FLOAT_COUNTS = {20: 120, 30: 180, 40: 240}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def proportional_quotas(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    exact = {tier: size * target / total for tier, size in group_sizes.items()}
    quotas = {tier: int(value) for tier, value in exact.items()}
    priority = sorted(
        group_sizes,
        key=lambda tier: (exact[tier] - quotas[tier], group_sizes[tier], tier),
        reverse=True,
    )
    for tier in priority[: target - sum(quotas.values())]:
        quotas[tier] += 1
    return quotas


def nested_replacement_ids(
    rows_by_id: dict[str, dict[str, Any]],
    stage_ids: Sequence[str],
    targets: dict[int, int],
    *,
    rng: random.Random,
) -> dict[int, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_id in stage_ids:
        grouped[rows_by_id[row_id]["metadata"]["tier"]].append(row_id)
    for tier_ids in grouped.values():
        tier_ids.sort()
        rng.shuffle(tier_ids)

    selected: dict[int, list[str]] = {}
    group_sizes = {tier: len(ids) for tier, ids in grouped.items()}
    for percentage in MIX_PERCENTAGES:
        quotas = proportional_quotas(group_sizes, targets[percentage])
        selected[percentage] = sorted(
            row_id
            for tier, tier_ids in grouped.items()
            for row_id in tier_ids[: quotas[tier]]
        )
    for smaller, larger in zip(MIX_PERCENTAGES, MIX_PERCENTAGES[1:]):
        if not set(selected[smaller]) < set(selected[larger]):
            raise AssertionError(f"{smaller}% replacements are not nested in {larger}%")
    return selected


def generate_replacement_pool(
    selected_ids: Sequence[str],
    rows_by_id: dict[str, dict[str, Any]],
    *,
    seed: int,
    seen_expressions: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    replacements: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for row_id in selected_ids:
        original = rows_by_id[row_id]
        required_tier = original["metadata"]["tier"]
        for _ in range(20_000):
            expression = random_float_ood_candidate(rng)
            rendered = expression.render()
            if (
                rendered in seen_expressions
                or expression_structure(expression)["tier"] != required_tier
                or not is_useful_candidate(expression)
            ):
                continue
            replacement = build_record(
                expression,
                record_id=row_id,
                rng=rng,
                split="train",
                ood_type="float",
            )
            replacement["metadata"]["training_mix"] = "float_pool"
            validate_record(replacement)
            replacements[row_id] = replacement
            seen_expressions.add(rendered)
            manifest_rows.append(
                {
                    "id": row_id,
                    "tier": required_tier,
                    "original_expression_sha256": canonical_sha256(
                        original["expression"]
                    ),
                    "float_expression": rendered,
                    "float_expression_sha256": canonical_sha256(rendered),
                }
            )
            break
        else:
            raise RuntimeError(f"Unable to generate a {required_tier} row for {row_id}")
    return replacements, manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("data/calculator_qwen3")
    )
    parser.add_argument(
        "--sft-ids", type=Path, default=Path("data/calculator_qwen3_sft200_ids.txt")
    )
    parser.add_argument(
        "--grpo-ids",
        type=Path,
        default=Path("data/calculator_qwen3_grpo600_ids.txt"),
    )
    parser.add_argument("--output-parent", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/calculator_qwen3_float_mix_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = {
        percentage: args.output_parent / f"calculator_qwen3_float{percentage}"
        for percentage in MIX_PERCENTAGES
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Remove existing outputs explicitly: {existing}")

    source = load_from_disk(str(args.source))
    if set(source) != {"train", *UNCHANGED_SPLITS}:
        raise AssertionError(f"Unexpected source splits: {set(source)}")
    source_rows = {
        split: [copy.deepcopy(dict(row)) for row in source[split]]
        for split in source
    }
    train_rows = source_rows["train"]
    if len(train_rows) != 800:
        raise AssertionError(f"Expected 800 train rows, found {len(train_rows)}")
    rows_by_id = {row["id"]: row for row in train_rows}
    if len(rows_by_id) != 800:
        raise AssertionError("Source train IDs are not unique")

    sft_ids = read_ids(args.sft_ids)
    grpo_ids = read_ids(args.grpo_ids)
    if len(sft_ids) != 200 or len(grpo_ids) != 600:
        raise AssertionError("Expected the frozen 200/600 partition")
    if set(sft_ids) & set(grpo_ids) or set(sft_ids) | set(grpo_ids) != set(rows_by_id):
        raise AssertionError("Frozen IDs do not partition source train")

    selection_rng = random.Random(args.seed)
    sft_selected = nested_replacement_ids(
        rows_by_id, sft_ids, SFT_FLOAT_COUNTS, rng=selection_rng
    )
    grpo_selected = nested_replacement_ids(
        rows_by_id, grpo_ids, GRPO_FLOAT_COUNTS, rng=selection_rng
    )
    if set(sft_selected[40]) & set(grpo_selected[40]):
        raise AssertionError("Maximum SFT/GRPO replacement pools overlap")

    seen_expressions = {
        row["expression"] for rows in source_rows.values() for row in rows
    }
    sft_pool, sft_manifest = generate_replacement_pool(
        sft_selected[40],
        rows_by_id,
        seed=args.seed + 1,
        seen_expressions=seen_expressions,
    )
    grpo_pool, grpo_manifest = generate_replacement_pool(
        grpo_selected[40],
        rows_by_id,
        seed=args.seed + 2,
        seen_expressions=seen_expressions,
    )
    replacement_pool = {**sft_pool, **grpo_pool}

    arm_manifests: dict[str, Any] = {}
    for percentage in MIX_PERCENTAGES:
        selected_ids = set(sft_selected[percentage]) | set(
            grpo_selected[percentage]
        )
        mixed_train: list[dict[str, Any]] = []
        for original in train_rows:
            if original["id"] not in selected_ids:
                mixed_train.append(copy.deepcopy(original))
                continue
            replacement = copy.deepcopy(replacement_pool[original["id"]])
            replacement["metadata"]["training_mix"] = f"float{percentage}"
            mixed_train.append(replacement)

        if sum(row["id"] in selected_ids for row in mixed_train) != 8 * percentage:
            raise AssertionError(f"Incorrect total replacement count for {percentage}%")
        if sum(row_id in selected_ids for row_id in sft_ids) != SFT_FLOAT_COUNTS[percentage]:
            raise AssertionError(f"Incorrect SFT replacement count for {percentage}%")
        if sum(row_id in selected_ids for row_id in grpo_ids) != GRPO_FLOAT_COUNTS[percentage]:
            raise AssertionError(f"Incorrect GRPO replacement count for {percentage}%")
        for row_id in selected_ids:
            operands = replacement_pool[row_id]["metadata"]["source_operands"]
            if not any(isinstance(value, float) and not value.is_integer() for value in operands):
                raise AssertionError(f"Replacement {row_id} lacks a fractional operand")

        output_rows = {"train": mixed_train}
        output_rows.update(
            {split: copy.deepcopy(source_rows[split]) for split in UNCHANGED_SPLITS}
        )
        dataset = DatasetDict(
            {split: Dataset.from_list(rows) for split, rows in output_rows.items()}
        )
        output_path = output_paths[percentage]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        reloaded = load_from_disk(str(output_path))
        for split in UNCHANGED_SPLITS:
            if canonical_sha256([dict(row) for row in source[split]]) != canonical_sha256(
                [dict(row) for row in reloaded[split]]
            ):
                raise AssertionError(f"{percentage}% mutated unchanged split {split}")
        arm_manifests[str(percentage)] = {
            "output": str(output_path),
            "stage_counts": {
                "sft": {
                    "total": 200,
                    "id": 200 - SFT_FLOAT_COUNTS[percentage],
                    "float": SFT_FLOAT_COUNTS[percentage],
                },
                "grpo": {
                    "total": 600,
                    "id": 600 - GRPO_FLOAT_COUNTS[percentage],
                    "float": GRPO_FLOAT_COUNTS[percentage],
                },
            },
            "sft_replacement_ids": sft_selected[percentage],
            "grpo_replacement_ids": grpo_selected[percentage],
            "split_fingerprints": {
                split: reloaded[split]._fingerprint for split in reloaded
            },
            "file_sha256": {
                str(path.relative_to(output_path)): sha256_file(path)
                for path in sorted(output_path.rglob("*"))
                if path.is_file()
            },
        }

    manifest = {
        "name": "floating-point-20-30-40-percent-ablation",
        "seed": args.seed,
        "source": str(args.source),
        "nested": True,
        "sft_pool": sft_manifest,
        "grpo_pool": grpo_manifest,
        "arms": arm_manifests,
        "unchanged_split_sha256": {
            split: canonical_sha256([dict(row) for row in source[split]])
            for split in UNCHANGED_SPLITS
        },
        "input_sha256": {
            "sft_ids": sha256_file(args.sft_ids),
            "grpo_ids": sha256_file(args.grpo_ids),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {percentage: arm_manifests[str(percentage)]["stage_counts"] for percentage in MIX_PERCENTAGES},
            sort_keys=True,
        )
    )
    print("FLOAT_SWEEP_DATASETS_READY")
    print(f"MANIFEST_READY {args.manifest}")


if __name__ == "__main__":
    main()
