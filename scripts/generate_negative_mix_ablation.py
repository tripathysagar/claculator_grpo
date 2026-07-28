"""Build the controlled 10% negative-operand SFT/GRPO ablation dataset."""

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
    random_negative_ood_candidate,
    validate_record,
)

UNCHANGED_SPLITS = ("eval", "test", "ood_length", "ood_negative", "ood_float")


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


def choose_replacements(
    rows_by_id: dict[str, dict[str, Any]],
    stage_ids: Sequence[str],
    *,
    count: int,
    rng: random.Random,
) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_id in stage_ids:
        grouped[rows_by_id[row_id]["metadata"]["tier"]].append(row_id)
    quotas = proportional_quotas(
        {tier: len(ids) for tier, ids in grouped.items()}, count
    )
    selected: list[str] = []
    for tier in sorted(grouped):
        tier_ids = sorted(grouped[tier])
        rng.shuffle(tier_ids)
        selected.extend(tier_ids[: quotas[tier]])
    return sorted(selected)


def generate_replacements(
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
            expression = random_negative_ood_candidate(rng)
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
                ood_type="negative",
            )
            replacement["metadata"]["training_mix"] = "negative10"
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
                    "negative_expression": rendered,
                    "negative_expression_sha256": canonical_sha256(rendered),
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
    parser.add_argument(
        "--output", type=Path, default=Path("data/calculator_qwen3_negative10")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/calculator_qwen3_negative10_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists; remove it explicitly before regeneration"
        )

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
    if len(rows_by_id) != len(train_rows):
        raise AssertionError("Source train IDs are not unique")

    sft_ids = read_ids(args.sft_ids)
    grpo_ids = read_ids(args.grpo_ids)
    if len(sft_ids) != 200 or len(grpo_ids) != 600:
        raise AssertionError("Expected the frozen 200/600 SFT/GRPO partition")
    if set(sft_ids) & set(grpo_ids) or set(sft_ids) | set(grpo_ids) != set(rows_by_id):
        raise AssertionError("Frozen SFT/GRPO IDs do not partition source train")

    selection_rng = random.Random(args.seed)
    sft_replaced_ids = choose_replacements(
        rows_by_id, sft_ids, count=20, rng=selection_rng
    )
    grpo_replaced_ids = choose_replacements(
        rows_by_id, grpo_ids, count=60, rng=selection_rng
    )
    if set(sft_replaced_ids) & set(grpo_replaced_ids):
        raise AssertionError("SFT and GRPO negative replacements overlap")

    seen_expressions = {
        row["expression"] for rows in source_rows.values() for row in rows
    }
    sft_replacements, sft_manifest = generate_replacements(
        sft_replaced_ids,
        rows_by_id,
        seed=args.seed + 1,
        seen_expressions=seen_expressions,
    )
    grpo_replacements, grpo_manifest = generate_replacements(
        grpo_replaced_ids,
        rows_by_id,
        seed=args.seed + 2,
        seen_expressions=seen_expressions,
    )
    replacements = {**sft_replacements, **grpo_replacements}
    mixed_train = [
        replacements.get(row["id"], copy.deepcopy(row)) for row in train_rows
    ]

    negative_ids = set(replacements)
    if sum(row["id"] in negative_ids for row in mixed_train) != 80:
        raise AssertionError("Mixed train split must contain exactly 80 negative rows")
    if sum(row_id in negative_ids for row_id in sft_ids) != 20:
        raise AssertionError("SFT partition is not exactly 10% negative")
    if sum(row_id in negative_ids for row_id in grpo_ids) != 60:
        raise AssertionError("GRPO partition is not exactly 10% negative")
    if not all(
        any(value < 0 for value in replacements[row_id]["metadata"]["source_operands"])
        for row_id in negative_ids
    ):
        raise AssertionError("A replacement lacks a negative source operand")

    output_rows = {"train": mixed_train}
    output_rows.update(
        {split: copy.deepcopy(source_rows[split]) for split in UNCHANGED_SPLITS}
    )
    output = DatasetDict(
        {split: Dataset.from_list(rows) for split, rows in output_rows.items()}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save_to_disk(str(args.output))

    reloaded = load_from_disk(str(args.output))
    for split in UNCHANGED_SPLITS:
        before = [dict(row) for row in source[split]]
        after = [dict(row) for row in reloaded[split]]
        if canonical_sha256(before) != canonical_sha256(after):
            raise AssertionError(f"Unchanged split mutated during serialization: {split}")

    manifest = {
        "name": "negative-operand-10-percent-ablation",
        "seed": args.seed,
        "source": str(args.source),
        "output": str(args.output),
        "stage_counts": {
            "sft": {"total": 200, "id": 180, "negative": 20},
            "grpo": {"total": 600, "id": 540, "negative": 60},
        },
        "sft_replacements": sft_manifest,
        "grpo_replacements": grpo_manifest,
        "source_split_fingerprints": {
            split: source[split]._fingerprint for split in source
        },
        "output_split_fingerprints": {
            split: reloaded[split]._fingerprint for split in reloaded
        },
        "unchanged_split_sha256": {
            split: canonical_sha256([dict(row) for row in source[split]])
            for split in UNCHANGED_SPLITS
        },
        "input_sha256": {
            "sft_ids": sha256_file(args.sft_ids),
            "grpo_ids": sha256_file(args.grpo_ids),
        },
        "output_file_sha256": {
            str(path.relative_to(args.output)): sha256_file(path)
            for path in sorted(args.output.rglob("*"))
            if path.is_file()
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["stage_counts"], sort_keys=True))
    print(f"DATASET_READY {args.output}")
    print(f"MANIFEST_READY {args.manifest}")


if __name__ == "__main__":
    main()
