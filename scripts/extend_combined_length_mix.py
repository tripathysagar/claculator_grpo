"""Extend the combined 10%-length dataset to a nested 20%-length arm."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from generate_calculator_dataset import random_length_ood_candidate
from generate_combined_generalization_mix import (
    UNCHANGED_SPLITS,
    canonical_sha256,
    generate_domain_rows,
    read_ids,
    sha256_file,
)

STAGE_COUNTS = {
    "sft": {"id": 100, "negative": 20, "float": 40, "length": 40},
    "grpo": {"id": 300, "negative": 60, "float": 120, "length": 120},
}
ADDITIONAL_LENGTH_COUNTS = {"sft": 20, "grpo": 60}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/calculator_qwen3_combined_60_10_20_10"),
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("data/calculator_qwen3_combined_60_10_20_10_manifest.json"),
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
        "--output",
        type=Path,
        default=Path("data/calculator_qwen3_combined_50_10_20_20"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/calculator_qwen3_combined_50_10_20_20_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def classify(row: dict[str, Any]) -> str:
    return row["metadata"].get("training_domain") or "id"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Remove {args.output} explicitly before regeneration")

    base = load_from_disk(str(args.base))
    base_manifest = json.loads(args.base_manifest.read_text())
    if set(base) != {"train", *UNCHANGED_SPLITS}:
        raise AssertionError(f"Unexpected base splits: {set(base)}")
    base_rows = {
        split: [copy.deepcopy(dict(row)) for row in base[split]] for split in base
    }
    train_rows = base_rows["train"]
    rows_by_id = {row["id"]: row for row in train_rows}
    if len(train_rows) != 800 or len(rows_by_id) != 800:
        raise AssertionError("Expected 800 unique base training rows")

    stage_ids = {
        "sft": read_ids(args.sft_ids),
        "grpo": read_ids(args.grpo_ids),
    }
    selection_rng = random.Random(args.seed)
    selected: dict[str, list[str]] = {}
    for stage, ids in stage_ids.items():
        eligible = sorted(
            row_id
            for row_id in ids
            if classify(rows_by_id[row_id]) == "id"
            and rows_by_id[row_id]["metadata"]["tier"] == "hard"
        )
        selection_rng.shuffle(eligible)
        count = ADDITIONAL_LENGTH_COUNTS[stage]
        selected[stage] = sorted(eligible[:count])
        if len(selected[stage]) != count:
            raise AssertionError(f"Insufficient hard ID rows for {stage}")
    if set(selected["sft"]) & set(selected["grpo"]):
        raise AssertionError("Additional SFT/GRPO length IDs overlap")

    seen_expressions = {
        row["expression"] for rows in base_rows.values() for row in rows
    }
    replacements: dict[str, dict[str, Any]] = {}
    replacement_manifest: dict[str, Any] = {}
    for offset, stage in enumerate(("sft", "grpo"), start=1):
        rows, rows_manifest = generate_domain_rows(
            "length",
            selected[stage],
            rows_by_id,
            candidate_factory=random_length_ood_candidate,
            mix_label="50_10_20_20",
            seed=args.seed + offset,
            seen_expressions=seen_expressions,
        )
        replacements.update(rows)
        replacement_manifest[stage] = rows_manifest

    mixed_train: list[dict[str, Any]] = []
    for row in train_rows:
        output_row = copy.deepcopy(replacements.get(row["id"], row))
        if classify(output_row) != "id":
            output_row["metadata"]["training_mix"] = "combined_50_10_20_20"
        mixed_train.append(output_row)

    output_by_id = {row["id"]: row for row in mixed_train}
    for stage, ids in stage_ids.items():
        counts = {
            domain: sum(classify(output_by_id[row_id]) == domain for row_id in ids)
            for domain in ("id", "negative", "float", "length")
        }
        if counts != STAGE_COUNTS[stage]:
            raise AssertionError((stage, counts, STAGE_COUNTS[stage]))

    output_rows = {"train": mixed_train}
    output_rows.update(
        {split: copy.deepcopy(base_rows[split]) for split in UNCHANGED_SPLITS}
    )
    output = DatasetDict(
        {split: Dataset.from_list(rows) for split, rows in output_rows.items()}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save_to_disk(str(args.output))
    reloaded = load_from_disk(str(args.output))
    for split in UNCHANGED_SPLITS:
        before = canonical_sha256([dict(row) for row in base[split]])
        after = canonical_sha256([dict(row) for row in reloaded[split]])
        if before != after:
            raise AssertionError(f"Unchanged split mutated: {split}")

    base_domain_ids = {
        domain: sorted(
            row["id"] for row in train_rows if classify(row) == domain
        )
        for domain in ("negative", "float", "length")
    }
    output_domain_ids = {
        domain: sorted(
            row["id"] for row in mixed_train if classify(row) == domain
        )
        for domain in ("negative", "float", "length")
    }
    if output_domain_ids["negative"] != base_domain_ids["negative"]:
        raise AssertionError("Negative treatment changed from the 10%-length arm")
    if output_domain_ids["float"] != base_domain_ids["float"]:
        raise AssertionError("Float treatment changed from the 10%-length arm")
    if not set(base_domain_ids["length"]) < set(output_domain_ids["length"]):
        raise AssertionError("The 10%-length treatment is not nested in the 20% arm")

    manifest = {
        "name": "combined-generalization-50-10-20-20",
        "seed": args.seed,
        "base": str(args.base),
        "output": str(args.output),
        "stage_counts": STAGE_COUNTS,
        "additional_length_ids": selected,
        "additional_length_replacements": replacement_manifest,
        "nested_domain_ids": {
            "base": base_domain_ids,
            "output": output_domain_ids,
        },
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "input_sha256": {
            "sft_ids": sha256_file(args.sft_ids),
            "grpo_ids": sha256_file(args.grpo_ids),
        },
        "unchanged_split_sha256": {
            split: canonical_sha256([dict(row) for row in base[split]])
            for split in UNCHANGED_SPLITS
        },
        "output_split_fingerprints": {
            split: reloaded[split]._fingerprint for split in reloaded
        },
        "output_file_sha256": {
            str(path.relative_to(args.output)): sha256_file(path)
            for path in sorted(args.output.rglob("*"))
            if path.is_file()
        },
        "base_stage_counts": base_manifest["stage_counts"],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(STAGE_COUNTS, sort_keys=True))
    print(f"LENGTH20_DATASET_READY {args.output}")
    print(f"MANIFEST_READY {args.manifest}")


if __name__ == "__main__":
    main()
