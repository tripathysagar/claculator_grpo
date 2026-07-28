"""Build the 60% ID / 10% negative / 20% float / 10% length dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from datasets import Dataset, DatasetDict, load_from_disk

from generate_calculator_dataset import (
    Expression,
    build_record,
    expression_structure,
    is_useful_candidate,
    random_float_ood_candidate,
    random_length_ood_candidate,
    random_negative_ood_candidate,
    validate_record,
)

UNCHANGED_SPLITS = ("eval", "test", "ood_length", "ood_negative", "ood_float")
FIXED_STAGE_COUNTS = {
    "sft": {"negative": 20, "float": 40},
    "grpo": {"negative": 60, "float": 120},
}
STAGE_TOTALS = {"sft": 200, "grpo": 600}


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


def select_tier_stratified(
    rows_by_id: dict[str, dict[str, Any]],
    available_ids: Sequence[str],
    *,
    count: int,
    rng: random.Random,
) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_id in available_ids:
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


def select_stage_domains(
    rows_by_id: dict[str, dict[str, Any]],
    stage_ids: Sequence[str],
    *,
    counts: dict[str, int],
    rng: random.Random,
) -> dict[str, list[str]]:
    hard_ids = [
        row_id
        for row_id in stage_ids
        if rows_by_id[row_id]["metadata"]["tier"] == "hard"
    ]
    rng.shuffle(hard_ids)
    selected = {"length": sorted(hard_ids[: counts["length"]])}
    remaining = sorted(set(stage_ids) - set(selected["length"]))
    selected["negative"] = select_tier_stratified(
        rows_by_id, remaining, count=counts["negative"], rng=rng
    )
    remaining = sorted(set(remaining) - set(selected["negative"]))
    selected["float"] = select_tier_stratified(
        rows_by_id, remaining, count=counts["float"], rng=rng
    )
    if any(
        set(selected[left]) & set(selected[right])
        for left, right in (("length", "negative"), ("length", "float"), ("negative", "float"))
    ):
        raise AssertionError("Domain replacement IDs overlap")
    return selected


def generate_domain_rows(
    domain: str,
    selected_ids: Sequence[str],
    rows_by_id: dict[str, dict[str, Any]],
    *,
    candidate_factory: Callable[[random.Random], Expression],
    mix_label: str,
    seed: int,
    seen_expressions: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    replacements: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for row_id in selected_ids:
        original = rows_by_id[row_id]
        required_tier = original["metadata"]["tier"]
        for _ in range(30_000):
            expression = candidate_factory(rng)
            rendered = expression.render()
            if rendered in seen_expressions or not is_useful_candidate(expression):
                continue
            if expression_structure(expression)["tier"] != required_tier:
                continue
            replacement = build_record(
                expression,
                record_id=row_id,
                rng=rng,
                split="train",
                ood_type=domain,
            )
            replacement["metadata"]["training_mix"] = f"combined_{mix_label}"
            replacement["metadata"]["training_domain"] = domain
            validate_record(replacement)
            replacements[row_id] = replacement
            seen_expressions.add(rendered)
            manifest_rows.append(
                {
                    "id": row_id,
                    "tier": required_tier,
                    "domain": domain,
                    "original_expression_sha256": canonical_sha256(
                        original["expression"]
                    ),
                    "replacement_expression": rendered,
                    "replacement_expression_sha256": canonical_sha256(rendered),
                }
            )
            break
        else:
            raise RuntimeError(f"Unable to generate {domain} replacement for {row_id}")
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
    parser.add_argument("--length-percent", type=int, choices=(10, 20), default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_percent = 70 - args.length_percent
    mix_label = f"{id_percent}_10_20_{args.length_percent}"
    args.output = args.output or Path(f"data/calculator_qwen3_combined_{mix_label}")
    args.manifest = args.manifest or Path(
        f"data/calculator_qwen3_combined_{mix_label}_manifest.json"
    )
    stage_counts = {
        stage: {
            "id": total
            - FIXED_STAGE_COUNTS[stage]["negative"]
            - FIXED_STAGE_COUNTS[stage]["float"]
            - total * args.length_percent // 100,
            **FIXED_STAGE_COUNTS[stage],
            "length": total * args.length_percent // 100,
        }
        for stage, total in STAGE_TOTALS.items()
    }
    if args.output.exists():
        raise FileExistsError(f"Remove {args.output} explicitly before regeneration")

    source = load_from_disk(str(args.source))
    if set(source) != {"train", *UNCHANGED_SPLITS}:
        raise AssertionError(f"Unexpected source splits: {set(source)}")
    source_rows = {
        split: [copy.deepcopy(dict(row)) for row in source[split]]
        for split in source
    }
    train_rows = source_rows["train"]
    rows_by_id = {row["id"]: row for row in train_rows}
    if len(train_rows) != 800 or len(rows_by_id) != 800:
        raise AssertionError("Expected 800 unique source training rows")

    sft_ids = read_ids(args.sft_ids)
    grpo_ids = read_ids(args.grpo_ids)
    if len(sft_ids) != 200 or len(grpo_ids) != 600:
        raise AssertionError("Expected frozen 200/600 stage IDs")
    if set(sft_ids) & set(grpo_ids) or set(sft_ids) | set(grpo_ids) != set(rows_by_id):
        raise AssertionError("Frozen stage IDs do not partition source train")

    selection_rng = random.Random(args.seed)
    stage_ids = {"sft": sft_ids, "grpo": grpo_ids}
    selected = {
        stage: select_stage_domains(
            rows_by_id,
            ids,
            counts=stage_counts[stage],
            rng=selection_rng,
        )
        for stage, ids in stage_ids.items()
    }

    seen_expressions = {
        row["expression"] for rows in source_rows.values() for row in rows
    }
    factories = {
        "negative": random_negative_ood_candidate,
        "float": random_float_ood_candidate,
        "length": random_length_ood_candidate,
    }
    replacements: dict[str, dict[str, Any]] = {}
    replacement_manifest: dict[str, Any] = {}
    seed_offset = 1
    for stage in ("sft", "grpo"):
        replacement_manifest[stage] = {}
        for domain in ("negative", "float", "length"):
            rows, rows_manifest = generate_domain_rows(
                domain,
                selected[stage][domain],
                rows_by_id,
                candidate_factory=factories[domain],
                mix_label=mix_label,
                seed=args.seed + seed_offset,
                seen_expressions=seen_expressions,
            )
            seed_offset += 1
            if set(replacements) & set(rows):
                raise AssertionError("Replacement pools overlap")
            replacements.update(rows)
            replacement_manifest[stage][domain] = rows_manifest

    mixed_train = [
        copy.deepcopy(replacements.get(row["id"], row)) for row in train_rows
    ]
    for stage, ids in stage_ids.items():
        counts = {
            domain: sum(
                replacements.get(row_id, {}).get("metadata", {}).get("training_domain")
                == domain
                for row_id in ids
            )
            for domain in ("negative", "float", "length")
        }
        counts["id"] = len(ids) - sum(counts.values())
        if counts != stage_counts[stage]:
            raise AssertionError((stage, counts, stage_counts[stage]))

    output_rows = {"train": mixed_train}
    output_rows.update(
        {split: copy.deepcopy(source_rows[split]) for split in UNCHANGED_SPLITS}
    )
    dataset = DatasetDict(
        {split: Dataset.from_list(rows) for split, rows in output_rows.items()}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output))
    reloaded = load_from_disk(str(args.output))
    for split in UNCHANGED_SPLITS:
        before = canonical_sha256([dict(row) for row in source[split]])
        after = canonical_sha256([dict(row) for row in reloaded[split]])
        if before != after:
            raise AssertionError(f"Unchanged split mutated: {split}")

    manifest = {
        "name": f"combined-generalization-{mix_label.replace('_', '-')}",
        "seed": args.seed,
        "source": str(args.source),
        "output": str(args.output),
        "stage_counts": stage_counts,
        "selected_ids": selected,
        "replacements": replacement_manifest,
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
    print(json.dumps(stage_counts, sort_keys=True))
    print(f"COMBINED_DATASET_READY {args.output}")
    print(f"MANIFEST_READY {args.manifest}")


if __name__ == "__main__":
    main()
