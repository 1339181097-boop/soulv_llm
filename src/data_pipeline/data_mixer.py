from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    log_info,
    log_success,
    log_warn,
    read_json,
    resolve_path,
    validate_chatml_dataset,
    write_json,
)

DEFAULT_OUTPUT_PATH = "data/final/stage1_general_sft.json"
DEFAULT_REPORT_PATH = "data/final/stage_mix_report.json"
DEFAULT_SEED = 42
DEFAULT_SPECS = {
    "sft_guide_generation.json": 0.50,
    "sft_dialogue.json": 0.30,
    "sft_roleplay_safety.json": 0.20,
}


@dataclass(frozen=True)
class DatasetBucket:
    filename: str
    weight: float
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class StageRecipe:
    name: str
    output_path: str
    total_samples: int
    seed: int
    specs: dict[str, float]


DEFAULT_STAGE_RECIPES = (
    StageRecipe(
        name="stage1_general_sft",
        output_path="data/final/stage1_general_sft.json",
        total_samples=960,
        seed=42,
        specs={
            "sft_guide_generation.json": 0.50,
            "sft_dialogue.json": 0.30,
            "sft_roleplay_safety.json": 0.20,
        },
    ),
)


def _parse_specs(raw_specs: list[str]) -> dict[str, float]:
    specs: dict[str, float] = {}
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"Invalid spec: {raw_spec}. Expected filename=weight")
        filename, raw_weight = raw_spec.split("=", 1)
        weight = float(raw_weight)
        if weight <= 0:
            raise ValueError(f"Weight must be > 0: {raw_spec}")
        specs[filename.strip()] = weight
    return specs


def _load_bucket(filename: str, weight: float) -> DatasetBucket | None:
    path = resolve_path(f"data/processed/{filename}")
    if not path.exists():
        log_warn(f"Missing processed dataset, skipped: {path}")
        return None

    dataset = read_json(path)
    errors = validate_chatml_dataset(dataset)
    if errors:
        log_warn(f"Invalid ChatML dataset, skipped: {path}")
        for error in errors[:5]:
            log_warn(error)
        return None

    return DatasetBucket(filename=filename, weight=weight, records=list(dataset))


def _resolve_target_counts(buckets: list[DatasetBucket], total_samples: int) -> dict[str, int]:
    total_weight = sum(bucket.weight for bucket in buckets)
    if total_weight <= 0:
        raise ValueError("Total weight must be > 0")

    exact_counts = []
    allocated = 0
    for index, bucket in enumerate(buckets):
        exact = total_samples * (bucket.weight / total_weight)
        base = math.floor(exact)
        exact_counts.append((index, bucket.filename, base, exact - base))
        allocated += base

    remainder = total_samples - allocated
    exact_counts.sort(key=lambda item: (-item[3], item[0]))

    requested: dict[str, int] = {filename: base for _, filename, base, _ in exact_counts}
    for _, filename, _, _ in exact_counts[:remainder]:
        requested[filename] += 1
    return requested


def _sample_records(records: list[dict[str, Any]], target: int, rng: random.Random) -> tuple[list[dict[str, Any]], int]:
    if target <= 0:
        return [], 0
    if not records:
        return [], 0
    if target <= len(records):
        return rng.sample(records, target), 0

    sampled: list[dict[str, Any]] = []
    duplicates = 0
    full_cycles, remainder = divmod(target, len(records))
    for _ in range(full_cycles):
        sampled.extend(rng.sample(records, len(records)))
    if remainder:
        sampled.extend(rng.sample(records, remainder))
    duplicates = target - len(records)
    return sampled, duplicates


def _message_content(sample: dict[str, Any], role: str) -> str:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _percentile(lengths: list[int], fraction: float) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min": 0, "avg": 0.0, "p50": 0, "p90": 0, "max": 0}
    return {
        "min": min(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
        "p50": _percentile(lengths, 0.5),
        "p90": _percentile(lengths, 0.9),
        "max": max(lengths),
    }


def build_mixed_dataset(
    output_json_path: str,
    *,
    seed: int,
    total_samples: int | None,
    specs: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_console_output()
    rng = random.Random(seed)
    buckets: list[DatasetBucket] = []

    for filename, weight in specs.items():
        bucket = _load_bucket(filename, weight)
        if bucket is not None and bucket.records:
            buckets.append(bucket)

    if not buckets:
        log_warn("No processed datasets available for mixing.")
        return [], {"sample_count": 0, "dataset_counts": {}}

    requested_counts = _resolve_target_counts(buckets, total_samples) if total_samples is not None else {
        bucket.filename: len(bucket.records) for bucket in buckets
    }

    mixed: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    dataset_sizes: dict[str, int] = {}
    oversample_counts: dict[str, int] = {}

    for bucket in buckets:
        dataset_sizes[bucket.filename] = len(bucket.records)
        target = requested_counts[bucket.filename]
        sampled, duplicates = _sample_records(bucket.records, target, rng)
        mixed.extend(sampled)
        dataset_counts[bucket.filename] = len(sampled)
        oversample_counts[bucket.filename] = duplicates
        if duplicates > 0:
            log_warn(
                f"Oversampled {bucket.filename}: requested {target}, unique {len(bucket.records)}, duplicated {duplicates}"
            )

    rng.shuffle(mixed)
    output_path = write_json(output_json_path, mixed)

    user_lengths = [len(_message_content(sample, "user")) for sample in mixed]
    assistant_lengths = [len(_message_content(sample, "assistant")) for sample in mixed]
    report = {
        "output_path": str(output_path),
        "seed": seed,
        "target_total_samples": total_samples,
        "sample_count": len(mixed),
        "dataset_counts": dataset_counts,
        "dataset_sizes": dataset_sizes,
        "oversample_counts": oversample_counts,
        "user_length": _length_summary(user_lengths),
        "assistant_length": _length_summary(assistant_lengths),
    }

    log_success(f"Mixed dataset written: {output_path} ({len(mixed)} samples)")
    log_info(f"Dataset counts: {dataset_counts}")
    return mixed, report


def mix_datasets(
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    seed: int = DEFAULT_SEED,
    total_samples: int | None = None,
    specs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    effective_specs = specs or DEFAULT_SPECS
    mixed, _ = build_mixed_dataset(
        output_json_path,
        seed=seed,
        total_samples=total_samples,
        specs=effective_specs,
    )
    return mixed


def build_stage_datasets(
    recipes: tuple[StageRecipe, ...] = DEFAULT_STAGE_RECIPES,
    *,
    report_path: str = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    configure_console_output()
    stage_reports: dict[str, Any] = {}
    for recipe in recipes:
        _, report = build_mixed_dataset(
            recipe.output_path,
            seed=recipe.seed,
            total_samples=recipe.total_samples,
            specs=recipe.specs,
        )
        report["name"] = recipe.name
        report["specs"] = recipe.specs
        stage_reports[recipe.name] = report

    payload = {
        "stages": stage_reports,
    }
    write_json(report_path, payload)
    log_success(f"Stage mix report written: {resolve_path(report_path)}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix processed ChatML datasets.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path for one-off mixing.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--total-samples", type=int, default=None, help="Target sample count for one-off mixing.")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Dataset spec in filename=weight format. May be passed multiple times.",
    )
    parser.add_argument(
        "--stage",
        choices=("none", "current"),
        default="none",
        help="Build the current stage dataset recipe instead of a one-off mixed dataset.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help="Stage report output path when --stage current is used.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.stage == "current":
        build_stage_datasets(report_path=args.report)
        return

    specs = _parse_specs(args.spec) if args.spec else DEFAULT_SPECS
    mix_datasets(args.output, seed=args.seed, total_samples=args.total_samples, specs=specs)


if __name__ == "__main__":
    main()
