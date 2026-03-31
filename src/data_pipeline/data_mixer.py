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
GUIDE_ASSISTANT_SHARE_THRESHOLD = 0.45
LONG_FORM_COMBINED_ASSISTANT_SHARE_THRESHOLD = 0.58

# Weighted mixing remains available for ad-hoc runs, but stage1 should prefer
# explicit six-bucket counts that follow the current docs recipe.
DEFAULT_SPECS = {
    "sft_guide_generation.json": 0.20,
    "sft_travel_qa.json": 0.25,
    "sft_hotel_recommendation.json": 0.19,
    "sft_traffic_planning.json": 0.12,
    "sft_persona_understanding_strict.json": 0.12,
    "sft_multi_turn_dialogue.json": 0.12,
}

DOCS_BASELINE_TARGET_COUNTS = {
    "sft_guide_generation.json": 800,
    "sft_travel_qa.json": 1000,
    "sft_hotel_recommendation.json": 750,
    "sft_traffic_planning.json": 500,
    "sft_persona_understanding_strict.json": 500,
    "sft_multi_turn_dialogue.json": 500,
}

ASSISTANT_AWARE_CORRECTED_TARGET_COUNTS = {
    "sft_guide_generation.json": 250,
    "sft_travel_qa.json": 1250,
    "sft_hotel_recommendation.json": 750,
    "sft_traffic_planning.json": 794,
    "sft_persona_understanding_strict.json": 500,
    "sft_multi_turn_dialogue.json": 350,
}

DEFAULT_TARGET_COUNTS = DOCS_BASELINE_TARGET_COUNTS


@dataclass(frozen=True)
class DatasetBucket:
    filename: str
    weight: float
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class StageRecipe:
    name: str
    output_path: str
    total_samples: int | None
    seed: int
    specs: dict[str, float]
    target_counts: dict[str, int] | None = None


DEFAULT_STAGE_RECIPES = (
    StageRecipe(
        name="stage1_general_sft",
        output_path=DEFAULT_OUTPUT_PATH,
        total_samples=None,
        seed=DEFAULT_SEED,
        specs={},
        target_counts=DOCS_BASELINE_TARGET_COUNTS,
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


def _parse_counts(raw_counts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_count in raw_counts:
        if "=" not in raw_count:
            raise ValueError(f"Invalid count: {raw_count}. Expected filename=count")
        filename, raw_value = raw_count.split("=", 1)
        count = int(raw_value)
        if count < 0:
            raise ValueError(f"Count must be >= 0: {raw_count}")
        counts[filename.strip()] = count
    return counts


def _role_total_length(sample: dict[str, Any], role: str) -> int:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def _sample_total_length(sample: dict[str, Any]) -> int:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def _task_type_for_records(records: list[dict[str, Any]], filename: str) -> str:
    for sample in records:
        task_type = sample.get("task_type")
        if isinstance(task_type, str) and task_type:
            return task_type
    return filename.removeprefix("sft_").removesuffix(".json")


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


def _bucket_length_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    user_lengths = [_role_total_length(sample, "user") for sample in records]
    assistant_lengths = [_role_total_length(sample, "assistant") for sample in records]
    total_lengths = [_sample_total_length(sample) for sample in records]
    return {
        "user_length": _length_summary(user_lengths),
        "assistant_length": _length_summary(assistant_lengths),
        "total_length": _length_summary(total_lengths),
        "avg_total_length": round(sum(total_lengths) / len(total_lengths), 2) if total_lengths else 0.0,
        "avg_assistant_length": round(sum(assistant_lengths) / len(assistant_lengths), 2) if assistant_lengths else 0.0,
    }


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


def _load_requested_buckets(
    *,
    target_counts: dict[str, int] | None,
    specs: dict[str, float] | None,
    total_samples: int | None,
) -> tuple[list[DatasetBucket], dict[str, int], list[str]]:
    buckets: list[DatasetBucket] = []
    missing_requested: list[str] = []

    if target_counts:
        requested_counts = {filename: count for filename, count in target_counts.items() if count > 0}
        for filename in requested_counts:
            bucket = _load_bucket(filename, 1.0)
            if bucket is None or not bucket.records:
                missing_requested.append(filename)
                continue
            buckets.append(bucket)
        return buckets, requested_counts, missing_requested

    effective_specs = specs or DEFAULT_SPECS
    for filename, weight in effective_specs.items():
        bucket = _load_bucket(filename, weight)
        if bucket is not None and bucket.records:
            buckets.append(bucket)

    if total_samples is not None:
        requested_counts = _resolve_target_counts(buckets, total_samples)
    else:
        requested_counts = {bucket.filename: len(bucket.records) for bucket in buckets}
    return buckets, requested_counts, missing_requested


def _build_bucket_audit(
    buckets: list[DatasetBucket],
    requested_counts: dict[str, int],
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    projected_total_length_all = 0.0
    projected_assistant_length_all = 0.0

    for bucket in buckets:
        length_report = _bucket_length_report(bucket.records)
        target_count = requested_counts.get(bucket.filename, 0)
        projected_total_length = round(length_report["avg_total_length"] * target_count, 2)
        projected_assistant_length = round(length_report["avg_assistant_length"] * target_count, 2)
        projected_total_length_all += projected_total_length
        projected_assistant_length_all += projected_assistant_length
        audits[bucket.filename] = {
            "task_type": _task_type_for_records(bucket.records, bucket.filename),
            "source_sample_count": len(bucket.records),
            "target_sample_count": target_count,
            "length_stats": length_report,
            "projected_total_length": projected_total_length,
            "projected_assistant_length": projected_assistant_length,
        }

    for audit in audits.values():
        audit["projected_total_length_share"] = round(
            audit["projected_total_length"] / projected_total_length_all,
            4,
        ) if projected_total_length_all > 0 else 0.0
        audit["projected_assistant_share"] = round(
            audit["projected_assistant_length"] / projected_assistant_length_all,
            4,
        ) if projected_assistant_length_all > 0 else 0.0

    return audits


def _choose_assistant_aware_target_counts(
    bucket_audits: dict[str, Any],
    baseline_counts: dict[str, int],
) -> tuple[dict[str, int], dict[str, Any]]:
    guide_share = bucket_audits.get("sft_guide_generation.json", {}).get("projected_assistant_share", 0.0)
    multi_share = bucket_audits.get("sft_multi_turn_dialogue.json", {}).get("projected_assistant_share", 0.0)
    long_form_combined_share = round(guide_share + multi_share, 4)

    use_corrected_counts = (
        guide_share > GUIDE_ASSISTANT_SHARE_THRESHOLD
        or long_form_combined_share > LONG_FORM_COMBINED_ASSISTANT_SHARE_THRESHOLD
    )
    final_counts = ASSISTANT_AWARE_CORRECTED_TARGET_COUNTS if use_corrected_counts else baseline_counts

    decision = {
        "mode": "docs_baseline" if not use_corrected_counts else "assistant_aware_correction",
        "guide_assistant_share_threshold": GUIDE_ASSISTANT_SHARE_THRESHOLD,
        "long_form_combined_assistant_share_threshold": LONG_FORM_COMBINED_ASSISTANT_SHARE_THRESHOLD,
        "docs_baseline_counts": baseline_counts,
        "corrected_counts": ASSISTANT_AWARE_CORRECTED_TARGET_COUNTS,
        "baseline_guide_assistant_share": round(guide_share, 4),
        "baseline_multi_turn_assistant_share": round(multi_share, 4),
        "baseline_long_form_combined_assistant_share": long_form_combined_share,
        "reason": (
            "Docs baseline is already reasonable on the assistant side, so keep it."
            if not use_corrected_counts
            else "Docs baseline still lets long-form assistant outputs dominate SFT supervision, so apply a one-step assistant-aware correction without oversampling."
        ),
    }
    return final_counts, decision


def _build_report(
    *,
    output_path: Path,
    seed: int,
    total_samples: int | None,
    requested_counts: dict[str, int],
    dataset_counts: dict[str, int],
    dataset_sizes: dict[str, int],
    oversample_counts: dict[str, int],
    missing_requested: list[str],
    mixed: list[dict[str, Any]],
    bucket_audits: dict[str, Any],
    strategy: dict[str, Any] | None,
) -> dict[str, Any]:
    user_lengths = [_role_total_length(sample, "user") for sample in mixed]
    assistant_lengths = [_role_total_length(sample, "assistant") for sample in mixed]
    total_lengths = [_sample_total_length(sample) for sample in mixed]
    task_type_counts: dict[str, int] = {}
    for sample in mixed:
        task_type = sample.get("task_type")
        if isinstance(task_type, str):
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    return {
        "output_path": str(output_path),
        "seed": seed,
        "target_total_samples": total_samples,
        "requested_counts": requested_counts,
        "sample_count": len(mixed),
        "dataset_counts": dataset_counts,
        "task_type_counts": task_type_counts,
        "dataset_sizes": dataset_sizes,
        "oversample_counts": oversample_counts,
        "missing_requested": missing_requested,
        "bucket_audits": bucket_audits,
        "user_length": _length_summary(user_lengths),
        "assistant_length": _length_summary(assistant_lengths),
        "total_length": _length_summary(total_lengths),
        "strategy": strategy,
    }


def build_mixed_dataset(
    output_json_path: str,
    *,
    seed: int,
    total_samples: int | None,
    specs: dict[str, float] | None = None,
    target_counts: dict[str, int] | None = None,
    strategy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_console_output()
    rng = random.Random(seed)
    buckets, requested_counts, missing_requested = _load_requested_buckets(
        target_counts=target_counts,
        specs=specs,
        total_samples=total_samples,
    )

    if not buckets:
        log_warn("No processed datasets available for mixing.")
        return [], {"sample_count": 0, "dataset_counts": {}, "missing_requested": missing_requested}

    bucket_audits = _build_bucket_audit(buckets, requested_counts)
    mixed: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    dataset_sizes: dict[str, int] = {}
    oversample_counts: dict[str, int] = {}

    for bucket in buckets:
        dataset_sizes[bucket.filename] = len(bucket.records)
        target = requested_counts.get(bucket.filename, 0)
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
    report = _build_report(
        output_path=output_path,
        seed=seed,
        total_samples=total_samples,
        requested_counts=requested_counts,
        dataset_counts=dataset_counts,
        dataset_sizes=dataset_sizes,
        oversample_counts=oversample_counts,
        missing_requested=missing_requested,
        mixed=mixed,
        bucket_audits=bucket_audits,
        strategy=strategy,
    )

    if missing_requested:
        log_warn(f"Missing requested datasets: {missing_requested}")
    log_success(f"Mixed dataset written: {output_path} ({len(mixed)} samples)")
    log_info(f"Dataset counts: {dataset_counts}")
    return mixed, report


def mix_datasets(
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    seed: int = DEFAULT_SEED,
    total_samples: int | None = None,
    specs: dict[str, float] | None = None,
    target_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    effective_target_counts = target_counts
    effective_specs = specs
    if effective_target_counts is None and effective_specs is None and total_samples is None:
        effective_target_counts = DEFAULT_TARGET_COUNTS

    mixed, _ = build_mixed_dataset(
        output_json_path,
        seed=seed,
        total_samples=total_samples,
        specs=effective_specs,
        target_counts=effective_target_counts,
    )
    return mixed


def _build_stage_dataset(recipe: StageRecipe) -> dict[str, Any]:
    baseline_counts = recipe.target_counts or DOCS_BASELINE_TARGET_COUNTS
    baseline_buckets, _, missing_requested = _load_requested_buckets(
        target_counts=baseline_counts,
        specs=recipe.specs,
        total_samples=recipe.total_samples,
    )
    baseline_audits = _build_bucket_audit(baseline_buckets, baseline_counts)
    final_counts, strategy = _choose_assistant_aware_target_counts(baseline_audits, baseline_counts)
    strategy["baseline_bucket_audits"] = baseline_audits
    strategy["missing_requested"] = missing_requested

    _, report = build_mixed_dataset(
        recipe.output_path,
        seed=recipe.seed,
        total_samples=recipe.total_samples,
        specs=recipe.specs,
        target_counts=final_counts,
        strategy=strategy,
    )
    report["name"] = recipe.name
    report["specs"] = recipe.specs
    report["target_counts"] = final_counts
    return report


def build_stage_datasets(
    recipes: tuple[StageRecipe, ...] = DEFAULT_STAGE_RECIPES,
    *,
    report_path: str = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    configure_console_output()
    stage_reports: dict[str, Any] = {}
    for recipe in recipes:
        stage_reports[recipe.name] = _build_stage_dataset(recipe)

    payload = {"stages": stage_reports}
    write_json(report_path, payload)
    log_success(f"Stage mix report written: {resolve_path(report_path)}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix processed ChatML datasets.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path for one-off mixing.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--total-samples", type=int, default=None, help="Target sample count for weighted mixing.")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Dataset spec in filename=weight format. May be passed multiple times.",
    )
    parser.add_argument(
        "--count",
        action="append",
        default=[],
        help="Dataset target in filename=count format. May be passed multiple times.",
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

    if args.spec and args.count:
        raise ValueError("Use either --spec or --count in one run, not both.")

    specs = _parse_specs(args.spec) if args.spec else None
    target_counts = _parse_counts(args.count) if args.count else None
    mix_datasets(
        args.output,
        seed=args.seed,
        total_samples=args.total_samples,
        specs=specs,
        target_counts=target_counts,
    )


if __name__ == "__main__":
    main()

