from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    log_error,
    log_info,
    log_success,
    log_warn,
    read_json,
    resolve_path,
    validate_chatml_dataset,
    write_json,
)
from src.data_pipeline.global_cleaner import normalize_text

DEFAULT_REPORT_PATH = "data/reports/processed_dataset_audit.json"
DEFAULT_DATASET_PATHS = (
    "data/processed/sft_guide_generation.json",
    "data/processed/sft_dialogue.json",
    "data/processed/sft_roleplay_safety.json",
)

USER_LENGTH_BUCKETS = (16, 32, 64, 128, 256)
ASSISTANT_LENGTH_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)
REAL_LIKE_SOURCE_PREFIXES = ("real", "pseudo_real")


def _get_message_content(sample: dict[str, Any], role: str) -> str:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == role:
            return normalize_text(message.get("content"))
    return ""


def _bucket_label(lower: int, upper: int | None) -> str:
    if upper is None:
        return f">{lower}"
    return f"{lower + 1}-{upper}"


def _bucketize_lengths(lengths: list[int], cutoffs: tuple[int, ...]) -> dict[str, int]:
    buckets: dict[str, int] = {}
    previous = -1
    for cutoff in cutoffs:
        buckets[_bucket_label(previous, cutoff)] = 0
        previous = cutoff
    buckets[_bucket_label(previous, None)] = 0

    for length in lengths:
        previous = -1
        assigned = False
        for cutoff in cutoffs:
            if length <= cutoff:
                buckets[_bucket_label(previous, cutoff)] += 1
                assigned = True
                break
            previous = cutoff
        if not assigned:
            buckets[_bucket_label(previous, None)] += 1
    return buckets


def _percentile(lengths: list[int], fraction: float) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths)
    index = math.ceil(fraction * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def _summarize_lengths(lengths: list[int], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "avg": 0.0,
            "p50": 0,
            "p90": 0,
            "buckets": _bucketize_lengths([], cutoffs),
        }

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
        "p50": _percentile(lengths, 0.5),
        "p90": _percentile(lengths, 0.9),
        "buckets": _bucketize_lengths(lengths, cutoffs),
    }


def _classify_source_origin(source: str) -> str:
    normalized = normalize_text(source).lower()
    if not normalized:
        return "unknown"
    if normalized.startswith(REAL_LIKE_SOURCE_PREFIXES):
        return "real_like"
    return "synthetic"


def summarize_dataset(dataset: list[dict[str, Any]], *, dataset_name: str) -> dict[str, Any]:
    user_lengths: list[int] = []
    assistant_lengths: list[int] = []
    source_counts: Counter[str] = Counter()
    source_origin_counts: Counter[str] = Counter()

    for sample in dataset:
        user_content = _get_message_content(sample, "user")
        assistant_content = _get_message_content(sample, "assistant")
        user_lengths.append(len(user_content))
        assistant_lengths.append(len(assistant_content))

        source = normalize_text(sample.get("source"))
        if source:
            source_counts[source] += 1
            source_origin_counts[_classify_source_origin(source)] += 1

    return {
        "dataset_name": dataset_name,
        "sample_count": len(dataset),
        "user_length": _summarize_lengths(user_lengths, USER_LENGTH_BUCKETS),
        "assistant_length": _summarize_lengths(assistant_lengths, ASSISTANT_LENGTH_BUCKETS),
        "source_counts": dict(source_counts),
        "source_origin_counts": dict(source_origin_counts),
        "source_stats_available": bool(source_counts),
    }


def audit_dataset_file(dataset_path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(dataset_path)
    dataset_name = resolved.name
    result: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }

    if not resolved.exists():
        result["valid_chatml"] = False
        result["errors"] = [f"文件不存在: {resolved}"]
        return result

    dataset = read_json(resolved)
    errors = validate_chatml_dataset(dataset)
    result["valid_chatml"] = not errors
    result["errors"] = errors[:20]
    if errors:
        return result

    result["summary"] = summarize_dataset(dataset, dataset_name=dataset_name)
    return result


def audit_processed_datasets(
    dataset_paths: list[str | Path] | None = None,
    *,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    configure_console_output()
    effective_paths = list(dataset_paths or DEFAULT_DATASET_PATHS)
    report: dict[str, Any] = {
        "datasets": {},
        "all_valid_chatml": True,
    }

    for dataset_path in effective_paths:
        resolved = resolve_path(dataset_path)
        log_info(f"开始验收数据集: {resolved}")
        dataset_report = audit_dataset_file(resolved)
        report["datasets"][resolved.name] = dataset_report
        if not dataset_report.get("valid_chatml", False):
            report["all_valid_chatml"] = False

    output_path = write_json(report_path, report)
    report["report_path"] = str(output_path)
    return report


def _log_dataset_summary(dataset_name: str, report: dict[str, Any]) -> None:
    if not report.get("exists"):
        log_error(f"{dataset_name}: 文件不存在")
        return
    if not report.get("valid_chatml"):
        log_error(f"{dataset_name}: ChatML 校验失败")
        for error in report.get("errors", [])[:5]:
            log_error(f"  - {error}")
        return

    summary = report["summary"]
    log_success(f"{dataset_name}: ChatML 校验通过，样本量 {summary['sample_count']}")
    log_info(
        f"{dataset_name}: user 长度 p50/p90={summary['user_length']['p50']}/{summary['user_length']['p90']}，"
        f"assistant 长度 p50/p90={summary['assistant_length']['p50']}/{summary['assistant_length']['p90']}"
    )
    if summary.get("source_stats_available"):
        log_info(f"{dataset_name}: 来源分布 {summary['source_counts']}")
        if summary.get("source_origin_counts"):
            log_info(f"{dataset_name}: 真实/合成分布 {summary['source_origin_counts']}")
    else:
        log_warn(f"{dataset_name}: 当前无法统计来源分布，样本内未保留 source 元数据")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验收 processed ChatML 数据集并生成统计报告。")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="待验收的 processed JSON 文件，可多次传入；默认验收当前主训练集。",
    )
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="验收报告输出路径。")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = audit_processed_datasets(args.file or None, report_path=args.report)
    for dataset_name, dataset_report in report["datasets"].items():
        _log_dataset_summary(dataset_name, dataset_report)

    log_info(f"验收报告已写入: {report['report_path']}")
    return 0 if report["all_valid_chatml"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
