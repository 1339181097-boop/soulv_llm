from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    iter_jsonl,
    log_info,
    log_success,
    log_warn,
    resolve_path,
    write_json,
    write_jsonl,
)
from src.data_pipeline.dedup_badcase_repairs import (
    DEFAULT_ANSWER_THRESHOLD,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_PROMPT_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    deduplicate_records,
)
from src.data_pipeline.global_cleaner import normalize_text

DEFAULT_INPUT_PATH = "data/badcase_fix/qwen3_32b_stage1_text_badcase_repair_expanded_v2.jsonl"
DEFAULT_SOURCE_BADCASES_PATH = "data/badcase_fix/qwen3_32b_stage1_text_full_badcases_v5.jsonl"
ALLOWED_TASK_TYPES = {
    "guide_generation",
    "hotel_recommendation",
    "multi_turn_dialogue",
    "persona_understanding",
    "traffic_planning",
    "travel_qa",
}
TASK_ANSWER_MAX_LENGTH = {
    "guide_generation": 2600,
    "hotel_recommendation": 1800,
    "multi_turn_dialogue": 2200,
    "persona_understanding": 1600,
    "traffic_planning": 1800,
    "travel_qa": 1400,
}
DEFAULT_MAX_ANSWER_LENGTH = 1800
MIN_ANSWER_LENGTH = 30
MIN_USER_LENGTH = 6

FORBIDDEN_TRACE_PATTERNS = (
    (re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE), "contains_think_tag"),
    (re.compile(r"\b(function_call|tool_call|arguments)\b", re.IGNORECASE), "contains_tool_trace"),
    (re.compile(r"```+\s*json", re.IGNORECASE), "contains_json_block"),
    (re.compile(r"(内部推理|思考过程|工具调用|函数调用|调用工具|输出\s*JSON)"), "contains_internal_or_tool_text"),
)
LIVE_FACT_PATTERNS = (
    (re.compile(r"([￥¥]\s*\d+|\d+(?:\.\d+)?\s*(?:元|块钱|人民币|RMB))", re.IGNORECASE), "contains_exact_price"),
    (re.compile(r"\b(?:G|D|C|Z|T|K)\d{2,5}\b|(?:高铁|动车|火车)\s*(?:G|D|C|Z|T|K)\d{2,5}", re.IGNORECASE), "contains_train_number"),
    (re.compile(r"\b[A-Z]{2}\d{3,4}\b"), "contains_flight_number"),
    (re.compile(r"\d{1,2}:\d{2}\s*(?:-|–|—|~|～|到|至)\s*\d{1,2}:\d{2}"), "contains_opening_hours"),
    (re.compile(r"\d{1,2}\s*点(?:开门|闭馆|关门|营业|停止入场)"), "contains_opening_hours"),
    (re.compile(r"(?:库存|余票|有票|售罄|满房|有房|余房)"), "contains_inventory_claim"),
    (re.compile(r"\d+(?:\.\d+)?\s*(?:分钟|小时|h|min)\b", re.IGNORECASE), "contains_precise_duration"),
)
META_ANSWER_PATTERNS = (
    (re.compile(r"(作为(?:一个)?AI|我是(?:一个)?AI|我不能提供|无法提供任何)"), "ai_meta_answer"),
    (re.compile(r"(示例答案|修复答案|以下是.*答案)"), "answer_meta_label"),
    (re.compile(r"(保持不变|本次调整|修改如下|调整如下)"), "answer_patch_label"),
)


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _with_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}")
    return path.with_name(f"{path.name}{suffix}")


def _normalized_for_exact_match(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def _cjk_ratio(text: str) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in normalized)
    return cjk_count / len(normalized)


def _message_contents(record: dict[str, Any], role: str | None = None) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []

    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if role is not None and message.get("role") != role:
            continue
        content = normalize_text(message.get("content"))
        if content:
            contents.append(content)
    return contents


def _role_counts(record: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    messages = record.get("messages")
    if not isinstance(messages, list):
        return counter
    for message in messages:
        if isinstance(message, dict):
            role = normalize_text(message.get("role"))
            if role:
                counter[role] += 1
    return counter


def _load_source_prompts(path: str | Path) -> dict[str, str]:
    source_path = resolve_path(path)
    prompts: dict[str, str] = {}
    for _, record in iter_jsonl(source_path):
        badcase_id = normalize_text(record.get("badcase_id"))
        if not badcase_id:
            continue
        prompts[badcase_id] = "\n".join(_message_contents(record, "user"))
    return prompts


def _has_json_like_answer(answer: str) -> bool:
    stripped = answer.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    if stripped.endswith("}") and stripped.count("{") >= 1:
        return True
    return False


def _validate_required_fields(record: dict[str, Any], source_prompts: dict[str, str]) -> list[str]:
    errors: list[str] = []
    source_badcase_id = normalize_text(record.get("source_badcase_id"))
    repair_type = normalize_text(record.get("repair_type"))
    task_type = normalize_text(record.get("task_type"))
    answer = normalize_text(record.get("answer"))
    messages = record.get("messages")
    allow_assistant_history = task_type == "multi_turn_dialogue" or repair_type == "multi_turn"

    if not source_badcase_id:
        errors.append("missing_source_badcase_id")
    elif source_badcase_id not in source_prompts:
        errors.append("source_badcase_id_not_found")
    if not repair_type:
        errors.append("missing_repair_type")
    if task_type not in ALLOWED_TASK_TYPES:
        errors.append("invalid_task_type")
    if not isinstance(messages, list) or not messages:
        errors.append("invalid_messages")
    if not answer:
        errors.append("missing_answer")

    if isinstance(messages, list):
        role_counts = _role_counts(record)
        if role_counts["system"] < 1:
            errors.append("missing_system_message")
        if role_counts["user"] < 1:
            errors.append("missing_user_message")
        if role_counts["assistant"] > 0 and not allow_assistant_history:
            errors.append("assistant_message_should_be_answer_field")
        for message in messages:
            if not isinstance(message, dict):
                errors.append("invalid_message_item")
                continue
            valid_roles = {"system", "user", "assistant"} if allow_assistant_history else {"system", "user"}
            if normalize_text(message.get("role")) not in valid_roles:
                errors.append("invalid_message_role")
            if not normalize_text(message.get("content")):
                errors.append("empty_message_content")

    fix_tags = record.get("fix_tags")
    if not isinstance(fix_tags, list) or not all(isinstance(item, str) and item.strip() for item in fix_tags):
        errors.append("invalid_fix_tags")
    return errors


def _validate_content(record: dict[str, Any], source_prompts: dict[str, str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    task_type = normalize_text(record.get("task_type"))
    answer = normalize_text(record.get("answer"))
    user_text = "\n".join(_message_contents(record, "user"))
    source_badcase_id = normalize_text(record.get("source_badcase_id"))
    source_prompt = source_prompts.get(source_badcase_id, "")

    if len(user_text) < MIN_USER_LENGTH:
        errors.append("user_prompt_too_short")
    if len(answer) < MIN_ANSWER_LENGTH:
        errors.append("answer_too_short")
    max_answer_length = TASK_ANSWER_MAX_LENGTH.get(task_type, DEFAULT_MAX_ANSWER_LENGTH)
    if len(answer) > max_answer_length:
        errors.append("answer_too_long")

    if user_text and _cjk_ratio(user_text) < 0.25:
        errors.append("user_prompt_low_chinese_ratio")
    if answer and _cjk_ratio(answer) < 0.18:
        errors.append("answer_low_chinese_ratio")

    if source_prompt and _normalized_for_exact_match(user_text) == _normalized_for_exact_match(source_prompt):
        errors.append("copies_source_prompt")

    if _has_json_like_answer(answer):
        errors.append("answer_looks_like_json")

    for pattern, reason in FORBIDDEN_TRACE_PATTERNS:
        if pattern.search(answer):
            errors.append(reason)
    for pattern, reason in META_ANSWER_PATTERNS:
        if pattern.search(answer):
            errors.append(reason)
    for pattern, reason in LIVE_FACT_PATTERNS:
        if pattern.search(answer):
            errors.append(reason)

    if "以官方/导航为准" not in answer and ("交通" in answer or "票务" in answer or "门票" in answer):
        warnings.append("missing_official_or_navigation_caveat")
    return errors, warnings


def validate_record(record: dict[str, Any], source_prompts: dict[str, str]) -> ValidationResult:
    errors = _validate_required_fields(record, source_prompts)
    if not errors:
        content_errors, warnings = _validate_content(record, source_prompts)
        errors.extend(content_errors)
        return ValidationResult(errors=tuple(dict.fromkeys(errors)), warnings=tuple(dict.fromkeys(warnings)))
    return ValidationResult(errors=tuple(dict.fromkeys(errors)), warnings=())


def clean_records(
    records: list[dict[str, Any]],
    *,
    source_prompts: dict[str, str],
    line_numbers: list[int | None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    effective_line_numbers = line_numbers or [None] * len(records)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    error_counter: Counter[str] = Counter()
    warning_counter: Counter[str] = Counter()

    for record, line_number in zip(records, effective_line_numbers, strict=True):
        result = validate_record(record, source_prompts)
        if result.warnings:
            warning_counter.update(result.warnings)
        if result.errors:
            rejected_record = copy.deepcopy(record)
            rejected_record["_cleaning"] = {
                "line_number": line_number,
                "errors": list(result.errors),
                "warnings": list(result.warnings),
            }
            rejected.append(rejected_record)
            error_counter.update(result.errors)
            continue
        kept.append(record)

    summary = {
        "input_count": len(records),
        "local_kept_count": len(kept),
        "local_rejected_count": len(rejected),
        "local_error_counts": dict(error_counter.most_common()),
        "local_warning_counts": dict(warning_counter.most_common()),
        "input_by_task_type": dict(Counter(normalize_text(record.get("task_type")) or "unknown" for record in records)),
        "local_kept_by_task_type": dict(Counter(normalize_text(record.get("task_type")) or "unknown" for record in kept)),
        "local_rejected_by_task_type": dict(Counter(normalize_text(record.get("task_type")) or "unknown" for record in rejected)),
    }
    return kept, rejected, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict local cleaning for badcase repair expansion JSONL records.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input badcase repair JSONL path.")
    parser.add_argument("--source-badcases", default=DEFAULT_SOURCE_BADCASES_PATH, help="Original badcase JSONL used to verify traceability and copied prompts.")
    parser.add_argument("--output", default=None, help="Final locally cleaned JSONL. Defaults to <input>.local_clean_final.jsonl.")
    parser.add_argument("--rejected", default=None, help="Rejected JSONL. Defaults to <input>.local_clean_rejected.jsonl.")
    parser.add_argument("--summary", default=None, help="Summary JSON. Defaults to <input>.local_clean_summary.json.")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip exact/similarity dedup after local validation.")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--prompt-threshold", type=float, default=DEFAULT_PROMPT_THRESHOLD)
    parser.add_argument("--answer-threshold", type=float, default=DEFAULT_ANSWER_THRESHOLD)
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    parser.add_argument("--similarity-scope", choices=["global", "source", "task"], default="task")
    return parser


def main() -> int:
    configure_console_output()
    args = _build_arg_parser().parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output) if args.output else _with_suffix(input_path, ".local_clean_final.jsonl")
    rejected_path = resolve_path(args.rejected) if args.rejected else _with_suffix(input_path, ".local_clean_rejected.jsonl")
    summary_path = resolve_path(args.summary) if args.summary else _with_suffix(input_path, ".local_clean_summary.json")

    source_prompts = _load_source_prompts(args.source_badcases)
    if not source_prompts:
        log_warn("No source badcases loaded; traceability validation will reject all records.")
    records_with_lines = list(iter_jsonl(input_path))
    records = [record for _, record in records_with_lines]
    line_numbers = [line_number for line_number, _ in records_with_lines]
    log_info(f"Loaded {len(records)} repair records from {input_path}")
    log_info(f"Loaded {len(source_prompts)} source badcase prompts from {resolve_path(args.source_badcases)}")

    local_kept, local_rejected, summary = clean_records(records, source_prompts=source_prompts, line_numbers=line_numbers)

    dedup_rejected: list[dict[str, Any]] = []
    if args.skip_dedup:
        final_kept = local_kept
        summary["dedup"] = {"skipped": True}
    else:
        final_kept, dedup_rejected, dedup_summary = deduplicate_records(
            local_kept,
            similarity_threshold=args.similarity_threshold,
            prompt_threshold=args.prompt_threshold,
            answer_threshold=args.answer_threshold,
            ngram_size=args.ngram_size,
            similarity_scope=args.similarity_scope,
        )
        summary["dedup"] = dedup_summary

    rejected = local_rejected + dedup_rejected
    summary.update(
        {
            "final_kept_count": len(final_kept),
            "final_rejected_count": len(rejected),
            "output_path": str(output_path),
            "rejected_path": str(rejected_path),
            "summary_path": str(summary_path),
            "input_path": str(input_path),
            "source_badcases_path": str(resolve_path(args.source_badcases)),
        }
    )

    write_jsonl(output_path, final_kept)
    write_jsonl(rejected_path, rejected)
    write_json(summary_path, summary)

    log_success(f"Final kept {len(final_kept)} records: {output_path}")
    log_success(f"Rejected {len(rejected)} records: {rejected_path}")
    log_success(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
