from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    iter_jsonl,
    log_info,
    log_success,
    resolve_path,
    validate_chatml_dataset,
    write_json,
    write_jsonl,
)
from src.data_pipeline.global_cleaner import normalize_text

DEFAULT_INPUT_PATH = "data/badcase_fix/qwen3_32b_stage1_text_badcase_repair_expanded_v2.final_kept.jsonl"
DEFAULT_JSON_OUTPUT_PATH = "data/final/stage1_badcase_repair_sft.json"
DEFAULT_JSONL_OUTPUT_PATH = "data/processed_stage1/sft_badcase_repair_expanded_v2_final_kept.jsonl"
DEFAULT_REPORT_PATH = "data/final/stage1_badcase_repair_sft_report.json"
DATASET_SOURCE = "badcase_repair_expanded_v2_final_kept"
ALLOWED_ROLES = {"system", "user", "assistant"}
FORBIDDEN_TOP_LEVEL_KEYS = {
    "answer",
    "quality_judge",
    "raw_response",
    "tool_calls",
    "function_call",
    "function_calls",
    "tools",
    "messages_with_answer",
}
FORBIDDEN_MESSAGE_KEYS = {"tool_calls", "function_call", "function_calls"}


def _record_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _clean_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = normalize_text(message.get("role"))
    content = normalize_text(message.get("content"))
    if role not in ALLOWED_ROLES or not content:
        return None
    return {"role": role, "content": content}


def _clean_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []

    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        cleaned_message = _clean_message(message)
        if cleaned_message is not None:
            cleaned.append(cleaned_message)

    answer = normalize_text(record.get("answer"))
    if answer:
        if not cleaned or cleaned[-1].get("role") != "assistant" or cleaned[-1].get("content") != answer:
            cleaned.append({"role": "assistant", "content": answer})
    return cleaned


def _quality_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    judge = record.get("quality_judge")
    if not isinstance(judge, dict):
        return {}
    scores = judge.get("scores")
    scores = scores if isinstance(scores, dict) else {}
    return {
        "quality_model": judge.get("model"),
        "quality_overall_score": scores.get("overall_score"),
        "quality_confidence": judge.get("confidence"),
        "quality_verdict": judge.get("verdict"),
    }


def convert_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    messages = _clean_messages(record)
    source_badcase_id = normalize_text(record.get("source_badcase_id"))
    repair_type = normalize_text(record.get("repair_type"))
    task_type = normalize_text(record.get("task_type"))
    identity_payload = {
        "source_badcase_id": source_badcase_id,
        "repair_type": repair_type,
        "task_type": task_type,
        "messages": messages,
    }
    digest = _record_hash(identity_payload)[:12]
    quality = _quality_snapshot(record)
    fix_tags = record.get("fix_tags")
    if not isinstance(fix_tags, list):
        fix_tags = []

    return {
        "id": f"badcase_repair_{index:06d}_{digest}",
        "record_id": f"badcase_repair_{index:06d}",
        "task_type": task_type,
        "scene": "badcase_repair",
        "source": DATASET_SOURCE,
        "source_badcase_id": source_badcase_id,
        "repair_type": repair_type,
        "fix_tags": [tag for tag in fix_tags if isinstance(tag, str) and tag.strip()],
        **quality,
        "messages": messages,
    }


def _strict_chatml_errors(dataset: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for sample_index, sample in enumerate(dataset):
        forbidden_top = sorted(FORBIDDEN_TOP_LEVEL_KEYS.intersection(sample.keys()))
        if forbidden_top:
            errors.append(f"sample {sample_index} has forbidden top-level keys: {forbidden_top}")

        messages = sample.get("messages")
        if not isinstance(messages, list):
            continue
        if messages and messages[-1].get("role") != "assistant":
            errors.append(f"sample {sample_index} does not end with assistant")

        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in ALLOWED_ROLES:
                errors.append(f"sample {sample_index} message {message_index} has invalid role: {role}")
            forbidden_message = sorted(FORBIDDEN_MESSAGE_KEYS.intersection(message.keys()))
            if forbidden_message:
                errors.append(
                    f"sample {sample_index} message {message_index} has forbidden message keys: {forbidden_message}"
                )
            if set(message.keys()) != {"role", "content"}:
                errors.append(f"sample {sample_index} message {message_index} has non-standard keys: {sorted(message.keys())}")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"sample {sample_index} message {message_index} has empty content")
    return errors


def _length_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "avg": 0.0, "max": 0}
    return {"min": min(values), "avg": round(sum(values) / len(values), 2), "max": max(values)}


def _build_report(dataset: list[dict[str, Any]], *, input_path: Path, json_output_path: Path, jsonl_output_path: Path) -> dict[str, Any]:
    task_counts = Counter(str(record.get("task_type")) for record in dataset)
    repair_counts = Counter(str(record.get("repair_type")) for record in dataset)
    assistant_lengths: list[int] = []
    turn_counts: list[int] = []
    for record in dataset:
        messages = record.get("messages", [])
        if isinstance(messages, list):
            turn_counts.append(len(messages))
            assistant_lengths.append(
                sum(len(message.get("content", "")) for message in messages if isinstance(message, dict) and message.get("role") == "assistant")
            )

    chatml_errors = validate_chatml_dataset(dataset)
    strict_errors = _strict_chatml_errors(dataset)
    return {
        "input_path": str(input_path),
        "json_output_path": str(json_output_path),
        "jsonl_output_path": str(jsonl_output_path),
        "output_format": "stage1_chatml_json_array_and_jsonl",
        "sample_count": len(dataset),
        "chatml_valid": not chatml_errors,
        "strict_stage1_valid": not strict_errors,
        "chatml_error_count": len(chatml_errors),
        "strict_error_count": len(strict_errors),
        "chatml_errors_preview": chatml_errors[:10],
        "strict_errors_preview": strict_errors[:10],
        "task_counts": dict(task_counts),
        "repair_type_counts": dict(repair_counts),
        "assistant_length": _length_summary(assistant_lengths),
        "message_turn_count": _length_summary(turn_counts),
        "source": DATASET_SOURCE,
    }


def prepare_badcase_repair_sft(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    json_output_path: str | Path = DEFAULT_JSON_OUTPUT_PATH,
    jsonl_output_path: str | Path = DEFAULT_JSONL_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    resolved_input = resolve_path(input_path)
    resolved_json_output = resolve_path(json_output_path)
    resolved_jsonl_output = resolve_path(jsonl_output_path)
    resolved_report = resolve_path(report_path)

    records = [record for _, record in iter_jsonl(resolved_input)]
    log_info(f"Loaded {len(records)} kept badcase repair records from {resolved_input}")
    dataset = [convert_record(record, index) for index, record in enumerate(records, start=1)]
    report = _build_report(
        dataset,
        input_path=resolved_input,
        json_output_path=resolved_json_output,
        jsonl_output_path=resolved_jsonl_output,
    )
    if not report["chatml_valid"] or not report["strict_stage1_valid"]:
        write_json(report_path, report)
        raise ValueError(f"Badcase repair SFT output failed validation. See report: {resolved_report}")

    write_json(resolved_json_output, dataset)
    write_jsonl(resolved_jsonl_output, dataset)
    write_json(resolved_report, report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert kept badcase repair JSONL into Stage1 ChatML SFT data.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input final_kept JSONL.")
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT_PATH, help="Output JSON array for LLaMA-Factory.")
    parser.add_argument("--jsonl-output", default=DEFAULT_JSONL_OUTPUT_PATH, help="Output JSONL for mixer/debug use.")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Output report JSON.")
    return parser


def main() -> int:
    configure_console_output()
    args = _build_arg_parser().parse_args()
    report = prepare_badcase_repair_sft(
        input_path=args.input,
        json_output_path=args.json_output,
        jsonl_output_path=args.jsonl_output,
        report_path=args.report,
    )
    log_success(f"Prepared {report['sample_count']} badcase repair SFT records")
    log_success(f"JSON output: {report['json_output_path']}")
    log_success(f"JSONL output: {report['jsonl_output_path']}")
    log_success(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
