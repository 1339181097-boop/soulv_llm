from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    iter_jsonl,
    log_error,
    log_info,
    log_success,
    resolve_path,
    write_json,
)
from src.data_pipeline.global_cleaner import clean_text, normalize_text
from src.data_pipeline.handlers.handler_dialogue import (
    _has_residual_promo,
    _looks_style_heavy,
)
from src.data_pipeline.system_prompt_loader import load_system_prompt

DEFAULT_INPUT_PATH = "data/raw/multi_turn_dialogue_raw_3_25.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_multi_turn_dialogue.json"

_DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "你是专业的中文旅行规划助手。请基于完整对话上下文回答用户问题，"
    "先理解前文约束，再根据用户新增条件做补充、改写或调整。"
    "避免营销口吻，不输出工具调用或路由信息。"
)

DEFAULT_SYSTEM_PROMPT = load_system_prompt("multi_turn_dialogue", _DEFAULT_SYSTEM_PROMPT_FALLBACK)

MIN_TURN_PAIRS = 3
MIN_ASSISTANT_LENGTH = 60
MAX_USER_LENGTH = 1200
MAX_ASSISTANT_LENGTH = 2400

EDITORIAL_TAIL_MARKERS = (
    "其余内容不变",
    "其他内容不变",
    "其余安排不变",
    "其余安排保持不变",
    "其余预算与住宿安排保持不变",
    "其余预算与住宿保持不变",
    "行程结构与自然体验内容完全保留",
    "行程结构完全保留",
    "其余行程安排保持不变",
)
HARD_PLACEHOLDER_MARKERS = (
    "后续同上",
    "继续同上",
    "续同上",
    "起完全保留",
    "完全保留）",
)
TRAILING_PUNCTUATION_PATTERN = re.compile(r"^[，,；;：:\s]+|[，,；;：:\s]+$")


def _soft_truncate(text: str, max_length: int) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_length:
        return normalized

    head = normalized[:max_length]
    best_cut = max(head.rfind(token) for token in ("\n\n", "\n", "。", "！", "？", "；", ".", "!", "?"))
    if best_cut >= max_length // 2:
        return head[: best_cut + 1].strip()
    return head.rstrip()


def _trim_editorial_tail(text: str) -> str:
    candidate = normalize_text(text)
    for marker in EDITORIAL_TAIL_MARKERS:
        index = candidate.find(marker)
        if index != -1:
            candidate = candidate[:index].rstrip()
            candidate = TRAILING_PUNCTUATION_PATTERN.sub("", candidate).rstrip()
            break
    return candidate


def _contains_hard_placeholder(text: str) -> bool:
    normalized = normalize_text(text)
    return any(marker in normalized for marker in HARD_PLACEHOLDER_MARKERS)


def _clean_multiturn_message(role: str, content: Any) -> str:
    max_length = MAX_USER_LENGTH if role == "user" else MAX_ASSISTANT_LENGTH
    cleaned = clean_text(content, max_length=max_length * 2, mask_sensitive=False)
    cleaned = normalize_text(cleaned)
    cleaned = _trim_editorial_tail(cleaned)
    return _soft_truncate(cleaned, max_length=max_length)


def _normalize_constraint_changes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [clean_text(item, max_length=300, mask_sensitive=False) for item in value]
    return [item for item in cleaned if item]


def _roles_are_valid(messages: list[dict[str, Any]]) -> bool:
    if len(messages) < MIN_TURN_PAIRS * 2 or len(messages) % 2 != 0:
        return False
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or normalize_text(message.get("role")).lower() != expected_role:
            return False
    return True


def _has_excessive_rewrite(messages: list[dict[str, str]]) -> bool:
    assistants = [message["content"] for message in messages if message["role"] == "assistant"]
    appended_rewrite_count = 0

    for previous, current in zip(assistants, assistants[1:]):
        similarity = SequenceMatcher(None, previous, current).ratio()
        shared_prefix = len(os.path.commonprefix([previous, current]))
        min_length = min(len(previous), len(current))
        max_length = max(len(previous), len(current))
        if min_length == 0:
            continue

        prefix_ratio = shared_prefix / min_length
        near_same_length = min_length >= max_length * 0.69

        if similarity >= 0.89 and near_same_length:
            return True
        if (similarity >= 0.77 or (shared_prefix >= 60 and prefix_ratio >= 0.60)) and near_same_length:
            appended_rewrite_count += 1

    return appended_rewrite_count >= 2


def _sample_fingerprint(messages: list[dict[str, str]]) -> str:
    payload = "\n###\n".join(f"{message['role']}::{message['content']}" for message in messages)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _build_multiturn_sample(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if normalize_text(record.get("task_type")) != "multi_turn_dialogue":
        return None, "wrong_task_type"

    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or not _roles_are_valid(raw_messages):
        return None, "invalid_structure"

    constraint_changes = _normalize_constraint_changes(record.get("constraint_changes"))
    if constraint_changes and len(constraint_changes) - 1 != len(raw_messages) // 2 - 1:
        return None, "constraint_mismatch"

    cleaned_messages: list[dict[str, str]] = []
    for item in raw_messages:
        role = normalize_text(item.get("role")).lower()
        content = _clean_multiturn_message(role, item.get("content"))
        if not content:
            return None, "empty_message"
        if role == "assistant":
            if len(content) < MIN_ASSISTANT_LENGTH:
                return None, "assistant_too_short"
            if _contains_hard_placeholder(content):
                return None, "editorial_placeholder"
            if _has_residual_promo(content):
                return None, "promo"
            if _looks_style_heavy(content):
                return None, "style_heavy"
        cleaned_messages.append({"role": role, "content": content})

    if _has_excessive_rewrite(cleaned_messages):
        return None, "rewrite_heavy"

    sample: dict[str, Any] = {
        "id": clean_text(record.get("record_id") or _sample_fingerprint(cleaned_messages), max_length=80, mask_sensitive=False),
        "record_id": clean_text(record.get("record_id"), max_length=80, mask_sensitive=False),
        "task_type": "multi_turn_dialogue",
        "scene": "travel_consultation",
        "source": clean_text(record.get("source"), max_length=120, mask_sensitive=False),
        "conversation_id": clean_text(record.get("conversation_id"), max_length=120, mask_sensitive=False),
        "city": clean_text(record.get("city"), max_length=120, mask_sensitive=False),
        "topic": clean_text(record.get("topic"), max_length=200, mask_sensitive=False),
        "destination": clean_text(record.get("destination"), max_length=200, mask_sensitive=False),
        "people_count": record.get("people_count"),
        "constraint_changes": constraint_changes,
        "messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *cleaned_messages],
    }

    updated_at = clean_text(record.get("updated_at"), max_length=40, mask_sensitive=False)
    note = clean_text(record.get("note"), max_length=200, mask_sensitive=False)
    if updated_at:
        sample["updated_at"] = updated_at
    if note:
        sample["note"] = note

    return sample, None


def build_multiturn_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    sample, _ = _build_multiturn_sample(record)
    return sample


def process_multiturn_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    configure_console_output()
    log_info(f"开始清洗 multi_turn_dialogue 原始数据: {resolve_path(input_file_path)}")

    processed_data: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    total_records = 0
    skipped_by_reason: dict[str, int] = {
        "invalid_structure": 0,
        "constraint_mismatch": 0,
        "empty_message": 0,
        "assistant_too_short": 0,
        "editorial_placeholder": 0,
        "promo": 0,
        "style_heavy": 0,
        "rewrite_heavy": 0,
        "duplicate_record_id": 0,
        "duplicate_content": 0,
        "wrong_task_type": 0,
    }

    try:
        for _, record in iter_jsonl(input_file_path):
            total_records += 1
            record_id = clean_text(record.get("record_id"), max_length=80, mask_sensitive=False)
            if record_id:
                if record_id in seen_record_ids:
                    skipped_by_reason["duplicate_record_id"] += 1
                    continue
                seen_record_ids.add(record_id)

            sample, skip_reason = _build_multiturn_sample(record)
            if sample is None:
                skipped_by_reason[skip_reason or "invalid_structure"] = (
                    skipped_by_reason.get(skip_reason or "invalid_structure", 0) + 1
                )
                continue

            fingerprint = _sample_fingerprint(sample["messages"][1:])
            if fingerprint in seen_fingerprints:
                skipped_by_reason["duplicate_content"] += 1
                continue

            seen_fingerprints.add(fingerprint)
            processed_data.append(sample)
    except FileNotFoundError:
        log_error(f"未找到输入文件: {resolve_path(input_file_path)}")
        return []

    output_path = write_json(output_json_path, processed_data)
    log_success(
        "multi_turn_dialogue 原始数据清洗完成。"
        f"读取 {total_records} 条，输出 {len(processed_data)} 条，"
        f"结构问题 {skipped_by_reason['invalid_structure']} 条，"
        f"字段不对齐 {skipped_by_reason['constraint_mismatch']} 条，"
        f"assistant 过短 {skipped_by_reason['assistant_too_short']} 条，"
        f"占位式续写 {skipped_by_reason['editorial_placeholder']} 条，"
        f"营销口吻 {skipped_by_reason['promo']} 条，"
        f"风格噪声 {skipped_by_reason['style_heavy']} 条，"
        f"高相似重写 {skipped_by_reason['rewrite_heavy']} 条，"
        f"重复 record_id {skipped_by_reason['duplicate_record_id']} 条，"
        f"内容去重 {skipped_by_reason['duplicate_content']} 条。"
    )
    log_info(f"输出文件: {output_path}")
    return processed_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="清洗真实 multi_turn_dialogue 原始 JSONL 并转换为 ChatML。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="原始 multi_turn_dialogue JSONL 路径。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 ChatML JSON 路径。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_multiturn_data(args.input, args.output)


if __name__ == "__main__":
    main()




