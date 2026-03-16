from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_utils import (
    configure_console_output,
    load_records,
    log_error,
    log_info,
    log_success,
    log_warn,
    resolve_path,
    write_json,
)
from pipeline.global_cleaner import clean_text

DEFAULT_INPUT_PATH = "data/raw/multiturn.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_multiturn.json"
DEFAULT_SYSTEM_PROMPT = "你是搜旅 AI 旅行管家“小奇”，请结合上下文连续回答用户问题。"
ALLOWED_ROLES = {"system", "user", "assistant", "tool", "observation"}


def _normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages") or record.get("conversation")
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip()
        content = clean_text(item.get("content"), max_length=6000)
        if not role or role not in ALLOWED_ROLES or not content:
            continue
        normalized.append({"role": role, "content": content})

    return normalized


def build_multiturn_sample(record: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    messages = _normalize_messages(record)
    if not messages:
        return None

    roles = [message["role"] for message in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    return {"messages": messages}


def process_multiturn_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    log_info(f"开始处理多轮对话数据: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"未找到多轮对话原始数据，先跳过: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, list[dict[str, str]]]] = []
    skipped = 0
    for record in raw_records:
        sample = build_multiturn_sample(record)
        if sample is None:
            skipped += 1
            continue
        processed.append(sample)

    output_path = write_json(output_json_path, processed)
    log_success(f"多轮对话数据处理完成，输出 {len(processed)} 条，跳过 {skipped} 条。")
    log_info(f"输出文件: {output_path}")
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将多轮对话数据转换为 ChatML。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="原始多轮对话数据路径，支持 JSON/JSONL。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 ChatML JSON 路径。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_multiturn_data(args.input, args.output)


if __name__ == "__main__":
    main()
