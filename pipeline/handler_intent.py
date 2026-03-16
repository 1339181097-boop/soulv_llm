from __future__ import annotations

import argparse
import json
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
from pipeline.global_cleaner import clean_text, normalize_text

DEFAULT_INPUT_PATH = "data/raw/intent.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_intent.json"
DEFAULT_SYSTEM_PROMPT = (
    "你是一个多功能智能旅行助手，能够精准识别用户意图，只能出现一种用户意图，"
    'json格式输出：{"intentionName":"FUNCTION_FLIGHTS_SEARCH_STRATEGY"}'
)

# 公司当前正在使用的意图规范。后续所有清洗和训练都以这份枚举为准。
CANONICAL_INTENT_NAMES = {
    "FUNCTION_FLIGHTS_SEARCH_STRATEGY",
    "FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY",
    "FUNCTION_FLIGHTS_PASSENGER_STRATEGY",
    "FUNCTION_HOTELS_STRATEGY",
    "TRAVEL_STRATEGY",
    "TRAVEL_LOCATION_STRATEGY",
    "FUNCTION_TICKETS_STRATEGY",
    "FUNCTION_CAR_RENTAL_STRATEGY",
    "FUNCTION_VISA_STRATEGY",
    "DEFAULT_STRATEGY",
}


def _normalize_intent_response(record: dict[str, Any]) -> str:
    assistant_response = record.get("assistant_response") or record.get("tool_call") or record.get("tool_json")
    if assistant_response:
        if isinstance(assistant_response, str):
            return clean_text(assistant_response, max_length=4000, mask_sensitive=False)
        return json.dumps(assistant_response, ensure_ascii=False)

    intention_name = normalize_text(record.get("intentionName"))
    if intention_name:
        return json.dumps({"intentionName": intention_name}, ensure_ascii=False)

    return ""


def build_intent_sample(record: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    user_query = clean_text(record.get("user_query") or record.get("query"), max_length=2000)
    assistant_content = _normalize_intent_response(record)

    if not user_query or not assistant_content:
        return None

    system_prompt = clean_text(
        record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        max_length=3000,
        mask_sensitive=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def process_intent_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    log_info(f"开始处理意图数据: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"未找到意图原始数据，先跳过: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, list[dict[str, str]]]] = []
    skipped = 0
    non_canonical = 0

    for record in raw_records:
        intention_name = normalize_text(record.get("intentionName"))
        if intention_name and intention_name not in CANONICAL_INTENT_NAMES:
            non_canonical += 1
            log_warn(f"发现未收录的意图名: {intention_name}")

        sample = build_intent_sample(record)
        if sample is None:
            skipped += 1
            continue
        processed.append(sample)

    output_path = write_json(output_json_path, processed)
    log_success(
        f"意图数据处理完成，输出 {len(processed)} 条，跳过 {skipped} 条。"
        f"未收录意图名 {non_canonical} 条。"
    )
    log_info(f"输出文件: {output_path}")
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将意图识别数据转换为 ChatML。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="原始意图数据路径，支持 JSON/JSONL。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 ChatML JSON 路径。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_intent_data(args.input, args.output)


if __name__ == "__main__":
    main()
