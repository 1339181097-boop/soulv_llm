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

DEFAULT_INPUT_PATH = "data/raw/basic_qa.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_basic_qa.json"
DEFAULT_SYSTEM_PROMPT = "你是搜旅景点百科助手，请基于事实回答用户的旅行问答。"


def build_basic_qa_sample(record: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    question = clean_text(record.get("question") or record.get("user_query"), max_length=2000)
    answer = clean_text(record.get("answer") or record.get("assistant_response"), max_length=5000)
    if not question or not answer:
        return None

    context = clean_text(record.get("context"), max_length=2000)
    if context:
        question = f"参考信息：{context}\n\n用户问题：{question}"

    system_prompt = clean_text(
        record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        max_length=1000,
        mask_sensitive=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def process_basic_qa_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    log_info(f"开始处理基础 QA 数据: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"未找到基础 QA 原始数据，先跳过: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, list[dict[str, str]]]] = []
    skipped = 0
    for record in raw_records:
        sample = build_basic_qa_sample(record)
        if sample is None:
            skipped += 1
            continue
        processed.append(sample)

    output_path = write_json(output_json_path, processed)
    log_success(f"基础 QA 数据处理完成，输出 {len(processed)} 条，跳过 {skipped} 条。")
    log_info(f"输出文件: {output_path}")
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将基础 QA 数据转换为 ChatML。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="原始 QA 数据路径，支持 JSON/JSONL。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 ChatML JSON 路径。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_basic_qa_data(args.input, args.output)


if __name__ == "__main__":
    main()
