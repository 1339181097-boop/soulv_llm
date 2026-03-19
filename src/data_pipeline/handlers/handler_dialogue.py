from __future__ import annotations

import argparse
import hashlib
import re
import sys
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

DEFAULT_INPUT_PATH = "data/raw/dialogue_data.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_dialogue.json"
DEFAULT_SYSTEM_PROMPT = (
    "你是专业的中文旅行规划助手，请基于用户需求提供清晰、实用、可执行的行程建议。"
    "避免营销口吻，优先给出真正有帮助的信息。"
)
MIN_ASSISTANT_LENGTH = 80

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]+", re.UNICODE)
TRAILING_SPACE_BEFORE_NEWLINE_PATTERN = re.compile(r"[ \t]+\n")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")

INTRO_NOISE_PATTERNS = (
    "欢迎你来到",
    "让我们这个旅游界的",
    "让我这个旅游界的",
    "带你一起开启",
    "等待着你的探索",
)
PROMO_BLOCK_KEYWORDS = (
    "出行准备",
    "常用app",
)
PROMO_LINE_KEYWORDS = (
    "点击“商城”",
    '点击"商城"',
    "很多朋友经常问旅行有办法省钱吗",
    "现在打开",
    "小橙序",
)
CLOSING_PATTERNS = (
    "希望这份攻略能帮助您",
    "祝您旅途愉快",
    "欢迎随时咨询",
    "欢迎随时来问",
)
BRAND_REPLACEMENTS = (
    ("tripai小奇旅行app", "平台"),
    ("tripai小奇旅行", "平台"),
    ("tripai", "平台"),
    ("小奇旅行", "平台"),
)


def _decode_literal_escapes(text: str) -> str:
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _strip_intro_block(blocks: list[str]) -> list[str]:
    if not blocks:
        return blocks

    first_block = blocks[0]
    if any(pattern in first_block for pattern in INTRO_NOISE_PATTERNS):
        return blocks[1:]
    return blocks


def _is_promo_block(block: str) -> bool:
    normalized = EMOJI_PATTERN.sub("", normalize_text(block)).lower().replace(" ", "")
    if any(keyword in normalized for keyword in PROMO_BLOCK_KEYWORDS):
        return True
    return any(keyword in block for keyword in PROMO_LINE_KEYWORDS)


def _drop_noise_blocks(text: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    blocks = _strip_intro_block(blocks)
    kept_blocks = [block for block in blocks if not _is_promo_block(block)]

    while kept_blocks and any(pattern in kept_blocks[-1] for pattern in CLOSING_PATTERNS):
        kept_blocks.pop()

    return "\n\n".join(kept_blocks)


def _replace_brand_terms(text: str) -> str:
    for source, target in BRAND_REPLACEMENTS:
        text = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
    return text


def _clean_dialogue_content(value: Any, *, max_length: int) -> str:
    text = normalize_text(value)
    text = _decode_literal_escapes(text)
    text = EMOJI_PATTERN.sub("", text)
    text = text.replace("\u200d", "").replace("\ufe0f", "")
    text = _replace_brand_terms(text)
    text = _drop_noise_blocks(text)
    text = TRAILING_SPACE_BEFORE_NEWLINE_PATTERN.sub("\n", text)
    text = MULTI_BLANK_LINE_PATTERN.sub("\n\n", text)
    return clean_text(text, max_length=max_length)


def _extract_turns(record: dict[str, Any]) -> tuple[str, str] | None:
    dialogue = record.get("dialogue")
    if not isinstance(dialogue, list) or len(dialogue) < 2:
        return None

    user_turn = None
    assistant_turn = None
    for item in dialogue:
        if not isinstance(item, dict):
            continue
        role = normalize_text(item.get("role")).lower()
        content = item.get("content")
        if role == "user" and user_turn is None:
            user_turn = content
        elif role == "assistant":
            assistant_turn = content

    if user_turn is None or assistant_turn is None:
        return None
    return str(user_turn), str(assistant_turn)


def build_dialogue_sample(record: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    extracted = _extract_turns(record)
    if extracted is None:
        return None

    raw_user, raw_assistant = extracted
    user_query = _clean_dialogue_content(raw_user, max_length=2000)
    assistant_content = _clean_dialogue_content(raw_assistant, max_length=20000)

    if not user_query or len(assistant_content) < MIN_ASSISTANT_LENGTH:
        return None

    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _sample_fingerprint(sample: dict[str, list[dict[str, str]]]) -> str:
    user_query = sample["messages"][1]["content"]
    assistant_content = sample["messages"][2]["content"]
    payload = f"{user_query}\n###\n{assistant_content}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def process_dialogue_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    log_info(f"开始处理单轮 dialogue 数据: {resolve_path(input_file_path)}")

    processed_data: list[dict[str, list[dict[str, str]]]] = []
    seen_fingerprints: set[str] = set()
    total_records = 0
    skipped_invalid = 0
    skipped_duplicate = 0

    try:
        for _, record in iter_jsonl(input_file_path):
            total_records += 1
            sample = build_dialogue_sample(record)
            if sample is None:
                skipped_invalid += 1
                continue

            fingerprint = _sample_fingerprint(sample)
            if fingerprint in seen_fingerprints:
                skipped_duplicate += 1
                continue

            seen_fingerprints.add(fingerprint)
            processed_data.append(sample)
    except FileNotFoundError:
        log_error(f"未找到输入文件: {resolve_path(input_file_path)}")
        return []

    output_path = write_json(output_json_path, processed_data)
    log_success(
        "dialogue 数据处理完成。"
        f"读取 {total_records} 条，输出 {len(processed_data)} 条，"
        f"跳过无效 {skipped_invalid} 条，去重 {skipped_duplicate} 条。"
    )
    log_info(f"输出文件: {output_path}")
    return processed_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将单轮 dialogue JSONL 清洗并转换为 ChatML。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="原始 dialogue JSONL 路径。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 ChatML JSON 路径。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_dialogue_data(args.input, args.output)


if __name__ == "__main__":
    main()

