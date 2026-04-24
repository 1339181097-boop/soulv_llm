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
    "你是专业的中文旅行咨询助手。请基于用户问题提供清晰、自然、实用的回答，"
    "避免营销口吻，不输出平台导购、二维码、下单或预订指引。"
)

MIN_ASSISTANT_LENGTH = 40
MAX_ASSISTANT_LENGTH = 900
MAX_USER_LENGTH = 1000

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]+", re.UNICODE)
DAY_MARKER_PATTERN = re.compile(r"(?:^|\n)\s*(?:第[一二三四五六七八九十0-9]+天|day\s*\d+)", re.IGNORECASE)
TIME_SLOT_PATTERN = re.compile(r"(?:^|\n)\s*(?:上午|中午|下午|傍晚|晚上|早餐|午餐|晚餐|入住|抵达)[：:]?", re.IGNORECASE)
SECTION_HEADING_PATTERN = re.compile(r"(?:^|\n)\s*(?:#{1,6}|\d+[.)]|[①②③④⑤⑥⑦⑧⑨⑩]|\-\s)")
EMPHASIS_PATTERN = re.compile(r"\*\*[^*]+\*\*")
BULLET_PATTERN = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[①②③④⑤⑥⑦⑧⑨⑩]|\-\s|\*\s)")
DURATION_PATTERN = re.compile(r"[一二三四五六七八九十0-9]{1,3}\s*(?:天|日)\s*[一二三四五六七八九十0-9]{0,3}\s*晚?")
TRAILING_SPACE_BEFORE_NEWLINE_PATTERN = re.compile(r"[ \t]+\n")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
PLATFORM_LISTING_PATTERN = re.compile(
    r"^\*{0,2}\s*(?:住宿|酒店|机票|航班|门票|订房|推荐)\*{0,2}\s*[：:]\s*(?:tripai|平台).*$",
    re.IGNORECASE,
)

BRAND_REPLACEMENTS = (
    ("tripai小奇旅行app", "平台"),
    ("tripai小奇旅行", "平台"),
    ("tripai", "平台"),
    ("TripAI", "平台"),
    ("小奇旅行", "平台"),
)
INTRO_PATTERNS = (
    "你好，我是tripai",
    "你好，我是平台",
    "您好，我是tripai",
    "您好，我是平台",
    "我是tripai",
    "我是平台",
    "我是小奇",
    "你的专属智能助手",
    "专属智能助手",
    "旅游搭子",
)
PROMO_KEYWORDS = (
    "二维码",
    "扫码",
    "小程序",
    "商城",
    "下载app",
    "打开app",
    "立即预订",
    "马上预订",
    "立即预约",
    "下单",
    "比价",
    "返现",
    "联系客服",
    "咨询客服",
    "优惠航线",
)
CLOSING_PATTERNS = (
    "希望这份攻略能帮助你",
    "希望这份攻略能帮助您",
    "祝你旅途愉快",
    "祝您旅途愉快",
    "欢迎随时咨询",
    "欢迎随时来问",
)
ITINERARY_REQUEST_KEYWORDS = (
    "攻略",
    "行程",
    "路线",
    "规划",
    "安排",
    "自由行",
    "一日游",
    "二日游",
    "三日游",
    "四日游",
    "五日游",
    "六日游",
    "七日游",
)
SELF_REFERENCE_KEYWORDS = (
    "我是平台",
    "你的专属智能助手",
    "专属智能助手",
    "旅游搭子",
)


def _decode_literal_escapes(text: str) -> str:
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _replace_brand_terms(text: str) -> str:
    for source, target in BRAND_REPLACEMENTS:
        text = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
    return text


def _normalize_for_match(text: str) -> str:
    return EMOJI_PATTERN.sub("", normalize_text(text)).lower().replace(" ", "")


def _is_intro_line(line: str) -> bool:
    normalized = _normalize_for_match(line)
    if not normalized:
        return False
    if any(pattern in normalized for pattern in INTRO_PATTERNS):
        return True
    return normalized.startswith(("哈喽", "你好呀", "嘿旅行者", "哇哦", "嗨"))


def _is_promo_line(line: str) -> bool:
    normalized = _normalize_for_match(line)
    if PLATFORM_LISTING_PATTERN.match(normalize_text(line)):
        return True
    if any(keyword in normalized for keyword in PROMO_KEYWORDS):
        return True
    if "平台" in normalize_text(line) and any(
        keyword in normalize_text(line) for keyword in ("预订", "预约", "订票", "比价", "优惠", "下载", "打开")
    ):
        return True
    return False


def _is_closing_line(line: str) -> bool:
    normalized = normalize_text(line)
    return any(pattern in normalized for pattern in CLOSING_PATTERNS)


def _drop_noise_blocks(text: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    kept_blocks: list[str] = []

    for block in blocks:
        kept_lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip() and not _is_intro_line(line) and not _is_promo_line(line) and not _is_closing_line(line)
        ]
        cleaned_block = "\n".join(kept_lines).strip()
        if cleaned_block:
            kept_blocks.append(cleaned_block)

    while kept_blocks and _is_closing_line(kept_blocks[-1]):
        kept_blocks.pop()

    return "\n\n".join(kept_blocks)


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


def _looks_like_itinerary_request(user_query: str) -> bool:
    normalized = normalize_text(user_query).lower()
    if any(keyword in normalized for keyword in ITINERARY_REQUEST_KEYWORDS):
        return True
    return bool(DURATION_PATTERN.search(normalized))


def _looks_like_itinerary(user_query: str, assistant_content: str) -> bool:
    normalized = normalize_text(assistant_content)
    lowered = normalized.lower()
    day_markers = len(DAY_MARKER_PATTERN.findall(normalized))
    time_slots = len(TIME_SLOT_PATTERN.findall(normalized))
    heading_count = len(SECTION_HEADING_PATTERN.findall(normalized))

    if _looks_like_itinerary_request(user_query) and len(normalized) >= 120:
        return True
    if day_markers >= 1:
        return True
    if time_slots >= 3 and len(normalized) >= 80:
        return True
    if heading_count >= 5 and len(normalized) >= 400:
        return True
    if len(normalized) >= 900 and ("行程总览" in lowered or "路线" in lowered):
        return True
    return False


def _has_residual_promo(assistant_content: str) -> bool:
    normalized = normalize_text(assistant_content)
    lowered = normalized.lower()
    if any(keyword in normalized for keyword in ("二维码", "扫码", "小程序", "商城", "优惠航线")):
        return True
    if "tripai" in lowered:
        return True
    if "平台" in normalized and any(
        keyword in normalized for keyword in ("预订", "预约", "订票", "比价", "优惠", "下载", "打开", "返现")
    ):
        return True
    if "app" in lowered and any(keyword in normalized for keyword in ("下载", "打开", "预订", "优惠")):
        return True
    return False


def _looks_style_heavy(assistant_content: str) -> bool:
    normalized = normalize_text(assistant_content)
    heading_count = len(SECTION_HEADING_PATTERN.findall(normalized))
    bullet_count = len(BULLET_PATTERN.findall(normalized))
    emphasis_count = len(EMPHASIS_PATTERN.findall(normalized))
    exclamation_count = sum(normalized.count(mark) for mark in ("！", "!", "？", "?"))
    self_reference_hits = sum(keyword in normalized for keyword in SELF_REFERENCE_KEYWORDS)

    if self_reference_hits >= 1 and (heading_count + bullet_count) >= 3:
        return True
    if "推荐清单" in normalized and bullet_count >= 3:
        return True
    if heading_count >= 1 and bullet_count >= 3 and len(normalized) >= 80:
        return True
    if heading_count >= 3 and bullet_count >= 4 and len(normalized) >= 140:
        return True
    if emphasis_count >= 4 and len(normalized) >= 180:
        return True
    if exclamation_count >= 6 and len(normalized) >= 160:
        return True
    return False


def _classify_dialogue_sample(user_query: str, assistant_content: str) -> str | None:
    if not user_query or len(assistant_content) < MIN_ASSISTANT_LENGTH:
        return "invalid"
    if _looks_like_itinerary(user_query, assistant_content):
        return "itinerary_like"
    if _has_residual_promo(assistant_content):
        return "promo"
    if _looks_style_heavy(assistant_content):
        return "style_heavy"
    if len(assistant_content) > MAX_ASSISTANT_LENGTH:
        return "overlong"
    return None


def _build_dialogue_sample(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    extracted = _extract_turns(record)
    if extracted is None:
        return None, "invalid"

    raw_user, raw_assistant = extracted
    user_query = _clean_dialogue_content(raw_user, max_length=MAX_USER_LENGTH)
    assistant_content = _clean_dialogue_content(raw_assistant, max_length=MAX_ASSISTANT_LENGTH * 2)
    skip_reason = _classify_dialogue_sample(user_query, assistant_content)
    if skip_reason is not None:
        return None, skip_reason

    session_id = clean_text(record.get("session_id"), max_length=80, mask_sensitive=False)
    sample_id = session_id or "dialogue_" + hashlib.md5(
        f"{user_query}\n###\n{assistant_content}".encode("utf-8")
    ).hexdigest()[:12]
    source = clean_text(record.get("source"), max_length=120, mask_sensitive=False) or "pseudo_real_dialogue"

    sample = {
        "id": sample_id,
        "task_type": "dialogue",
        "scene": "travel_consultation",
        "source": source,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_content},
        ],
    }
    if session_id:
        sample["session_id"] = session_id
    return sample, None


def build_dialogue_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    sample, _ = _build_dialogue_sample(record)
    return sample


def _sample_fingerprint(sample: dict[str, Any]) -> str:
    messages = sample["messages"]
    payload = f"{messages[1]['content']}\n###\n{messages[2]['content']}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def process_dialogue_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    configure_console_output()
    log_info(f"开始处理单轮 dialogue 数据: {resolve_path(input_file_path)}")

    processed_data: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    total_records = 0
    skipped_duplicate = 0
    skipped_by_reason = {
        "invalid": 0,
        "promo": 0,
        "itinerary_like": 0,
        "style_heavy": 0,
        "overlong": 0,
    }

    try:
        for _, record in iter_jsonl(input_file_path):
            total_records += 1
            sample, skip_reason = _build_dialogue_sample(record)
            if sample is None:
                skipped_by_reason[skip_reason or "invalid"] = skipped_by_reason.get(skip_reason or "invalid", 0) + 1
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
        f"无效/过短 {skipped_by_reason['invalid']} 条，"
        f"营销/平台导购 {skipped_by_reason['promo']} 条，"
        f"攻略化/模板化 {skipped_by_reason['itinerary_like']} 条，"
        f"风格噪声 {skipped_by_reason['style_heavy']} 条，"
        f"超长 {skipped_by_reason['overlong']} 条，"
        f"去重 {skipped_duplicate} 条。"
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
