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
MIN_ASSISTANT_LENGTH = 40
MAX_ASSISTANT_LENGTH = 900

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]+", re.UNICODE)
TRAILING_SPACE_BEFORE_NEWLINE_PATTERN = re.compile(r"[ \t]+\n")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
DAY_MARKER_PATTERN = re.compile(r"(?:^|\n)\s*(?:[*#-]\s*)?(?:第[一二三四五六七八九十0-9]+天|day\s*[0-9]+)", re.IGNORECASE)
SECTION_HEADING_PATTERN = re.compile(r"(?:^|\n)\s*(?:#{1,6}|\*\*|[0-9]+[.)]|[①②③④⑤⑥⑦⑧⑨⑩])")
PRICE_PATTERN = re.compile(r"(?:¥|￥|RMB|CNY|港币|欧元|美元|人均)\s*\d+", re.IGNORECASE)
TIME_SLOT_PATTERN = re.compile(r"(?:^|\n)\s*(?:上午|中午|下午|傍晚|晚上|早餐|午餐|晚餐|入住|抵达)", re.IGNORECASE)
DURATION_PATTERN = re.compile(r"[一二三四五六七八九十0-9]{1,3}\s*(?:天|日)\s*[一二三四五六七八九十0-9]{0,3}\s*晚?")
BULLET_LINE_PATTERN = re.compile(r"(?:^|\n)\s*(?:[-*?]|[0-9]+[.)]|[??????????]|[0-9]?)")
EMPHASIS_PATTERN = re.compile(r"\*\*[^*]+\*\*")

INTRO_NOISE_PATTERNS = (
    "欢迎你来到",
    "让我们这个旅游界的",
    "让我这个旅游界的",
    "带你一起开启",
    "等待着你的探索",
)
SELF_INTRO_PATTERNS = (
    "我是tripai",
    "我是tripAI",
    "我是小奇",
    "我是平台",
    "你的专属智能助手",
    "专属智能助手",
    "你的专属智能旅行搭子",
    "旅行搭子",
    "贴心的旅游搭子",
)
HYPE_INTRO_PATTERNS = (
    "????",
    "???",
    "???????",
    "??????",
    "???????????",
    "????????????????",
    "???256G??",
    "???256g??",
    "???????????",
    "?????",
    "?????????",
)
PROMO_BLOCK_KEYWORDS = (
    "出行准备",
    "常用app",
    "交通指南",
    "行程总览",
    "总原则",
    "天气参考",
    "必备物品",
    "住宿建议",
    "美食地图",
)
PROMO_LINE_KEYWORDS = (
    "点击“商城”",
    '点击"商城"',
    "很多朋友经常问旅行有办法省钱吗",
    "现在打开",
    "小橙序",
    "小程序",
    "二维码",
    "扫码",
    "优惠航线",
    "一站式",
)
CLOSING_PATTERNS = (
    "??????????",
    "??????",
    "??????",
    "??????",
    "???????",
    "?????????",
    "?????????",
    "??????",
    "??????",
    "???????",
)
GENERIC_PLATFORM_LISTING_PATTERN = re.compile(
    r"^\*{0,2}\s*(?:住宿|酒店|机票|航班)\*{0,2}\s*[：:]\s*平台\s*$",
    re.IGNORECASE,
)
BRAND_REPLACEMENTS = (
    ("tripai小奇旅行app", "平台"),
    ("tripai小奇旅行", "平台"),
    ("tripai", "平台"),
    ("小奇旅行", "平台"),
)
ITINERARY_SECTION_KEYWORDS = (
    "交通指南",
    "出行准备",
    "常用app",
    "行程总览",
    "住宿建议",
    "天气参考",
    "美食地图",
    "day 1",
    "第一天",
)
ITINERARY_REQUEST_KEYWORDS = (
    "攻略",
    "行程",
    "自由行",
    "路线",
    "自驾",
    "安排",
    "规划",
    "一日游",
    "二日游",
    "三日游",
    "四日游",
    "五日游",
    "六日游",
    "七日游",
)
ITINERARY_RESPONSE_PHRASES = (
    "为您规划一份",
    "为你规划一份",
    "量身定制一份",
    "深度游攻略",
    "轻松路线",
    "行程概览",
    "核心路线",
    "day 0",
    "day 1",
    "第一天",
    "第二天",
)
SELF_REFERENCE_PATTERNS = (
    "????",
    "????",
    "??????",
    "?????",
    "?????",
    "????",
    "???????",
    "????????",
    "??????",
    "???256G??",
    "???256g??",
    "?????",
    "?????",
)


def _decode_literal_escapes(text: str) -> str:
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _normalized_for_match(text: str) -> str:
    return EMOJI_PATTERN.sub("", normalize_text(text)).lower().replace(" ", "")


def _replace_brand_terms(text: str) -> str:
    for source, target in BRAND_REPLACEMENTS:
        text = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
    return text


def _is_intro_block(block: str) -> bool:
    normalized = normalize_text(block)
    lowered = normalized.lower()
    if len(normalized) > 260:
        return False
    if any(pattern in normalized for pattern in INTRO_NOISE_PATTERNS):
        return True
    if any(pattern.lower() in lowered for pattern in SELF_INTRO_PATTERNS):
        return True
    if any(pattern in normalized for pattern in HYPE_INTRO_PATTERNS):
        return True
    return normalized.startswith(("哈喽", "嘿，旅行者", "你好呀", "哎呀", "哇哦"))


def _is_intro_line(line: str) -> bool:
    normalized = normalize_text(line)
    lowered = normalized.lower()
    if not normalized:
        return False
    if any(pattern in normalized for pattern in INTRO_NOISE_PATTERNS):
        return True
    if any(pattern.lower() in lowered for pattern in SELF_INTRO_PATTERNS):
        return True
    if any(pattern in normalized for pattern in HYPE_INTRO_PATTERNS):
        return True
    return normalized.startswith(("哈喽", "嘿，旅行者", "你好呀", "哎呀", "哇哦"))


def _strip_intro_block(blocks: list[str]) -> list[str]:
    while len(blocks) > 1 and _is_intro_block(blocks[0]):
        blocks = blocks[1:]
    return blocks


def _is_promo_line(line: str) -> bool:
    normalized = _normalized_for_match(line)
    if any(keyword in normalized for keyword in PROMO_BLOCK_KEYWORDS):
        return True
    if any(keyword in line for keyword in PROMO_LINE_KEYWORDS):
        return True
    if GENERIC_PLATFORM_LISTING_PATTERN.match(normalize_text(line)):
        return True
    if "平台" in line and any(keyword in line for keyword in ("预订", "预约", "订票", "扫码", "优惠", "查看", "比价")):
        return True
    return False


def _is_promo_block(block: str) -> bool:
    normalized = _normalized_for_match(block)
    if any(keyword in normalized for keyword in PROMO_BLOCK_KEYWORDS):
        return True
    if any(keyword in block for keyword in PROMO_LINE_KEYWORDS):
        return True
    if "平台" in block and any(keyword in block for keyword in ("预订", "预约", "订票", "优惠", "商城", "比价")):
        return True
    return False


def _is_closing_block(block: str) -> bool:
    return any(pattern in block for pattern in CLOSING_PATTERNS)


def _drop_noise_blocks(text: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    blocks = _strip_intro_block(blocks)

    kept_blocks: list[str] = []
    for block in blocks:
        cleaned_lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip() and not _is_intro_line(line) and not _is_promo_line(line)
        ]
        cleaned_block = "\n".join(cleaned_lines).strip()
        if not cleaned_block or _is_promo_block(cleaned_block):
            continue
        kept_blocks.append(cleaned_block)

    while kept_blocks and _is_closing_block(kept_blocks[-1]):
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
    normalized_user = normalize_text(user_query)
    lowered_user = normalized_user.lower()
    if any(keyword in lowered_user for keyword in ITINERARY_REQUEST_KEYWORDS):
        return True
    return bool(DURATION_PATTERN.search(normalized_user))


def _looks_like_itinerary(user_query: str, assistant_content: str) -> bool:
    normalized_assistant = normalize_text(assistant_content)
    lowered_assistant = normalized_assistant.lower()
    block_count = len([block for block in normalized_assistant.split("\n\n") if block.strip()])
    day_markers = len(DAY_MARKER_PATTERN.findall(normalized_assistant))
    section_keywords = sum(keyword in lowered_assistant for keyword in ITINERARY_SECTION_KEYWORDS)
    heading_count = len(SECTION_HEADING_PATTERN.findall(normalized_assistant))
    price_mentions = len(PRICE_PATTERN.findall(normalized_assistant))
    time_slots = len(TIME_SLOT_PATTERN.findall(normalized_assistant))
    explicit_itinerary_request = _looks_like_itinerary_request(user_query)
    response_phrases = sum(phrase in lowered_assistant for phrase in ITINERARY_RESPONSE_PHRASES)

    if explicit_itinerary_request:
        return True
    if day_markers >= 2:
        return True
    if day_markers >= 1 and len(normalized_assistant) >= 700:
        return True
    if section_keywords >= 3 and len(normalized_assistant) >= 700:
        return True
    if time_slots >= 3 and len(normalized_assistant) >= 80:
        return True
    if response_phrases >= 2 and len(normalized_assistant) >= 400:
        return True
    if block_count >= 8 and heading_count >= 8 and len(normalized_assistant) >= 1000:
        return True
    if price_mentions >= 3 and heading_count >= 6 and len(normalized_assistant) >= 1000:
        return True
    if len(normalized_assistant) >= 900 and (block_count >= 5 or heading_count >= 5):
        return True
    return False


def _has_residual_promo(assistant_content: str) -> bool:
    normalized = normalize_text(assistant_content)
    lowered = normalized.lower()
    if any(pattern in normalized for pattern in ("商城", "优惠航线", "二维码", "扫码", "小程序")):
        return True
    if "tripai" in lowered:
        return True
    if "平台" in normalized and any(keyword in normalized for keyword in ("预订", "预约", "订票", "优惠", "比价", "下载", "打开")):
        return True
    if "app" in lowered and any(keyword in normalized for keyword in ("下载", "打开", "预订", "优惠")):
        return True
    return False


def _looks_style_heavy(assistant_content: str) -> bool:
    normalized = normalize_text(assistant_content)
    heading_count = len(SECTION_HEADING_PATTERN.findall(normalized))
    bullet_count = len(BULLET_LINE_PATTERN.findall(normalized))
    emphasis_count = len(EMPHASIS_PATTERN.findall(normalized))
    exclamation_count = sum(normalized.count(mark) for mark in ("?", "!", "?"))
    self_reference_hits = sum(pattern in normalized for pattern in SELF_REFERENCE_PATTERNS)

    if self_reference_hits >= 1 and len(normalized) >= 220:
        return True
    if heading_count >= 4 and bullet_count >= 4 and len(normalized) >= 450:
        return True
    if emphasis_count >= 6 and len(normalized) >= 450:
        return True
    if exclamation_count >= 6 and len(normalized) >= 300:
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


def _build_dialogue_sample(record: dict[str, Any]) -> tuple[dict[str, list[dict[str, str]]] | None, str | None]:
    extracted = _extract_turns(record)
    if extracted is None:
        return None, "invalid"

    raw_user, raw_assistant = extracted
    user_query = _clean_dialogue_content(raw_user, max_length=1000)
    assistant_content = _clean_dialogue_content(raw_assistant, max_length=4000)
    skip_reason = _classify_dialogue_sample(user_query, assistant_content)
    if skip_reason is not None:
        return None, skip_reason

    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_content},
        ]
    }, None


def build_dialogue_sample(record: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    sample, _ = _build_dialogue_sample(record)
    return sample


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
    skipped_duplicate = 0
    skipped_by_reason = {
        "invalid": 0,
        "overlong": 0,
        "promo": 0,
        "itinerary_like": 0,
        "style_heavy": 0,
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
        f"超长 {skipped_by_reason['overlong']} 条，"
        f"营销/平台导购 {skipped_by_reason['promo']} 条，"
        f"攻略化/模板化 {skipped_by_reason['itinerary_like']} 条，"
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

