from __future__ import annotations

import argparse
import hashlib
import random
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
    log_warn,
    resolve_path,
    write_json,
    write_jsonl,
)
from src.data_pipeline.global_cleaner import clean_text, normalize_text

DEFAULT_SOURCE_RAW_PATH = "data/raw/guide_generation_raw.jsonl"
DEFAULT_PREPARED_RAW_PATH = "data/raw/guide_generation_raw_expanded.jsonl"
DEFAULT_INPUT_PATH = DEFAULT_PREPARED_RAW_PATH
DEFAULT_OUTPUT_PATH = "data/processed/sft_guide_generation.json"
DEFAULT_SEED = 42
DEFAULT_TARGET_RAW_COUNT = 1500
DEFAULT_TARGET_CLEANABLE_COUNT = 1500
DEFAULT_MAX_WINDOW_DAYS = 3
MIN_ASSISTANT_LENGTH = 240
MAX_INPUT_LENGTH = 8000
MAX_ASSISTANT_LENGTH = 2400
MAX_UNSTRUCTURED_LENGTH = 1800

SYSTEM_PROMPTS = (
    "\u4f60\u662f\u4e13\u4e1a\u7684\u4e2d\u6587\u65c5\u884c\u89c4\u5212\u52a9\u624b\uff0c\u8bf7\u6839\u636e\u7528\u6237\u7ed9\u51fa\u7684\u76ee\u7684\u5730\u548c\u51fa\u884c\u5929\u6570\uff0c\u63d0\u4f9b\u7ed3\u6784\u6e05\u6670\u3001\u53ef\u6267\u884c\u3001\u4fe1\u606f\u514b\u5236\u7684\u5206\u65e5\u884c\u7a0b\u5efa\u8bae\u3002\u907f\u514d\u8425\u9500\u53e3\u543b\uff0c\u4e0d\u505a\u4ef7\u683c\u6216\u4ea7\u54c1\u627f\u8bfa\u3002",
)

USER_QUERY_TEMPLATES = (
    "\u5e2e\u6211\u89c4\u5212\u4e00\u4efd {destination} {days} \u5929\u7684\u65c5\u884c\u653b\u7565\uff0c\u6309\u5929\u5b89\u6392\u91cd\u70b9\u3002",
    "\u6211\u6253\u7b97\u53bb {destination} \u73a9 {days} \u5929\uff0c\u8bf7\u7ed9\u6211\u4e00\u4efd\u6e05\u6670\u7684\u5206\u65e5\u884c\u7a0b\u3002",
    "\u51c6\u5907\u53bb {destination} \u65c5\u884c {days} \u5929\uff0c\u9ebb\u70e6\u6309\u5929\u5e2e\u6211\u5b89\u6392\u884c\u7a0b\u3002",
)

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]+", re.UNICODE)
TRAILING_SPACE_BEFORE_NEWLINE_PATTERN = re.compile(r"[ \t]+\n")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
DAY_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:(?:#{1,6}|[*-])\s*)?(?:\u7b2c[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u53410-9]+\u5929|day\s*[0-9]+)",
    re.IGNORECASE,
)
PRICE_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:\u5143|\u65e5\u5143|\u7f8e\u5143|\u6b27\u5143|\u82f1\u9551|\u6e2f\u5e01|\u745e\u90ce|\u8fea\u62c9\u59c6|\u6bd4\u7d22|NOK|CHF|HKD|RMB|CNY)|(?:\u00a5|\uffe5|RMB|CNY|AED|USD|JPY|CHF|HKD)\s*\d+)",
    re.IGNORECASE,
)
CLOCK_TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\b")
NUMBERED_LISTING_PATTERN = re.compile(r"^\s*(?:[-*]\s*)?(?:\*{0,2}\s*)?(?:\d+[.)\u3001]|[\u2460-\u2469])")
GENERIC_PLATFORM_LISTING_PATTERN = re.compile(
    r"^\*{0,2}\s*(?:\u4f4f\u5bbf|\u9152\u5e97|\u673a\u7968|\u822a\u73ed)\*{0,2}\s*[\uff1a:]\s*\u5e73\u53f0\s*$",
    re.IGNORECASE,
)
PLACEHOLDER_TOKEN_PATTERN = re.compile(r"\[(?:PHONE|EMAIL|ID_CARD|TRUNCATED)\]")

INTRO_NOISE_PATTERNS = (
    "\u6b22\u8fce\u4f60\u6765\u5230",
    "\u8ba9\u6211\u4eec\u8fd9\u4e2a\u65c5\u6e38\u754c\u7684",
    "\u8ba9\u6211\u8fd9\u4e2a\u65c5\u6e38\u754c\u7684",
    "\u5e26\u4f60\u4e00\u8d77\u5f00\u542f",
    "\u7b49\u5f85\u7740\u4f60\u7684\u63a2\u7d22",
)
SELF_INTRO_PATTERNS = (
    "\u6211\u662ftripai",
    "\u6211\u662ftripAI",
    "\u6211\u662f\u5c0f\u5947",
    "\u6211\u662f\u5e73\u53f0",
    "\u4f60\u7684\u4e13\u5c5e\u667a\u80fd\u52a9\u624b",
    "\u65c5\u884c\u642d\u5b50",
    "\u8d34\u5fc3\u7684\u65c5\u6e38\u642d\u5b50",
)
PROMO_LINE_KEYWORDS = (
    "\u70b9\u51fb\u201c\u5546\u57ce\u201d",
    '\u70b9\u51fb"\u5546\u57ce"',
    "\u5f88\u591a\u670b\u53cb\u7ecf\u5e38\u95ee\u65c5\u884c\u6709\u529e\u6cd5\u7701\u94b1\u5417",
    "\u73b0\u5728\u6253\u5f00",
    "\u5c0f\u7a0b\u5e8f",
    "\u4e8c\u7ef4\u7801",
    "\u626b\u7801",
    "\u4f18\u60e0\u822a\u7ebf",
    "\u4e00\u7ad9\u5f0f",
    "\u8fd4\u73b0",
)
PREFACE_SECTION_KEYWORDS = (
    "\u51fa\u884c\u51c6\u5907",
    "\u5e38\u7528app",
    "\u5e38\u7528App",
    "\u5e38\u7528APP",
    "\u5e38\u7528\u5de5\u5177",
    "\u4ea4\u901a",
    "\u4f4f\u5bbf",
    "\u9910\u5385",
    "\u7f8e\u98df",
    "\u5929\u6c14",
    "\u7279\u4ea7",
    "\u624b\u4fe1",
    "\u8d2d\u7269",
)
TAIL_NOISE_SECTION_KEYWORDS = (
    "\u5929\u6c14\u9884\u62a5",
    "\u5929\u6c14\u63d0\u793a",
    "\u7279\u4ea7\u624b\u4fe1",
    "\u8d2d\u7269\u63a8\u8350",
    "\u4f34\u624b\u793c",
    "\u7b7e\u8bc1",
    "\u5b9e\u7528\u4fe1\u606f",
    "\u5b89\u5168\u4e0e\u5e94\u6025",
    "\u5b63\u8282\u4e0e\u7a7f\u7740\u5efa\u8bae",
    "\u793e\u4ea4\u793c\u4eea",
    "\u516c\u5171\u884c\u4e3a",
    "\u7528\u9910\u793c\u4eea",
    "\u9000\u7a0e",
    "\u5b97\u6559\u573a\u6240",
    "\u63d2\u5934\u4e0e\u7535\u538b",
)
CLOSING_PATTERNS = (
    "\u5e0c\u671b\u8fd9\u4efd\u653b\u7565",
    "\u795d\u60a8\u65c5\u9014\u6109\u5feb",
    "\u6b22\u8fce\u968f\u65f6\u54a8\u8be2",
    "\u6b22\u8fce\u968f\u65f6\u6765\u95ee",
    "\u65c5\u884c\u7684\u610f\u4e49\u5728\u4e8e",
    "\u8fd9\u5ea7\u57ce\u5e02\u7684\u9b45\u529b\u5c06\u6c38\u8fdc\u7559\u5728\u4f60\u7684\u8bb0\u5fc6\u4e2d",
    "\u671f\u5f85\u4f60\u7684\u4e0b\u6b21\u5230\u6765",
    "\u611f\u8c22\u4f60\u9009\u62e9",
)
FACTUAL_RISK_KEYWORDS = (
    "\u8fd4\u73b0",
    "\u5b98\u65b9\u6700\u4f4e\u4ef7",
    "\u5168\u7f51\u6700\u4f4e",
    "\u4fdd\u8bc1\u4e70\u5230",
    "\u95ed\u773c\u51b2",
    "\u7a33\u8d5a",
    "\u7edd\u4e0d\u8e29\u96f7",
)
DETAIL_LINE_KEYWORDS = (
    "\u8425\u4e1a\u65f6\u95f4",
    "\u5f00\u653e\u65f6\u95f4",
    "\u6700\u665a\u5165\u56ed",
    "\u505c\u6b62\u552e\u7968",
    "\u95e8\u7968",
    "\u7968\u4ef7",
    "\u4ef7\u683c",
    "\u4eba\u5747",
    "\u5730\u5740",
    "\u4f4d\u7f6e",
    "\u7535\u8bdd",
    "\u8054\u7cfb\u65b9\u5f0f",
    "\u9700\u652f\u4ed8\u8d39\u7528",
)
MEAL_LISTING_KEYWORDS = (
    "\u9910\u5385",
    "\u996d\u5e97",
    "\u9152\u697c",
    "\u5496\u5561\u9986",
    "\u7f8e\u98df",
    "\u5c0f\u5403",
    "\u5e02\u573a",
)
BRAND_REPLACEMENTS = (
    ("tripai\u5c0f\u5947\u65c5\u884capp", "\u5e73\u53f0"),
    ("tripai\u5c0f\u5947\u65c5\u884c", "\u5e73\u53f0"),
    ("tripai", "\u5e73\u53f0"),
    ("\u5c0f\u5947\u65c5\u884c", "\u5e73\u53f0"),
)


def synthesize_user_query(destination: str, days: str, rng: random.Random) -> str:
    template = rng.choice(USER_QUERY_TEMPLATES)
    return template.format(destination=destination, days=days)



def _decode_literal_escapes(text: str) -> str:
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")



def _replace_brand_terms(text: str) -> str:
    for source, target in BRAND_REPLACEMENTS:
        text = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
    return text



def _normalized_for_match(text: str) -> str:
    return EMOJI_PATTERN.sub("", normalize_text(text)).lower().replace(" ", "")



def _is_intro_block(block: str) -> bool:
    normalized = normalize_text(block)
    lowered = normalized.lower()
    if len(normalized) > 260:
        return False
    if any(pattern in normalized for pattern in INTRO_NOISE_PATTERNS):
        return True
    if any(pattern.lower() in lowered for pattern in SELF_INTRO_PATTERNS):
        return True
    return normalized.startswith(("\u54c8\u55bd", "\u563f\uff0c\u65c5\u884c\u8005", "\u4f60\u597d\u5440", "\u54ce\u5440", "\u54c7\u54e6"))



def _is_intro_line(line: str) -> bool:
    normalized = normalize_text(line)
    lowered = normalized.lower()
    if not normalized:
        return False
    if any(pattern in normalized for pattern in INTRO_NOISE_PATTERNS):
        return True
    if any(pattern.lower() in lowered for pattern in SELF_INTRO_PATTERNS):
        return True
    return normalized.startswith(("\u54c8\u55bd", "\u563f\uff0c\u65c5\u884c\u8005", "\u4f60\u597d\u5440", "\u54ce\u5440", "\u54c7\u54e6"))



def _is_promo_line(line: str) -> bool:
    normalized = normalize_text(line)
    lowered = normalized.lower()
    if any(keyword in normalized for keyword in PROMO_LINE_KEYWORDS):
        return True
    if GENERIC_PLATFORM_LISTING_PATTERN.match(normalized):
        return True
    if "\u5e73\u53f0" in normalized and any(
        keyword in normalized
        for keyword in ("\u9884\u8ba2", "\u9884\u7ea6", "\u8ba2\u7968", "\u8ba2\u4f4d", "\u4f18\u60e0", "\u6bd4\u4ef7", "\u4e0b\u8f7d", "\u6253\u5f00")
    ):
        return True
    if "app" in lowered and any(keyword in normalized for keyword in ("\u4e0b\u8f7d", "\u9884\u8ba2", "\u6253\u5f00", "\u4f18\u60e0")):
        return True
    return False



def _is_stale_detail_line(line: str) -> bool:
    normalized = normalize_text(line)
    if not normalized or DAY_MARKER_PATTERN.search(normalized):
        return False
    if PLACEHOLDER_TOKEN_PATTERN.search(normalized):
        return True
    if CLOCK_TIME_PATTERN.search(normalized):
        return True
    if PRICE_PATTERN.search(normalized):
        return True
    return any(keyword in normalized for keyword in DETAIL_LINE_KEYWORDS)



def _is_meal_listing_line(line: str) -> bool:
    normalized = normalize_text(line)
    if not NUMBERED_LISTING_PATTERN.search(normalized):
        return False
    return any(keyword in normalized for keyword in MEAL_LISTING_KEYWORDS)



def _block_heading_text(block: str) -> str:
    first_line = ""
    for line in block.splitlines():
        stripped = normalize_text(line).strip()
        if stripped:
            first_line = stripped
            break
    lowered = first_line.lower().replace(" ", "")
    return lowered.lstrip("#*-0123456789.()[]:?")

def _looks_like_preface_block(block: str) -> bool:
    normalized = normalize_text(block)
    if DAY_MARKER_PATTERN.search(normalized):
        return False
    heading = _block_heading_text(block)
    return any(heading.startswith(keyword.lower().replace(" ", "")) for keyword in PREFACE_SECTION_KEYWORDS)



def _looks_like_tail_noise_block(block: str) -> bool:
    normalized = normalize_text(block)
    if DAY_MARKER_PATTERN.search(normalized):
        return False
    heading = _block_heading_text(block)
    return any(heading.startswith(keyword.lower().replace(" ", "")) for keyword in TAIL_NOISE_SECTION_KEYWORDS)



def _is_closing_block(block: str) -> bool:
    normalized = normalize_text(block)
    return any(pattern in normalized for pattern in CLOSING_PATTERNS)



def _strip_intro_blocks(blocks: list[str]) -> list[str]:
    while len(blocks) > 1 and _is_intro_block(blocks[0]):
        blocks = blocks[1:]
    return blocks



def _trim_to_first_day_block(blocks: list[str]) -> list[str]:
    for index, block in enumerate(blocks):
        if DAY_MARKER_PATTERN.search(normalize_text(block)):
            return blocks[index:]
    return blocks



def _drop_noise_blocks(text: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    blocks = _strip_intro_blocks(blocks)
    blocks = _trim_to_first_day_block(blocks)

    kept_blocks: list[str] = []
    for block in blocks:
        cleaned_lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip() and not _is_intro_line(line) and not _is_promo_line(line) and not _is_stale_detail_line(line) and not _is_meal_listing_line(line)
        ]
        cleaned_block = "\n".join(cleaned_lines).strip()
        if not cleaned_block:
            continue
        if _looks_like_preface_block(cleaned_block) or _looks_like_tail_noise_block(cleaned_block):
            continue
        kept_blocks.append(cleaned_block)

    while kept_blocks and _is_closing_block(kept_blocks[-1]):
        kept_blocks.pop()

    return "\n\n".join(kept_blocks)



def _clean_itinerary_content(value: Any) -> str:
    text = normalize_text(value)
    text = _decode_literal_escapes(text)
    text = EMOJI_PATTERN.sub("", text)
    text = text.replace("\u200d", "").replace("\ufe0f", "")
    text = _replace_brand_terms(text)
    text = _drop_noise_blocks(text)
    text = TRAILING_SPACE_BEFORE_NEWLINE_PATTERN.sub("\n", text)
    text = MULTI_BLANK_LINE_PATTERN.sub("\n\n", text)
    return clean_text(text, max_length=MAX_INPUT_LENGTH)



def _has_residual_promo(content: str) -> bool:
    normalized = normalize_text(content)
    lowered = normalized.lower()
    if PLACEHOLDER_TOKEN_PATTERN.search(normalized):
        return True
    if any(keyword in normalized for keyword in ("\u5546\u57ce", "\u4f18\u60e0\u822a\u7ebf", "\u4e8c\u7ef4\u7801", "\u626b\u7801", "\u5c0f\u7a0b\u5e8f", "\u8fd4\u73b0")):
        return True
    if "tripai" in lowered:
        return True
    if "\u5e73\u53f0" in normalized and any(
        keyword in normalized
        for keyword in ("\u9884\u8ba2", "\u9884\u7ea6", "\u8ba2\u7968", "\u8ba2\u4f4d", "\u4f18\u60e0", "\u6bd4\u4ef7", "\u4e0b\u8f7d", "\u6253\u5f00", "\u8fd4\u73b0")
    ):
        return True
    if "app" in lowered and any(keyword in normalized for keyword in ("\u4e0b\u8f7d", "\u9884\u8ba2", "\u6253\u5f00", "\u4f18\u60e0")):
        return True
    return False



def _is_price_heavy(content: str) -> bool:
    normalized = normalize_text(content)
    price_mentions = len(PRICE_PATTERN.findall(normalized))
    day_markers = len(DAY_MARKER_PATTERN.findall(normalized))
    if price_mentions >= 8:
        return True
    if price_mentions >= 5 and day_markers <= 1:
        return True
    return False



def _is_factually_risky(content: str) -> bool:
    normalized = normalize_text(content)
    return any(keyword in normalized for keyword in FACTUAL_RISK_KEYWORDS)



def _is_detail_heavy(content: str) -> bool:
    normalized = normalize_text(content)
    day_markers = len(DAY_MARKER_PATTERN.findall(normalized))
    clock_times = len(CLOCK_TIME_PATTERN.findall(normalized))
    detail_hits = sum(normalized.count(keyword) for keyword in DETAIL_LINE_KEYWORDS)
    meal_hits = normalized.count("\u63a8\u8350\u7f8e\u98df") + normalized.count("\u7279\u8272\u7f8e\u98df")
    line_count = len([line for line in normalized.splitlines() if line.strip()])

    if clock_times >= 4:
        return True
    if detail_hits >= 6:
        return True
    if meal_hits >= 4 and len(normalized) >= 1600:
        return True
    if len(normalized) >= 1800 and line_count >= 35 and day_markers <= 4:
        return True
    if day_markers >= 1 and len(normalized) / day_markers > 900:
        return True
    return False



def _classify_itinerary_sample(content: str) -> str | None:
    if len(content) < MIN_ASSISTANT_LENGTH:
        return "invalid"
    if _has_residual_promo(content) or _is_factually_risky(content):
        return "promo_or_risky"
    if _is_price_heavy(content):
        return "price_heavy"
    if _is_detail_heavy(content):
        return "detail_heavy"
    if len(content) > MAX_ASSISTANT_LENGTH:
        return "overlong"
    if not DAY_MARKER_PATTERN.search(content) and len(content) > MAX_UNSTRUCTURED_LENGTH:
        return "unstructured"
    return None



def _sample_fingerprint(sample: dict[str, list[dict[str, str]]]) -> str:
    payload = f"{sample['messages'][1]['content']}\n###\n{sample['messages'][2]['content']}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()



def _build_itinerary_sample(
    record: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, list[dict[str, str]]] | None, str | None]:
    destination = normalize_text(record.get("destination") or "\u672a\u77e5\u76ee\u7684\u5730")
    days = normalize_text(record.get("days") or "3")
    content = _clean_itinerary_content(record.get("itinerary_content"))
    skip_reason = _classify_itinerary_sample(content)
    if skip_reason is not None:
        return None, skip_reason

    sample = {
        "id": "guide_generation_" + hashlib.md5((destination + "|" + days + "|" + content).encode("utf-8")).hexdigest()[:12],
        "task_type": "guide_generation",
        "scene": "city_travel",
        "source": "tripai_guide_generation_raw",
        "brand_style": "content_first",
        "difficulty": "medium" if len(content) <= 1400 else "high",
        "messages": [
            {"role": "system", "content": rng.choice(SYSTEM_PROMPTS)},
            {"role": "user", "content": synthesize_user_query(destination, days, rng)},
            {"role": "assistant", "content": content},
        ],
    }
    return sample, None



def build_itinerary_sample(record: dict[str, Any], rng: random.Random) -> dict[str, list[dict[str, str]]] | None:
    sample, _ = _build_itinerary_sample(record, rng)
    return sample



def process_itinerary_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    rng = random.Random(seed)
    processed_data: list[dict[str, list[dict[str, str]]]] = []
    seen_fingerprints: set[str] = set()
    total_records = 0
    skipped_duplicate = 0
    skipped_by_reason = {
        "invalid": 0,
        "promo_or_risky": 0,
        "price_heavy": 0,
        "detail_heavy": 0,
        "overlong": 0,
        "unstructured": 0,
    }

    log_info(f"\u5f00\u59cb\u5904\u7406 guide_generation \u6570\u636e: {resolve_path(input_file_path)}")

    try:
        for _, record in iter_jsonl(input_file_path):
            total_records += 1
            sample, skip_reason = _build_itinerary_sample(record, rng)
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
        log_error(f"\u672a\u627e\u5230\u8f93\u5165\u6587\u4ef6: {resolve_path(input_file_path)}")
        return []

    output_path = write_json(output_json_path, processed_data)
    log_success(
        "guide_generation \u6570\u636e\u5904\u7406\u5b8c\u6210\u3002"
        f"\u8bfb\u53d6 {total_records} \u6761\uff0c\u8f93\u51fa {len(processed_data)} \u6761\uff0c"
        f"\u65e0\u6548/\u8fc7\u77ed {skipped_by_reason['invalid']} \u6761\uff0c"
        f"\u5e7f\u544a\u6216\u98ce\u9669\u627f\u8bfa {skipped_by_reason['promo_or_risky']} \u6761\uff0c"
        f"\u4ef7\u683c\u8fc7\u5bc6 {skipped_by_reason['price_heavy']} \u6761\uff0c"
        f"\u8d85\u957f {skipped_by_reason['overlong']} \u6761\uff0c"
        f"\u7ed3\u6784\u4e0d\u8db3 {skipped_by_reason['unstructured']} \u6761\uff0c"
        f"\u53bb\u91cd {skipped_duplicate} \u6761\u3002"
    )
    log_info(f"\u8f93\u51fa\u6587\u4ef6: {output_path}")
    return processed_data





def build_guide_generation_sample(record: dict[str, Any], rng: random.Random) -> dict[str, list[dict[str, str]]] | None:
    return build_itinerary_sample(record, rng)


def process_guide_generation_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, list[dict[str, str]]]]:
    return process_itinerary_data(input_file_path, output_json_path, seed)


def _normalize_raw_content(value: Any) -> str:
    text = normalize_text(value)
    text = _decode_literal_escapes(text)
    text = TRAILING_SPACE_BEFORE_NEWLINE_PATTERN.sub("\n", text)
    text = MULTI_BLANK_LINE_PATTERN.sub("\n\n", text)
    return text.strip()


def _raw_record_fingerprint(record: dict[str, Any]) -> str:
    payload = "\n###\n".join(
        (
            normalize_text(record.get("destination")),
            normalize_text(record.get("days")),
            _normalize_raw_content(record.get("itinerary_content")),
        )
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _split_day_blocks(text: str) -> tuple[str, list[str]]:
    normalized = _normalize_raw_content(text)
    matches = list(DAY_MARKER_PATTERN.finditer(normalized))
    if not matches:
        return normalized, []

    prefix = normalized[:matches[0].start()].strip()
    day_blocks: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        block = normalized[start:end].strip()
        if block:
            day_blocks.append(block)
    return prefix, day_blocks


def _compose_window_content(prefix: str, day_blocks: list[str], start_index: int, window_size: int) -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.extend(day_blocks[start_index : start_index + window_size])
    return "\n\n".join(part for part in parts if part).strip()


def _strip_internal_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _estimate_cleanable(record: dict[str, Any]) -> bool:
    sample, _ = _build_itinerary_sample(record, random.Random(DEFAULT_SEED))
    return sample is not None


def _build_raw_candidate_bundle(record: dict[str, Any], source_record_id: int) -> list[dict[str, Any]]:
    destination = normalize_text(record.get("destination") or "Unknown destination")
    source_days = normalize_text(record.get("days") or "3")
    raw_content = _normalize_raw_content(record.get("itinerary_content"))
    if not raw_content:
        return []

    original = {
        "destination": destination,
        "days": source_days,
        "itinerary_content": raw_content,
        "source_record_id": source_record_id,
        "source_days": source_days,
        "variant_type": "original",
        "_priority": (0, 0, 0),
    }
    original["_estimated_cleanable"] = _estimate_cleanable(original)

    prefix, day_blocks = _split_day_blocks(raw_content)
    seen_fingerprints = {_raw_record_fingerprint(original)}
    bundle: list[dict[str, Any]] = [original]

    if not day_blocks:
        return bundle

    max_window_days = min(DEFAULT_MAX_WINDOW_DAYS, len(day_blocks))
    for window_size in range(max_window_days, 0, -1):
        for start_index in range(0, len(day_blocks) - window_size + 1):
            content = _compose_window_content(prefix, day_blocks, start_index, window_size)
            if not content:
                continue

            candidate = {
                "destination": destination,
                "days": str(window_size),
                "itinerary_content": content,
                "source_record_id": source_record_id,
                "source_days": source_days,
                "variant_type": "contiguous_day_window",
                "window_start_day": start_index + 1,
                "window_end_day": start_index + window_size,
                "_priority": (1, -window_size, start_index),
            }
            fingerprint = _raw_record_fingerprint(candidate)
            if fingerprint in seen_fingerprints:
                continue

            seen_fingerprints.add(fingerprint)
            candidate["_estimated_cleanable"] = _estimate_cleanable(candidate)
            bundle.append(candidate)

    return sorted(
        bundle,
        key=lambda item: (
            item["_priority"][0],
            0 if item.get("_estimated_cleanable") else 1,
            item["_priority"][1],
            item["_priority"][2],
        ),
    )


def prepare_guide_generation_raw(
    source_input_path: str = DEFAULT_SOURCE_RAW_PATH,
    output_file_path: str = DEFAULT_PREPARED_RAW_PATH,
    *,
    target_raw_count: int = DEFAULT_TARGET_RAW_COUNT,
    target_cleanable_count: int = DEFAULT_TARGET_CLEANABLE_COUNT,
) -> list[dict[str, Any]]:
    configure_console_output()
    log_info(f"Preparing guide_generation raw: {resolve_path(source_input_path)}")

    candidate_bundles: list[list[dict[str, Any]]] = []
    total_records = 0
    total_candidates = 0
    try:
        for line_number, record in iter_jsonl(source_input_path):
            total_records += 1
            bundle = _build_raw_candidate_bundle(record, line_number)
            if not bundle:
                continue
            total_candidates += len(bundle)
            candidate_bundles.append(bundle)
    except FileNotFoundError:
        log_error(f"Input file not found: {resolve_path(source_input_path)}")
        return []

    selected_records: list[dict[str, Any]] = []
    selected_cleanable_count = 0
    seen_fingerprints: set[str] = set()
    max_depth = max((len(bundle) for bundle in candidate_bundles), default=0)

    stop_selection = False
    for depth in range(max_depth):
        added_in_round = False
        for bundle in candidate_bundles:
            if depth >= len(bundle):
                continue

            candidate = bundle[depth]
            fingerprint = _raw_record_fingerprint(candidate)
            if fingerprint in seen_fingerprints:
                continue

            seen_fingerprints.add(fingerprint)
            added_in_round = True
            if candidate.get("_estimated_cleanable"):
                selected_cleanable_count += 1
            selected_records.append(_strip_internal_fields(candidate))

            if len(selected_records) >= target_raw_count and selected_cleanable_count >= target_cleanable_count:
                stop_selection = True
                break
        if stop_selection:
            break
        if not added_in_round:
            break

    output_path = write_jsonl(output_file_path, selected_records)
    if len(selected_records) < target_raw_count:
        log_warn(f"guide_generation raw did not reach target_raw_count: target={target_raw_count}, current={len(selected_records)}")
    if selected_cleanable_count < target_cleanable_count:
        log_warn(
            f"guide_generation raw did not reach estimated cleanable target: target={target_cleanable_count}, current={selected_cleanable_count}"
        )

    log_success(
        "guide_generation raw preparation complete. "
        f"source_records={total_records}, candidate_records={total_candidates}, "
        f"output_raw={len(selected_records)}, estimated_cleanable={selected_cleanable_count}."
    )
    log_info(f"Output file: {output_path}")
    return selected_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand guide_generation_raw and clean it into ChatML.")
    parser.add_argument(
        "--mode",
        choices=("prepare-raw", "clean", "all"),
        default="all",
        help="prepare-raw only expands raw, clean only converts raw to ChatML, all runs both steps.",
    )
    parser.add_argument("--source-raw", default=DEFAULT_SOURCE_RAW_PATH, help="Source guide_generation raw JSONL path.")
    parser.add_argument(
        "--prepared-raw",
        default=DEFAULT_PREPARED_RAW_PATH,
        help="Expanded guide_generation raw JSONL output path.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Raw input path for clean mode; defaults to prepared-raw.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="guide_generation ChatML JSON output path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible output.")
    parser.add_argument("--target-raw", type=int, default=DEFAULT_TARGET_RAW_COUNT, help="Target raw sample count.")
    parser.add_argument(
        "--target-cleanable",
        type=int,
        default=DEFAULT_TARGET_CLEANABLE_COUNT,
        help="Target estimated cleanable count during raw expansion.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    clean_input = args.input or args.prepared_raw

    if args.mode in {"prepare-raw", "all"}:
        prepare_guide_generation_raw(
            args.source_raw,
            args.prepared_raw,
            target_raw_count=args.target_raw,
            target_cleanable_count=args.target_cleanable,
        )

    if args.mode in {"clean", "all"}:
        process_guide_generation_data(clean_input, args.output, args.seed)


if __name__ == "__main__":
    main()
