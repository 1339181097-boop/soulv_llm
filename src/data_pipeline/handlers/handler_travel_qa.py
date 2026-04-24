from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    load_records,
    log_error,
    log_info,
    log_success,
    log_warn,
    resolve_path,
    write_json,
    write_jsonl,
)
from src.data_pipeline.global_cleaner import clean_text, normalize_text
from src.data_pipeline.system_prompt_loader import load_system_prompt

STAGE1_TRAVEL_QA_FINAL_TARGET = 3250
STAGE1_TRAVEL_QA_CANDIDATE_MIN = 8000
STAGE1_TRAVEL_QA_CANDIDATE_MAX = 12000

DEFAULT_INPUT_PATH = "data/raw/travel_qa_raw_2026_04_22.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_travel_qa.json"
DEFAULT_JSONL_OUTPUT_PATH = "data/processed/sft_travel_qa_2026_04_22_strict.jsonl"
DEFAULT_REPORT_PATH = "data/reports/travel_qa_2026_04_22_strict_report.json"
DEFAULT_TOTAL_SAMPLES = STAGE1_TRAVEL_QA_FINAL_TARGET
DEFAULT_CITY_CAP = 80
DEFAULT_ANSWER_CAP = 1
SPOT_RATIO = 0.55
CITY_RATIO = 0.25
TRAFFIC_RATIO = 0.20
_DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "\u4f60\u662f\u4e13\u4e1a\u7684\u4e2d\u6587\u65c5\u6e38\u95ee\u7b54\u52a9\u624b\u3002"
    "\u8bf7\u9488\u5bf9\u7528\u6237\u63d0\u51fa\u7684\u666f\u70b9\u3001\u57ce\u5e02\u3001\u76ee\u7684"
    "\u5730\u6216\u4ea4\u901a\u76f8\u5173\u95ee\u9898\uff0c\u7ed9\u51fa\u51c6\u786e\u3001\u81ea\u7136"
    "\u3001\u7b80\u6d01\u3001\u76f4\u63a5\u7684\u56de\u7b54\u3002\u907f\u514d\u8425\u9500\u53e3\u543b"
    "\uff0c\u4e0d\u505a\u4ef7\u683c\u3001\u5e93\u5b58\u6216\u4ea7\u54c1\u627f\u8bfa\u3002"
)
DEFAULT_SYSTEM_PROMPT = load_system_prompt("travel_qa", _DEFAULT_SYSTEM_PROMPT_FALLBACK)
TIME_SENSITIVE_ADVISORY = (
    "\u5177\u4f53\u5f00\u653e\u3001\u6f14\u51fa\u3001\u4ea4\u901a\u6216\u6d3b\u52a8\u5b89\u6392"
    "\u5efa\u8bae\u4ee5\u5f53\u65e5\u516c\u544a\u6216\u5b98\u65b9\u4fe1\u606f\u4e3a\u51c6\u3002"
)
CONTEXT_PREFIX = "\u53c2\u8003\u4fe1\u606f\uff1a"
USER_QUESTION_PREFIX = "\u7528\u6237\u95ee\u9898\uff1a"
COMMON_LOCATION_SUFFIXES = (
    "\u5e02",
    "\u5730\u533a",
    "\u81ea\u6cbb\u5dde",
    "\u81ea\u6cbb\u53bf",
    "\u53bf",
    "\u533a",
    "\u65b0\u533a",
    "\u666f\u533a",
    "\u98ce\u666f\u533a",
    "\u65c5\u6e38\u533a",
    "\u5ea6\u5047\u533a",
    "\u516c\u56ed",
    "\u53e4\u9547",
    "\u535a\u7269\u9986",
    "\u4e50\u56ed",
    "\u6d77\u6d0b\u9986",
    "\u7d22\u9053",
    "\u56fd\u5bb6\u516c\u56ed",
    "\u98ce\u666f\u540d\u80dc\u533a",
)
GENERIC_QUESTION_PREFIXES = (
    "\u600e\u4e48\u53bb",
    "\u4ece\u5e02\u533a\u600e\u4e48\u53bb",
    "\u4ece\u54ea\u91cc\u8d70",
    "\u4f4f\u54ea\u91cc",
    "\u4f4f\u54ea\u4e2a\u533a\u57df",
    "\u9002\u5408",
    "\u503c\u5f97\u53bb",
    "\u597d\u73a9\u5417",
    "\u6709\u4ec0\u4e48",
    "\u73a9\u4ec0\u4e48",
    "\u665a\u4e0a\u6709\u4ec0\u4e48",
)
GENERIC_QUESTION_TOKENS = (
    "\u6700\u65b9\u4fbf",
    "\u600e\u4e48\u5750\u8f66",
    "\u5750\u4ec0\u4e48\u8f66",
    "\u9002\u5408\u5e26\u5b69\u5b50",
    "\u9002\u5408\u8001\u4eba",
    "\u9002\u5408\u4eb2\u5b50",
    "\u600e\u4e48\u5b89\u6392",
)

CLOCK_TIME_PATTERN = re.compile(
    r"(?:\b\d{1,2}[:\uff1a]\d{2}\b|\d{1,2}\s*\u70b9(?:\d{1,2}\s*\u5206|\u534a)?(?:\u5de6\u53f3)?)"
)
PRICE_PATTERN = re.compile(
    r"(?:[\d.]+\s*(?:\u5143|\u65e5\u5143|\u7f8e\u5143|\u6b27\u5143|\u82f1\u9551|\u6e2f\u5e01|"
    r"\u6bd4\u7d22|\u8fea\u62c9\u59c6))(?:/\u4eba|/\u4f4d|/\u5f20)?"
)
DISTANCE_PATTERN = re.compile(r"(?:\u603b\u8ddd\u79bb)?\u7ea6?\s*[\d.]+\s*\u516c\u91cc")
DURATION_PATTERN = re.compile(r"(?:\u516c\u5171\u4ea4\u901a)?\u7ea6?\s*[\d.]+\s*\u5206\u949f")
VOLATILE_SEGMENT_PATTERN = re.compile(
    r"(?:\u6bcf\u5929|\u6bcf\u65e5|\u6bcf\u5468|\u5de5\u4f5c\u65e5|\u5468\u672b|\u8282\u5047\u65e5|"
    r"\u5f53\u65e5|\u4ee5\u5f53\u65e5|\u5f00\u653e\u65f6\u95f4|\u8425\u4e1a\u65f6\u95f4|\u7968\u4ef7|"
    r"\u95e8\u7968|\u6f14\u51fa\u65f6\u957f|\u573a\u6b21|\u4eae\u706f|\u516c\u544a|\u9884\u7ea6|"
    r"\u9884\u7ea6\u65b9\u5f0f|\u73ed\u6b21|\u9996\u73ed|\u672b\u73ed)"
)
TIME_OF_DAY_PATTERN = re.compile(
    r"(?:\u65e9\u4e0a|\u4e0a\u5348|\u4e2d\u5348|\u4e0b\u5348|\u508d\u665a|\u665a\u4e0a|\u51cc\u6668)"
)
DETAIL_CLAUSE_PATTERN = re.compile(
    r"(?:\u6bcf(?:\u5929|\u65e5|\u5468)|\u5de5\u4f5c\u65e5|\u5468\u672b|\u8282\u5047\u65e5|"
    r"\u5f53\u65e5|\u4ee5\u5f53\u65e5|\u5f00\u653e\u65f6\u95f4|\u8425\u4e1a\u65f6\u95f4|\u7968\u4ef7|"
    r"\u95e8\u7968|\u6f14\u51fa\u65f6\u957f|\u573a\u6b21|\u4eae\u706f|\u516c\u544a|\u9884\u7ea6|"
    r"\u73ed\u6b21|\u9996\u73ed|\u672b\u73ed)[^\u3002\uff1b;]*"
)
TRAILING_PUNCTUATION_PATTERN = re.compile(r"^[\uff0c,;；:\uff1a\s]+|[\uff0c,;；:\uff1a\s]+$")
PII_TOKEN_PATTERN = re.compile(r"\[(?:PHONE|EMAIL|ID_CARD|TRUNCATED)\]")
BOOKING_NOISE_PATTERN = re.compile(r"(?:\u5e93\u5b58|\u4f59\u7968|\u6709\u623f|\u9884\u8ba2|\u9884\u7ea6|\u7acb\u5373\u8d2d\u4e70|\u9650\u65f6\u62a2|\u4f18\u60e0\u4ef7)")
STRICT_TICKET_QUESTION_TYPES = {"\u7968\u52a1\u4fe1\u606f"}
STRICT_MARKDOWN_PATTERN = re.compile(r"(?:\*\*|```|^#{1,6}\s*)", re.M)
STRICT_PRICE_OR_TICKET_PATTERN = re.compile(
    r"(?:\d+(?:[.,]\d+)?\s*(?:\u5143|\u97e9\u5143|\u65e5\u5143|\u9a6c\u5e01|\u5362\u6bd4|"
    r"\u7f8e\u5143|USD|RM|THB|SGD|\$|€)|"
    r"(?:USD|RM|THB|SGD|\$|€)\s*\d+(?:[.,]\d+)?|"
    r"\u7968\u4ef7|\u6210\u4eba\u7968|\u513f\u7ae5\u7968|\u8d2d\u7968|\u4e70\u7968|\u552e\u7968|"
    r"\u5305\u65e5\u4ef7|\u8d77\u6b65\u4ef7|\u542b\u5168\u9669\u7ea6|"
    r"\u4eba\u5747\u7ea6?\s*\d+)"
)
STRICT_HOURS_OR_SCHEDULE_PATTERN = re.compile(
    r"(?:\u8425\u4e1a\u65f6\u95f4|\u5f00\u653e\u65f6\u95f4|\u95ed\u9986|"
    r"\u73ed\u6b21|\u8f66\u6b21|\u822a\u73ed|\u9996\u73ed|\u672b\u73ed|\u53d1\u8f66|"
    r"\u6bcf\s*\d+\s*\u5206\u949f\s*\u4e00\u73ed|\d+\s*\u5206\u949f\s*\u4e00\u73ed|"
    r"\u5b9e\u65f6\u67e5\u8be2|\u73b0\u573a\u8bae\u4ef7|KakaoMap|Naver Map|T-Money)"
)
STRICT_TRANSACTION_PATTERN = re.compile(
    r"(?:\u5e93\u5b58|\u4f59\u7968|\u6709\u623f|\u623f\u6001|\u9884\u8ba2|\u4e0b\u5355|"
    r"\u652f\u4ed8|\u8ba2\u5355|\u7acb\u5373\u8d2d\u4e70|\u626b\u7801|\u5145\u503c)"
)
STRICT_TOOL_OR_JSON_PATTERN = re.compile(
    r"(?:tool_calls?|function_call|```json|\"\s*tool\s*\"|\{\s*\"(?:intent|intentionName)\")",
    re.IGNORECASE,
)
STRICT_PROMO_PATTERN = re.compile(
    r"(?:\u5b98\u65b9\u9884\u8ba2|\u7acb\u5373\u62a2\u8d2d|\u4e13\u4eab\u4f18\u60e0|"
    r"\u8054\u7cfb\u5ba2\u670d|\u4e0b\u8f7dAPP|\u6253\u5f00APP|\u5bfc\u6d41)"
)
STRICT_MOJIBAKE_PATTERN = re.compile(r"\ufffd|(?:ç|è|é|å|æ|ä|ï|¼|½|œ|‰|¤|»){4,}")
MAX_STRICT_USER_CHARS = 260
MAX_STRICT_ASSISTANT_CHARS = 420


def _name_aliases(name: str) -> set[str]:
    normalized = normalize_text(name).replace(" ", "")
    if not normalized:
        return set()

    aliases = {normalized}
    for suffix in COMMON_LOCATION_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            aliases.add(normalized[: -len(suffix)])
    return {alias for alias in aliases if alias}


def _mentions_known_place(question: str, city: str, entity_name: str) -> bool:
    normalized_question = normalize_text(question).replace(" ", "")
    aliases = _name_aliases(city) | _name_aliases(entity_name)
    return any(alias and alias in normalized_question for alias in aliases)


def _needs_context(question: str) -> bool:
    normalized_question = normalize_text(question)
    if len(normalized_question) <= 10:
        return True
    if normalized_question.startswith(GENERIC_QUESTION_PREFIXES):
        return True
    return any(token in normalized_question for token in GENERIC_QUESTION_TOKENS)


def _build_contextual_user_query(question: str, city: str, entity_name: str) -> str:
    context_parts: list[str] = []
    if city:
        context_parts.append(f"\u57ce\u5e02\uff1a{city}")
    if entity_name and entity_name != city:
        context_parts.append(f"\u5bf9\u8c61\uff1a{entity_name}")
    if not context_parts:
        return question
    return f"{CONTEXT_PREFIX}{'\uff1b'.join(context_parts)}\n\n{USER_QUESTION_PREFIX}{question}"


def _self_contained_user_query(question: str, city: str, entity_name: str) -> str:
    if not question:
        return ""
    if _mentions_known_place(question, city, entity_name):
        return question
    if not _needs_context(question):
        return question
    return _build_contextual_user_query(question, city, entity_name)


def _classify_trip_length(minutes: float | None, distance_km: float | None) -> str:
    if minutes is not None:
        if minutes <= 40:
            return "\u6574\u4f53\u8def\u7a0b\u4e0d\u7b97\u8fdc"
        if minutes <= 90:
            return "\u6574\u4f53\u8def\u7a0b\u9002\u4e2d"
        return "\u6574\u4f53\u8def\u7a0b\u76f8\u5bf9\u8f83\u8fdc"
    if distance_km is not None:
        if distance_km <= 10:
            return "\u6574\u4f53\u8def\u7a0b\u4e0d\u7b97\u8fdc"
        if distance_km <= 30:
            return "\u6574\u4f53\u8def\u7a0b\u9002\u4e2d"
        return "\u6574\u4f53\u8def\u7a0b\u76f8\u5bf9\u8f83\u8fdc"
    return "\u516c\u5171\u4ea4\u901a\u901a\u5e38\u53ef\u4ee5\u5230\u8fbe"


def _extract_first_number(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    number_match = re.search(r"[\d.]+", match.group(0))
    if not number_match:
        return None
    return float(number_match.group(0))


def _traffic_mode_summary(text: str) -> str:
    has_subway = "\u5730\u94c1" in text
    has_bus = any(token in text for token in ("\u516c\u4ea4", "\u5df4\u58eb", "\u4e13\u7ebf", "\u89c2\u5149\u4e13\u7ebf"))
    has_train = any(token in text for token in ("\u9ad8\u94c1", "\u706b\u8f66", "\u94c1\u8def"))
    if has_subway and has_bus:
        return "\u4e00\u822c\u53ef\u4f18\u5148\u8003\u8651\u5730\u94c1\u6362\u4e58\u516c\u4ea4"
    if has_subway:
        return "\u4e00\u822c\u53ef\u4f18\u5148\u8003\u8651\u5730\u94c1\u51fa\u884c"
    if has_bus:
        return "\u4e00\u822c\u53ef\u4f18\u5148\u8003\u8651\u516c\u4ea4\u6216\u65c5\u6e38\u4e13\u7ebf"
    if has_train:
        return "\u901a\u5e38\u53ef\u4ee5\u5148\u4e58\u5750\u706b\u8f66\u6216\u9ad8\u94c1\u5230\u9644\u8fd1\uff0c\u518d\u8854\u63a5\u5e02\u5185\u4ea4\u901a"
    return "\u4e00\u822c\u53ef\u4f18\u5148\u8003\u8651\u516c\u5171\u4ea4\u901a\u6216\u6253\u8f66\u524d\u5f80"


def _destination_phrase(record: dict[str, Any]) -> str:
    entity_name = normalize_text(record.get("entity_name"))
    city = normalize_text(record.get("city"))
    if entity_name and entity_name != city:
        return entity_name
    if city:
        return city
    return "\u76ee\u7684\u5730"


def _simplify_traffic_answer(record: dict[str, Any], answer: str) -> str:
    normalized = normalize_text(answer)
    minutes = _extract_first_number(DURATION_PATTERN, normalized)
    distance_km = _extract_first_number(DISTANCE_PATTERN, normalized)
    mode_summary = _traffic_mode_summary(normalized)
    trip_summary = _classify_trip_length(minutes, distance_km)
    destination = _destination_phrase(record)
    return (
        f"\u53bb{destination}\u65f6\uff0c{mode_summary}\uff0c{trip_summary}\u3002"
        "\u5982\u679c\u66f4\u770b\u91cd\u7701\u5fc3\u6216\u540c\u884c\u4eba\u6570\u8f83\u591a\uff0c\u4e5f\u53ef\u4ee5\u76f4\u63a5\u6253\u8f66\u3002"
        "\u51fa\u53d1\u524d\u5efa\u8bae\u518d\u786e\u8ba4\u5f53\u65e5\u7ebf\u8def\u3001\u7ad9\u70b9\u548c\u8fd0\u8425\u60c5\u51b5\u3002"
    )


def _strip_time_sensitive_detail(segment: str) -> str:
    cleaned = CLOCK_TIME_PATTERN.sub("", segment)
    cleaned = PRICE_PATTERN.sub("", cleaned)
    cleaned = DETAIL_CLAUSE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\d+\s*(?:\u5206\u949f|\u5c0f\u65f6|\u5929)", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = TRAILING_PUNCTUATION_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"(?:\u5f00\u59cb|\u5f00\u653e|\u8425\u4e1a|\u4e0a\u6f14|\u4e0d\u7b49)$", "", cleaned)
    return cleaned


def _looks_stable_segment(segment: str) -> bool:
    normalized = normalize_text(segment)
    if len(normalized) < 4:
        return False
    if PRICE_PATTERN.search(normalized):
        return False
    if VOLATILE_SEGMENT_PATTERN.search(normalized):
        return False
    if CLOCK_TIME_PATTERN.search(normalized):
        return False
    return True


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _soften_time_sensitive_answer(answer: str) -> str:
    normalized = normalize_text(answer)
    segments = [segment.strip() for segment in re.split(r"[\u3002\uff1b;]", normalized) if segment.strip()]
    kept_segments: list[str] = []
    for segment in segments:
        if (
            CLOCK_TIME_PATTERN.search(segment)
            or PRICE_PATTERN.search(segment)
            or VOLATILE_SEGMENT_PATTERN.search(segment)
            or TIME_OF_DAY_PATTERN.search(segment)
        ):
            stripped = _strip_time_sensitive_detail(segment)
            if _looks_stable_segment(stripped):
                kept_segments.append(stripped)
            continue
        if _looks_stable_segment(segment):
            kept_segments.append(segment)

    stable_segments = _dedupe_preserve_order(kept_segments)
    if stable_segments:
        return f"{'\uff1b'.join(stable_segments)}\u3002{TIME_SENSITIVE_ADVISORY}"
    return TIME_SENSITIVE_ADVISORY


def _clean_answer(record: dict[str, Any], answer: str) -> str:
    task_type = normalize_text(record.get("task_type")).lower()
    entity_type = normalize_text(record.get("entity_type")).lower()
    is_time_sensitive = bool(record.get("is_time_sensitive"))
    normalized_answer = normalize_text(answer)

    if PII_TOKEN_PATTERN.search(normalized_answer) or BOOKING_NOISE_PATTERN.search(normalized_answer):
        return ""

    if task_type == "traffic_qa" or entity_type == "traffic":
        return _simplify_traffic_answer(record, normalized_answer)
    if is_time_sensitive:
        return _soften_time_sensitive_answer(normalized_answer)
    return normalized_answer


def _clean_list(raw_value: Any, *, max_items: int = 8, item_length: int = 40) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_value:
        text = clean_text(item, max_length=item_length, mask_sensitive=False)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _normalize_assistant_style(answer: str) -> str:
    cleaned = STRICT_MARKDOWN_PATTERN.sub("", answer)
    cleaned = normalize_text(cleaned)
    if cleaned and not cleaned.endswith(("\u3002", "\uff1f", "\uff01", ".", "?", "!")):
        cleaned += "\u3002"
    return cleaned


def _message_content(sample: dict[str, Any], role: str) -> str:
    for message in sample.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def classify_strict_filter_reason(sample: dict[str, Any]) -> str | None:
    source_task_type = normalize_text(sample.get("source_task_type"))
    if source_task_type not in {"spot_qa", "city_qa", "traffic_qa"}:
        return "wrong_source_task_type"

    question_type = normalize_text(sample.get("question_type"))
    if question_type in STRICT_TICKET_QUESTION_TYPES:
        return "ticket_question_type"

    user_text = _message_content(sample, "user")
    assistant_text = _message_content(sample, "assistant")
    combined = f"{user_text}\n{assistant_text}"
    if not user_text or not assistant_text:
        return "empty_chatml_content"
    if len(user_text) > MAX_STRICT_USER_CHARS:
        return "overlong_user_query"
    if len(assistant_text) > MAX_STRICT_ASSISTANT_CHARS:
        return "overlong_assistant_answer"
    if assistant_text == TIME_SENSITIVE_ADVISORY:
        return "generic_time_sensitive_only"
    if "[TRUNCATED]" in combined:
        return "truncated_content"
    if STRICT_MOJIBAKE_PATTERN.search(combined):
        return "mojibake_content"
    if STRICT_TOOL_OR_JSON_PATTERN.search(combined):
        return "tool_or_json_content"
    if STRICT_PROMO_PATTERN.search(combined):
        return "promotional_content"
    if STRICT_TRANSACTION_PATTERN.search(combined):
        return "booking_or_transaction_content"
    if STRICT_PRICE_OR_TICKET_PATTERN.search(combined):
        return "price_or_ticket_content"
    if STRICT_HOURS_OR_SCHEDULE_PATTERN.search(combined):
        return "schedule_or_hours_content"
    return None


def build_travel_qa_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    city = clean_text(record.get("city"), max_length=100)
    entity_name = clean_text(record.get("entity_name"), max_length=200)
    question = clean_text(record.get("user_query") or record.get("question"), max_length=2000)
    raw_answer = clean_text(
        record.get("assistant_content") or record.get("assistant_response") or record.get("answer"),
        max_length=5000,
    )
    if not question or not raw_answer:
        return None

    user_query = _self_contained_user_query(question, city, entity_name)
    assistant_answer = clean_text(_normalize_assistant_style(_clean_answer(record, raw_answer)), max_length=2500)
    if not user_query or not assistant_answer:
        return None

    system_prompt = clean_text(
        record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        max_length=1000,
        mask_sensitive=False,
    )
    sample_id = "travel_qa_" + hashlib.md5(
        f"{city}|{entity_name}|{user_query}|{assistant_answer}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": sample_id,
        "record_id": clean_text(record.get("record_id"), max_length=100, mask_sensitive=False),
        "task_type": "travel_qa",
        "scene": "travel_qa",
        "source": clean_text(record.get("source") or "tripai_db", max_length=100, mask_sensitive=False),
        "source_dataset": "travel_qa_raw_2026_04_22",
        "source_id": clean_text(record.get("source_id"), max_length=100, mask_sensitive=False),
        "updated_at": clean_text(record.get("updated_at"), max_length=40, mask_sensitive=False),
        "source_task_type": clean_text(record.get("task_type"), max_length=100, mask_sensitive=False),
        "city": city,
        "entity_name": entity_name,
        "entity_type": clean_text(record.get("entity_type"), max_length=50, mask_sensitive=False),
        "question_type": clean_text(record.get("question_type"), max_length=50, mask_sensitive=False),
        "tags": _clean_list(record.get("tags"), max_items=8, item_length=40),
        "is_time_sensitive": bool(record.get("is_time_sensitive")),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_answer},
        ],
    }


def _fingerprint(sample: dict[str, Any]) -> str:
    messages = sample["messages"]
    return hashlib.md5(
        f"{messages[1]['content']}\n###\n{messages[2]['content']}".encode("utf-8")
    ).hexdigest()


def _answer_fingerprint(sample: dict[str, Any]) -> str:
    return hashlib.md5(normalize_text(sample["messages"][2]["content"]).encode("utf-8")).hexdigest()


def _calculate_category_targets(total_samples: int) -> dict[str, int]:
    spot_target = round(total_samples * SPOT_RATIO)
    city_target = round(total_samples * CITY_RATIO)
    traffic_target = total_samples - spot_target - city_target
    return {
        "spot_qa": spot_target,
        "city_qa": city_target,
        "traffic_qa": traffic_target,
    }


def _select_balanced_subset(
    samples: list[dict[str, Any]],
    *,
    target_count: int,
    city_cap: int,
) -> list[dict[str, Any]]:
    if target_count <= 0 or len(samples) <= target_count:
        return list(samples)

    city_buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for sample in samples:
        city_key = sample.get("city") or "__unknown__"
        city_buckets[city_key].append(sample)

    city_order = sorted(city_buckets, key=lambda city: (-len(city_buckets[city]), city))
    city_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    active_cities = list(city_order)
    while active_cities and len(selected) < target_count:
        next_active: list[str] = []
        for city in active_cities:
            if len(selected) >= target_count:
                break
            if city_cap > 0 and city_counts[city] >= city_cap:
                continue
            bucket = city_buckets[city]
            if not bucket:
                continue
            selected.append(bucket.popleft())
            city_counts[city] += 1
            if bucket and (city_cap <= 0 or city_counts[city] < city_cap):
                next_active.append(city)
        active_cities = next_active
    return selected


def _balance_categories(
    processed: list[dict[str, Any]],
    *,
    total_samples: int,
    city_cap: int,
) -> list[dict[str, Any]]:
    if total_samples <= 0 or len(processed) <= total_samples:
        return processed

    targets = _calculate_category_targets(total_samples)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in processed:
        grouped[sample.get("source_task_type")].append(sample)

    balanced: list[dict[str, Any]] = []
    for source_task_type in ("spot_qa", "city_qa", "traffic_qa"):
        balanced.extend(
            _select_balanced_subset(
                grouped.get(source_task_type, []),
                target_count=targets.get(source_task_type, 0),
                city_cap=city_cap,
            )
        )

    if len(balanced) < total_samples:
        selected_ids = {sample["id"] for sample in balanced}
        for sample in processed:
            if len(balanced) >= total_samples:
                break
            if sample["id"] in selected_ids:
                continue
            balanced.append(sample)
            selected_ids.add(sample["id"])

    return balanced


def _limit_repeated_answers(samples: list[dict[str, Any]], *, answer_cap: int) -> list[dict[str, Any]]:
    if answer_cap <= 0:
        return list(samples)

    answer_counts: Counter[str] = Counter()
    limited: list[dict[str, Any]] = []
    for sample in samples:
        fingerprint = _answer_fingerprint(sample)
        if answer_counts[fingerprint] >= answer_cap:
            continue
        answer_counts[fingerprint] += 1
        limited.append(sample)
    return limited


def _percentile(lengths: list[int], fraction: float) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths)
    index = max(0, min(len(ordered) - 1, round(len(ordered) * fraction) - 1))
    return ordered[index]


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min": 0, "avg": 0.0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
    return {
        "min": min(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "max": max(lengths),
    }


def _summarize_raw_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    required_fields = (
        "record_id",
        "task_type",
        "source",
        "source_id",
        "city",
        "entity_name",
        "entity_type",
        "user_query",
        "assistant_content",
        "is_time_sensitive",
        "updated_at",
    )
    missing_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    for record in records:
        for field in required_fields:
            if field not in record:
                missing_counts[field] += 1
            elif record[field] in (None, "", []):
                empty_counts[field] += 1

    return {
        "count": len(records),
        "candidate_count_range": [STAGE1_TRAVEL_QA_CANDIDATE_MIN, STAGE1_TRAVEL_QA_CANDIDATE_MAX],
        "meets_candidate_count_range": STAGE1_TRAVEL_QA_CANDIDATE_MIN <= len(records) <= STAGE1_TRAVEL_QA_CANDIDATE_MAX,
        "task_type_counts": dict(Counter(record.get("task_type") for record in records)),
        "source_counts": dict(Counter(record.get("source") for record in records)),
        "field_names": sorted({field for record in records for field in record}),
        "missing_required_counts": dict(missing_counts),
        "empty_required_counts": dict(empty_counts),
        "city_count": len({record.get("city") for record in records if record.get("city")}),
        "top_cities": dict(Counter(record.get("city") for record in records).most_common(20)),
        "question_type_counts": dict(Counter(record.get("question_type") for record in records).most_common(20)),
    }


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    user_lengths: list[int] = []
    assistant_lengths: list[int] = []
    city_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for sample in samples:
        user = _message_content(sample, "user")
        assistant = _message_content(sample, "assistant")
        user_lengths.append(len(user))
        assistant_lengths.append(len(assistant))
        answer_counts[normalize_text(assistant)] += 1
        pair_counts[(normalize_text(user), normalize_text(assistant))] += 1
        city_counts[sample.get("city") or "__unknown__"] += 1

    return {
        "count": len(samples),
        "final_target_count": STAGE1_TRAVEL_QA_FINAL_TARGET,
        "meets_final_target": len(samples) >= STAGE1_TRAVEL_QA_FINAL_TARGET,
        "task_type_counts": dict(Counter(sample.get("task_type") for sample in samples)),
        "source_task_type_counts": dict(Counter(sample.get("source_task_type") for sample in samples)),
        "entity_type_counts": dict(Counter(sample.get("entity_type") for sample in samples)),
        "question_type_counts": dict(Counter(sample.get("question_type") for sample in samples).most_common(20)),
        "time_sensitive_counts": dict(Counter(bool(sample.get("is_time_sensitive")) for sample in samples)),
        "city_count": len([city for city in city_counts if city != "__unknown__"]),
        "top_cities": dict(city_counts.most_common(20)),
        "user_length": _length_summary(user_lengths),
        "assistant_length": _length_summary(assistant_lengths),
        "duplicate_answer_extra": sum(count - 1 for count in answer_counts.values() if count > 1),
        "duplicate_pair_extra": sum(count - 1 for count in pair_counts.values() if count > 1),
    }


def process_travel_qa_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    jsonl_output_path: str | None = None,
    report_path: str | None = None,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    city_cap: int = DEFAULT_CITY_CAP,
    answer_cap: int = DEFAULT_ANSWER_CAP,
) -> list[dict[str, Any]]:
    configure_console_output()
    log_info(f"\u5f00\u59cb\u5904\u7406 travel_qa \u6570\u636e: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"\u672a\u627e\u5230 travel_qa \u539f\u59cb\u6570\u636e\uff0c\u5148\u8df3\u8fc7: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    skipped = 0
    deduped = 0
    filter_reasons: Counter[str] = Counter()
    removed_examples: list[dict[str, Any]] = []
    for record in raw_records:
        sample = build_travel_qa_sample(record)
        if sample is None:
            skipped += 1
            filter_reasons["build_failed"] += 1
            continue
        filter_reason = classify_strict_filter_reason(sample)
        if filter_reason is not None:
            skipped += 1
            filter_reasons[filter_reason] += 1
            if len(removed_examples) < 60:
                removed_examples.append(
                    {
                        "record_id": record.get("record_id"),
                        "reason": filter_reason,
                        "source_task_type": record.get("task_type"),
                        "question_type": record.get("question_type"),
                        "user": _message_content(sample, "user"),
                        "assistant": _message_content(sample, "assistant"),
                    }
                )
            continue
        fingerprint = _fingerprint(sample)
        if fingerprint in seen_fingerprints:
            deduped += 1
            filter_reasons["duplicate_pair"] += 1
            continue
        seen_fingerprints.add(fingerprint)
        processed.append(sample)

    capped = _limit_repeated_answers(processed, answer_cap=answer_cap)
    answer_cap_trimmed = len(processed) - len(capped)
    balanced = _balance_categories(capped, total_samples=total_samples, city_cap=city_cap)
    output_path = write_json(output_json_path, balanced)
    jsonl_output_file = write_jsonl(jsonl_output_path, balanced) if jsonl_output_path else None

    report_file = None
    if report_path:
        city_counts = Counter(sample.get("city") or "__unknown__" for sample in balanced)
        report = {
            "input_path": str(resolve_path(input_file_path)),
            "output_path": str(output_path),
            "jsonl_output_path": str(jsonl_output_file) if jsonl_output_file else None,
            "raw_summary": _summarize_raw_records(raw_records),
            "processed_candidate_summary": _summarize_samples(processed),
            "answer_capped_summary": _summarize_samples(capped),
            "output_summary": _summarize_samples(balanced),
            "target_category_counts": _calculate_category_targets(total_samples) if total_samples > 0 else None,
            "final_target_count": STAGE1_TRAVEL_QA_FINAL_TARGET,
            "requested_total_samples": total_samples,
            "city_cap": city_cap,
            "answer_cap": answer_cap,
            "skipped_count": skipped,
            "deduped_count": deduped,
            "answer_cap_trimmed_count": answer_cap_trimmed,
            "selected_count": len(balanced),
            "filter_reasons": dict(filter_reasons),
            "city_cap_violations": {
                city: count for city, count in city_counts.items() if city_cap > 0 and count > city_cap
            },
            "removed_examples": removed_examples,
        }
        report_file = write_json(report_path, report)

    log_success(
        "\u5904\u7406 travel_qa \u6570\u636e\u5b8c\u6210\u3002"
        f"\u8f93\u51fa {len(balanced)} \u6761\uff0c"
        f"\u8df3\u8fc7 {skipped} \u6761\uff0c"
        f"\u53bb\u91cd {deduped} \u6761\uff0c"
        f"\u539f\u59cb\u5165\u9009 {len(processed)} \u6761\uff0c"
        f"\u76f8\u540c\u7b54\u6848\u9650\u9891\u540e {len(capped)} \u6761\u3002"
    )
    log_info(
        "\u76ee\u6807\u5206\u5e03: "
        f"{_calculate_category_targets(total_samples) if total_samples > 0 else 'keep_all'}; "
        f"\u5355\u57ce\u5e02\u4e0a\u9650: {city_cap if city_cap > 0 else 'unlimited'}; "
        f"\u76f8\u540c\u7b54\u6848\u4e0a\u9650: {answer_cap if answer_cap > 0 else 'unlimited'}"
    )
    log_info(f"\u8fc7\u6ee4\u539f\u56e0: {dict(filter_reasons)}")
    log_info(f"\u8f93\u51fa\u6587\u4ef6: {output_path}")
    if jsonl_output_file:
        log_info(f"JSONL \u8f93\u51fa\u6587\u4ef6: {jsonl_output_file}")
    if report_file:
        log_info(f"\u6e05\u6d17\u62a5\u544a: {report_file}")
    return balanced


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="\u5c06 travel_qa \u539f\u59cb\u6570\u636e\u6e05\u6d17\u4e3a ChatML\u3002")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="travel_qa \u539f\u59cb\u6570\u636e\u8def\u5f84\uff0c\u652f\u6301 JSON/JSONL\u3002")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="travel_qa ChatML JSON \u8f93\u51fa\u8def\u5f84\u3002")
    parser.add_argument("--jsonl-output", default=DEFAULT_JSONL_OUTPUT_PATH, help="travel_qa ChatML JSONL \u8f93\u51fa\u8def\u5f84\u3002")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="travel_qa \u6e05\u6d17\u62a5\u544a\u8f93\u51fa\u8def\u5f84\u3002")
    parser.add_argument("--total-samples", type=int, default=DEFAULT_TOTAL_SAMPLES, help=f"\u76ee\u6807\u6837\u672c\u6570\uff0c\u9ed8\u8ba4 {DEFAULT_TOTAL_SAMPLES}\uff1b\u8bbe\u4e3a 0 \u4fdd\u7559\u5168\u91cf\u3002")
    parser.add_argument("--city-cap", type=int, default=DEFAULT_CITY_CAP, help="\u5355\u57ce\u5e02\u6700\u591a\u4fdd\u7559\u6761\u6570\uff0c\u8bbe\u4e3a 0 \u4e0d\u9650\u5236\u3002")
    parser.add_argument("--answer-cap", type=int, default=DEFAULT_ANSWER_CAP, help="\u76f8\u540c assistant \u7b54\u6848\u6700\u591a\u4fdd\u7559\u6761\u6570\uff0c\u8bbe\u4e3a 0 \u4e0d\u9650\u5236\u3002")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_travel_qa_data(
        args.input,
        args.output,
        jsonl_output_path=args.jsonl_output,
        report_path=args.report,
        total_samples=args.total_samples,
        city_cap=args.city_cap,
        answer_cap=args.answer_cap,
    )


if __name__ == "__main__":
    main()
