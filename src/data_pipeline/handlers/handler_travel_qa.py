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
)
from src.data_pipeline.global_cleaner import clean_text, normalize_text
from src.data_pipeline.system_prompt_loader import load_system_prompt

DEFAULT_INPUT_PATH = "data/raw/travel_qa_raw_3_23.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_travel_qa.json"
DEFAULT_TOTAL_SAMPLES = 1250
DEFAULT_CITY_CAP = 80
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


def _simplify_traffic_answer(answer: str) -> str:
    normalized = normalize_text(answer)
    minutes = _extract_first_number(DURATION_PATTERN, normalized)
    distance_km = _extract_first_number(DISTANCE_PATTERN, normalized)
    mode_summary = _traffic_mode_summary(normalized)
    trip_summary = _classify_trip_length(minutes, distance_km)
    return (
        f"{mode_summary}\uff0c{trip_summary}\u3002"
        "\u5982\u679c\u66f4\u770b\u91cd\u7701\u5fc3\u6216\u540c\u884c\u4eba\u6570\u8f83\u591a\uff0c"
        "\u4e5f\u53ef\u4ee5\u76f4\u63a5\u6253\u8f66\u3002"
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

    if task_type == "traffic_qa" or entity_type == "traffic":
        return _simplify_traffic_answer(answer)
    if is_time_sensitive:
        return _soften_time_sensitive_answer(answer)
    return answer


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
    assistant_answer = clean_text(_clean_answer(record, raw_answer), max_length=2500)
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
        "task_type": "travel_qa",
        "scene": "travel_qa",
        "source": "tripai_travel_qa_raw_3_23",
        "source_task_type": clean_text(record.get("task_type"), max_length=100, mask_sensitive=False),
        "city": city,
        "entity_name": entity_name,
        "entity_type": clean_text(record.get("entity_type"), max_length=50, mask_sensitive=False),
        "question_type": clean_text(record.get("question_type"), max_length=50, mask_sensitive=False),
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


def process_travel_qa_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    city_cap: int = DEFAULT_CITY_CAP,
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
    for record in raw_records:
        sample = build_travel_qa_sample(record)
        if sample is None:
            skipped += 1
            continue
        fingerprint = _fingerprint(sample)
        if fingerprint in seen_fingerprints:
            deduped += 1
            continue
        seen_fingerprints.add(fingerprint)
        processed.append(sample)

    balanced = _balance_categories(processed, total_samples=total_samples, city_cap=city_cap)
    output_path = write_json(output_json_path, balanced)
    log_success(
        "\u5904\u7406 travel_qa \u6570\u636e\u5b8c\u6210\u3002"
        f"\u8f93\u51fa {len(balanced)} \u6761\uff0c"
        f"\u8df3\u8fc7 {skipped} \u6761\uff0c"
        f"\u53bb\u91cd {deduped} \u6761\uff0c"
        f"\u539f\u59cb\u5165\u9009 {len(processed)} \u6761\u3002"
    )
    log_info(
        "\u76ee\u6807\u5206\u5e03: "
        f"{_calculate_category_targets(total_samples) if total_samples > 0 else 'keep_all'}; "
        f"\u5355\u57ce\u5e02\u4e0a\u9650: {city_cap if city_cap > 0 else 'unlimited'}"
    )
    log_info(f"\u8f93\u51fa\u6587\u4ef6: {output_path}")
    return balanced


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="\u5c06 travel_qa \u539f\u59cb\u6570\u636e\u6e05\u6d17\u4e3a ChatML\u3002")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="travel_qa \u539f\u59cb\u6570\u636e\u8def\u5f84\uff0c\u652f\u6301 JSON/JSONL\u3002")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="travel_qa ChatML JSON \u8f93\u51fa\u8def\u5f84\u3002")
    parser.add_argument("--total-samples", type=int, default=DEFAULT_TOTAL_SAMPLES, help="\u76ee\u6807\u6837\u672c\u6570\uff0c\u9ed8\u8ba4 1250\uff1b\u8bbe\u4e3a 0 \u4fdd\u7559\u5168\u91cf\u3002")
    parser.add_argument("--city-cap", type=int, default=DEFAULT_CITY_CAP, help="\u5355\u57ce\u5e02\u6700\u591a\u4fdd\u7559\u6761\u6570\uff0c\u8bbe\u4e3a 0 \u4e0d\u9650\u5236\u3002")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_travel_qa_data(
        args.input,
        args.output,
        total_samples=args.total_samples,
        city_cap=args.city_cap,
    )


if __name__ == "__main__":
    main()
