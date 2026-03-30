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

DEFAULT_INPUT_PATH = "data/raw/hotel_recommendation_0330.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_hotel_recommendation.json"
DEFAULT_TOTAL_SAMPLES = 750
DEFAULT_CITY_CAP = 60
DEFAULT_QUERY_CAP = 1

_DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "你是专业的中文旅行住宿顾问。请根据用户的人群、预算、位置、出行目的和偏好，"
    "提供清晰、自然的住宿建议，并说明推荐理由。"
    "避免营销口吻，不编造实时价格、库存或预订结果。"
)
DEFAULT_SYSTEM_PROMPT = load_system_prompt(
    "hotel_recommendation", _DEFAULT_SYSTEM_PROMPT_FALLBACK
)

CONTEXT_PREFIX = "参考信息："
USER_QUESTION_PREFIX = "用户问题："

COMMON_LOCATION_SUFFIXES = (
    "市",
    "区",
    "县",
    "镇",
    "乡",
    "村",
    "新区",
    "景区",
    "风景区",
    "度假区",
    "古城",
    "古镇",
    "机场",
    "火车站",
    "高铁站",
    "地铁站",
    "商圈",
    "片区",
    "大道",
    "路",
)
GENERIC_QUERY_PREFIXES = (
    "住哪里",
    "住哪儿",
    "有推荐吗",
    "有吗",
    "方便吗",
    "适合吗",
    "怎么选",
)
GENERIC_QUERY_TOKENS = (
    "附近住哪里",
    "附近有什么推荐",
    "方便中转",
    "方便赶早班机",
    "适合短住",
    "适合带娃",
    "适合情侣",
    "适合商务",
)

PRICE_OR_BOOKING_PATTERN = re.compile(
    r"(今日价格|实时房价|库存|房态|剩余\d+间|立即预订|帮我下单|支付结果|明晚还有没有房|"
    r"优惠价|限时抢|扫码预订|点击预订|下单购买)"
)
PROMO_PATTERN = re.compile(
    r"(官方预订|立即抢购|专享优惠|联系客服|咨询客服|下载APP|打开APP|商城|比价订房)"
)
MARKDOWN_PATTERN = re.compile(r"^\s*#{1,6}\s*", re.M)
MULTI_PUNCT_PATTERN = re.compile(r"[，。；、]{2,}")
TRAILING_NOTE_PATTERN = re.compile(r"(?:预订前|下单前)[^。；]*确认[^。；]*[。；]?$")
EMPTY_NOTE_PATTERN = re.compile(r"^(?:是的|可以的|可以|当然可以)[，。；]*$")

RATIONALE_MARKERS = (
    "适合",
    "更适合",
    "不太适合",
    "不适合",
    "因为",
    "因此",
    "所以",
    "核心",
    "优势",
    "优点",
    "缺点",
    "更方便",
)
CONTRAST_MARKERS = ("但", "不过", "需注意", "要注意", "同时也", "另一方面")
DEMAND_MARKERS = (
    "中转",
    "亲子",
    "情侣",
    "老人",
    "家庭",
    "商务",
    "预算",
    "地铁",
    "公交",
    "机场",
    "高铁",
    "安静",
    "方便",
)
OPERATIONAL_TOKENS = (
    "24小时前台",
    "行李寄存",
    "快速入住",
    "快速退房",
    "免费班车",
    "接送机",
    "停车场",
    "会议室",
)

STYLE_TARGET_RATIO = {
    "transit_stop": 0.22,
    "family_resort": 0.19,
    "business_commute": 0.16,
    "city_explore": 0.14,
    "couple_vacation": 0.12,
    "multi_spot_trip": 0.08,
    "budget_trip": 0.06,
    "senior_relaxed": 0.03,
}
STYLE_PRIORITY = (
    "transit_stop",
    "family_resort",
    "business_commute",
    "city_explore",
    "couple_vacation",
    "multi_spot_trip",
    "budget_trip",
    "senior_relaxed",
)


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


def _name_aliases(name: str) -> set[str]:
    normalized = normalize_text(name).replace(" ", "")
    if not normalized:
        return set()

    aliases = {normalized}
    for suffix in COMMON_LOCATION_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            aliases.add(normalized[: -len(suffix)])
    return {alias for alias in aliases if alias}


def _mentions_context(query: str, city: str, district: str, hotel_name: str) -> bool:
    normalized_query = normalize_text(query).replace(" ", "")
    aliases = _name_aliases(city) | _name_aliases(district) | _name_aliases(hotel_name)
    return any(alias and alias in normalized_query for alias in aliases)


def _needs_context(query: str) -> bool:
    normalized_query = normalize_text(query)
    if len(normalized_query) <= 12:
        return True
    if normalized_query.startswith(GENERIC_QUERY_PREFIXES):
        return True
    return any(token in normalized_query for token in GENERIC_QUERY_TOKENS)


def _build_contextual_user_query(query: str, city: str, district: str) -> str:
    context_parts: list[str] = []
    if city:
        context_parts.append(f"城市：{city}")
    if district:
        context_parts.append(f"区域：{district}")
    if not context_parts:
        return query
    return f"{CONTEXT_PREFIX}{'；'.join(context_parts)}\n\n{USER_QUESTION_PREFIX}{query}"


def _self_contained_user_query(query: str, city: str, district: str, hotel_name: str) -> str:
    if not query:
        return ""
    if _mentions_context(query, city, district, hotel_name) and len(query) >= 10:
        return query
    if not _needs_context(query):
        return query
    return _build_contextual_user_query(query, city, district)


def _clean_hotel_answer(answer: str) -> str:
    cleaned = clean_text(answer, max_length=2200)
    cleaned = MARKDOWN_PATTERN.sub("", cleaned)
    cleaned = TRAILING_NOTE_PATTERN.sub("", cleaned).strip()
    cleaned = MULTI_PUNCT_PATTERN.sub("。", cleaned)
    cleaned = cleaned.replace("适老/亲子", "适老或亲子")
    return cleaned


def _has_hard_violation(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(
        PRICE_OR_BOOKING_PATTERN.search(normalized) or PROMO_PATTERN.search(normalized)
    )


def _has_basic_reasoning(answer: str) -> bool:
    rationale_hits = sum(marker in answer for marker in RATIONALE_MARKERS)
    contrast_hits = sum(marker in answer for marker in CONTRAST_MARKERS)
    return rationale_hits >= 2 and contrast_hits >= 1


def _quality_score(sample: dict[str, Any]) -> tuple[int, int]:
    user_content = sample["messages"][1]["content"]
    answer = sample["messages"][2]["content"]
    score = 0

    answer_length = len(answer)
    if 140 <= answer_length <= 420:
        score += 4
    elif 110 <= answer_length <= 520:
        score += 2

    score += sum(marker in answer for marker in RATIONALE_MARKERS[:6])
    score += min(2, sum(marker in answer for marker in CONTRAST_MARKERS))
    score += min(2, sum(marker in answer for marker in DEMAND_MARKERS if marker in user_content and marker in answer))

    if sample.get("question_mode") == "open_recommendation":
        score += 1
    if sample.get("travel_style") in STYLE_TARGET_RATIO:
        score += 1

    operational_hits = sum(token in answer for token in OPERATIONAL_TOKENS)
    if operational_hits > 2:
        score -= operational_hits - 2

    return score, -answer_length


def build_hotel_recommendation_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    if clean_text(record.get("task_type"), max_length=80, mask_sensitive=False) != "hotel_recommendation":
        return None

    city = clean_text(record.get("city"), max_length=40, mask_sensitive=False)
    district = clean_text(record.get("district"), max_length=40, mask_sensitive=False)
    hotel_name = clean_text(record.get("hotel_name"), max_length=120, mask_sensitive=False)
    question = clean_text(record.get("user_query"), max_length=300)
    raw_answer = record.get("assistant_content")

    if not city or not district or not question or not raw_answer:
        return None

    answer = _clean_hotel_answer(str(raw_answer))
    if not answer or len(answer) < 110:
        return None

    combined_text = "\n".join(
        [
            question,
            answer,
            clean_text(record.get("reason_text"), max_length=300),
        ]
    )
    if _has_hard_violation(combined_text):
        return None
    if EMPTY_NOTE_PATTERN.fullmatch(answer):
        return None
    if not _has_basic_reasoning(answer):
        return None

    user_query = _self_contained_user_query(question, city, district, hotel_name)
    if not user_query:
        return None

    system_prompt = clean_text(
        record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        max_length=1000,
        mask_sensitive=False,
    )
    audience = _clean_list(record.get("audience"), max_items=5, item_length=30)
    hotel_tags = _clean_list(record.get("hotel_tags"), max_items=6, item_length=30)
    suitable_for = _clean_list(record.get("suitable_for"), max_items=6, item_length=40)
    not_suitable_for = _clean_list(record.get("not_suitable_for"), max_items=6, item_length=40)

    sample_id = "hotel_recommendation_" + hashlib.md5(
        f"{city}|{district}|{user_query}|{answer}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": sample_id,
        "task_type": "hotel_recommendation",
        "scene": "hotel_recommendation",
        "source": "tripai_hotel_recommendation_0330",
        "source_id": clean_text(record.get("source_id"), max_length=80, mask_sensitive=False),
        "city": city,
        "district": district,
        "hotel_name": hotel_name,
        "budget_level": clean_text(record.get("budget_level"), max_length=30, mask_sensitive=False),
        "audience": audience,
        "hotel_tags": hotel_tags,
        "suitable_for": suitable_for,
        "not_suitable_for": not_suitable_for,
        "travel_style": clean_text(record.get("travel_style"), max_length=40, mask_sensitive=False),
        "query_intent": clean_text(record.get("query_intent"), max_length=40, mask_sensitive=False),
        "question_mode": clean_text(record.get("question_mode"), max_length=40, mask_sensitive=False),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer},
        ],
    }


def _fingerprint(sample: dict[str, Any]) -> str:
    messages = sample["messages"]
    return hashlib.md5(
        f"{messages[1]['content']}\n###\n{messages[2]['content']}".encode("utf-8")
    ).hexdigest()


def _calculate_style_targets(total_samples: int) -> dict[str, int]:
    targets = {
        style: round(total_samples * ratio)
        for style, ratio in STYLE_TARGET_RATIO.items()
    }
    diff = total_samples - sum(targets.values())
    priority = list(STYLE_PRIORITY)
    index = 0
    while diff > 0:
        targets[priority[index % len(priority)]] += 1
        diff -= 1
        index += 1
    return targets


def _select_balanced_subset(
    samples: list[dict[str, Any]],
    *,
    target_count: int,
    city_cap: int,
) -> list[dict[str, Any]]:
    if target_count <= 0 or len(samples) <= target_count:
        return list(samples)

    city_buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for sample in sorted(samples, key=_quality_score, reverse=True):
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


def _balance_styles(
    processed: list[dict[str, Any]],
    *,
    total_samples: int,
    city_cap: int,
) -> list[dict[str, Any]]:
    if total_samples <= 0 or len(processed) <= total_samples:
        return sorted(processed, key=_quality_score, reverse=True)

    targets = _calculate_style_targets(total_samples)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in processed:
        grouped[sample.get("travel_style") or "__unknown__"].append(sample)

    balanced: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for style in STYLE_PRIORITY:
        bucket = grouped.get(style, [])
        picked = _select_balanced_subset(bucket, target_count=targets.get(style, 0), city_cap=city_cap)
        for sample in picked:
            if sample["id"] in selected_ids:
                continue
            balanced.append(sample)
            selected_ids.add(sample["id"])

    leftovers = [
        sample
        for sample in sorted(processed, key=_quality_score, reverse=True)
        if sample["id"] not in selected_ids
    ]
    city_counts = Counter(sample.get("city") or "__unknown__" for sample in balanced)
    for sample in leftovers:
        if len(balanced) >= total_samples:
            break
        city_key = sample.get("city") or "__unknown__"
        if city_cap > 0 and city_counts[city_key] >= city_cap:
            continue
        balanced.append(sample)
        selected_ids.add(sample["id"])
        city_counts[city_key] += 1

    if len(balanced) < total_samples:
        for sample in leftovers:
            if len(balanced) >= total_samples:
                break
            if sample["id"] in selected_ids:
                continue
            balanced.append(sample)
            selected_ids.add(sample["id"])
    return balanced


def process_hotel_recommendation_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    city_cap: int = DEFAULT_CITY_CAP,
    query_cap: int = DEFAULT_QUERY_CAP,
) -> list[dict[str, Any]]:
    configure_console_output()
    log_info(f"开始处理 hotel_recommendation 数据: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"未找到 hotel_recommendation 原始数据，先跳过: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    skipped = 0
    deduped = 0
    for record in raw_records:
        sample = build_hotel_recommendation_sample(record)
        if sample is None:
            skipped += 1
            continue
        fingerprint = _fingerprint(sample)
        if fingerprint in seen_fingerprints:
            deduped += 1
            continue
        seen_fingerprints.add(fingerprint)
        processed.append(sample)

    ranked = sorted(processed, key=_quality_score, reverse=True)
    query_filtered: list[dict[str, Any]] = []
    query_counts: Counter[str] = Counter()
    query_trimmed = 0
    for sample in ranked:
        query_key = sample["messages"][1]["content"]
        if query_cap > 0 and query_counts[query_key] >= query_cap:
            query_trimmed += 1
            continue
        query_counts[query_key] += 1
        query_filtered.append(sample)

    balanced = _balance_styles(query_filtered, total_samples=total_samples, city_cap=city_cap)
    output_path = write_json(output_json_path, balanced)

    style_counts = Counter(sample.get("travel_style") or "__unknown__" for sample in balanced)
    city_counts = Counter(sample.get("city") or "__unknown__" for sample in balanced)
    log_success(
        "处理 hotel_recommendation 数据完成。"
        f"输出 {len(balanced)} 条，"
        f"跳过 {skipped} 条，"
        f"去重 {deduped} 条，"
        f"同问裁剪 {query_trimmed} 条，"
        f"原始入选 {len(processed)} 条。"
    )
    log_info(
        f"目标样本数: {total_samples if total_samples > 0 else 'keep_all'}; "
        f"单城市上限: {city_cap if city_cap > 0 else 'unlimited'}; "
        f"同问上限: {query_cap if query_cap > 0 else 'unlimited'}"
    )
    log_info(f"travel_style 分布: {dict(style_counts)}")
    log_info(f"城市 Top 10: {city_counts.most_common(10)}")
    log_info(f"输出文件: {output_path}")
    return balanced


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 hotel_recommendation 原始数据清洗为高质量 ChatML。"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="hotel_recommendation 原始数据路径，支持 JSON/JSONL。",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="hotel_recommendation ChatML JSON 输出路径。",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=DEFAULT_TOTAL_SAMPLES,
        help="目标样本数，默认 750；设为 0 保留全量。",
    )
    parser.add_argument(
        "--city-cap",
        type=int,
        default=DEFAULT_CITY_CAP,
        help="单城市最多保留条数，设为 0 不限制。",
    )
    parser.add_argument(
        "--query-cap",
        type=int,
        default=DEFAULT_QUERY_CAP,
        help="同一用户问法最多保留条数，设为 0 不限制。",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_hotel_recommendation_data(
        args.input,
        args.output,
        total_samples=args.total_samples,
        city_cap=args.city_cap,
        query_cap=args.query_cap,
    )


if __name__ == "__main__":
    main()
