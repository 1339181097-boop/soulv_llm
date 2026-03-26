from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
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

DEFAULT_INPUT_PATH = "data/raw/traffic_planning_raw_2026_03_25.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_traffic_planning.json"
STRICT_OUTPUT_PATH = "data/processed/sft_traffic_planning_strict.json"

_DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "你是专业的中文旅行交通规划助手。请根据用户的出发地、目的地和出行约束，"
    "提供清晰、可执行的交通建议，并简要说明推荐理由。"
    "避免营销口吻，不编造实时票价、班次或余票信息。"
)

DEFAULT_SYSTEM_PROMPT = load_system_prompt("traffic_planning", _DEFAULT_SYSTEM_PROMPT_FALLBACK)

CONTEXT_PREFIX = "参考信息："
USER_QUESTION_PREFIX = "用户问题："
TRANSFER_CONFIRMATION = "出发前建议再确认线路、站点、首末班或发车安排，并预留换乘时间。"
CAR_CONFIRMATION = "出发前留意路况、上下客点或停车安排会更稳妥。"

MODE_PRIORITY = (
    "地铁",
    "高铁",
    "火车",
    "机场大巴",
    "公交车",
    "网约车",
    "出租车",
    "自驾",
    "步行",
)

PUBLIC_TRANSPORT_MODES = {"地铁", "高铁", "火车", "机场大巴", "公交车"}
CAR_MODES = {"网约车", "出租车", "自驾"}
TRANSPORT_TOKENS = (
    "地铁",
    "公交",
    "公交车",
    "高铁",
    "火车",
    "城际",
    "机场大巴",
    "大巴",
    "步行",
    "打车",
    "网约车",
    "出租车",
    "自驾",
    "驾车",
    "换乘",
    "直达",
    "接驳",
    "乘坐",
    "坐到",
    "下车",
    "出站",
)
PUBLIC_QUERY_KEYWORDS = ("公共交通", "地铁", "公交", "高铁", "火车", "机场大巴")
NO_DRIVE_KEYWORDS = ("不自己开车", "不自驾", "不考虑自驾", "不想开车")
RECOMMEND_PATTERNS = (
    ("地铁", ("推荐选择地铁", "建议选择地铁", "优先选择地铁", "地铁更方便", "搭乘地铁更便利", "推荐乘坐地铁")),
    ("自驾", ("推荐选择驾车", "推荐选择驾车方式", "推荐自驾", "自驾更方便", "驾车更方便", "更建议自驾", "自驾相对灵活")),
    ("网约车", ("推荐打车", "建议打车", "网约车更方便", "出租车更方便", "门到门更省心")),
    ("机场大巴", ("推荐选择机场大巴", "机场大巴更方便", "机场大巴方式")),
    ("公交车", ("推荐选择公交", "建议公交", "公交更方便")),
    ("高铁", ("先乘坐高铁", "高铁转", "高铁+")),
)
SPECIAL_AUDIENCE_KEYWORDS = ("老人", "小孩", "孩子", "亲子", "带娃", "家庭", "长辈")
BAGGAGE_KEYWORDS = ("行李", "行李箱", "拖着", "大箱子", "背着箱")
EASY_PRIORITY_KEYWORDS = ("方便", "省心", "轻松", "少走弯路", "少换乘", "不绕远路")
TIME_PRIORITY_KEYWORDS = ("最快", "节省时间", "省时间", "赶时间", "更快")

MARKDOWN_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s*", re.M)
EMPHASIS_PATTERN = re.compile(r"\*\*([^*]+)\*\*")
LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+)")
EMPTY_HEADING_PATTERN = re.compile(
    r"^(?:方案[一二三四五六七八九十]|第[一二三四五六七八九十]+步|交通方式及(?:路线|分析)|注意事项|中间注意事项)\s*[:：]?$"
)
VOLATILE_CLAUSE_PATTERN = re.compile(r"(票价|余票|库存|二维码|扫码|优惠|预订|商城|班次时间|实时交通|实时路况)")
DISTANCE_PATTERN = re.compile(r"(?:全程|驾车距离|距离)?约?\s*\d+(?:\.\d+)?\s*(?:公里|km)")
TIME_PATTERN = re.compile(r"(?:预计|大概|大约|约|近)?\s*\d+(?:\.\d+)?\s*(?:分钟|分|小时)(?:左右|上下)?")
RANGE_TIME_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*小时")
STOP_COUNT_PATTERN = re.compile(r"(?:乘坐|坐|经)\s*约?\d+\s*(?:站|个站)")
EXIT_PATTERN = re.compile(r"[A-Z]口(?:或[B-Z]口)?")
SOFT_DURATION_PATTERN = re.compile(r"(?:耗时|时长)[^，。；;]*")

STRICT_MIN_ANSWER_LENGTH = 90
STRICT_SHORT_TEMPLATE_LENGTH = 120
TIME_DURATION_QUERY_KEYWORDS = (
    "多久",
    "多长时间",
    "耗时",
    "要多久",
    "需要多久",
    "多久能到",
    "多久到",
    "大概多久",
    "需要多长时间",
)
TIME_COMPARISON_QUERY_KEYWORDS = (
    "更节省时间",
    "节省时间",
    "更快",
    "最快",
    "时间更短",
    "时间更少",
    "更省时间",
)
TIME_DURATION_ANSWER_MARKERS = (
    "路程不算远",
    "路程适中",
    "相对较远",
    "半小时",
    "一小时",
    "两小时",
    "通常要",
    "一般要",
    "建议预留",
    "整体路程",
)
TIME_COMPARISON_ANSWER_MARKERS = (
    "节省时间",
    "更快",
    "最快",
    "一路直达",
    "更省时间",
    "通常会更合适",
)
ROUTE_DETAIL_MARKERS = (
    "号线",
    "地铁",
    "公交",
    "高铁",
    "火车",
    "机场大巴",
    "轮渡",
    "码头",
    "班车",
    "客运",
    "汽车站",
    "换乘",
    "乘坐",
    "下车",
    "出站",
    "步行",
    "接驳",
)
STRICT_BOILERPLATE_SENTENCES = (
    "更建议优先走地铁或城际轨道，路线通常更稳，也更容易控制换乘节奏。",
    "从机场进城时，先走机场大巴或轨道交通会更顺。",
    "如果不赶时间，可以优先考虑公交接驳，整体更省预算。",
    "如果更看重少换乘和门到门体验，直接打车通常会更省心。",
    "如果更看重灵活性和一路直达，自驾会更合适。",
    "如果更看重节省时间和一路直达，自驾通常会更合适。",
    "跨城段更建议先用高铁衔接，到站后再转市内交通。",
    "如果更看重门到门和少换乘，也可以直接打车。",
    "如果更看重稳定性，也可以改走地铁或公交接驳。",
    TRANSFER_CONFIRMATION,
    CAR_CONFIRMATION,
)


def _has_promo_noise(text: str) -> bool:
    normalized = normalize_text(text)
    lowered = normalized.lower()
    if any(pattern in normalized for pattern in ("??", "????", "???", "??", "???")):
        return True
    if "tripai" in lowered:
        return True
    if "??" in normalized and any(keyword in normalized for keyword in ("??", "??", "??", "??", "??", "??", "??")):
        return True
    if "app" in lowered and any(keyword in normalized for keyword in ("??", "??", "??", "??")):
        return True
    return False


def _name_aliases(name: str) -> set[str]:
    normalized = normalize_text(name).replace(" ", "")
    if not normalized:
        return set()

    aliases = {normalized}
    suffixes = ("市", "区", "站", "机场", "景区", "公园", "古镇", "博物馆", "步行街")
    for suffix in suffixes:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            aliases.add(normalized[: -len(suffix)])
    return {alias for alias in aliases if alias}


def _mentions_context(query: str, city: str, origin: str, destination: str) -> bool:
    normalized_query = normalize_text(query).replace(" ", "")
    aliases = _name_aliases(city) | _name_aliases(origin) | _name_aliases(destination)
    return any(alias and alias in normalized_query for alias in aliases)


def _build_contextual_user_query(query: str, city: str, origin: str, destination: str) -> str:
    context_parts: list[str] = []
    if city:
        context_parts.append(f"城市：{city}")
    if origin:
        context_parts.append(f"出发地：{origin}")
    if destination:
        context_parts.append(f"目的地：{destination}")
    if not context_parts:
        return query
    return f"{CONTEXT_PREFIX}{'；'.join(context_parts)}\n\n{USER_QUESTION_PREFIX}{query}"


def _self_contained_user_query(query: str, city: str, origin: str, destination: str) -> str:
    if not query:
        return ""
    if _mentions_context(query, city, origin, destination) and len(query) >= 10:
        return query
    return _build_contextual_user_query(query, city, origin, destination)


def _normalize_modes(raw_modes: Any) -> list[str]:
    if not isinstance(raw_modes, list):
        return []

    normalized: list[str] = []
    for item in raw_modes:
        mode = clean_text(item, max_length=20, mask_sensitive=False)
        if not mode:
            continue
        if mode == "出租车":
            mode = "网约车"
        if mode == "火车":
            mode = "高铁"
        normalized.append(mode)

    seen: set[str] = set()
    ordered: list[str] = []
    for mode in normalized:
        if mode in seen:
            continue
        seen.add(mode)
        ordered.append(mode)
    return ordered


def _strip_markdown(text: str) -> str:
    normalized = normalize_text(text)
    normalized = MARKDOWN_HEADING_PATTERN.sub("", normalized)
    normalized = EMPHASIS_PATTERN.sub(r"\1", normalized)
    cleaned_lines: list[str] = []
    for line in normalized.splitlines():
        stripped = LIST_MARKER_PATTERN.sub("", line).strip()
        if stripped:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def _simplify_sentence(sentence: str) -> str:
    cleaned = normalize_text(sentence)
    if not cleaned:
        return ""

    cleaned = DISTANCE_PATTERN.sub("", cleaned)
    cleaned = RANGE_TIME_PATTERN.sub("", cleaned)
    cleaned = TIME_PATTERN.sub("", cleaned)
    cleaned = STOP_COUNT_PATTERN.sub("", cleaned)
    cleaned = EXIT_PATTERN.sub("出站口", cleaned)
    cleaned = SOFT_DURATION_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"按照导航(?:路线|指引)?", "", cleaned)
    cleaned = re.sub(r"可通过导航软件规划路线", "", cleaned)
    cleaned = re.sub(r"直接通过导航软件输入[^，。；;]*", "", cleaned)
    cleaned = re.sub(r"直接打开打车软件[^，。；;]*", "直接打车前往", cleaned)
    cleaned = re.sub(r"可先查找", "可先", cleaned)
    cleaned = re.sub(r"先找到", "先到", cleaned)
    cleaned = re.sub(r"[^，。；;]*?附近能到达地铁的公交站点", "附近公交站", cleaned)
    cleaned = re.sub(r"[（(][^)）]*?(?:方向|外环|内环)[)）]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[，,；;]{2,}", "，", cleaned)
    cleaned = cleaned.strip(" ，,；;：:")
    return cleaned


def _extract_route_sentences(answer: str, query: str, primary_mode: str | None) -> list[str]:
    cleaned_answer = _strip_markdown(answer)
    raw_segments = re.split(r"[\n。！？；;]+", cleaned_answer)
    normalized_query = normalize_text(query)
    prefer_public = primary_mode in PUBLIC_TRANSPORT_MODES or any(keyword in normalized_query for keyword in PUBLIC_QUERY_KEYWORDS)
    avoid_drive = any(keyword in normalized_query for keyword in NO_DRIVE_KEYWORDS) or "公共交通" in normalized_query

    result: list[str] = []
    seen: set[str] = set()
    for raw_segment in raw_segments:
        segment = _simplify_sentence(raw_segment)
        if not segment:
            continue
        if EMPTY_HEADING_PATTERN.fullmatch(segment):
            continue
        if VOLATILE_CLAUSE_PATTERN.search(segment):
            continue
        if any(marker in segment for marker in ("预计", "相比公共交通", "更节省时间", "成本较低")):
            continue
        if len(segment) < 8:
            continue
        is_car_segment = any(token in segment for token in ("打车", "网约车", "出租车", "自驾", "驾车", "导航"))
        is_public_segment = any(token in segment for token in ("地铁", "公交", "高铁", "火车", "机场大巴", "换乘", "接驳", "乘坐"))
        if avoid_drive and any(token in segment for token in ("自驾", "驾车", "导航")):
            continue
        if prefer_public and is_car_segment and not is_public_segment:
            continue
        if not any(token in segment for token in TRANSPORT_TOKENS):
            continue
        if segment in seen:
            continue
        seen.add(segment)
        result.append(segment)
        if len(result) >= 3:
            break
    return result


def _infer_primary_mode(record: dict[str, Any], answer: str, modes: list[str]) -> str | None:
    normalized_answer = normalize_text(answer)
    for mode, phrases in RECOMMEND_PATTERNS:
        if any(phrase in normalized_answer for phrase in phrases):
            return "网约车" if mode == "出租车" else mode

    query = normalize_text(record.get("user_query"))
    has_special_need = any(keyword in query for keyword in SPECIAL_AUDIENCE_KEYWORDS) or any(
        keyword in query for keyword in BAGGAGE_KEYWORDS
    )
    if has_special_need and any(mode in modes for mode in CAR_MODES):
        return "网约车"

    if any(keyword in query for keyword in NO_DRIVE_KEYWORDS) or "公共交通" in query:
        for mode in MODE_PRIORITY:
            mapped_mode = "网约车" if mode == "出租车" else mode
            if mapped_mode in modes and mapped_mode in PUBLIC_TRANSPORT_MODES:
                return mapped_mode

    for mode in MODE_PRIORITY:
        mapped_mode = "网约车" if mode == "出租车" else mode
        if mapped_mode in modes:
            return mapped_mode
    return None


def _build_opening(primary_mode: str | None, query: str, scenario: str) -> str:
    normalized_query = normalize_text(query)
    time_first = any(keyword in normalized_query for keyword in TIME_PRIORITY_KEYWORDS)
    if primary_mode == "地铁":
        return "更建议优先走地铁或城际轨道，路线通常更稳，也更容易控制换乘节奏。"
    if primary_mode == "机场大巴":
        return "从机场进城时，先走机场大巴或轨道交通会更顺。"
    if primary_mode == "公交车":
        return "如果不赶时间，可以优先考虑公交接驳，整体更省预算。"
    if primary_mode == "网约车":
        return "如果更看重少换乘和门到门体验，直接打车通常会更省心。"
    if primary_mode == "自驾":
        if time_first:
            return "如果更看重节省时间和一路直达，自驾通常会更合适。"
        return "如果更看重灵活性和一路直达，自驾会更合适。"
    if primary_mode == "高铁":
        return "跨城段更建议先用高铁衔接，到站后再转市内交通。"

    if scenario == "airport_to_city":
        return "这类机场到市区的出行，通常优先选轨道交通或机场大巴会更稳。"
    if scenario == "train_to_city":
        return "这类火车站到市区的出行，通常优先接入地铁会更省心。"
    if any(keyword in normalized_query for keyword in EASY_PRIORITY_KEYWORDS):
        return "更建议优先选择换乘更少、路径更直接的方式。"
    return "建议优先选择更稳定、换乘更少的交通方式。"


def _build_alternative_sentence(query: str, primary_mode: str | None, modes: list[str]) -> str:
    normalized_query = normalize_text(query)
    has_special_need = any(keyword in normalized_query for keyword in SPECIAL_AUDIENCE_KEYWORDS) or any(
        keyword in normalized_query for keyword in BAGGAGE_KEYWORDS
    )
    has_car = any(mode in modes for mode in CAR_MODES)
    has_public = any(mode in modes for mode in PUBLIC_TRANSPORT_MODES)

    if has_special_need and has_car and primary_mode not in CAR_MODES:
        return "如果同行有老人、小孩或行李较多，也可以直接打车，能少换乘一些。"
    if primary_mode in PUBLIC_TRANSPORT_MODES and has_car:
        return "如果更看重门到门和少换乘，也可以直接打车。"
    if primary_mode in CAR_MODES and has_public:
        return "如果更看重稳定性，也可以改走地铁或公交接驳。"
    return ""


def _build_confirmation_sentence(primary_mode: str | None, modes: list[str], route_summary: str) -> str:
    route_mentions_public = any(token in route_summary for token in ("地铁", "公交", "高铁", "火车", "机场大巴", "换乘", "接驳"))
    has_public = primary_mode in PUBLIC_TRANSPORT_MODES or route_mentions_public or any(mode in modes for mode in PUBLIC_TRANSPORT_MODES)
    if has_public:
        return TRANSFER_CONFIRMATION
    return CAR_CONFIRMATION


def _build_avoid_sentence(avoid_text: str) -> str:
    cleaned = normalize_text(avoid_text)
    if not cleaned or "无特殊" in cleaned:
        return ""
    if cleaned.endswith(("。", "！", "？")):
        return cleaned
    return cleaned + "。"


def _join_sentences(sentences: list[str]) -> str:
    cleaned = [normalize_text(sentence).rstrip("。；;，, ") for sentence in sentences if normalize_text(sentence)]
    if not cleaned:
        return ""
    return "；".join(cleaned) + "。"


def _build_assistant_answer(record: dict[str, Any]) -> str:
    raw_answer = clean_text(record.get("assistant_content"), max_length=5000, mask_sensitive=False)
    avoid_text = clean_text(record.get("avoid_text"), max_length=200, mask_sensitive=False)
    query = clean_text(record.get("user_query"), max_length=1000, mask_sensitive=False)
    scenario = clean_text(record.get("scenario"), max_length=80, mask_sensitive=False)
    modes = _normalize_modes(record.get("transport_modes"))

    primary_mode = _infer_primary_mode(record, raw_answer, modes)
    opening = _build_opening(primary_mode, query, scenario)
    route_sentences = _extract_route_sentences(raw_answer, query, primary_mode)
    route_summary = _join_sentences(route_sentences[:2])
    alternative = _build_alternative_sentence(query, primary_mode, modes)
    confirmation = _build_confirmation_sentence(primary_mode, modes, route_summary)
    avoid_sentence = _build_avoid_sentence(avoid_text)

    parts = [opening]
    if route_summary:
        parts.append(route_summary)
    if alternative:
        parts.append(alternative)
    parts.append(confirmation)
    if avoid_sentence:
        parts.append(avoid_sentence)

    answer = " ".join(part.strip() for part in parts if part.strip())
    return clean_text(answer, max_length=700, mask_sensitive=False)


def _assistant_content(sample: dict[str, Any]) -> str:
    return normalize_text(sample["messages"][2]["content"])


def _user_content(sample: dict[str, Any]) -> str:
    return normalize_text(sample["messages"][1]["content"])


def _looks_like_duration_question(query: str) -> bool:
    return any(keyword in query for keyword in TIME_DURATION_QUERY_KEYWORDS)


def _looks_like_time_comparison_question(query: str) -> bool:
    return any(keyword in query for keyword in TIME_COMPARISON_QUERY_KEYWORDS)


def _answers_time_core(query: str, answer: str) -> bool:
    if _looks_like_duration_question(query):
        if any(marker in answer for marker in TIME_DURATION_ANSWER_MARKERS):
            return True
        return False
    if _looks_like_time_comparison_question(query):
        if any(marker in answer for marker in TIME_COMPARISON_ANSWER_MARKERS):
            return True
        return False
    return True


def _route_detail_hits(answer: str) -> int:
    return sum(marker in answer for marker in ROUTE_DETAIL_MARKERS)


def _has_executable_route(answer: str) -> bool:
    if _route_detail_hits(answer) >= 2:
        return True
    if "从" in answer and any(marker in answer for marker in ("乘坐", "换乘", "下车", "出站", "步行")):
        return True
    return False


def _boilerplate_hits(answer: str) -> int:
    return sum(sentence in answer for sentence in STRICT_BOILERPLATE_SENTENCES)


def _classify_strict_filter_reason(sample: dict[str, Any]) -> str | None:
    query = _user_content(sample)
    answer = _assistant_content(sample)

    if not _answers_time_core(query, answer):
        return "time_core_not_answered"
    if not _has_executable_route(answer):
        return "non_executable"
    if len(answer) < STRICT_MIN_ANSWER_LENGTH:
        return "short_template"
    if len(answer) < STRICT_SHORT_TEMPLATE_LENGTH and _boilerplate_hits(answer) >= 2:
        return "short_template"
    return None


def filter_traffic_planning_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    filtered: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()

    for sample in samples:
        reason = _classify_strict_filter_reason(sample)
        if reason is not None:
            skip_reasons[reason] += 1
            continue
        filtered.append(sample)

    return filtered, skip_reasons


def _fingerprint(user_query: str, assistant_answer: str) -> str:
    payload = f"{user_query}\n###\n{assistant_answer}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _build_sample(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if clean_text(record.get("task_type"), max_length=80, mask_sensitive=False) != "traffic_planning":
        return None, "wrong_task_type"

    raw_answer = clean_text(record.get("assistant_content"), max_length=5000, mask_sensitive=False)
    if _has_promo_noise(raw_answer):
        return None, "promo"

    city = clean_text(record.get("city"), max_length=100, mask_sensitive=False)
    origin = clean_text(record.get("origin"), max_length=200, mask_sensitive=False)
    destination = clean_text(record.get("destination"), max_length=200, mask_sensitive=False)
    user_query = clean_text(record.get("user_query"), max_length=1200, mask_sensitive=False)
    if not user_query or not raw_answer:
        return None, "empty_core_field"

    user_query = _self_contained_user_query(user_query, city, origin, destination)
    assistant_answer = _build_assistant_answer(record)
    if not assistant_answer:
        return None, "empty_answer"
    if not any(token in assistant_answer for token in TRANSPORT_TOKENS):
        return None, "transport_signal_missing"

    sample_id = "traffic_planning_" + _fingerprint(user_query, assistant_answer)[:12]
    transport_modes = _normalize_modes(record.get("transport_modes"))
    suitable_for = [
        clean_text(item, max_length=50, mask_sensitive=False)
        for item in record.get("suitable_for", [])
        if clean_text(item, max_length=50, mask_sensitive=False)
    ]

    sample: dict[str, Any] = {
        "id": sample_id,
        "record_id": clean_text(record.get("record_id"), max_length=80, mask_sensitive=False),
        "task_type": "traffic_planning",
        "scene": clean_text(record.get("scenario"), max_length=80, mask_sensitive=False) or "traffic_planning",
        "source": "tripai_traffic_planning_raw_2026_03_25",
        "source_id": clean_text(record.get("source_id"), max_length=80, mask_sensitive=False),
        "city": city,
        "origin": origin,
        "destination": destination,
        "transport_modes": transport_modes,
        "suitable_for": suitable_for,
        "updated_at": clean_text(record.get("updated_at"), max_length=40, mask_sensitive=False),
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_answer},
        ],
    }
    return sample, None


def build_traffic_planning_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    sample, _ = _build_sample(record)
    return sample


def process_traffic_planning_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    strict_output_json_path: str | None = None,
) -> list[dict[str, Any]]:
    configure_console_output()
    log_info(f"开始清洗 traffic_planning 原始数据: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_error(f"未找到 traffic_planning 原始数据: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    skip_reasons: Counter[str] = Counter()

    for record in raw_records:
        sample, skip_reason = _build_sample(record)
        if sample is None:
            skip_reasons[skip_reason or "unknown"] += 1
            continue

        record_id = sample.get("record_id")
        if record_id and record_id in seen_record_ids:
            skip_reasons["duplicate_record_id"] += 1
            continue

        fingerprint = _fingerprint(sample["messages"][1]["content"], sample["messages"][2]["content"])
        if fingerprint in seen_fingerprints:
            skip_reasons["duplicate_content"] += 1
            continue

        if record_id:
            seen_record_ids.add(record_id)
        seen_fingerprints.add(fingerprint)
        processed.append(sample)

    output_path = write_json(output_json_path, processed)
    log_success(f"traffic_planning 清洗完成，输出 {len(processed)} 条样本。")
    if skip_reasons:
        log_warn(f"跳过原因统计: {dict(skip_reasons)}")
    log_info(f"输出文件: {output_path}")

    if strict_output_json_path:
        filtered, strict_skip_reasons = filter_traffic_planning_samples(processed)
        strict_output_path = write_json(strict_output_json_path, filtered)
        log_success(f"traffic_planning 严格筛洗完成，输出 {len(filtered)} 条样本。")
        if strict_skip_reasons:
            log_warn(f"严格筛洗跳过原因统计: {dict(strict_skip_reasons)}")
        log_info(f"严格版输出文件: {strict_output_path}")

    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 traffic_planning 原始数据清洗为 ChatML。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="traffic_planning 原始数据路径，支持 JSON/JSONL。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="traffic_planning ChatML JSON 输出路径。")
    parser.add_argument(
        "--strict-output",
        default="",
        help="可选：输出严格筛洗后的 traffic_planning ChatML JSON 路径。",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_traffic_planning_data(
        args.input,
        args.output,
        strict_output_json_path=args.strict_output or None,
    )


if __name__ == "__main__":
    main()
