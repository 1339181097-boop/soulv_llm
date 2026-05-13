from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.clean_stage2_single_tool_call import (
    _answer_is_clean,
    _conversation_value,
    _geocode_answer,
    _json_dumps,
    _loads_object,
    _normalize_text,
    _route_answer,
    _to_source_dataset,
    _write_json,
    _write_jsonl,
)
from src.data_pipeline.clean_stage2_slot_filling_tool_call import _poi_answer
from src.data_pipeline.data_utils import configure_console_output
from src.tool_use.datasets import (
    export_tool_use_dataset_to_sharegpt,
    validate_sharegpt_tool_dataset,
    validate_tool_use_source_dataset,
)
from src.tool_use.protocol import TRIPAI_TOOL_USE_SYSTEM_PROMPT

DEFAULT_INPUTS = [
    Path("data/raw_stage2/clarify_then_call.jsonl"),
    Path("data/raw_stage2/clarify_then_call_v2.jsonl"),
]
DEFAULT_OUTPUT = Path("data/processed_stage2/clarify_then_call.jsonl")
DEFAULT_REPORT = Path("data/processed_stage2/clarify_then_call_report.json")

TARGET_COUNTS = {
    "route_missing_origin": 120,
    "route_missing_destination": 120,
    "route_missing_city_or_ambiguous_city": 120,
    "route_missing_mode": 120,
    "poi_missing_city": 160,
    "poi_missing_anchor": 160,
    "poi_missing_keyword": 160,
    "geocode_missing_city": 160,
    "geocode_ambiguous_same_name": 80,
}
SUBTYPE_ORDER = tuple(TARGET_COUNTS)

SUBTYPE_TO_TOOL = {
    "route_missing_origin": ("amap_plan_route", "amap_plan_route"),
    "route_missing_destination": ("amap_plan_route", "amap_plan_route"),
    "route_missing_city_or_ambiguous_city": ("amap_plan_route", "amap_plan_route"),
    "route_missing_mode": ("amap_plan_route", "amap_plan_route"),
    "poi_missing_city": ("amap_search_poi", "amap_search_poi"),
    "poi_missing_anchor": ("amap_search_poi", "amap_search_poi"),
    "poi_missing_keyword": ("amap_search_poi", "amap_search_poi"),
    "geocode_missing_city": ("amap_geocode", "amap_geocode"),
    "geocode_ambiguous_same_name": ("amap_geocode", "amap_geocode"),
}
ARG_ORDER = {
    "amap_plan_route": ("origin", "destination", "mode", "city"),
    "amap_search_poi": ("keyword", "city", "around_location", "radius_m"),
    "amap_geocode": ("address", "city"),
}
ALLOWED_TOOL_ARGS = {
    "amap_plan_route": {"origin", "destination", "mode", "city"},
    "amap_search_poi": {"keyword", "city", "around_location", "radius_m"},
    "amap_geocode": {"address", "city"},
}
REQUIRED_TOOL_ARGS_BY_SUBTYPE = {
    "route_missing_origin": {"origin", "destination", "mode", "city"},
    "route_missing_destination": {"origin", "destination", "mode", "city"},
    "route_missing_city_or_ambiguous_city": {"origin", "destination", "mode", "city"},
    "route_missing_mode": {"origin", "destination", "mode", "city"},
    "poi_missing_city": {"keyword", "city", "around_location"},
    "poi_missing_anchor": {"keyword", "city", "around_location"},
    "poi_missing_keyword": {"keyword", "city", "around_location"},
    "geocode_missing_city": {"address", "city"},
    "geocode_ambiguous_same_name": {"address", "city"},
}
ROUTE_MODES = {"transit", "driving", "walking", "bicycling"}
ROUTE_MODE_TERMS = {
    "transit": ("\u516c\u4ea4", "\u5730\u94c1", "\u516c\u5171\u4ea4\u901a"),
    "driving": ("\u9a7e\u8f66", "\u5f00\u8f66", "\u81ea\u9a7e", "\u6253\u8f66"),
    "walking": ("\u6b65\u884c", "\u8d70\u8def"),
    "bicycling": ("\u9a91\u884c", "\u9a91\u8f66", "\u81ea\u884c\u8f66"),
}
NEARBY_TERMS = ("\u9644\u8fd1", "\u5468\u8fb9", "\u65c1\u8fb9", "\u5468\u56f4")
VAGUE_POI_TERMS = (
    "\u5730\u65b9",
    "\u65b9\u4fbf",
    "\u5403\u996d",
    "\u901b",
    "\u4e70",
    "\u73a9",
    "\u4f11\u606f",
    "\u597d\u53bb\u5904",
    "\u5e97",
)
POI_SUCCESS_MIN_RESULTS = 2
POI_SUCCESS_MAX_RESULTS = 5
MAX_DUPLICATE_RATE = 0.02

CLARIFY_TOKENS_BY_SUBTYPE = {
    "route_missing_origin": ("出发", "从哪里", "哪里走", "当前位置", "城市"),
    "route_missing_destination": ("去哪里", "目的地", "终点", "要去"),
    "route_missing_city_or_ambiguous_city": ("城市", "哪个城市", "同名", "具体"),
    "route_missing_mode": ("出行方式", "交通方式", "公交", "地铁", "开车", "步行", "骑行"),
    "poi_missing_city": ("城市", "哪个城市", "具体城市"),
    "poi_missing_anchor": (
        "位置",
        "当前位置",
        "参考点",
        "地标",
        "哪里附近",
        "哪个位置",
        "具体位置",
        "具体地址",
        "哪个区域",
        "区域",
        "地点",
        "商圈",
        "经纬度",
        "附近范围",
    ),
    "poi_missing_keyword": ("找什么", "什么类型", "具体类型", "类型", "关键词", "想找", "哪类", "哪一类", "具体品牌", "店铺类型", "哪种"),
    "geocode_missing_city": ("城市", "哪个城市", "同名", "具体"),
    "geocode_ambiguous_same_name": ("城市", "哪个城市", "同名", "具体", "附近地标"),
}
FIRST_GPT_FORBIDDEN_FACTS = (
    "\u9a7e\u8f66\u7ea6",
    "\u6b65\u884c\u7ea6",
)
FIRST_GPT_FORBIDDEN_FACT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:\u516c\u91cc|\u5206\u949f|\u7c73)|"
    r"(?:\u5750\u6807|\u7ecf\u5ea6|\u7eac\u5ea6)\s*[:\uff1a]?\s*-?\d"
)
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
MARKDOWN_RE = re.compile(r"```|\*\*|^\s{0,3}#{1,6}\s|^\s*[-*+]\s+", re.MULTILINE)
THINK_RE = re.compile(r"<think>|</think>|reasoning_content", re.IGNORECASE)
COORD_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*$")


@dataclass(frozen=True)
class RawRow:
    source_file: str
    line_number: int
    item: dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    source_file: str
    line_number: int
    raw_id: str
    subtype: str
    score: int
    argument_signature: str
    item: dict[str, Any]


def _read_jsonl_files(paths: list[Path]) -> list[RawRow]:
    rows: list[RawRow] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object.")
            rows.append(RawRow(source_file=path.name, line_number=line_number, item=payload))
    return rows


def _parse_conversations(item: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None, "missing_conversations"
    if len(conversations) != 7:
        return None, "conversation_length_not_7"
    if not all(isinstance(message, dict) for message in conversations):
        return None, "conversation_message_not_object"
    roles = tuple(message.get("from") for message in conversations)
    if roles != ("system", "human", "gpt", "human", "function_call", "observation", "gpt"):
        return None, "bad_role_sequence"
    if any(not _normalize_text(message.get("value")) for message in conversations):
        return None, "empty_conversation_value"
    return conversations, None


def _is_coordinate(value: Any) -> bool:
    return isinstance(value, str) and bool(COORD_RE.match(value))


def _city_variants(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        return ()
    city = value.strip()
    variants = [city]
    if city.endswith("\u5e02") or city.endswith("\u533a"):
        variants.append(city[:-1])
    return tuple(variant for variant in variants if len(variant) >= 2)


def _strict_semantic_rejection_reason(subtype: str, question: str, args: dict[str, Any]) -> str | None:
    if subtype == "route_missing_origin":
        origin = args.get("origin")
        if isinstance(origin, str) and len(origin) >= 2 and origin in question:
            return "semantic_origin_present_in_first_user"
    elif subtype == "route_missing_destination":
        destination = args.get("destination")
        if isinstance(destination, str) and len(destination) >= 2 and destination in question:
            return "semantic_destination_present_in_first_user"
    elif subtype == "route_missing_city_or_ambiguous_city":
        if any(city in question for city in _city_variants(args.get("city"))):
            return "semantic_city_present_in_first_user"
    elif subtype == "route_missing_mode":
        if any(term in question for term in ROUTE_MODE_TERMS.get(str(args.get("mode")), ())):
            return "semantic_mode_present_in_first_user"
    elif subtype == "poi_missing_city":
        if any(city in question for city in _city_variants(args.get("city"))):
            return "semantic_city_present_in_first_user"
        if not args.get("around_location"):
            return "poi_missing_city_missing_anchor_argument"
    elif subtype == "poi_missing_anchor":
        if not any(term in question for term in NEARBY_TERMS):
            return "semantic_anchor_question_not_nearby"
    elif subtype == "poi_missing_keyword":
        keyword = args.get("keyword")
        if isinstance(keyword, str) and len(keyword) >= 2 and keyword in question:
            return "semantic_keyword_present_in_first_user"
        if not any(term in question for term in VAGUE_POI_TERMS):
            return "semantic_keyword_question_not_vague"
        if not args.get("around_location"):
            return "poi_missing_keyword_missing_anchor_argument"
    elif subtype == "geocode_missing_city":
        if any(city in question for city in _city_variants(args.get("city"))):
            return "semantic_city_present_in_first_user"
    return None


def _normalize_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in ARG_ORDER[tool_name]:
        if key not in args:
            continue
        value = args[key]
        if key == "radius_m":
            if isinstance(value, int) and value > 0:
                normalized[key] = value
            elif isinstance(value, str) and value.strip().isdigit():
                normalized[key] = int(value.strip())
            continue
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized


def _trim_success_observation(tool_name: str, observation: dict[str, Any], fixes: Counter[str]) -> dict[str, Any] | None:
    if observation.get("status") != "success":
        return observation
    data = observation.get("data")
    if not isinstance(data, dict):
        return None
    normalized = {"status": "success", "data": dict(data)}
    if tool_name == "amap_search_poi":
        pois = data.get("pois")
        if not isinstance(pois, list):
            return None
        valid_pois = [
            poi
            for poi in pois
            if (
                isinstance(poi, dict)
                and str(poi.get("name") or "").strip()
                and str(poi.get("address") or "").strip()
                and str(poi.get("distance") or "").strip()
            )
        ]
        if len(valid_pois) < POI_SUCCESS_MIN_RESULTS:
            return None
        if len(valid_pois) > POI_SUCCESS_MAX_RESULTS:
            valid_pois = valid_pois[:POI_SUCCESS_MAX_RESULTS]
            fixes["trimmed_poi_results"] += 1
        normalized["data"]["pois"] = valid_pois
        normalized["data"]["count"] = min(int(normalized["data"].get("count") or len(valid_pois)), len(valid_pois))
    return normalized


def _clarify_text_is_valid(subtype: str, text: str) -> bool:
    if any(token in text for token in FIRST_GPT_FORBIDDEN_FACTS):
        return False
    if FIRST_GPT_FORBIDDEN_FACT_RE.search(text):
        return False
    tokens = CLARIFY_TOKENS_BY_SUBTYPE[subtype]
    return any(token in text for token in tokens)


def _fallback_answer(tool_name: str, status: str) -> str:
    if tool_name == "amap_plan_route":
        return (
            "刚才路线工具没有返回可靠结果，我先不编造具体路线。"
            "你可以再确认起点、终点、城市和出行方式，或稍后用实时地图重新查询。"
        )
    if tool_name == "amap_search_poi":
        return (
            "刚才周边搜索没有返回可靠结果，我先不编造店名或距离。"
            "你可以换一个更具体的位置或关键词，或稍后再用地图实时搜索。"
        )
    reason = "没有返回可靠结果" if status == "empty" else "暂时查询失败"
    return f"刚才位置查询{reason}，我先不编造坐标。你可以补充城市、区县或附近地标后再查一次。"


def _style_variant_question(base: str, variant_index: int, template_count: int) -> str:
    if variant_index < template_count:
        return base
    prefixes = (
        "麻烦帮我查一下，",
        "先帮我看下，",
        "我想确认一下，",
        "帮我快速查下，",
        "能不能帮我看看，",
        "请帮我查查，",
    )
    suffixes = (
        "",
        "先给我确认下。",
        "我想先看个结果。",
        "给我几个可参考的结果。",
        "查到后简单说下。",
        "先按这个条件查。",
    )
    style_index = max(0, variant_index - template_count)
    prefix = prefixes[style_index % len(prefixes)]
    suffix = suffixes[(style_index // len(prefixes)) % len(suffixes)]
    trimmed = base.rstrip(" 。！？!?")
    if suffix:
        return prefix + trimmed + "，" + suffix
    return prefix + trimmed + "。"


def _rewrite_answer(
    tool_name: str,
    question: str,
    args: dict[str, Any],
    observation: dict[str, Any],
) -> str | None:
    status = observation.get("status")
    if status == "success":
        data = observation.get("data")
        if not isinstance(data, dict):
            return None
        if tool_name == "amap_plan_route":
            return _route_answer(args, data)
        if tool_name == "amap_search_poi":
            return _poi_answer(args, data)
        if tool_name == "amap_geocode":
            return _geocode_answer(question, args, data)
        return None
    if status in {"empty", "error"}:
        return _fallback_answer(tool_name, status)
    return None


def _quality_tags(subtype: str, raw_tags: Any, *, fixed_subset: bool, status: str) -> list[str]:
    tags = ["clarify_first", "single_tool", subtype, "strict_arguments", "human_readable_units"]
    if status == "success":
        tags.extend(["success_observation", "grounded_answer"])
    else:
        tags.extend(["fallback_after_tool_failure", "no_hallucination"])
    if fixed_subset:
        tags.append("rebuilt_expected_arguments_subset")
    if isinstance(raw_tags, list):
        tags.extend(str(tag).strip() for tag in raw_tags if isinstance(tag, str) and tag.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def _candidate_score(
    *,
    source_file: str,
    subtype: str,
    tool_name: str,
    args: dict[str, Any],
    observation: dict[str, Any],
    answer: str,
) -> int:
    score = 100
    if source_file.endswith("_v2.jsonl"):
        score += 5
    if observation.get("status") == "success":
        score += 20
    if tool_name == "amap_search_poi":
        pois = observation.get("data", {}).get("pois")
        if isinstance(pois, list):
            score += min(len(pois), 5) * 5
            if any(str(poi.get("distance") or "").strip() for poi in pois if isinstance(poi, dict)):
                score += 5
    elif tool_name == "amap_plan_route":
        if args.get("mode") in {"transit", "walking", "bicycling"}:
            score += 5
        data = observation.get("data") if isinstance(observation.get("data"), dict) else {}
        if isinstance(data.get("duration_min"), (int, float)):
            score += 5
        if isinstance(data.get("distance_km"), (int, float)):
            score += 5
    elif tool_name == "amap_geocode":
        data = observation.get("data") if isinstance(observation.get("data"), dict) else {}
        if data.get("district"):
            score += 5
        if data.get("formatted_address"):
            score += 5
    if 50 <= len(answer) <= 260:
        score += 4
    return score


def _clean_one(row: RawRow, rejected: Counter[str], fixes: Counter[str]) -> Candidate | None:
    item = row.item
    raw_id = _normalize_text(item.get("id"))
    subtype = _normalize_text(item.get("subtype"))
    if item.get("task_type") != "clarify_then_call":
        rejected["bad_task_type"] += 1
        return None
    if subtype not in SUBTYPE_TO_TOOL:
        rejected["bad_subtype"] += 1
        return None
    expected_scene, expected_tool = SUBTYPE_TO_TOOL[subtype]
    if item.get("scene") != expected_scene:
        rejected["scene_subtype_mismatch"] += 1
        return None
    if item.get("expected_behavior") != "should_clarify":
        rejected["bad_expected_behavior"] += 1
        return None
    if item.get("expected_tool_chain") != [expected_tool]:
        rejected["bad_expected_tool_chain"] += 1
        return None

    conversations, conversation_error = _parse_conversations(item)
    if conversations is None:
        rejected[conversation_error or "bad_conversations"] += 1
        return None
    serialized = json.dumps(conversations, ensure_ascii=False)
    if THINK_RE.search(serialized):
        rejected["contains_thinking_marker"] += 1
        return None
    if EMOJI_RE.search(serialized):
        rejected["contains_emoji"] += 1
        return None

    question = _conversation_value(conversations, "human")
    first_clarify = _normalize_text(conversations[2].get("value"))
    followup = _normalize_text(conversations[3].get("value"))
    if MARKDOWN_RE.search(first_clarify):
        rejected["first_clarify_has_markdown"] += 1
        return None
    if not _clarify_text_is_valid(subtype, first_clarify):
        rejected["first_gpt_not_strict_clarify"] += 1
        return None

    function_payload = _loads_object(_conversation_value(conversations, "function_call"))
    observation = _loads_object(_conversation_value(conversations, "observation"))
    if not function_payload:
        rejected["bad_function_call_json"] += 1
        return None
    if not observation:
        rejected["bad_observation_json"] += 1
        return None

    tool_name = function_payload.get("name")
    args = function_payload.get("arguments")
    if tool_name != expected_tool:
        rejected["function_tool_mismatch"] += 1
        return None
    if not isinstance(args, dict):
        rejected["arguments_not_object"] += 1
        return None
    if set(args) - ALLOWED_TOOL_ARGS[tool_name]:
        rejected["disallowed_arguments"] += 1
        return None
    normalized_args = _normalize_args(tool_name, args)
    required_args = REQUIRED_TOOL_ARGS_BY_SUBTYPE[subtype]
    missing_required_args = sorted(arg for arg in required_args if arg not in normalized_args)
    if missing_required_args:
        rejected[f"missing_required_argument_{missing_required_args[0]}"] += 1
        return None
    if tool_name == "amap_plan_route" and normalized_args.get("mode") not in ROUTE_MODES:
        rejected["bad_route_mode"] += 1
        return None
    semantic_rejection_reason = _strict_semantic_rejection_reason(subtype, question, normalized_args)
    if semantic_rejection_reason:
        rejected[semantic_rejection_reason] += 1
        return None

    fixed_subset = item.get("expected_arguments_subset") != normalized_args
    if fixed_subset:
        fixes["rebuilt_expected_arguments_subset"] += 1

    normalized_observation = _trim_success_observation(tool_name, observation, fixes)
    if normalized_observation is None:
        rejected["bad_or_insufficient_observation"] += 1
        return None

    answer = _rewrite_answer(tool_name, question, normalized_args, normalized_observation)
    if answer is None:
        rejected["answer_rewrite_failed"] += 1
        return None
    answer = _normalize_text(answer)
    if MARKDOWN_RE.search(answer) or not _answer_is_clean(answer):
        rejected["rewritten_answer_not_clean"] += 1
        return None

    function_call_value = _json_dumps({"name": tool_name, "arguments": normalized_args})
    argument_signature = json.dumps({"name": tool_name, "arguments": normalized_args}, ensure_ascii=False, sort_keys=True)
    cleaned_item = {
        "id": raw_id,
        "task_type": "clarify_then_call",
        "subtype": subtype,
        "scene": expected_scene,
        "expected_behavior": "should_clarify",
        "expected_tool_chain": [expected_tool],
        "expected_arguments_subset": normalized_args,
        "quality_tags": _quality_tags(
            subtype,
            item.get("quality_tags"),
            fixed_subset=fixed_subset,
            status=str(normalized_observation.get("status")),
        ),
        "conversations": [
            {"from": "system", "value": TRIPAI_TOOL_USE_SYSTEM_PROMPT},
            {"from": "human", "value": question},
            {"from": "gpt", "value": first_clarify},
            {"from": "human", "value": followup},
            {"from": "function_call", "value": function_call_value},
            {"from": "observation", "value": _json_dumps(normalized_observation)},
            {"from": "gpt", "value": answer},
        ],
    }
    score = _candidate_score(
        source_file=row.source_file,
        subtype=subtype,
        tool_name=tool_name,
        args=normalized_args,
        observation=normalized_observation,
        answer=answer,
    )
    return Candidate(
        source_file=row.source_file,
        line_number=row.line_number,
        raw_id=raw_id,
        subtype=subtype,
        score=score,
        argument_signature=argument_signature,
        item=cleaned_item,
    )


def _variant_opening(subtype: str, args: dict[str, Any], variant_index: int) -> str:
    if subtype == "poi_missing_anchor":
        keyword = str(args.get("keyword") or "周边地点")
        city = str(args.get("city") or "")
        templates = (
            "我想在附近找{keyword}，先帮我看看，城市是{city}。",
            "{city}这边附近有没有合适的{keyword}？",
            "帮我查附近的{keyword}，我在{city}。",
        )
        base = templates[variant_index % len(templates)].format(keyword=keyword, city=city)
        return _style_variant_question(base, variant_index, len(templates))
    if subtype == "poi_missing_keyword":
        around = str(args.get("around_location") or "附近")
        city = str(args.get("city") or "")
        templates = (
            "我在{city}{around}附近，想找个方便的地方。",
            "{city}{around}周边有什么合适的地方可以去？",
            "帮我看看{around}附近有什么方便去的地方，城市是{city}。",
        )
        base = templates[variant_index % len(templates)].format(around=around, city=city)
        return _style_variant_question(base, variant_index, len(templates))
    if subtype == "poi_missing_city":
        keyword = str(args.get("keyword") or "地点")
        around = str(args.get("around_location") or "万达广场")
        templates = (
            "{around}附近有没有{keyword}？",
            "帮我找一下{around}周边的{keyword}。",
            "我想查{around}附近的{keyword}，先帮我看看。",
            "附近有什么{keyword}？",
            "附近有哪些{keyword}？",
            "附近有{keyword}吗？",
            "周边有什么{keyword}？",
            "周边有哪些{keyword}？",
            "想找附近的{keyword}。",
            "帮我查一下附近的{keyword}。",
            "帮我看看附近有没有{keyword}。",
            "哪里有{keyword}？",
            "有没有推荐的{keyword}？",
            "找一家{keyword}，先查一下。",
            "附近能找到{keyword}吗？",
            "我想找{keyword}，但还没说具体城市。",
            "查一下{keyword}，城市我还没补充。",
            "先帮我看看{keyword}，还需要确认城市。",
            "想查{keyword}，具体城市稍后补。",
            "我想找个{keyword}，先问下能不能查。",
            "帮我看看有没有合适的{keyword}。",
            "附近有没有合适的{keyword}可以去？",
            "周边有没有口碑不错的{keyword}？",
            "想查周边的{keyword}，先帮我确认一下。",
            "我需要找{keyword}，但位置城市还没说清楚。",
            "帮我搜索{keyword}，先确认缺的城市信息。",
            "想找{keyword}，你先帮我问清城市。",
            "查{keyword}之前是不是要先确认城市？",
            "{around}周边有适合去的{keyword}吗？",
            "我想看看{around}附近的{keyword}。",
        )
        base = templates[variant_index % len(templates)].format(around=around, keyword=keyword)
        return _style_variant_question(base, variant_index, len(templates))
    if subtype.startswith("geocode"):
        address = str(args.get("address") or "这个地点")
        templates = (
            "{address}具体在哪里？",
            "帮我查一下{address}的位置。",
            "{address}在什么地方？",
        )
        base = templates[variant_index % len(templates)].format(address=address)
        return _style_variant_question(base, variant_index, len(templates))

    origin = str(args.get("origin") or "")
    destination = str(args.get("destination") or "")
    city = str(args.get("city") or "")
    mode = str(args.get("mode") or "transit")
    mode_text = {"transit": "公交地铁", "driving": "开车", "walking": "步行", "bicycling": "骑行"}.get(mode, "出行")
    if subtype == "route_missing_origin":
        templates = (
            "我想去{destination}，帮我查{mode_text}怎么走。",
            "帮我规划一下去{destination}的{mode_text}路线。",
            "{city}这边去{destination}怎么走？我倾向{mode_text}。",
        )
    elif subtype == "route_missing_destination":
        templates = (
            "我从{origin}出发，想查一下{mode_text}路线。",
            "我现在在{origin}，帮我规划个{mode_text}方案。",
            "{city}{origin}出发，帮我看看怎么走。",
        )
    elif subtype == "route_missing_mode":
        templates = (
            "我从{origin}去{destination}，帮我看一下怎么走。",
            "{origin}到{destination}怎么走比较合适？",
            "帮我查一下{city}{origin}到{destination}的路线。",
        )
    else:
        templates = (
            "我从{origin}去{destination}，帮我查{mode_text}路线。",
            "帮我看一下从{origin}到{destination}的{mode_text}方案。",
            "{origin}到{destination}怎么走？",
        )
    base = templates[variant_index % len(templates)].format(
        origin=origin,
        destination=destination,
        city=city,
        mode_text=mode_text,
    )
    return _style_variant_question(base, variant_index, len(templates))


def _make_unique_question(
    candidate: Candidate,
    used_questions: set[str],
    used_dialogue_keys: set[str],
    variant_counts: Counter[str],
    fixes: Counter[str],
) -> dict[str, Any]:
    item = json.loads(json.dumps(candidate.item, ensure_ascii=False))
    question = item["conversations"][1]["value"]
    followup = item["conversations"][3]["value"]
    dialogue_key = question + "\0" + followup
    if question not in used_questions and dialogue_key not in used_dialogue_keys:
        return item
    args = json.loads(item["conversations"][4]["value"])["arguments"]
    for _ in range(80):
        variant_index = variant_counts[candidate.argument_signature]
        variant_counts[candidate.argument_signature] += 1
        rewritten = _variant_opening(candidate.subtype, args, variant_index)
        if rewritten and rewritten not in used_questions and rewritten + "\0" + followup not in used_dialogue_keys:
            item["conversations"][1]["value"] = rewritten
            item["quality_tags"] = [*item["quality_tags"], "rewritten_duplicate_question"]
            fixes["rewritten_duplicate_question"] += 1
            return item
    raise RuntimeError(f"could not create unique question for {candidate.raw_id}")


def _dialogue_key(item: dict[str, Any]) -> str:
    return item["conversations"][1]["value"] + "\0" + item["conversations"][3]["value"]


def _select_for_subtype(
    subtype: str,
    candidates: list[Candidate],
    target_count: int,
    *,
    used_questions: set[str],
    used_dialogue_keys: set[str],
    function_call_counts: Counter[str],
    duplicate_budget: int,
    allow_partial: bool,
    fixes: Counter[str],
) -> list[dict[str, Any]]:
    selected: list[Candidate] = []
    selected_keys: set[tuple[str, int, str]] = set()
    reserved_questions = set(used_questions)
    reserved_dialogue_keys = set(used_dialogue_keys)
    def duplicate_function_calls() -> int:
        return sum(count - 1 for count in function_call_counts.values() if count > 1)

    def can_take_function_call(function_call: str, *, require_unique_function_call: bool) -> bool:
        if require_unique_function_call and function_call_counts[function_call] > 0:
            return False
        if function_call_counts[function_call] > 0 and duplicate_function_calls() >= duplicate_budget:
            return False
        return True

    def add_phase(*, require_unique_question: bool, require_unique_dialogue: bool, require_unique_function_call: bool) -> None:
        if len(selected) >= target_count:
            return
        for candidate in candidates:
            if len(selected) >= target_count:
                return
            key = (candidate.source_file, candidate.line_number, candidate.raw_id)
            if key in selected_keys:
                continue
            question = candidate.item["conversations"][1]["value"]
            dialogue_key = _dialogue_key(candidate.item)
            function_call = candidate.item["conversations"][4]["value"]
            if require_unique_question and question in reserved_questions:
                continue
            if require_unique_dialogue and dialogue_key in reserved_dialogue_keys:
                continue
            if not can_take_function_call(function_call, require_unique_function_call=require_unique_function_call):
                continue
            selected.append(candidate)
            selected_keys.add(key)
            reserved_questions.add(question)
            reserved_dialogue_keys.add(dialogue_key)
            function_call_counts[function_call] += 1

    add_phase(require_unique_question=True, require_unique_dialogue=True, require_unique_function_call=True)
    add_phase(require_unique_question=True, require_unique_dialogue=True, require_unique_function_call=False)
    add_phase(require_unique_question=False, require_unique_dialogue=True, require_unique_function_call=False)

    if len(selected) < target_count:
        for candidate in candidates:
            if len(selected) >= target_count:
                break
            key = (candidate.source_file, candidate.line_number, candidate.raw_id)
            if key in selected_keys:
                continue
            function_call = candidate.item["conversations"][4]["value"]
            if not can_take_function_call(function_call, require_unique_function_call=False):
                continue
            selected.append(candidate)
            selected_keys.add(key)
            function_call_counts[function_call] += 1

    if len(selected) < target_count and not allow_partial:
        raise RuntimeError(f"{subtype} has only {len(selected)} selectable candidates, target={target_count}")

    variant_counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for candidate in selected:
        item = _make_unique_question(candidate, used_questions, used_dialogue_keys, variant_counts, fixes)
        used_questions.add(item["conversations"][1]["value"])
        used_dialogue_keys.add(_dialogue_key(item))
        output.append(item)
    return output


def _renumber_selected(rows_by_subtype: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for subtype in SUBTYPE_ORDER:
        for index, row in enumerate(rows_by_subtype[subtype], start=1):
            item = dict(row)
            item["id"] = f"stage2_v2_clarify_then_call_{subtype}_{index:06d}"
            cleaned.append(item)
    return cleaned


def _validate_processed_rows(rows: list[dict[str, Any]]) -> list[str]:
    source_dataset = _to_source_dataset(rows)
    errors = validate_tool_use_source_dataset(source_dataset)
    if errors:
        return errors
    sharegpt_dataset = export_tool_use_dataset_to_sharegpt(source_dataset)
    return validate_sharegpt_tool_dataset(sharegpt_dataset)


def clean_clarify_then_call(
    rows: list[RawRow],
    *,
    target_counts: dict[str, int] | None = None,
    max_duplicate_rate: float = MAX_DUPLICATE_RATE,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_counts = dict(target_counts or TARGET_COUNTS)
    rejected: Counter[str] = Counter()
    fixes: Counter[str] = Counter()
    candidates_by_subtype: dict[str, list[Candidate]] = {subtype: [] for subtype in SUBTYPE_ORDER}
    raw_counts = Counter(row.item.get("subtype") for row in rows)
    source_counts = Counter(row.source_file for row in rows)

    for row in rows:
        candidate = _clean_one(row, rejected, fixes)
        if candidate is not None:
            candidates_by_subtype[candidate.subtype].append(candidate)

    for subtype, candidates in candidates_by_subtype.items():
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.argument_signature, candidate.raw_id, candidate.source_file, candidate.line_number))
        if len(candidates) < target_counts[subtype] and not allow_partial:
            raise RuntimeError(f"{subtype} has only {len(candidates)} clean candidates, target={target_counts[subtype]}")

    used_questions: set[str] = set()
    used_dialogue_keys: set[str] = set()
    function_call_counts_for_selection: Counter[str] = Counter()
    duplicate_budget = 0 if allow_partial else int(sum(target_counts.values()) * max_duplicate_rate)
    selected_by_subtype: dict[str, list[dict[str, Any]]] = {}
    for subtype in SUBTYPE_ORDER:
        selected_by_subtype[subtype] = _select_for_subtype(
            subtype,
            candidates_by_subtype[subtype],
            target_counts[subtype],
            used_questions=used_questions,
            used_dialogue_keys=used_dialogue_keys,
            function_call_counts=function_call_counts_for_selection,
            duplicate_budget=duplicate_budget,
            allow_partial=allow_partial,
            fixes=fixes,
        )

    cleaned = _renumber_selected(selected_by_subtype)
    validation_errors = _validate_processed_rows(cleaned)
    if validation_errors:
        raise RuntimeError(f"processed clarify_then_call failed validation: {validation_errors[:5]}")

    selected_counts = Counter(item["subtype"] for item in cleaned)
    tool_counts = Counter()
    route_modes = Counter()
    observation_statuses = Counter()
    questions = Counter()
    dialogue_keys = Counter()
    function_calls = Counter()
    argument_signatures = Counter()
    answer_lengths = [len(item["conversations"][-1]["value"]) for item in cleaned]
    for item in cleaned:
        questions[item["conversations"][1]["value"]] += 1
        dialogue_keys[item["conversations"][1]["value"] + "\0" + item["conversations"][3]["value"]] += 1
        function_payload = json.loads(item["conversations"][4]["value"])
        observation = json.loads(item["conversations"][5]["value"])
        tool_name = function_payload["name"]
        args = function_payload["arguments"]
        tool_counts[tool_name] += 1
        observation_statuses[observation.get("status")] += 1
        function_calls[item["conversations"][4]["value"]] += 1
        argument_signatures[json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False, sort_keys=True)] += 1
        if tool_name == "amap_plan_route":
            route_modes[args.get("mode")] += 1

    duplicate_argument_signatures = sum(count - 1 for count in argument_signatures.values() if count > 1)
    duplicate_dialogue_keys = sum(count - 1 for count in dialogue_keys.values() if count > 1)
    duplicate_questions = sum(count - 1 for count in questions.values() if count > 1)
    duplicate_function_calls = sum(count - 1 for count in function_calls.values() if count > 1)
    duplicate_dialogue_key_rate = duplicate_dialogue_keys / len(cleaned) if cleaned else 0
    duplicate_function_call_rate = duplicate_function_calls / len(cleaned) if cleaned else 0
    if duplicate_dialogue_key_rate > max_duplicate_rate:
        raise RuntimeError(
            "processed clarify_then_call duplicate dialogue key rate "
            f"{duplicate_dialogue_key_rate:.4f} exceeds max_duplicate_rate={max_duplicate_rate}"
        )
    if duplicate_function_call_rate > max_duplicate_rate:
        raise RuntimeError(
            "processed clarify_then_call duplicate function call rate "
            f"{duplicate_function_call_rate:.4f} exceeds max_duplicate_rate={max_duplicate_rate}"
        )
    shortage_counts = {
        subtype: max(0, target_counts[subtype] - selected_counts.get(subtype, 0))
        for subtype in SUBTYPE_ORDER
    }
    report = {
        "input_records": len(rows),
        "input_files": dict(sorted(source_counts.items())),
        "target_counts": target_counts,
        "target_total": sum(target_counts.values()),
        "raw_subtype_counts": dict(sorted(raw_counts.items())),
        "candidate_counts": {subtype: len(candidates_by_subtype[subtype]) for subtype in SUBTYPE_ORDER},
        "selected_count": len(cleaned),
        "selected_counts": dict(sorted(selected_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "route_mode_counts": dict(sorted(route_modes.items())),
        "observation_status_counts": dict(sorted(observation_statuses.items())),
        "selected_duplicate_questions": duplicate_questions,
        "selected_duplicate_dialogue_keys": duplicate_dialogue_keys,
        "selected_duplicate_dialogue_key_rate": round(duplicate_dialogue_key_rate, 4),
        "selected_duplicate_function_calls": duplicate_function_calls,
        "selected_duplicate_function_call_rate": round(duplicate_function_call_rate, 4),
        "selected_duplicate_argument_signatures": duplicate_argument_signatures,
        "selected_duplicate_argument_signature_rate": round(duplicate_argument_signatures / len(cleaned), 4) if cleaned else 0,
        "max_duplicate_rate": max_duplicate_rate,
        "allow_partial": allow_partial,
        "shortage_counts": {subtype: count for subtype, count in shortage_counts.items() if count},
        "shortage_total": sum(shortage_counts.values()),
        "rejected": dict(sorted(rejected.items())),
        "fixes": dict(sorted(fixes.items())),
        "answer_length": {
            "min": min(answer_lengths, default=0),
            "max": max(answer_lengths, default=0),
            "avg": round(sum(answer_lengths) / len(answer_lengths), 1) if answer_lengths else 0,
        },
        "validation": {
            "source_validator_errors": 0,
            "sharegpt_validator_errors": 0,
        },
        "output_format": "processed_stage2_jsonl_with_conversations",
    }
    return cleaned, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean raw Stage2 clarify_then_call JSONL into processed Stage2 JSONL.")
    parser.add_argument("--input", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-duplicate-rate", type=float, default=MAX_DUPLICATE_RATE)
    parser.add_argument("--allow-partial", action="store_true", help="Write the strict usable subset when raw data cannot fill every target bucket.")
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    rows = _read_jsonl_files(args.input)
    cleaned, report = clean_clarify_then_call(rows, max_duplicate_rate=args.max_duplicate_rate, allow_partial=args.allow_partial)
    output_path = _write_jsonl(args.output, cleaned)
    report_path = _write_json(args.report, report)
    print(f"[OK] cleaned clarify_then_call written to: {output_path}")
    print(f"[OK] report written to: {report_path}")
    print(f"[OK] selected: {report['selected_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
