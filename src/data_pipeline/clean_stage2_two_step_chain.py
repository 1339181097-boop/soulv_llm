from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.clean_stage2_single_tool_call import (
    EMOJI_RE,
    RAW_LONG_METERS_RE,
    RAW_SECONDS_RE,
    _format_km,
    _format_meter_distance,
    _json_dumps,
    _loads_object,
    _normalize_text,
    _ordered_unique,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from src.data_pipeline.data_utils import configure_console_output
from src.tool_use.datasets import (
    export_tool_use_dataset_to_sharegpt,
    validate_sharegpt_tool_dataset,
    validate_tool_use_source_dataset,
)
from src.tool_use.protocol import TRIPAI_TOOL_USE_SYSTEM_PROMPT, build_amap_tool_schemas

DEFAULT_INPUTS = [
    Path("data/raw_stage2/two_step_chain_v2.jsonl"),
    Path("data/raw_stage2/two_step_chain_v2_repaired.jsonl"),
]
DEFAULT_OUTPUT = Path("data/processed_stage2/two_step_chain_v2_repaired.jsonl")
DEFAULT_REPORT = Path("data/processed_stage2/two_step_chain_v2_repaired_report.json")

TARGET_COUNTS = {
    "geocode_then_search_hotel": 120,
    "geocode_then_search_restaurant": 100,
    "geocode_then_search_subway": 80,
    "geocode_then_search_parking": 80,
    "geocode_then_search_mall_or_spot": 120,
    "geocode_then_route_transit": 140,
    "geocode_then_route_walking": 80,
    "geocode_then_route_driving_or_bicycling": 80,
}
SUBTYPE_ORDER = (
    "geocode_then_search_hotel",
    "geocode_then_search_restaurant",
    "geocode_then_search_subway",
    "geocode_then_search_parking",
    "geocode_then_search_mall_or_spot",
    "geocode_then_route_transit",
    "geocode_then_route_walking",
    "geocode_then_route_driving_or_bicycling",
)
SUBTYPE_TO_CHAIN = {
    "geocode_then_search_hotel": ("amap_geocode", "amap_search_poi"),
    "geocode_then_search_restaurant": ("amap_geocode", "amap_search_poi"),
    "geocode_then_search_subway": ("amap_geocode", "amap_search_poi"),
    "geocode_then_search_parking": ("amap_geocode", "amap_search_poi"),
    "geocode_then_search_mall_or_spot": ("amap_geocode", "amap_search_poi"),
    "geocode_then_route_transit": ("amap_geocode", "amap_plan_route"),
    "geocode_then_route_walking": ("amap_geocode", "amap_plan_route"),
    "geocode_then_route_driving_or_bicycling": ("amap_geocode", "amap_plan_route"),
}
ALLOWED_TOOL_ARGS = {
    "amap_geocode": {"address", "city"},
    "amap_search_poi": {"keyword", "city", "around_location", "radius_m"},
    "amap_plan_route": {"origin", "destination", "mode", "city"},
}
ARG_ORDER = {
    "amap_geocode": ("address", "city"),
    "amap_search_poi": ("keyword", "city", "around_location", "radius_m"),
    "amap_plan_route": ("origin", "destination", "mode", "city"),
}
ROUTE_MODES = {"transit", "driving", "walking", "bicycling"}
ROUTE_MODE_LABELS = {
    "transit": "公共交通",
    "driving": "驾车",
    "walking": "步行",
    "bicycling": "骑行",
}
REQUIRED_QUALITY_TAGS = ("two_step", "geocode_anchor", "grounded_answer")

COORD_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")
SPACE_RE = re.compile(r"[ \t\u3000]+")
DISTANCE_TEXT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:千米|公里|米)")
ROAD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,18}(?:大道|高速|公路|快速路|立交桥|路桥|路|街|道|巷|桥|线)")


@dataclass(frozen=True)
class Candidate:
    source: str
    line_number: int
    raw_id: str
    subtype: str
    score: int
    semantic_key: str
    item: dict[str, Any]
    fixes: tuple[str, ...]


def _read_jsonl_inputs(paths: Iterable[Path]) -> list[tuple[str, int, dict[str, Any]]]:
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for path in paths:
        for line_number, item in _read_jsonl(path):
            rows.append((str(path), line_number, item))
    return rows


def _is_coordinate(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    match = COORD_RE.match(value)
    if not match:
        return False
    lon = float(match.group(1))
    lat = float(match.group(2))
    return -180 <= lon <= 180 and -90 <= lat <= 90


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _same_nonempty_text(left: Any, right: Any) -> bool:
    left_text = _compact(left)
    right_text = _compact(right)
    return bool(left_text and right_text and left_text == right_text)


def _normalize_question(value: Any) -> str:
    text = _normalize_text(value)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def _infer_anchor_side(question: str, address: str) -> str | None:
    compact_question = _compact(question)
    compact_address = _compact(address)
    if not compact_question or not compact_address or compact_address not in compact_question:
        return None

    index = compact_question.find(compact_address)
    before = compact_question[max(0, index - 12) : index]
    after = compact_question[index + len(compact_address) : index + len(compact_address) + 12]

    if "从" in before or after.startswith(("出发", "开始")):
        return "origin"
    if any(token in before for token in ("到", "去", "至", "前往", "抵达")):
        return "destination"
    if any(token in compact_question[max(0, index - 8) : index] for token in ("想去", "要去", "打算去", "准备去")):
        return "destination"
    if compact_question.rfind("从", 0, index) != -1 and any(
        compact_question.find(token, index) != -1 for token in ("去", "到", "前往")
    ):
        return "origin"
    return None


def _parse_conversations(item: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None, "missing_conversations"
    if len(conversations) != 7:
        return None, "conversation_length_not_7"
    if not all(isinstance(message, dict) for message in conversations):
        return None, "conversation_message_not_object"
    roles = tuple(message.get("from") for message in conversations)
    if roles != ("system", "human", "function_call", "observation", "function_call", "observation", "gpt"):
        return None, "bad_role_sequence"
    if any(not _normalize_text(message.get("value")) for message in conversations):
        return None, "empty_conversation_value"
    return conversations, None


def _normalize_arguments(tool_name: str, raw_args: Any) -> dict[str, Any] | None:
    if not isinstance(raw_args, dict):
        return None
    if set(raw_args) - ALLOWED_TOOL_ARGS[tool_name]:
        return None

    normalized: dict[str, Any] = {}
    for key in ARG_ORDER[tool_name]:
        if key not in raw_args:
            continue
        value = raw_args[key]
        if key == "radius_m":
            if not isinstance(value, int) or value <= 0:
                return None
            normalized[key] = value
            continue
        text = _normalize_text(value)
        if text:
            normalized[key] = text

    if tool_name == "amap_geocode" and "address" not in normalized:
        return None
    if tool_name == "amap_search_poi" and "keyword" not in normalized:
        return None
    if tool_name == "amap_plan_route":
        if "origin" not in normalized or "destination" not in normalized:
            return None
        if normalized.get("mode") not in ROUTE_MODES:
            return None
    return normalized


def _normalize_success_observation(raw_observation: Any) -> dict[str, Any] | None:
    if not isinstance(raw_observation, dict):
        return None
    if raw_observation.get("status") != "success":
        return None
    data = raw_observation.get("data")
    if not isinstance(data, dict):
        return None
    return {"status": "success", "data": dict(data)}


def _is_implausible_walking_route(args: dict[str, Any], observation: dict[str, Any]) -> bool:
    if args.get("mode") != "walking":
        return False
    data = observation.get("data", {})
    try:
        distance_km = float(data.get("distance_km"))
    except (TypeError, ValueError):
        distance_km = 0.0
    try:
        duration_min = float(data.get("duration_min"))
    except (TypeError, ValueError):
        duration_min = 0.0
    return distance_km > 8 or duration_min > 180


def _sync_route_observation(args: dict[str, Any], observation: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    fixes: list[str] = []
    data = dict(observation["data"])
    for key in ("origin", "destination"):
        if data.get(key) != args.get(key):
            data[key] = args.get(key)
            fixes.append(f"sync_route_observation_{key}")
    if data.get("origin_location") != args.get("origin"):
        data["origin_location"] = args.get("origin")
        fixes.append("sync_route_observation_origin_location")
    if data.get("destination_location") != args.get("destination"):
        data["destination_location"] = args.get("destination")
        fixes.append("sync_route_observation_destination_location")
    return {"status": "success", "data": data}, fixes


def _repair_route_args(
    question: str,
    geocode_args: dict[str, Any],
    geocode_observation: dict[str, Any],
    route_args: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    location = geocode_observation["data"].get("location")
    if not _is_coordinate(location):
        return None, []

    repaired = dict(route_args)
    fixes: list[str] = []
    side = _infer_anchor_side(question, str(geocode_args.get("address") or ""))
    current_side = None
    if repaired.get("origin") == location:
        current_side = "origin"
    elif repaired.get("destination") == location:
        current_side = "destination"

    if side is None:
        if current_side is None:
            return None, []
        fixes.append("kept_existing_geocode_side_uninferred")
        return repaired, fixes

    if current_side == side:
        return repaired, fixes

    target_other_key = "destination" if side == "origin" else "origin"
    if current_side is None:
        source_other_key = target_other_key
    else:
        source_other_key = "destination" if current_side == "origin" else "origin"
    current_other = repaired.get(source_other_key)
    if not isinstance(current_other, str) or not current_other.strip():
        return None, []
    repaired[side] = location
    repaired[target_other_key] = current_other
    fixes.append(f"set_geocode_location_as_{side}")
    return repaired, fixes


def _normalize_search_observation(
    args: dict[str, Any],
    observation: dict[str, Any],
    geocode_location: str,
) -> tuple[dict[str, Any], list[str]] | None:
    data = dict(observation["data"])
    pois = data.get("pois")
    if not isinstance(pois, list) or not pois:
        return None
    valid_pois = [poi for poi in pois if isinstance(poi, dict) and _normalize_text(poi.get("name"))]
    if len(valid_pois) < 2:
        return None

    fixes: list[str] = []
    if args.get("around_location") != geocode_location:
        args["around_location"] = geocode_location
        fixes.append("set_search_around_location_to_geocode_location")
    if data.get("around_location") != geocode_location:
        data["around_location"] = geocode_location
        fixes.append("sync_search_observation_around_location")
    data["keyword"] = args.get("keyword") or data.get("keyword")
    if args.get("city"):
        data["city"] = args["city"]
    data["count"] = len(valid_pois)
    data["pois"] = valid_pois
    return {"status": "success", "data": data}, fixes


def _format_duration(minutes: Any) -> str | None:
    if not isinstance(minutes, (int, float)) or minutes <= 0:
        return None
    total = int(round(minutes))
    if total >= 60:
        hours, mins = divmod(total, 60)
        if mins:
            return f"约{hours}小时{mins}分钟"
        return f"约{hours}小时"
    return f"约{total}分钟"


def _clean_step_text(step: Any) -> str:
    if isinstance(step, dict):
        text = " ".join(str(value) for value in step.values())
    else:
        text = str(step)
    text = DISTANCE_TEXT_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(" ，,。；;")
    return text


def _route_detail(mode: str, data: dict[str, Any]) -> str:
    steps = data.get("main_steps")
    if not isinstance(steps, list):
        return "主要按工具返回的导航路线行进"

    cleaned_steps = [_clean_step_text(step) for step in steps]
    cleaned_steps = [step for step in cleaned_steps if step and step not in {"乘坐", "步行"}]
    if mode == "transit":
        transit_steps = [
            step
            for step in cleaned_steps
            if any(token in step for token in ("乘坐", "换乘", "地铁", "公交", "站", "路", "线"))
        ]
        if transit_steps:
            return "主要路线：" + "；".join(transit_steps[:3])
        if cleaned_steps:
            return "主要路线：" + "；".join(cleaned_steps[:3])
        return "主要按公交地铁实时路线换乘"

    roads: list[str] = []
    for step in steps:
        text = " ".join(str(value) for value in step.values()) if isinstance(step, dict) else str(step)
        roads.extend(ROAD_RE.findall(text))
    roads = _ordered_unique(roads)
    if roads:
        return "主要经过" + "、".join(roads[:5])
    if cleaned_steps:
        return "主要路线：" + "；".join(cleaned_steps[:3])
    return "主要按工具返回的导航路线行进"


def _route_answer(
    question: str,
    geocode_args: dict[str, Any],
    geocode_observation: dict[str, Any],
    route_args: dict[str, Any],
    route_observation: dict[str, Any],
) -> str | None:
    data = route_observation.get("data")
    if not isinstance(data, dict):
        return None
    mode = route_args.get("mode") or data.get("mode")
    if mode not in ROUTE_MODE_LABELS:
        return None
    duration = _format_duration(data.get("duration_min"))
    distance = _format_km(data.get("distance_km"))
    if not duration or not distance:
        return None

    anchor = _normalize_text(geocode_observation.get("data", {}).get("query") or geocode_args.get("address"))
    if not anchor:
        return None
    side = _infer_anchor_side(question, str(geocode_args.get("address") or ""))
    if side == "origin":
        anchor_clause = f"我先定位了{anchor}，并以它作为起点规划路线。"
    elif side == "destination":
        anchor_clause = f"我先定位了{anchor}，并以它作为终点规划路线。"
    else:
        anchor_clause = f"我先定位了{anchor}，再用这个坐标规划路线。"

    label = ROUTE_MODE_LABELS[str(mode)]
    if mode == "transit":
        reminder = "出发前再确认实时班次和换乘信息。"
    elif mode == "driving":
        reminder = "出发前建议看一下实时路况，高峰时段多预留一点时间。"
    elif mode == "walking":
        reminder = "步行时留意路口和天气，夜间优先走照明更好的道路。"
    else:
        reminder = "骑行时注意非机动车道和路口转向，雨天建议改用其他交通方式。"
    detail = _route_detail(str(mode), data)
    return f"{anchor_clause}{label}全程{duration}，距离{distance}。{detail}。{reminder}"


def _poi_result_text(poi: dict[str, Any]) -> str:
    name = _normalize_text(poi.get("name"))
    address = _normalize_text(poi.get("address"))
    distance = _format_meter_distance(poi.get("distance"))
    parts = [name]
    if distance:
        parts.append(f"距离{distance}")
    if address:
        parts.append(f"地址在{address}")
    return "，".join(part for part in parts if part)


def _poi_answer(
    geocode_args: dict[str, Any],
    geocode_observation: dict[str, Any],
    search_args: dict[str, Any],
    search_observation: dict[str, Any],
) -> str | None:
    data = search_observation.get("data")
    if not isinstance(data, dict):
        return None
    anchor = _normalize_text(geocode_observation.get("data", {}).get("query") or geocode_args.get("address"))
    keyword = _normalize_text(data.get("keyword") or search_args.get("keyword"))
    pois = data.get("pois")
    if not anchor or not keyword or not isinstance(pois, list):
        return None
    valid_pois = [poi for poi in pois if isinstance(poi, dict) and _normalize_text(poi.get("name"))]
    if len(valid_pois) < 2:
        return None
    selected = valid_pois[: min(5, len(valid_pois))]
    result_text = "；".join(_poi_result_text(poi) for poi in selected)
    if not result_text:
        return None
    if len(selected) == 1:
        suffix = "目前工具只返回这一处结果，出发前建议再确认营业状态和现场情况。"
    else:
        suffix = "建议优先选距离更近、地址更匹配行程的一家。"
    return f"我先定位了{anchor}，再围绕这个坐标查找{keyword}。可以优先看：{result_text}。{suffix}"


def _answer_is_clean(answer: str) -> bool:
    if not answer or "\n" in answer:
        return False
    if EMOJI_RE.search(answer):
        return False
    if any(token in answer for token in ("<think>", "</think>", "reasoning_content", "```", "**")):
        return False
    if any(token in answer for token in ("tool_call", "observation", "formatted_address", '"status"')):
        return False
    if RAW_SECONDS_RE.search(answer) or RAW_LONG_METERS_RE.search(answer):
        return False
    return True


def _quality_tags(subtype: str, raw_tags: Any, fixes: list[str]) -> list[str]:
    tags = ["two_step", "geocode_anchor", "grounded_answer", "human_readable_units", subtype]
    if isinstance(raw_tags, list):
        tags.extend(_normalize_text(tag) for tag in raw_tags if isinstance(tag, str) and _normalize_text(tag))
    if fixes:
        tags.append("cleaned_deterministically")
    for fix in fixes:
        if fix.startswith("set_geocode_location_as_"):
            tags.append("fixed_route_anchor_side")
            break
    return _ordered_unique(tags)


def _semantic_key(item: dict[str, Any], first_call: dict[str, Any], second_call: dict[str, Any]) -> str:
    conversations = item.get("conversations") or []
    question = _compact(conversations[1].get("value") if len(conversations) > 1 and isinstance(conversations[1], dict) else "")
    first_args = first_call.get("arguments") if isinstance(first_call.get("arguments"), dict) else {}
    second_args = second_call.get("arguments") if isinstance(second_call.get("arguments"), dict) else {}
    payload = {
        "subtype": item.get("subtype"),
        "question": question,
        "geocode": first_args.get("address"),
        "second_tool": second_call.get("name"),
        "mode": second_args.get("mode"),
        "keyword": second_args.get("keyword"),
    }
    return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _candidate_score(subtype: str, second_tool: str, args: dict[str, Any], observation: dict[str, Any], fixes: list[str]) -> int:
    score = 100 - len(set(fixes)) * 3
    data = observation.get("data") if isinstance(observation.get("data"), dict) else {}
    if second_tool == "amap_search_poi":
        pois = data.get("pois")
        if isinstance(pois, list):
            score += min(len(pois), 5) * 4
            if any(_normalize_text(poi.get("distance")) for poi in pois if isinstance(poi, dict)):
                score += 6
        if args.get("radius_m"):
            score += 3
    elif second_tool == "amap_plan_route":
        if isinstance(data.get("duration_min"), (int, float)):
            score += 8
        if isinstance(data.get("distance_km"), (int, float)):
            score += 8
        if args.get("mode") == "transit":
            score += 6
        elif args.get("mode") in {"walking", "bicycling"}:
            score += 4
    if subtype == "geocode_then_route_walking":
        try:
            distance_km = float(data.get("distance_km"))
        except (TypeError, ValueError):
            distance_km = 0.0
        if 0.5 <= distance_km <= 4:
            score += 8
        elif distance_km > 6:
            score -= 10
    return score


def _clean_one(
    source: str,
    line_number: int,
    item: dict[str, Any],
    rejected: Counter[str],
    fixes: Counter[str],
) -> Candidate | None:
    raw_id = _normalize_text(item.get("id"))
    subtype = _normalize_text(item.get("subtype"))
    if item.get("task_type") != "two_step_chain":
        rejected["bad_task_type"] += 1
        return None
    if subtype not in SUBTYPE_TO_CHAIN:
        rejected["bad_subtype"] += 1
        return None
    if item.get("scene") != "two_step_chain":
        rejected["bad_scene"] += 1
        return None
    if item.get("expected_behavior") != "should_call_tool":
        rejected["bad_expected_behavior"] += 1
        return None
    expected_chain = SUBTYPE_TO_CHAIN[subtype]
    if tuple(item.get("expected_tool_chain") or ()) != expected_chain:
        rejected["bad_expected_tool_chain"] += 1
        return None

    conversations, conversation_error = _parse_conversations(item)
    if conversations is None:
        rejected[conversation_error or "bad_conversations"] += 1
        return None
    question = _normalize_question(conversations[1].get("value"))
    if not question or EMOJI_RE.search(question):
        rejected["bad_question"] += 1
        return None

    first_call = _loads_object(_normalize_text(conversations[2].get("value")))
    first_observation = _loads_object(_normalize_text(conversations[3].get("value")))
    second_call = _loads_object(_normalize_text(conversations[4].get("value")))
    second_observation = _loads_object(_normalize_text(conversations[5].get("value")))
    if not first_call or not second_call:
        rejected["bad_function_call_json"] += 1
        return None
    if not first_observation or not second_observation:
        rejected["bad_observation_json"] += 1
        return None
    if (first_call.get("name"), second_call.get("name")) != expected_chain:
        rejected["function_tool_chain_mismatch"] += 1
        return None

    geocode_args = _normalize_arguments("amap_geocode", first_call.get("arguments"))
    if geocode_args is None:
        rejected["bad_geocode_arguments"] += 1
        return None
    geocode_observation = _normalize_success_observation(first_observation)
    if geocode_observation is None:
        rejected["bad_geocode_observation"] += 1
        return None
    location = geocode_observation["data"].get("location")
    if not _is_coordinate(location):
        rejected["bad_geocode_location"] += 1
        return None

    second_tool = expected_chain[1]
    second_args = _normalize_arguments(second_tool, second_call.get("arguments"))
    if second_args is None:
        rejected["bad_second_tool_arguments"] += 1
        return None
    normalized_second_observation = _normalize_success_observation(second_observation)
    if normalized_second_observation is None:
        rejected["bad_second_observation"] += 1
        return None

    fix_list: list[str] = []
    if second_tool == "amap_search_poi":
        normalized_search = _normalize_search_observation(second_args, normalized_second_observation, str(location))
        if normalized_search is None:
            rejected["bad_search_observation"] += 1
            return None
        normalized_second_observation, search_fixes = normalized_search
        fix_list.extend(search_fixes)
        answer = _poi_answer(geocode_args, geocode_observation, second_args, normalized_second_observation)
    else:
        repaired_route_args, route_arg_fixes = _repair_route_args(
            question, geocode_args, geocode_observation, second_args
        )
        if repaired_route_args is None:
            rejected["route_anchor_side_unrepairable"] += 1
            return None
        second_args = repaired_route_args
        if _same_nonempty_text(second_args.get("origin"), second_args.get("destination")):
            rejected["route_same_origin_destination"] += 1
            return None
        fix_list.extend(route_arg_fixes)
        normalized_second_observation, route_observation_fixes = _sync_route_observation(
            second_args, normalized_second_observation
        )
        fix_list.extend(route_observation_fixes)
        if _is_implausible_walking_route(second_args, normalized_second_observation):
            rejected["implausible_walking_route"] += 1
            return None
        answer = _route_answer(question, geocode_args, geocode_observation, second_args, normalized_second_observation)

    if answer is None:
        rejected["answer_rewrite_failed"] += 1
        return None
    answer = _normalize_text(answer)
    if not _answer_is_clean(answer):
        rejected["rewritten_answer_not_clean"] += 1
        return None

    expected_subset = {
        "amap_geocode": dict(geocode_args),
        second_tool: dict(second_args),
    }
    if item.get("expected_arguments_subset") != expected_subset:
        fix_list.append("fixed_expected_arguments_subset")
    for fix in set(fix_list):
        fixes[fix] += 1

    cleaned_item = {
        "id": raw_id,
        "task_type": "two_step_chain",
        "subtype": subtype,
        "scene": "two_step_chain",
        "expected_behavior": "should_call_tool",
        "expected_tool_chain": list(expected_chain),
        "expected_arguments_subset": expected_subset,
        "quality_tags": _quality_tags(subtype, item.get("quality_tags"), fix_list),
        "conversations": [
            {"from": "system", "value": TRIPAI_TOOL_USE_SYSTEM_PROMPT},
            {"from": "human", "value": question},
            {"from": "function_call", "value": _json_dumps({"name": "amap_geocode", "arguments": geocode_args})},
            {"from": "observation", "value": _json_dumps(geocode_observation)},
            {"from": "function_call", "value": _json_dumps({"name": second_tool, "arguments": second_args})},
            {"from": "observation", "value": _json_dumps(normalized_second_observation)},
            {"from": "gpt", "value": answer},
        ],
    }
    semantic_key = _semantic_key(item, first_call, second_call)
    score = _candidate_score(subtype, second_tool, second_args, normalized_second_observation, fix_list)
    if source.endswith("two_step_chain_v2_repaired.jsonl"):
        score += 2
    return Candidate(
        source=source,
        line_number=line_number,
        raw_id=raw_id,
        subtype=subtype,
        score=score,
        semantic_key=semantic_key,
        item=cleaned_item,
        fixes=tuple(sorted(set(fix_list))),
    )


def _dedupe_candidates(candidates: list[Candidate], rejected: Counter[str]) -> list[Candidate]:
    best_by_key: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best_by_key.get(candidate.semantic_key)
        if current is None or (candidate.score, candidate.source.endswith("two_step_chain_v2_repaired.jsonl")) > (
            current.score,
            current.source.endswith("two_step_chain_v2_repaired.jsonl"),
        ):
            if current is not None:
                rejected["duplicate_semantic_candidate"] += 1
            best_by_key[candidate.semantic_key] = candidate
        else:
            rejected["duplicate_semantic_candidate"] += 1
    return list(best_by_key.values())


def _renumber_selected(rows_by_subtype: dict[str, list[Candidate]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for subtype in SUBTYPE_ORDER:
        for index, candidate in enumerate(rows_by_subtype[subtype], start=1):
            item = dict(candidate.item)
            item["id"] = f"stage2_v2_two_step_chain_{subtype}_{index:06d}"
            cleaned.append(item)
    return cleaned


def _to_source_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = build_amap_tool_schemas()
    source_rows: list[dict[str, Any]] = []
    for row in rows:
        messages_with_answer: list[dict[str, Any]] = []
        call_index = 0
        current_call_id = ""
        for message in row["conversations"]:
            role = message["from"]
            value = message["value"]
            if role == "system":
                messages_with_answer.append({"role": "system", "content": value})
            elif role == "human":
                messages_with_answer.append({"role": "user", "content": value})
            elif role == "function_call":
                call_index += 1
                current_call_id = f"call_{call_index:03d}"
                payload = json.loads(value)
                messages_with_answer.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": current_call_id,
                                "type": "function",
                                "function": {
                                    "name": payload["name"],
                                    "arguments": _json_dumps(payload["arguments"]),
                                },
                            }
                        ],
                    }
                )
            elif role == "observation":
                messages_with_answer.append({"role": "tool", "tool_call_id": current_call_id, "content": value})
            elif role == "gpt":
                messages_with_answer.append({"role": "assistant", "content": value})

        source_rows.append(
            {
                "id": row["id"],
                "task_type": row["task_type"],
                "scene": row["scene"],
                "expected_behavior": row["expected_behavior"],
                "tools": tools,
                "messages": messages_with_answer[:2],
                "messages_with_answer": messages_with_answer,
                "expected_tool_chain": row["expected_tool_chain"],
                "expected_arguments_subset": row["expected_arguments_subset"],
                "quality_tags": row["quality_tags"],
            }
        )
    return source_rows


def _validate_processed_rows(rows: list[dict[str, Any]]) -> list[str]:
    source_dataset = _to_source_dataset(rows)
    errors = validate_tool_use_source_dataset(source_dataset)
    if errors:
        return errors
    sharegpt_dataset = export_tool_use_dataset_to_sharegpt(source_dataset)
    return validate_sharegpt_tool_dataset(sharegpt_dataset)


def clean_two_step_chain(
    rows: list[tuple[str, int, dict[str, Any]]] | list[tuple[int, dict[str, Any]]],
    *,
    target_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = dict(target_counts or TARGET_COUNTS)
    if set(targets) != set(SUBTYPE_ORDER):
        raise ValueError(f"target_counts must contain exactly: {SUBTYPE_ORDER!r}")

    rejected: Counter[str] = Counter()
    fixes: Counter[str] = Counter()
    input_sources: Counter[str] = Counter()
    all_candidates: list[Candidate] = []

    for row in rows:
        if len(row) == 2:
            line_number, item = row  # type: ignore[misc]
            source = "<memory>"
        else:
            source, line_number, item = row  # type: ignore[misc]
        input_sources[str(source)] += 1
        candidate = _clean_one(str(source), int(line_number), item, rejected, fixes)
        if candidate is not None:
            all_candidates.append(candidate)

    deduped = _dedupe_candidates(all_candidates, rejected)
    candidates_by_subtype: dict[str, list[Candidate]] = {subtype: [] for subtype in SUBTYPE_ORDER}
    for candidate in deduped:
        candidates_by_subtype[candidate.subtype].append(candidate)

    selected_by_subtype: dict[str, list[Candidate]] = {}
    shortfall: dict[str, int] = {}
    for subtype in SUBTYPE_ORDER:
        candidates = candidates_by_subtype[subtype]
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.raw_id, candidate.line_number))
        target = targets[subtype]
        selected_by_subtype[subtype] = candidates[:target]
        if len(candidates) < target:
            shortfall[subtype] = target - len(candidates)

    cleaned = _renumber_selected(selected_by_subtype)
    validation_errors = _validate_processed_rows(cleaned)
    if validation_errors:
        raise RuntimeError(f"processed two_step_chain failed validation: {validation_errors[:5]}")

    selected_counts = Counter(item["subtype"] for item in cleaned)
    candidate_counts = {subtype: len(candidates_by_subtype[subtype]) for subtype in SUBTYPE_ORDER}
    raw_candidate_counts = Counter(candidate.subtype for candidate in all_candidates)
    route_modes = Counter()
    tool_counts = Counter()
    source_counts = Counter()
    selected_fix_counts = Counter()
    answer_lengths = [len(item["conversations"][-1]["value"]) for item in cleaned]
    for subtype, candidates in selected_by_subtype.items():
        for candidate in candidates:
            source_counts[candidate.source] += 1
            for fix in candidate.fixes:
                selected_fix_counts[fix] += 1
            final_call = json.loads(candidate.item["conversations"][4]["value"])
            tool_counts[final_call["name"]] += 1
            if final_call["name"] == "amap_plan_route":
                route_modes[final_call["arguments"].get("mode")] += 1

    report = {
        "input_records": len(rows),
        "input_sources": dict(sorted(input_sources.items())),
        "target_counts": targets,
        "target_total": sum(targets.values()),
        "raw_candidate_count": len(all_candidates),
        "raw_candidate_counts": dict(sorted(raw_candidate_counts.items())),
        "candidate_count_after_semantic_dedupe": len(deduped),
        "candidate_counts_after_semantic_dedupe": candidate_counts,
        "selected_count": len(cleaned),
        "selected_counts": dict(sorted(selected_counts.items())),
        "shortfall": shortfall,
        "selected_source_counts": dict(sorted(source_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "route_mode_counts": dict(sorted(route_modes.items())),
        "rejected": dict(sorted(rejected.items())),
        "fixes_all_candidates": dict(sorted(fixes.items())),
        "fixes_selected": dict(sorted(selected_fix_counts.items())),
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
        "complete_target_met": len(cleaned) == sum(targets.values()) and not shortfall,
    }
    return cleaned, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean raw Stage2 two_step_chain JSONL into processed Stage2 JSONL.")
    parser.add_argument("--input", type=Path, action="append", dest="inputs", help="Raw JSONL input. Can repeat.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    input_paths = args.inputs or DEFAULT_INPUTS
    rows = _read_jsonl_inputs(input_paths)
    cleaned, report = clean_two_step_chain(rows)
    output_path = _write_jsonl(args.output, cleaned)
    report_path = _write_json(args.report, report)
    print(f"[OK] cleaned two_step_chain written to: {output_path}")
    print(f"[OK] report written to: {report_path}")
    print(f"[OK] selected: {report['selected_count']} / {report['target_total']}")
    if report["shortfall"]:
        print(f"[WARN] shortfall: {json.dumps(report['shortfall'], ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
