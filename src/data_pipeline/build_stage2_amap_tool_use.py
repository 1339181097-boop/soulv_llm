from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    load_records,
    log_error,
    log_info,
    log_success,
    resolve_path,
    write_json,
)
from src.tool_use.datasets import (
    export_tool_use_dataset_to_sharegpt,
    validate_sharegpt_tool_dataset,
    validate_tool_use_source_dataset,
)
from src.tool_use.protocol import (
    TRIPAI_TOOL_USE_SYSTEM_PROMPT,
    build_amap_tool_schemas,
    build_tool_empty,
    build_tool_error,
    build_tool_success,
)

DEFAULT_SOURCE_OUTPUT = "data/tool_use/stage2_amap_tool_use_source.json"
DEFAULT_EXPORT_OUTPUT = "data/final/stage2_amap_tool_use_sft.json"
DEFAULT_REPORT_OUTPUT = "data/final/stage2_amap_tool_use_report.json"
DEFAULT_TRAFFIC_INPUT = "data/processed/sft_traffic_planning_strict_round2_final.jsonl"
DEFAULT_HOTEL_INPUT = "data/processed/sft_hotel_recommendation_0423_strict.jsonl"
DEFAULT_TRAVEL_INPUT = "data/processed/sft_travel_qa_2026_04_22_strict.jsonl"
DEFAULT_TOTAL_SAMPLES = 3200
DEFAULT_SEED = 42

TARGET_RATIOS = {
    "single_tool_call": 0.20,
    "slot_filling_tool_call": 0.18,
    "clarify_then_call": 0.18,
    "tool_result_grounded_answer": 0.22,
    "no_tool_needed": 0.12,
    "tool_failure_fallback": 0.10,
}

BUCKET_SUBPOOL_RATIOS = {
    "single_tool_call": {"route": 0.38, "geocode": 0.22, "poi": 0.40},
    "slot_filling_tool_call": {"route": 0.50, "poi": 0.50},
    "clarify_then_call": {"route": 0.55, "poi": 0.45},
    "tool_result_grounded_answer": {"route_chain": 0.45, "poi_chain": 0.55},
    "no_tool_needed": {"safe_travel_qa": 1.0},
    "tool_failure_fallback": {"route_failure": 0.45, "poi_failure": 0.40, "geocode_failure": 0.15},
}

ROUTE_MODE_CANDIDATES = ("transit", "driving", "walking", "bicycling")
ROUTE_MODE_LABELS = {
    "transit": "公共交通",
    "driving": "驾车",
    "walking": "步行",
    "bicycling": "骑行",
}
POI_CATEGORY_CONFIGS = {
    "hotel": {"keyword": "酒店", "friendly": "住一晚的酒店", "radius_m": 3000},
    "restaurant": {"keyword": "餐厅", "friendly": "吃饭的地方", "radius_m": 1500},
    "subway": {"keyword": "地铁站", "friendly": "最近的地铁站", "radius_m": 1800},
    "mall": {"keyword": "商场", "friendly": "顺路逛逛的商场", "radius_m": 3000},
    "parking": {"keyword": "停车场", "friendly": "方便停车的地方", "radius_m": 1500},
    "spot": {"keyword": "景点", "friendly": "还能顺路去的景点", "radius_m": 5000},
}
SAFE_NO_TOOL_EXCLUDED_QUESTION_TYPES = {"位置交通"}
SAFE_NO_TOOL_KEYWORDS = (
    "怎么去",
    "怎么走",
    "最方便",
    "路线",
    "路程",
    "打车",
    "公交",
    "地铁",
    "乘车",
    "坐什么车",
    "住哪",
    "住哪个",
    "住哪里",
    "住宿",
    "酒店",
    "停车",
    "附近有什么",
    "附近有啥",
    "具体位置",
    "位置在哪",
    "在哪里",
    "查路线",
)
TRANSIT_LINES = ("地铁1号线", "地铁2号线", "地铁4号线", "地铁环线", "公交快线", "旅游专线", "机场快线")
ROADS = ("城市快速路", "机场高速", "内环高架", "滨河大道", "景区连接线", "主干道")


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _stable_hash(*parts: Any) -> str:
    material = "||".join(str(part) for part in parts)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = _stable_hash(*parts)[:12]
    return f"{prefix}_{digest}"


def _stable_int(*parts: Any, low: int, high: int) -> int:
    if high < low:
        raise ValueError("high must be >= low")
    span = high - low + 1
    return low + (int(_stable_hash(*parts)[:12], 16) % span)


def _stable_choice(options: tuple[str, ...] | list[str], *parts: Any) -> str:
    if not options:
        raise ValueError("options must be non-empty")
    index = _stable_int(*parts, low=0, high=len(options) - 1)
    return options[index]


def _stable_location(*parts: Any) -> str:
    digest = hashlib.md5("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    lon = 73.0 + (int(digest[:8], 16) % 6100000) / 100000.0
    lat = 18.0 + (int(digest[8:16], 16) % 3500000) / 100000.0
    return f"{lon:.6f},{lat:.6f}"


def _build_tools() -> list[dict[str, Any]]:
    return build_amap_tool_schemas()


def _build_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": _json_dumps(arguments)},
    }


def _build_tool_message(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": _json_dumps(payload)}


def _sample_with_oversampling(records: list[dict[str, Any]], target_count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not records or target_count <= 0:
        return []
    if len(records) >= target_count:
        return rng.sample(records, target_count)

    sampled = records[:]
    while len(sampled) < target_count:
        sampled.extend(rng.sample(records, min(len(records), target_count - len(sampled))))
    return sampled[:target_count]


def _compute_target_counts(total_samples: int) -> dict[str, int]:
    exact = {name: total_samples * ratio for name, ratio in TARGET_RATIOS.items()}
    base = {name: int(value) for name, value in exact.items()}
    remainder = total_samples - sum(base.values())
    ranked = sorted(exact.items(), key=lambda item: item[1] - int(item[1]), reverse=True)
    for name, _ in ranked[:remainder]:
        base[name] += 1
    return base


def _compute_ratio_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    exact = {name: total * ratio for name, ratio in ratios.items()}
    base = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(base.values())
    ranked = sorted(exact.items(), key=lambda item: item[1] - int(item[1]), reverse=True)
    for name, _ in ranked[:remainder]:
        base[name] += 1
    return base


def _first_user_message(record: dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _last_assistant_message(record: dict[str, Any]) -> str:
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _record_city(record: dict[str, Any]) -> str:
    return str(record.get("city") or "")


def _record_anchor(record: dict[str, Any]) -> str:
    for field in ("district", "entity_name", "hotel_name", "destination", "origin", "city"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "目的地附近"


def _short_place(name: str) -> str:
    cleaned = (
        name.replace("风景名胜区", "")
        .replace("风景区", "")
        .replace("景区", "")
        .replace("博物馆", "")
        .replace("公园", "")
        .replace("旅游区", "")
        .replace("(", "")
        .replace(")", "")
        .replace("（", "")
        .replace("）", "")
        .strip()
    )
    if not cleaned:
        return name[:6]
    return cleaned[:8]


def _make_sample(
    *,
    sample_id: str,
    task_type: str,
    scene: str,
    difficulty: str,
    source_record_id: str,
    expected_behavior: str,
    messages: list[dict[str, Any]],
    messages_with_answer: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "task_type": task_type,
        "scene": scene,
        "difficulty": difficulty,
        "source_record_id": source_record_id,
        "tools": _build_tools(),
        "expected_behavior": expected_behavior,
        "messages": messages,
        "messages_with_answer": messages_with_answer,
    }


def _allowed_route_modes(record: dict[str, Any]) -> tuple[str, ...]:
    scene = str(record.get("scene") or "")
    if scene in {"airport_to_city", "train_to_city"}:
        return ("transit", "driving")
    if scene == "special_audience":
        return ("transit", "driving", "walking")
    return ROUTE_MODE_CANDIDATES


def _select_route_mode(record: dict[str, Any], bucket: str) -> str:
    return _stable_choice(_allowed_route_modes(record), record.get("id"), bucket, "route_mode")


def _route_query(origin: str, destination: str, mode: str, tag: str) -> str:
    mode_templates = {
        "transit": (
            "我从{origin}出发，想去{destination}，优先公共交通，帮我查一下路线。",
            "从{origin}到{destination}，麻烦按公交地铁方案帮我看看怎么走。",
            "我要去{destination}，现在在{origin}，想坐公共交通，给我规划下路线。",
        ),
        "driving": (
            "我准备从{origin}去{destination}，按驾车路线帮我查一下。",
            "从{origin}开车到{destination}怎么走更顺？",
            "帮我看一下从{origin}到{destination}的驾车路线。",
        ),
        "walking": (
            "我想从{origin}步行去{destination}，帮我看看路线。",
            "从{origin}走到{destination}大概怎么走？",
            "帮我查一下从{origin}到{destination}的步行方案。",
        ),
        "bicycling": (
            "我准备从{origin}骑行去{destination}，帮我规划一下路线。",
            "从{origin}骑车到{destination}怎么走比较顺？",
            "帮我看看从{origin}到{destination}的骑行路线。",
        ),
    }
    template = _stable_choice(mode_templates[mode], origin, destination, mode, tag, "route_query")
    return template.format(origin=origin, destination=destination)


def _route_slot_query(origin: str, destination: str, mode: str, tag: str) -> str:
    mode_templates = {
        "transit": (
            "{origin}过去{destination}，想少折腾点，帮我看看公交地铁怎么走。",
            "我现在在{origin}，去{destination}，优先地铁公交。",
        ),
        "driving": (
            "{origin}到{destination}我准备开车过去，帮我看路线。",
            "我从{origin}出发，去{destination}打算自驾，怎么走顺一点？",
        ),
        "walking": (
            "{origin}走到{destination}会怎么走？",
            "从{origin}步行去{destination}，帮我看一眼路线。",
        ),
        "bicycling": (
            "{origin}骑车到{destination}怎么走不容易绕？",
            "我想从{origin}骑行去{destination}，帮我查路线。",
        ),
    }
    template = _stable_choice(mode_templates[mode], origin, destination, mode, tag, "route_slot_query")
    return template.format(origin=origin, destination=destination)


def _route_clarify_scenario(record: dict[str, Any]) -> str:
    scenarios = ("missing_origin", "missing_mode", "missing_destination", "missing_city")
    return _stable_choice(scenarios, record.get("id"), "route_clarify_scenario")


def _route_clarify_opening(origin: str, destination: str, city: str, mode: str, scenario: str) -> str:
    mode_hint = ROUTE_MODE_LABELS[mode]
    if scenario == "missing_origin":
        return _stable_choice(
            ("帮我规划一下去{destination}的{mode_hint}路线。", "我想去{destination}，帮我查一下{mode_hint}怎么走。"),
            origin,
            destination,
            city,
            mode,
            scenario,
        ).format(destination=destination, mode_hint=mode_hint)
    if scenario == "missing_mode":
        return _stable_choice(
            ("我从{origin}去{destination}，帮我看一下怎么走。", "{origin}到{destination}怎么走比较合适？"),
            origin,
            destination,
            city,
            mode,
            scenario,
        ).format(origin=origin, destination=destination)
    if scenario == "missing_destination":
        return _stable_choice(
            ("我从{origin}出发，想查一下{mode_hint}路线。", "我现在在{origin}，帮我规划个{mode_hint}方案。"),
            origin,
            destination,
            city,
            mode,
            scenario,
        ).format(origin=origin, mode_hint=mode_hint)
    return _stable_choice(
        ("我从{origin}去{destination}，想查{mode_hint}路线。", "帮我看一下从{origin}到{destination}的{mode_hint}方案。"),
        origin,
        destination,
        city,
        mode,
        scenario,
    ).format(origin=origin, destination=destination, mode_hint=mode_hint)


def _route_clarify_prompt(scenario: str, mode: str, record: dict[str, Any]) -> str:
    prompt_map = {
        "missing_origin": (
            "可以，我先帮你查，但还缺出发地。你现在从哪里走，城市是哪座？",
            "没问题，不过我还需要你的出发点和所在城市，这样路线才查得准。",
            "能帮你规划，先补一下起点吧。你现在从哪里出发，在哪个城市？",
        ),
        "missing_mode": (
            "可以查路线。你更想走公共交通、驾车、步行还是骑行？如果方便，也补一下所在城市。",
            "我先确认一下出行方式：你是想坐公共交通、开车、步行还是骑行？城市也告诉我一下。",
            "路线可以查，不过我还缺偏好的出行方式。你想按公共交通、驾车、步行还是骑行来查？",
        ),
        "missing_destination": (
            "可以，我先补个关键信息：你准备去哪里？如果有偏好的出行方式也一起告诉我。",
            "没问题，不过目的地还没给我。你要去哪里，顺便说下想按什么方式走？",
            "能帮你规划，但还差目的地。你想去哪里，优先公共交通、驾车、步行还是骑行？",
        ),
        "missing_city": (
            "可以查，不过我还需要确认城市名，避免同名地点查偏。你所在或要查的是哪座城市？",
            "先补一下城市吧，同名地点不少。你这是在哪座城市查路线？",
            "能帮你看路线，但城市还没说。补一个城市名，我就继续往下查。",
        ),
    }
    return _stable_choice(prompt_map[scenario], record.get("id"), scenario, mode, "route_clarify_prompt")


def _route_clarify_followup(origin: str, destination: str, city: str, mode: str, scenario: str) -> str:
    mode_hint = ROUTE_MODE_LABELS[mode]
    followup_map = {
        "missing_origin": ("我从{origin}出发，在{city}，优先{mode_hint}。", "起点是{origin}，城市是{city}，想按{mode_hint}来查。"),
        "missing_mode": ("我从{origin}去{destination}，城市是{city}，优先{mode_hint}。", "{city}这边，从{origin}到{destination}，想按{mode_hint}来查。"),
        "missing_destination": ("我从{origin}出发，要去{destination}，城市是{city}，优先{mode_hint}。", "目的地是{destination}，我在{origin}，{city}这边，想按{mode_hint}查。"),
        "missing_city": ("城市是{city}，从{origin}到{destination}，优先{mode_hint}。", "在{city}，我想从{origin}去{destination}，按{mode_hint}来查。"),
    }
    template = _stable_choice(followup_map[scenario], origin, destination, city, mode, scenario, "route_followup")
    return template.format(origin=origin, destination=destination, city=city, mode_hint=mode_hint)


def _build_route_snapshot(
    origin: str,
    destination: str,
    city: str,
    mode: str,
    *,
    origin_display: str | None = None,
    destination_display: str | None = None,
) -> dict[str, Any]:
    origin_display = origin_display or origin
    destination_display = destination_display or destination
    distance_km = _stable_int(city, origin, destination, mode, "distance_km_tenths", low=8, high=280) / 10.0

    if mode == "transit":
        walk1 = _stable_int(city, origin, destination, mode, "walk1", low=120, high=520)
        walk2 = _stable_int(city, origin, destination, mode, "walk2", low=150, high=680)
        walk_total = walk1 + walk2
        transfer_count = _stable_int(city, origin, destination, mode, "transfer_count", low=0, high=2)
        line1 = _stable_choice(TRANSIT_LINES, city, origin, destination, mode, "line1")
        line2_candidates = [line for line in TRANSIT_LINES if line != line1]
        line2 = _stable_choice(line2_candidates, city, origin, destination, mode, "line2")
        start_stop = f"{_short_place(origin_display)}上车点"
        transfer_stop = f"{_short_place(destination_display)}换乘站"
        duration_min = int(distance_km * 2.2) + _stable_int(city, origin, destination, mode, "duration_bias", low=8, high=20)
        summary_tail = "无需换乘" if transfer_count == 0 else f"换乘{transfer_count}次"
        steps = [
            {"segment_type": "walk", "instruction": f"先从{origin_display}步行约{walk1}米到{start_stop}"},
            {"segment_type": "transit", "instruction": f"乘坐{line1}前往{transfer_stop if transfer_count else f'{destination_display}附近站点'}"},
        ]
        if transfer_count:
            steps.append({"segment_type": "transfer", "instruction": f"在{transfer_stop}换乘{line2}，继续往{destination_display}方向走"})
        steps.append({"segment_type": "walk", "instruction": f"下车后步行约{walk2}米到{destination_display}"})
        tips = [
            f"这条线更稳的是前段{line1}，高峰期给换乘多留几分钟。",
            f"临近{destination_display}时按站内指引走，最后一段步行会更顺。",
        ]
        extra = {
            "walk_distance_m": walk_total,
            "transfer_count": transfer_count,
            "estimated_cost_cny": _stable_int(city, origin, destination, mode, "cost", low=2, high=9),
        }
        summary = f"全程约{duration_min}分钟，步行约{walk_total}米，{summary_tail}"
    elif mode == "driving":
        duration_min = int(distance_km * 1.7) + _stable_int(city, origin, destination, mode, "duration_bias", low=4, high=14)
        road1 = _stable_choice(ROADS, city, origin, destination, mode, "road1")
        road2_candidates = [road for road in ROADS if road != road1]
        road2 = _stable_choice(road2_candidates, city, origin, destination, mode, "road2")
        parking_name = f"{_short_place(destination_display)}停车点"
        steps = [
            {"segment_type": "drive", "instruction": f"从{origin_display}出发后先驶入{road1}"},
            {"segment_type": "drive", "instruction": f"随后接入{road2}，朝{destination_display}方向继续"},
            {"segment_type": "drive", "instruction": f"接近目的地后留意{parking_name}周边道路指引"},
        ]
        tips = [
            f"{road1}入口段在高峰期容易放慢，早点并线会更稳。",
            f"如果你要就近停车，到{destination_display}前可以先留意{parking_name}。",
        ]
        extra = {"distance_km": distance_km, "estimated_toll_cny": _stable_int(city, origin, destination, mode, "toll", low=0, high=22)}
        summary = f"全程约{duration_min}分钟，距离约{distance_km:.1f}公里"
    elif mode == "walking":
        distance_km = min(distance_km, 6.2)
        duration_min = int(distance_km * 13) + _stable_int(city, origin, destination, mode, "duration_bias", low=2, high=8)
        waypoint = f"{_short_place(origin_display)}主路口"
        steps = [
            {"segment_type": "walk", "instruction": f"从{origin_display}出发后沿主路步行约{distance_km / 2:.1f}公里"},
            {"segment_type": "walk", "instruction": f"经过{waypoint}后继续按{destination_display}方向步行"},
            {"segment_type": "walk", "instruction": f"最后一段根据路边指示步行进入{destination_display}"},
        ]
        tips = [
            "这条路线步行段比较完整，建议穿舒适一点的鞋。",
            f"靠近{destination_display}前的最后几百米注意看指引牌就行。",
        ]
        extra = {"distance_km": distance_km}
        summary = f"全程步行约{duration_min}分钟，距离约{distance_km:.1f}公里"
    else:
        distance_km = min(distance_km, 12.5)
        duration_min = int(distance_km * 4.4) + _stable_int(city, origin, destination, mode, "duration_bias", low=2, high=6)
        bike_lane = f"{_short_place(destination_display)}骑行道"
        steps = [
            {"segment_type": "bicycling", "instruction": f"从{origin_display}出发后先沿城市慢行道骑行"},
            {"segment_type": "bicycling", "instruction": f"中段接上{bike_lane}，继续朝{destination_display}方向走"},
            {"segment_type": "bicycling", "instruction": f"接近目的地后按骑行指示进入{destination_display}周边道路"},
        ]
        tips = [
            "这条线骑行段比较连贯，通勤和景区间短接都合适。",
            f"靠近{destination_display}时注意和步行人流错开就行。",
        ]
        extra = {"distance_km": distance_km}
        summary = f"全程骑行约{duration_min}分钟，距离约{distance_km:.1f}公里"

    data = {
        "mode": mode,
        "city": city,
        "origin": origin,
        "destination": destination,
        "origin_display": origin_display,
        "destination_display": destination_display,
        "origin_location": _stable_location(city, origin, "origin"),
        "destination_location": _stable_location(city, destination, "destination"),
        "summary": summary,
        "route_steps": steps,
        "transfer_tips": tips,
        "duration_min": duration_min,
        **extra,
    }
    return build_tool_success(data)


def _route_answer_from_snapshot(snapshot: dict[str, Any]) -> str:
    data = snapshot["data"]
    steps = data.get("route_steps", [])
    tips = data.get("transfer_tips", [])
    step_sentence = "；".join(step["instruction"] for step in steps[:3] if isinstance(step.get("instruction"), str))
    tip_sentence = tips[0] if tips else ""
    mode_label = ROUTE_MODE_LABELS.get(data.get("mode"), "路线")
    return f"查到的{mode_label}方案是：{data['summary']}。可以按这几步走：{step_sentence}。{tip_sentence}"


def _grounded_route_destination(city: str, mode: str, record: dict[str, Any]) -> str:
    options_map = {
        "transit": (f"{city}火车站", f"{city}高铁站", f"{city}市中心"),
        "driving": (f"{city}高铁站", f"{city}机场", f"{city}会展中心"),
        "walking": (f"{city}游客中心", f"{city}老城区", f"{city}地铁站"),
        "bicycling": (f"{city}老城区", f"{city}滨江步道", f"{city}城市公园"),
    }
    return _stable_choice(options_map[mode], record.get("id"), city, mode, "grounded_route_destination")


def _poi_categories_for_record(record: dict[str, Any]) -> tuple[str, ...]:
    task_type = str(record.get("task_type") or "")
    if task_type == "hotel_recommendation":
        return ("hotel", "restaurant", "subway", "mall")
    return ("restaurant", "subway", "parking", "spot")


def _select_poi_category(record: dict[str, Any], bucket: str) -> str:
    return _stable_choice(_poi_categories_for_record(record), record.get("id"), bucket, "poi_category")


def _poi_keyword(category: str) -> str:
    return POI_CATEGORY_CONFIGS[category]["keyword"]


def _poi_friendly(category: str) -> str:
    return POI_CATEGORY_CONFIGS[category]["friendly"]


def _poi_radius(category: str) -> int:
    return int(POI_CATEGORY_CONFIGS[category]["radius_m"])


def _poi_name_candidates(category: str, anchor_label: str) -> list[str]:
    base = _short_place(anchor_label or "中心")
    if category == "hotel":
        return [f"{base}轻居酒店", f"{base}精选酒店", f"{base}城市客栈", f"{base}商务酒店", f"{base}驿站"]
    if category == "restaurant":
        return [f"{base}小馆", f"{base}食集", f"{base}饭堂", f"{base}风味餐厅", f"{base}面馆"]
    if category == "subway":
        return [f"{base}地铁站", f"{base}东站", f"{base}西站", f"{base}中心站"]
    if category == "mall":
        return [f"{base}广场", f"{base}天地", f"{base}里购物中心", f"{base}商场"]
    if category == "parking":
        return [f"{base}停车场", f"{base}P1停车楼", f"{base}社会停车场", f"{base}东侧停车点"]
    return [f"{base}游客中心", f"{base}文化街区", f"{base}公园", f"{base}展览馆", f"{base}观景点"]


def _poi_query(anchor_label: str, category: str, tag: str) -> str:
    if category == "hotel":
        templates = ("帮我找{anchor}附近适合住一晚的酒店。", "查一下{anchor}周边住哪儿方便，先看看酒店。")
    elif category == "restaurant":
        templates = ("帮我找{anchor}附近吃饭方便的餐厅。", "查一下{anchor}周边有什么适合吃饭的地方。")
    elif category == "subway":
        templates = ("查一下{anchor}附近最近的地铁站。", "帮我看看{anchor}周边地铁站在哪。")
    elif category == "mall":
        templates = ("想在{anchor}附近顺路逛逛商场，帮我搜一下。", "帮我找找{anchor}周边适合逛的商场。")
    elif category == "parking":
        templates = ("我开车去{anchor}，帮我看看附近停车场。", "查一下{anchor}周边哪里停车方便。")
    else:
        templates = ("除了{anchor}，附近还有什么景点可以顺路去？", "帮我搜一下{anchor}周边还能逛的景点。")
    template = _stable_choice(templates, anchor_label, category, tag, "poi_query")
    return template.format(anchor=anchor_label)


def _poi_slot_query(anchor_label: str, category: str, tag: str) -> str:
    if category == "hotel":
        templates = ("{anchor}附近住哪儿方便？", "{anchor}这边落脚的话住哪儿合适？")
    elif category == "restaurant":
        templates = ("{anchor}附近想吃饭，帮我看看。", "{anchor}周边想找个吃饭的地方。")
    elif category == "subway":
        templates = ("{anchor}附近怎么上地铁更近？", "{anchor}周边最近的地铁入口在哪？")
    elif category == "mall":
        templates = ("{anchor}附近想逛逛，有没有商场？", "{anchor}周边有什么适合顺路逛的商场？")
    elif category == "parking":
        templates = ("开车去{anchor}的话附近怎么停车？", "{anchor}这边开车过去停哪儿方便？")
    else:
        templates = ("{anchor}附近还能顺路逛点什么？", "{anchor}周边还有没有值得去的地方？")
    template = _stable_choice(templates, anchor_label, category, tag, "poi_slot_query")
    return template.format(anchor=anchor_label)


def _poi_clarify_scenario(record: dict[str, Any]) -> str:
    scenarios = ("missing_anchor", "missing_keyword", "missing_city")
    return _stable_choice(scenarios, record.get("id"), "poi_clarify_scenario")


def _poi_clarify_opening(anchor_label: str, category: str, scenario: str) -> str:
    friendly = _poi_friendly(category)
    if scenario == "missing_anchor":
        templates = ("帮我找一下{friendly}。", "想找个离得近的{friendly}。")
        return _stable_choice(templates, anchor_label, category, scenario, "poi_opening").format(friendly=friendly)
    if scenario == "missing_keyword":
        templates = ("帮我看看{anchor}附近有什么。", "查一下{anchor}周边都有什么。")
        return _stable_choice(templates, anchor_label, category, scenario, "poi_opening").format(anchor=anchor_label)
    templates = ("帮我找{anchor}附近的{friendly}。", "查一下{anchor}周边的{friendly}。")
    return _stable_choice(templates, anchor_label, category, scenario, "poi_opening").format(anchor=anchor_label, friendly=friendly)


def _poi_clarify_prompt(category: str, scenario: str, record: dict[str, Any]) -> str:
    prompt_map = {
        "missing_anchor": (
            "可以，我先帮你找，但还缺一个范围。你想在哪个城市或哪个地点附近找？",
            "能查，不过我还不知道你要围绕哪个地点来找。给我一个城市、商圈或景点吧。",
            "没问题，先补一个具体范围。你想在哪个城市、哪个区域或哪个地标附近找？",
        ),
        "missing_keyword": (
            "可以查，但我先确认一下类型：你想找酒店、餐厅、地铁站、商场、停车场还是景点？",
            "我能帮你搜，不过还差关键词。你想找酒店、吃饭的地方、地铁站还是别的？",
            "先确认下你要找什么类型的点位吧，比如酒店、餐厅、地铁站、商场或停车场。",
        ),
        "missing_city": (
            "可以，不过我还需要城市名，避免同名地点查偏。你要查的是哪座城市？",
            "先补一下城市吧，这样搜周边更稳。你说的是哪座城市？",
            "没问题，但还差城市信息。告诉我城市名，我再继续查。",
        ),
    }
    return _stable_choice(prompt_map[scenario], record.get("id"), category, scenario, "poi_clarify_prompt")


def _poi_clarify_followup(anchor_label: str, city: str, category: str, scenario: str) -> str:
    friendly = _poi_friendly(category)
    followup_map = {
        "missing_anchor": ("在{city}{anchor}附近，我想找{friendly}。", "我想围绕{city}{anchor}来找，先看{friendly}。"),
        "missing_keyword": ("我在{city}{anchor}附近，想找{friendly}。", "{city}{anchor}附近，帮我找{friendly}。"),
        "missing_city": ("城市是{city}，我想找{anchor}附近的{friendly}。", "在{city}，帮我看{anchor}附近的{friendly}。"),
    }
    template = _stable_choice(followup_map[scenario], anchor_label, city, category, scenario, "poi_followup")
    return template.format(anchor=anchor_label, city=city, friendly=friendly)


def _build_poi_snapshot(category: str, city: str, *, anchor_label: str, around_location: str | None = None) -> dict[str, Any]:
    keyword = _poi_keyword(category)
    candidate_names = _poi_name_candidates(category, anchor_label)
    start_index = _stable_int(city, anchor_label, category, "poi_start", low=0, high=len(candidate_names) - 1)
    count = _stable_int(city, anchor_label, category, "poi_count", low=2, high=min(4, len(candidate_names)))
    rotated = candidate_names[start_index:] + candidate_names[:start_index]
    pois: list[dict[str, Any]] = []
    for index, name in enumerate(rotated[:count]):
        distance = _stable_int(city, anchor_label, category, index, "poi_distance", low=180 + index * 120, high=520 + index * 260)
        pois.append(
            {
                "name": name,
                "address": f"{city}{anchor_label}片区{index + 1}号",
                "cityname": city,
                "adname": anchor_label,
                "type": keyword,
                "location": _stable_location(city, anchor_label, category, name, index),
                "distance": str(distance),
            }
        )

    data = {
        "keyword": keyword,
        "city": city,
        "around_label": anchor_label,
        "around_location": around_location or _stable_location(city, anchor_label, category, "center"),
        "count": len(pois),
        "pois": pois,
    }
    return build_tool_success(data)


def _poi_advice(keyword: str) -> str:
    if keyword == "酒店":
        return "如果你是住一晚，先从最近的两家开始比位置和通勤会更高效。"
    if keyword == "餐厅":
        return "如果赶时间，优先看最近的一家会更稳。"
    if keyword == "地铁站":
        return "优先选距离最近的站点，上手最省时间。"
    if keyword == "商场":
        return "如果只是顺路休息或吃饭，前两家就足够先筛一轮。"
    if keyword == "停车场":
        return "开车的话优先看距离近、入口清晰的停车点。"
    return "如果时间有限，先去最近的一个景点会更顺。"


def _poi_answer_from_snapshot(snapshot: dict[str, Any]) -> str:
    data = snapshot["data"]
    pois = data.get("pois", [])
    count = data.get("count", 0)
    if not pois:
        return f"这次在{data.get('around_label') or data.get('city') or '该区域'}附近没有查到合适结果。"
    first = pois[0]
    answer = (
        f"这次在{data.get('around_label') or data.get('city') or '该区域'}附近查到{count}个{data['keyword']}结果。"
        f"离得最近的是{first['name']}，约{first['distance']}米，在{first['address']}。"
    )
    if len(pois) > 1:
        second = pois[1]
        answer += f"其次可以看{second['name']}，约{second['distance']}米。"
    answer += _poi_advice(str(data['keyword']))
    return answer


def _build_geocode_snapshot(address: str, city: str | None) -> dict[str, Any]:
    location = _stable_location(city or "", address)
    data = {
        "query": address,
        "city": city,
        "formatted_address": f"{city or ''}{address}",
        "district": address,
        "location": location,
        "level": "兴趣点",
    }
    return build_tool_success(data)


def _geocode_answer_from_snapshot(snapshot: dict[str, Any]) -> str:
    data = snapshot["data"]
    return f"查到的位置是{data['formatted_address']}，坐标约为{data['location']}。如果你还要继续查路线或周边，我可以接着帮你处理。"


def _route_failure_result(record: dict[str, Any], mode: str) -> dict[str, Any]:
    failure_kind = _stable_choice(("error", "empty"), record.get("id"), mode, "route_failure_kind")
    if failure_kind == "error":
        return build_tool_error(_stable_choice(("amap_request_failed", "route_service_unavailable"), record.get("id"), mode, "route_error_reason"))
    return build_tool_empty(_stable_choice(("route_not_found", "no_feasible_route"), record.get("id"), mode, "route_empty_reason"))


def _route_failure_answer(origin: str, destination: str, mode: str, result: dict[str, Any]) -> str:
    mode_label = ROUTE_MODE_LABELS[mode]
    if result["status"] == "empty":
        return (
            f"这次没查到从{origin}到{destination}的{mode_label}结果，我先不硬给你编路线。"
            "你可以把起终点再说具体一点，或者换一种出行方式，我再继续帮你查。"
        )
    return (
        f"路线工具这次请求失败了，我现在拿不到从{origin}到{destination}的可靠{mode_label}结果。"
        "你可以稍后再试，或者给我更具体的起终点，我换一种方式继续帮你查。"
    )


def _poi_failure_result(record: dict[str, Any], category: str) -> dict[str, Any]:
    failure_kind = _stable_choice(("empty", "error"), record.get("id"), category, "poi_failure_kind")
    if failure_kind == "error":
        return build_tool_error(_stable_choice(("amap_request_failed", "poi_service_busy"), record.get("id"), category, "poi_error_reason"))
    return build_tool_empty(_stable_choice(("no_result", "poi_not_found"), record.get("id"), category, "poi_empty_reason"))


def _poi_failure_answer(anchor_label: str, category: str, result: dict[str, Any]) -> str:
    keyword = _poi_keyword(category)
    if result["status"] == "empty":
        return f"这次在{anchor_label}附近没查到合适的{keyword}结果。你可以换个更具体的关键词，或者把范围放大一点，我再帮你查。"
    return f"周边检索工具这次失败了，我现在拿不到{anchor_label}附近可靠的{keyword}结果，不想硬编。你稍后重试，或者换个地点我继续帮你看。"


def _geocode_failure_result(record: dict[str, Any]) -> dict[str, Any]:
    if _stable_choice(("empty", "error"), record.get("id"), "geocode_failure_kind") == "empty":
        return build_tool_empty("address_not_resolved")
    return build_tool_error("amap_request_failed")


def _geocode_failure_answer(address: str, result: dict[str, Any]) -> str:
    if result["status"] == "empty":
        return f"我暂时没把“{address}”解析成明确位置，先不给你瞎指路。可以补一个城市名、商圈名，或者给我更完整的地址。"
    return f"定位工具这次请求失败了，我现在没法可靠确认“{address}”的位置。你可以稍后重试，或者补充更完整的地址我再查。"


def _travel_no_tool(record: dict[str, Any], _: random.Random | None) -> dict[str, Any]:
    user = _first_user_message(record)
    assistant = _last_assistant_message(record)
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "travel_no_tool"),
        task_type="no_tool_needed",
        scene="travel_qa",
        difficulty="easy",
        source_record_id=record["id"],
        expected_behavior="should_answer_directly",
        messages=messages,
        messages_with_answer=[*messages, {"role": "assistant", "content": assistant}],
    )


def _is_safe_no_tool_record(record: dict[str, Any]) -> bool:
    question_type = str(record.get("question_type") or "")
    if question_type in SAFE_NO_TOOL_EXCLUDED_QUESTION_TYPES:
        return False
    if bool(record.get("is_time_sensitive")):
        return False
    if str(record.get("entity_type") or "") == "traffic":
        return False
    user = _first_user_message(record)
    return not any(keyword in user for keyword in SAFE_NO_TOOL_KEYWORDS)


def _route_single_sample(record: dict[str, Any]) -> dict[str, Any]:
    origin = str(record.get("origin") or "")
    destination = str(record.get("destination") or "")
    city = _record_city(record)
    mode = _select_route_mode(record, "route_single")
    call_id = _stable_id("call", record["id"], "route_single")
    route_result = _build_route_snapshot(origin, destination, city, mode)
    user = _route_query(origin, destination, mode, "single")
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "single_tool_call", "route"),
        task_type="single_tool_call",
        scene="amap_route_planning",
        difficulty="medium",
        source_record_id=record["id"],
        expected_behavior="should_call_tool",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_plan_route", {"origin": origin, "destination": destination, "mode": mode, "city": city})]},
            _build_tool_message(call_id, route_result),
            {"role": "assistant", "content": _route_answer_from_snapshot(route_result)},
        ],
    )


def _route_slot_sample(record: dict[str, Any]) -> dict[str, Any]:
    origin = str(record.get("origin") or "")
    destination = str(record.get("destination") or "")
    city = _record_city(record)
    mode = _select_route_mode(record, "route_slot")
    call_id = _stable_id("call", record["id"], "route_slot")
    route_result = _build_route_snapshot(origin, destination, city, mode)
    user = _route_slot_query(origin, destination, mode, "slot")
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "slot_filling_tool_call", "route"),
        task_type="slot_filling_tool_call",
        scene="amap_route_planning",
        difficulty="medium",
        source_record_id=record["id"],
        expected_behavior="should_call_tool",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_plan_route", {"origin": origin, "destination": destination, "mode": mode, "city": city})]},
            _build_tool_message(call_id, route_result),
            {"role": "assistant", "content": _route_answer_from_snapshot(route_result)},
        ],
    )


def _route_clarify_sample(record: dict[str, Any]) -> dict[str, Any]:
    origin = str(record.get("origin") or "")
    destination = str(record.get("destination") or "")
    city = _record_city(record)
    mode = _select_route_mode(record, "route_clarify")
    scenario = _route_clarify_scenario(record)
    call_id = _stable_id("call", record["id"], "route_clarify")
    route_result = _build_route_snapshot(origin, destination, city, mode)
    initial_user = _route_clarify_opening(origin, destination, city, mode, scenario)
    clarify_text = _route_clarify_prompt(scenario, mode, record)
    followup = _route_clarify_followup(origin, destination, city, mode, scenario)
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": initial_user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "clarify_then_call", "route"),
        task_type="clarify_then_call",
        scene="amap_route_planning",
        difficulty="hard",
        source_record_id=record["id"],
        expected_behavior="should_clarify",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "content": clarify_text},
            {"role": "user", "content": followup},
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_plan_route", {"origin": origin, "destination": destination, "mode": mode, "city": city})]},
            _build_tool_message(call_id, route_result),
            {"role": "assistant", "content": _route_answer_from_snapshot(route_result)},
        ],
    )


def _route_grounded_sample(record: dict[str, Any]) -> dict[str, Any]:
    city = _record_city(record)
    origin_label = _record_anchor(record)
    mode = _select_route_mode(record, "route_grounded")
    destination = _grounded_route_destination(city, mode, record)
    geocode_call_id = _stable_id("call", record["id"], "route_grounded_geocode")
    route_call_id = _stable_id("call", record["id"], "route_grounded_route")
    geocode_result = _build_geocode_snapshot(origin_label, city)
    route_result = _build_route_snapshot(geocode_result["data"]["location"], destination, city, mode, origin_display=origin_label, destination_display=destination)
    user = f"我现在在{origin_label}附近，想去{destination}，帮我看看{ROUTE_MODE_LABELS[mode]}怎么走。"
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "tool_result_grounded_answer", "route"),
        task_type="tool_result_grounded_answer",
        scene="amap_route_chain",
        difficulty="hard",
        source_record_id=record["id"],
        expected_behavior="should_call_tool",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(geocode_call_id, "amap_geocode", {"address": origin_label, "city": city})]},
            _build_tool_message(geocode_call_id, geocode_result),
            {"role": "assistant", "tool_calls": [_build_tool_call(route_call_id, "amap_plan_route", {"origin": geocode_result["data"]["location"], "destination": destination, "mode": mode, "city": city})]},
            _build_tool_message(route_call_id, route_result),
            {"role": "assistant", "content": _route_answer_from_snapshot(route_result)},
        ],
    )


def _route_failure_sample(record: dict[str, Any]) -> dict[str, Any]:
    origin = str(record.get("origin") or "")
    destination = str(record.get("destination") or "")
    city = _record_city(record)
    mode = _select_route_mode(record, "route_failure")
    call_id = _stable_id("call", record["id"], "route_failure")
    failure_result = _route_failure_result(record, mode)
    user = _route_query(origin, destination, mode, "failure")
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "tool_failure_fallback", "route"),
        task_type="tool_failure_fallback",
        scene="amap_route_planning",
        difficulty="medium",
        source_record_id=record["id"],
        expected_behavior="should_fallback",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_plan_route", {"origin": origin, "destination": destination, "mode": mode, "city": city})]},
            _build_tool_message(call_id, failure_result),
            {"role": "assistant", "content": _route_failure_answer(origin, destination, mode, failure_result)},
        ],
    )


def _travel_geocode_sample(record: dict[str, Any]) -> dict[str, Any]:
    entity_name = str(record.get("entity_name") or "")
    city = _record_city(record)
    call_id = _stable_id("call", record["id"], "geocode_single")
    geocode_result = _build_geocode_snapshot(entity_name, city)
    user = _stable_choice(
        (f"{entity_name}具体在{city}哪一带？", f"帮我确认一下{entity_name}的位置。", f"{entity_name}在哪里？我想先知道大概位置。"),
        record.get("id"),
        city,
        entity_name,
        "geocode_query",
    )
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "single_tool_call", "geocode"),
        task_type="single_tool_call",
        scene="amap_geocode",
        difficulty="easy",
        source_record_id=record["id"],
        expected_behavior="should_call_tool",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_geocode", {"address": entity_name, "city": city})]},
            _build_tool_message(call_id, geocode_result),
            {"role": "assistant", "content": _geocode_answer_from_snapshot(geocode_result)},
        ],
    )


def _poi_single_sample(record: dict[str, Any]) -> dict[str, Any]:
    city = _record_city(record)
    anchor_label = _record_anchor(record)
    category = _select_poi_category(record, "poi_single")
    call_id = _stable_id("call", record["id"], "poi_single")
    poi_result = _build_poi_snapshot(category, city, anchor_label=anchor_label)
    user = _poi_query(anchor_label, category, "single")
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "single_tool_call", "poi"),
        task_type="single_tool_call",
        scene="amap_poi_search",
        difficulty="medium",
        source_record_id=record["id"],
        expected_behavior="should_call_tool",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_search_poi", {"keyword": _poi_keyword(category), "city": city, "around_location": anchor_label, "radius_m": _poi_radius(category)})]},
            _build_tool_message(call_id, poi_result),
            {"role": "assistant", "content": _poi_answer_from_snapshot(poi_result)},
        ],
    )


def _poi_slot_sample(record: dict[str, Any]) -> dict[str, Any]:
    city = _record_city(record)
    anchor_label = _record_anchor(record)
    category = _select_poi_category(record, "poi_slot")
    call_id = _stable_id("call", record["id"], "poi_slot")
    poi_result = _build_poi_snapshot(category, city, anchor_label=anchor_label)
    user = _poi_slot_query(anchor_label, category, "slot")
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "slot_filling_tool_call", "poi"),
        task_type="slot_filling_tool_call",
        scene="amap_poi_search",
        difficulty="medium",
        source_record_id=record["id"],
        expected_behavior="should_call_tool",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_search_poi", {"keyword": _poi_keyword(category), "city": city, "around_location": anchor_label, "radius_m": _poi_radius(category)})]},
            _build_tool_message(call_id, poi_result),
            {"role": "assistant", "content": _poi_answer_from_snapshot(poi_result)},
        ],
    )


def _poi_clarify_sample(record: dict[str, Any]) -> dict[str, Any]:
    city = _record_city(record)
    anchor_label = _record_anchor(record)
    category = _select_poi_category(record, "poi_clarify")
    scenario = _poi_clarify_scenario(record)
    call_id = _stable_id("call", record["id"], "poi_clarify")
    poi_result = _build_poi_snapshot(category, city, anchor_label=anchor_label)
    initial_user = _poi_clarify_opening(anchor_label, category, scenario)
    clarify_text = _poi_clarify_prompt(category, scenario, record)
    followup = _poi_clarify_followup(anchor_label, city, category, scenario)
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": initial_user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "clarify_then_call", "poi"),
        task_type="clarify_then_call",
        scene="amap_poi_search",
        difficulty="hard",
        source_record_id=record["id"],
        expected_behavior="should_clarify",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "content": clarify_text},
            {"role": "user", "content": followup},
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_search_poi", {"keyword": _poi_keyword(category), "city": city, "around_location": anchor_label, "radius_m": _poi_radius(category)})]},
            _build_tool_message(call_id, poi_result),
            {"role": "assistant", "content": _poi_answer_from_snapshot(poi_result)},
        ],
    )


def _poi_grounded_sample(record: dict[str, Any]) -> dict[str, Any]:
    city = _record_city(record)
    anchor_label = _record_anchor(record)
    category = _select_poi_category(record, "poi_grounded")
    geocode_call_id = _stable_id("call", record["id"], "poi_grounded_geocode")
    search_call_id = _stable_id("call", record["id"], "poi_grounded_search")
    geocode_result = _build_geocode_snapshot(anchor_label, city)
    poi_result = _build_poi_snapshot(category, city, anchor_label=anchor_label, around_location=geocode_result["data"]["location"])
    user = _poi_query(anchor_label, category, "grounded")
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "tool_result_grounded_answer", "poi"),
        task_type="tool_result_grounded_answer",
        scene="amap_poi_chain",
        difficulty="hard",
        source_record_id=record["id"],
        expected_behavior="should_call_tool",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(geocode_call_id, "amap_geocode", {"address": anchor_label, "city": city})]},
            _build_tool_message(geocode_call_id, geocode_result),
            {"role": "assistant", "tool_calls": [_build_tool_call(search_call_id, "amap_search_poi", {"keyword": _poi_keyword(category), "city": city, "around_location": geocode_result["data"]["location"], "radius_m": _poi_radius(category)})]},
            _build_tool_message(search_call_id, poi_result),
            {"role": "assistant", "content": _poi_answer_from_snapshot(poi_result)},
        ],
    )


def _poi_failure_sample(record: dict[str, Any]) -> dict[str, Any]:
    city = _record_city(record)
    anchor_label = _record_anchor(record)
    category = _select_poi_category(record, "poi_failure")
    call_id = _stable_id("call", record["id"], "poi_failure")
    failure_result = _poi_failure_result(record, category)
    user = _poi_query(anchor_label, category, "failure")
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "tool_failure_fallback", "poi"),
        task_type="tool_failure_fallback",
        scene="amap_poi_search",
        difficulty="medium",
        source_record_id=record["id"],
        expected_behavior="should_fallback",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_search_poi", {"keyword": _poi_keyword(category), "city": city, "around_location": anchor_label, "radius_m": _poi_radius(category)})]},
            _build_tool_message(call_id, failure_result),
            {"role": "assistant", "content": _poi_failure_answer(anchor_label, category, failure_result)},
        ],
    )


def _geocode_failure_sample(record: dict[str, Any]) -> dict[str, Any]:
    city = _record_city(record)
    address = _record_anchor(record)
    call_id = _stable_id("call", record["id"], "geocode_failure")
    failure_result = _geocode_failure_result(record)
    user = _stable_choice((f"帮我确认一下{address}在哪。", f"{address}具体位置在哪？", f"我想先知道{address}的位置。"), record.get("id"), address, city, "geocode_failure_query")
    messages = [{"role": "system", "content": TRIPAI_TOOL_USE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return _make_sample(
        sample_id=_stable_id("tool", record["id"], "tool_failure_fallback", "geocode"),
        task_type="tool_failure_fallback",
        scene="amap_geocode",
        difficulty="medium",
        source_record_id=record["id"],
        expected_behavior="should_fallback",
        messages=messages,
        messages_with_answer=[
            *messages,
            {"role": "assistant", "tool_calls": [_build_tool_call(call_id, "amap_geocode", {"address": address, "city": city})]},
            _build_tool_message(call_id, failure_result),
            {"role": "assistant", "content": _geocode_failure_answer(address, failure_result)},
        ],
    )


def _safe_json_loads(value: str) -> dict[str, Any]:
    return json.loads(value)


def _final_answer(item: dict[str, Any]) -> str:
    for message in reversed(item["messages_with_answer"]):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _semantic_validation_errors(dataset: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(dataset):
        messages_with_answer = item["messages_with_answer"]
        final_answer = _final_answer(item)

        if item["task_type"] == "no_tool_needed":
            if any(message.get("role") == "assistant" and message.get("tool_calls") for message in messages_with_answer):
                errors.append(f"item {index} no_tool_needed should not contain tool_calls")
            continue

        if item["task_type"] == "clarify_then_call":
            if len(messages_with_answer) < 5:
                errors.append(f"item {index} clarify_then_call is too short")
            first_assistant = next((message for message in messages_with_answer if message.get("role") == "assistant"), None)
            if not first_assistant or not isinstance(first_assistant.get("content"), str):
                errors.append(f"item {index} clarify_then_call is missing the initial clarification turn")

        tool_call_names: dict[str, str] = {}
        last_tool_result: dict[str, Any] | None = None
        last_tool_name = ""

        for message in messages_with_answer:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls", []) or []:
                    tool_call_names[tool_call["id"]] = tool_call["function"]["name"]
            elif message.get("role") == "tool":
                tool_name = tool_call_names.get(str(message.get("tool_call_id")), "")
                payload = _safe_json_loads(str(message.get("content") or "{}"))
                last_tool_result = payload
                last_tool_name = tool_name

        if not last_tool_result:
            continue

        status = last_tool_result.get("status")
        if status == "success" and last_tool_name == "amap_search_poi":
            data = last_tool_result.get("data", {})
            pois = data.get("pois") or []
            count = str(data.get("count"))
            top_name = str(pois[0]["name"]) if pois else ""
            if count not in final_answer or top_name not in final_answer:
                errors.append(f"item {index} poi answer is not grounded to count/name")
        elif status == "success" and last_tool_name == "amap_plan_route":
            data = last_tool_result.get("data", {})
            summary = str(data.get("summary") or "")
            steps = data.get("route_steps") or []
            first_step = str(steps[0].get("instruction") or "") if steps else ""
            if summary not in final_answer or first_step not in final_answer:
                errors.append(f"item {index} route answer is not grounded to summary/step")
        elif status == "success" and last_tool_name == "amap_geocode":
            data = last_tool_result.get("data", {})
            if str(data.get("formatted_address")) not in final_answer and str(data.get("location")) not in final_answer:
                errors.append(f"item {index} geocode answer is not grounded to formatted_address/location")
        elif status == "empty":
            if not any(marker in final_answer for marker in ("没查到", "没找到", "暂时没", "没有查到")):
                errors.append(f"item {index} empty fallback answer does not acknowledge empty result")
        elif status == "error":
            if not any(marker in final_answer for marker in ("失败", "暂时", "拿不到", "重试")):
                errors.append(f"item {index} error fallback answer does not acknowledge tool error")

    return errors


def _summarize_dataset(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    tool_call_counter: Counter[str] = Counter()
    route_mode_counter: Counter[str] = Counter()
    search_keyword_counter: Counter[str] = Counter()
    envelope_status_counter: Counter[str] = Counter()
    clarify_openings: Counter[str] = Counter()

    for item in dataset:
        if item["task_type"] == "clarify_then_call":
            assistant_turns = [message for message in item["messages_with_answer"] if message.get("role") == "assistant" and isinstance(message.get("content"), str)]
            if assistant_turns:
                clarify_openings[assistant_turns[0]["content"]] += 1

        for message in item["messages_with_answer"]:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls", []) or []:
                    function = tool_call["function"]
                    name = function["name"]
                    args = _safe_json_loads(function["arguments"])
                    tool_call_counter[name] += 1
                    if name == "amap_plan_route":
                        route_mode_counter[str(args.get("mode") or "transit")] += 1
                    if name == "amap_search_poi":
                        search_keyword_counter[str(args.get("keyword") or "")] += 1
            elif message.get("role") == "tool":
                payload = _safe_json_loads(message["content"])
                envelope_status_counter[str(payload.get("status") or "unknown")] += 1

    return {
        "tool_call_distribution": dict(tool_call_counter),
        "route_mode_distribution": dict(route_mode_counter),
        "search_keyword_distribution": dict(search_keyword_counter),
        "envelope_status_distribution": dict(envelope_status_counter),
        "clarify_first_turn_unique_count": len(clarify_openings),
        "clarify_first_turn_top_examples": [{"text": text, "count": count} for text, count in clarify_openings.most_common(10)],
    }


def _build_candidate_subpools(
    traffic_records: list[dict[str, Any]],
    hotel_records: list[dict[str, Any]],
    travel_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    safe_travel_records = [record for record in travel_records if _is_safe_no_tool_record(record)]
    subpools = {
        "single_tool_call": {
            "route": [_route_single_sample(record) for record in traffic_records],
            "geocode": [_travel_geocode_sample(record) for record in travel_records],
            "poi": [_poi_single_sample(record) for record in [*hotel_records, *travel_records]],
        },
        "slot_filling_tool_call": {
            "route": [_route_slot_sample(record) for record in traffic_records],
            "poi": [_poi_slot_sample(record) for record in [*hotel_records, *travel_records]],
        },
        "clarify_then_call": {
            "route": [_route_clarify_sample(record) for record in traffic_records],
            "poi": [_poi_clarify_sample(record) for record in [*hotel_records, *travel_records]],
        },
        "tool_result_grounded_answer": {
            "route_chain": [_route_grounded_sample(record) for record in travel_records],
            "poi_chain": [_poi_grounded_sample(record) for record in [*hotel_records, *travel_records]],
        },
        "no_tool_needed": {
            "safe_travel_qa": [_travel_no_tool(record, None) for record in safe_travel_records],
        },
        "tool_failure_fallback": {
            "route_failure": [_route_failure_sample(record) for record in traffic_records],
            "poi_failure": [_poi_failure_sample(record) for record in [*hotel_records, *travel_records]],
            "geocode_failure": [_geocode_failure_sample(record) for record in travel_records],
        },
    }
    meta = {
        "source_record_counts": {
            "traffic_records": len(traffic_records),
            "hotel_records": len(hotel_records),
            "travel_records": len(travel_records),
            "safe_no_tool_records": len(safe_travel_records),
        }
    }
    return subpools, meta


def build_dataset(
    total_samples: int,
    seed: int,
    *,
    traffic_input: str | Path = DEFAULT_TRAFFIC_INPUT,
    hotel_input: str | Path = DEFAULT_HOTEL_INPUT,
    travel_input: str | Path = DEFAULT_TRAVEL_INPUT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    traffic_path = resolve_path(traffic_input)
    hotel_path = resolve_path(hotel_input)
    travel_path = resolve_path(travel_input)
    traffic_records = load_records(traffic_path)
    hotel_records = load_records(hotel_path)
    travel_records = load_records(travel_path)

    candidate_subpools, source_meta = _build_candidate_subpools(traffic_records, hotel_records, travel_records)

    target_counts = _compute_target_counts(total_samples)
    dataset: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "total_samples": total_samples,
        "seed": seed,
        "input_paths": {
            "traffic": str(traffic_path),
            "hotel": str(hotel_path),
            "travel": str(travel_path),
        },
        "targets": target_counts,
        "source_meta": source_meta,
        "buckets": {},
    }

    for task_type, target_count in target_counts.items():
        subpool_targets = _compute_ratio_counts(target_count, BUCKET_SUBPOOL_RATIOS[task_type])
        sampled_bucket: list[dict[str, Any]] = []
        report["buckets"][task_type] = {"sampled_count": 0, "subpools": {}}

        for subpool_name, sub_target in subpool_targets.items():
            candidates = candidate_subpools[task_type][subpool_name]
            sampled = _sample_with_oversampling(candidates, sub_target, rng)
            sampled_bucket.extend(sampled)
            report["buckets"][task_type]["subpools"][subpool_name] = {
                "candidate_count": len(candidates),
                "target_count": sub_target,
                "sampled_count": len(sampled),
            }

        dataset.extend(sampled_bucket)
        report["buckets"][task_type]["sampled_count"] = len(sampled_bucket)

    rng.shuffle(dataset)
    report["final_count"] = len(dataset)
    report["semantic_summary"] = _summarize_dataset(dataset)
    return dataset, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stage2 AMap tool-use dataset and export it to LLaMA-Factory.")
    parser.add_argument("--total-samples", type=int, default=DEFAULT_TOTAL_SAMPLES, help="Total stage2 samples.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--traffic-input", default=DEFAULT_TRAFFIC_INPUT, help="Strict traffic-planning records.")
    parser.add_argument("--hotel-input", default=DEFAULT_HOTEL_INPUT, help="Strict hotel-recommendation records.")
    parser.add_argument("--travel-input", default=DEFAULT_TRAVEL_INPUT, help="Strict travel-QA records.")
    parser.add_argument("--source-output", default=DEFAULT_SOURCE_OUTPUT, help="Source dataset output path.")
    parser.add_argument("--export-output", default=DEFAULT_EXPORT_OUTPUT, help="Exported sharegpt output path.")
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT, help="Builder report output path.")
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()

    dataset, report = build_dataset(
        args.total_samples,
        args.seed,
        traffic_input=args.traffic_input,
        hotel_input=args.hotel_input,
        travel_input=args.travel_input,
    )

    source_errors = validate_tool_use_source_dataset(dataset)
    if source_errors:
        log_error(f"Stage2 source dataset validation failed with {len(source_errors)} errors.")
        for error in source_errors[:10]:
            log_error(error)
        return 1

    semantic_errors = _semantic_validation_errors(dataset)
    report["semantic_validation"] = {"error_count": len(semantic_errors), "sample_errors": semantic_errors[:10]}
    if semantic_errors:
        log_error(f"Stage2 semantic validation failed with {len(semantic_errors)} errors.")
        for error in semantic_errors[:10]:
            log_error(error)
        return 1

    exported = export_tool_use_dataset_to_sharegpt(dataset)
    export_errors = validate_sharegpt_tool_dataset(exported)
    if export_errors:
        log_error(f"Stage2 exported dataset validation failed with {len(export_errors)} errors.")
        for error in export_errors[:10]:
            log_error(error)
        return 1

    source_path = write_json(args.source_output, dataset)
    export_path = write_json(args.export_output, exported)
    report_path = write_json(args.report_output, report)
    log_info(f"Source dataset written to: {source_path}")
    log_info(f"Exported dataset written to: {export_path}")
    log_success(f"Build report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
