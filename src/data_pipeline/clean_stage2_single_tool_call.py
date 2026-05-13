from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import configure_console_output
from src.tool_use.datasets import (
    export_tool_use_dataset_to_sharegpt,
    validate_sharegpt_tool_dataset,
    validate_tool_use_source_dataset,
)
from src.tool_use.protocol import TRIPAI_TOOL_USE_SYSTEM_PROMPT, build_amap_tool_schemas

DEFAULT_INPUT = Path("data/raw_stage2/single_tool_call.jsonl")
DEFAULT_OUTPUT = Path("data/processed_stage2/single_tool_call.jsonl")
DEFAULT_REPORT = Path("data/processed_stage2/single_tool_call_report.json")
TARGET_PER_SUBTYPE = 80

SUBTYPE_TO_TOOL = {
    "direct_route": ("amap_plan_route", "amap_plan_route"),
    "direct_geocode": ("amap_geocode", "amap_geocode"),
    "direct_poi": ("amap_search_poi", "amap_search_poi"),
}
SUBTYPE_ORDER = ("direct_route", "direct_geocode", "direct_poi")

ALLOWED_TOOL_ARGS = {
    "amap_geocode": {"address", "city"},
    "amap_search_poi": {"keyword", "city", "around_location", "radius_m"},
    "amap_plan_route": {"origin", "destination", "mode", "city"},
}
REQUIRED_TOOL_ARGS = {
    "amap_geocode": {"address", "city"},
    "amap_search_poi": {"keyword", "city"},
    "amap_plan_route": {"origin", "destination", "mode", "city"},
}
ROUTE_MODES = {"transit", "driving", "walking", "bicycling"}
ROUTE_MODE_LABELS = {
    "transit": "公共交通",
    "driving": "驾车",
    "walking": "步行",
    "bicycling": "骑行",
}

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
SPACE_RE = re.compile(r"[ \t\u3000]+")
COORD_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*$")
ROAD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,16}(?:大道|高速|公路|快速路|立交桥|路桥|路|街|道|巷|桥|线)")
DISTANCE_TEXT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:千米|公里|米)")
RAW_SECONDS_RE = re.compile(r"\d+\s*秒")
RAW_LONG_METERS_RE = re.compile(r"\d{4,}\s*米")


@dataclass(frozen=True)
class Candidate:
    line_number: int
    raw_id: str
    subtype: str
    score: int
    item: dict[str, Any]


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object.")
        rows.append((line_number, payload))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = SPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r" *\n+ *", "\n", cleaned)
    return cleaned.strip()


def _conversation_value(messages: list[dict[str, Any]], role: str, *, last: bool = False) -> str:
    iterable = reversed(messages) if last else messages
    for message in iterable:
        if message.get("from") == role:
            return _normalize_text(message.get("value"))
    return ""


def _parse_conversations(item: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None, "missing_conversations"
    if len(conversations) != 5:
        return None, "conversation_length_not_5"
    if not all(isinstance(message, dict) for message in conversations):
        return None, "conversation_message_not_object"
    roles = tuple(message.get("from") for message in conversations)
    if roles != ("system", "human", "function_call", "observation", "gpt"):
        return None, "bad_role_sequence"
    if any(not _normalize_text(message.get("value")) for message in conversations):
        return None, "empty_conversation_value"
    return conversations, None


def _loads_object(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _flatten_expected_subset(raw_subset: Any, tool_name: str) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(raw_subset, dict):
        return None, False
    if set(raw_subset) == {tool_name} and isinstance(raw_subset.get(tool_name), dict):
        return dict(raw_subset[tool_name]), True
    return dict(raw_subset), False


def _is_coordinate(value: Any) -> bool:
    return isinstance(value, str) and bool(COORD_RE.match(value))


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


def _format_km(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    if value < 1:
        return f"约{int(round(value * 1000))}米"
    formatted = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"约{formatted}公里"


def _format_meter_distance(value: Any) -> str:
    if value in {None, ""}:
        return ""
    try:
        meters = float(value)
    except (TypeError, ValueError):
        return ""
    if meters <= 0:
        return ""
    if meters >= 1000:
        km = meters / 1000
        return f"约{km:.1f}".rstrip("0").rstrip(".") + "公里"
    return f"约{int(round(meters))}米"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        value = value.strip(" ，,。；;")
        value = re.sub(r"^(?:沿|途径)", "", value).strip(" ，,。；;")
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _extract_roads(steps: Any) -> list[str]:
    if not isinstance(steps, list):
        return []
    roads: list[str] = []
    for step in steps:
        if isinstance(step, dict):
            step_text = " ".join(str(value) for value in step.values())
        else:
            step_text = str(step)
        roads.extend(ROAD_RE.findall(step_text))
    return _ordered_unique(roads)


def _clean_step_text(step: Any) -> str:
    if isinstance(step, dict):
        text = " ".join(str(value) for value in step.values())
    else:
        text = str(step)
    text = DISTANCE_TEXT_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(" ，,。；;")
    return text


def _route_detail(data: dict[str, Any]) -> str:
    roads = _extract_roads(data.get("main_steps"))
    if roads:
        return "主要经过" + "、".join(roads[:5])

    steps = data.get("main_steps")
    if isinstance(steps, list):
        cleaned_steps = [_clean_step_text(step) for step in steps[:3]]
        cleaned_steps = [step for step in cleaned_steps if step]
        if cleaned_steps:
            return "前半段可按导航提示：" + "；".join(cleaned_steps)
    return "途中按导航提示行进，留意连续转向和路口"


def _route_reminder(mode: str) -> str:
    if mode == "driving":
        return "出发前看一下实时路况，高峰时段建议多预留一点时间。"
    if mode == "transit":
        return "出发前确认首末班和换乘信息，高峰时段预留候车时间。"
    if mode == "walking":
        return "步行时注意路口和天气，夜间出行优先走照明更好的道路。"
    if mode == "bicycling":
        return "骑行时注意非机动车道和路口转向，雨天建议改用其他交通方式。"
    return "出发前再看一下实时导航，按现场路况调整。"


def _route_answer(args: dict[str, Any], data: dict[str, Any]) -> str | None:
    mode = str(args.get("mode") or data.get("mode") or "")
    if mode not in ROUTE_MODES:
        return None
    origin = str(data.get("origin") or args.get("origin") or "").strip()
    destination = str(data.get("destination") or args.get("destination") or "").strip()
    duration = _format_duration(data.get("duration_min"))
    distance = _format_km(data.get("distance_km"))
    if not origin or not destination or not duration or not distance:
        return None
    detail = _route_detail(data)
    label = ROUTE_MODE_LABELS[mode]
    return f"从{origin}到{destination}{label}{duration}，距离{distance}。{detail}。{_route_reminder(mode)}"


def _geocode_answer(question: str, args: dict[str, Any], data: dict[str, Any]) -> str | None:
    query = str(data.get("query") or args.get("address") or "").strip()
    formatted_address = str(data.get("formatted_address") or "").strip()
    city = str(data.get("city") or args.get("city") or "").strip()
    district = str(data.get("district") or "").strip()
    location = str(data.get("location") or "").strip()
    if not query or not (formatted_address or city):
        return None

    location_text = formatted_address or f"{city}{district}"
    location_text = location_text.strip()
    if not location_text:
        return None
    if location_text == query:
        if city and district:
            location_text = f"{city}{district}{query}"
        elif city:
            location_text = f"{city}{query}"

    if district and district not in location_text:
        answer = f"{query}位于{location_text}，所属区域是{district}。"
    else:
        answer = f"{query}位于{location_text}，可以作为地图搜索和导航定位参考。"

    wants_coordinate = any(token in question for token in ("坐标", "经纬", "经度", "纬度"))
    if wants_coordinate:
        parts = [part.strip() for part in location.split(",")]
        if len(parts) == 2 and all(parts):
            answer = answer.rstrip("。") + f"，坐标约为经度{parts[0]}、纬度{parts[1]}。"
    return answer


def _poi_result_text(poi: dict[str, Any]) -> str:
    name = str(poi.get("name") or "").strip()
    address = str(poi.get("address") or "").strip()
    distance = _format_meter_distance(poi.get("distance"))
    parts = [name]
    if distance:
        parts.append(f"距离{distance}")
    if address:
        parts.append(f"地址在{address}")
    return "，".join(part for part in parts if part)


def _poi_answer(args: dict[str, Any], data: dict[str, Any]) -> str | None:
    keyword = str(data.get("keyword") or args.get("keyword") or "").strip()
    city = str(data.get("city") or args.get("city") or "").strip()
    around = str(args.get("around_location") or data.get("around_location") or "").strip()
    pois = data.get("pois")
    if not keyword or not city or not isinstance(pois, list) or len(pois) < 2:
        return None

    valid_pois = [poi for poi in pois if isinstance(poi, dict) and str(poi.get("name") or "").strip()]
    if len(valid_pois) < 2:
        return None
    selected = valid_pois[: min(5, len(valid_pois))]
    result_text = "；".join(_poi_result_text(poi) for poi in selected)
    if not result_text:
        return None

    if around and not _is_coordinate(around):
        place = around if around.startswith(city) else f"{city}{around}"
        prefix = f"我查到{place}附近有这些{keyword}可以优先看："
    else:
        prefix = f"我查到{city}有这些{keyword}可以优先看："
    return prefix + result_text + "。建议优先选距离更近、地址更匹配行程的一家。"


def _rewrite_answer(tool_name: str, question: str, args: dict[str, Any], observation: dict[str, Any]) -> str | None:
    if observation.get("status") != "success":
        return None
    data = observation.get("data")
    if not isinstance(data, dict):
        return None
    if tool_name == "amap_plan_route":
        return _route_answer(args, data)
    if tool_name == "amap_geocode":
        return _geocode_answer(question, args, data)
    if tool_name == "amap_search_poi":
        return _poi_answer(args, data)
    return None


def _answer_is_clean(answer: str) -> bool:
    if not answer or "\n" in answer:
        return False
    if EMOJI_RE.search(answer):
        return False
    if any(token in answer for token in ("<think>", "</think>", "reasoning_content", "```", "**")):
        return False
    if any(token in answer for token in ("tool_call", "observation", "formatted_address", "route_steps", '"status"')):
        return False
    if RAW_SECONDS_RE.search(answer) or RAW_LONG_METERS_RE.search(answer):
        return False
    return True


def _quality_tags(subtype: str, raw_tags: Any, *, fixed_subset: bool) -> list[str]:
    tags: list[str] = ["single_tool", subtype, "success_observation", "grounded_answer", "human_readable_units"]
    if fixed_subset:
        tags.append("fixed_expected_arguments_subset")
    if isinstance(raw_tags, list):
        tags.extend(str(tag).strip() for tag in raw_tags if isinstance(tag, str) and tag.strip())
    return _ordered_unique(tags)


def _candidate_score(subtype: str, args: dict[str, Any], observation: dict[str, Any], answer: str) -> int:
    score = 100
    data = observation.get("data") if isinstance(observation.get("data"), dict) else {}
    if subtype == "direct_route":
        roads = _extract_roads(data.get("main_steps"))
        score += min(len(roads), 5) * 5
        if len(answer) >= 60:
            score += 8
        if args.get("mode") in {"transit", "walking", "bicycling"}:
            score += 3
    elif subtype == "direct_geocode":
        if data.get("district"):
            score += 8
        if data.get("formatted_address"):
            score += 8
    elif subtype == "direct_poi":
        pois = data.get("pois")
        if isinstance(pois, list):
            score += min(len(pois), 5) * 3
            if any(str(poi.get("distance") or "").strip() for poi in pois if isinstance(poi, dict)):
                score += 8
        if args.get("around_location"):
            score += 5
    if 40 <= len(answer) <= 260:
        score += 5
    return score


def _clean_one(line_number: int, item: dict[str, Any], rejected: Counter[str], fixes: Counter[str]) -> Candidate | None:
    raw_id = _normalize_text(item.get("id"))
    subtype = _normalize_text(item.get("subtype"))
    if item.get("task_type") != "single_tool_call":
        rejected["bad_task_type"] += 1
        return None
    if subtype not in SUBTYPE_TO_TOOL:
        rejected["bad_subtype"] += 1
        return None
    expected_scene, expected_tool = SUBTYPE_TO_TOOL[subtype]
    if item.get("scene") != expected_scene:
        rejected["scene_subtype_mismatch"] += 1
        return None
    if item.get("expected_behavior") != "should_call_tool":
        rejected["bad_expected_behavior"] += 1
        return None
    if item.get("expected_tool_chain") != [expected_tool]:
        rejected["bad_expected_tool_chain"] += 1
        return None

    conversations, conversation_error = _parse_conversations(item)
    if conversations is None:
        rejected[conversation_error or "bad_conversations"] += 1
        return None

    question = _conversation_value(conversations, "human")
    if not question or EMOJI_RE.search(question):
        rejected["bad_question"] += 1
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
    for arg_name in REQUIRED_TOOL_ARGS[tool_name]:
        value = args.get(arg_name)
        if not isinstance(value, (str, int)) or not str(value).strip():
            rejected[f"missing_required_argument_{arg_name}"] += 1
            return None
    if tool_name == "amap_plan_route" and args.get("mode") not in ROUTE_MODES:
        rejected["bad_route_mode"] += 1
        return None

    expected_subset, fixed_subset = _flatten_expected_subset(item.get("expected_arguments_subset"), tool_name)
    if expected_subset is None:
        rejected["bad_expected_arguments_subset"] += 1
        return None
    for key, value in expected_subset.items():
        if args.get(key) != value:
            rejected["expected_arguments_subset_mismatch"] += 1
            return None
    if fixed_subset:
        fixes["nested_expected_arguments_subset"] += 1

    answer = _rewrite_answer(tool_name, question, args, observation)
    if answer is None:
        rejected["answer_rewrite_failed"] += 1
        return None
    answer = _normalize_text(answer)
    if not _answer_is_clean(answer):
        rejected["rewritten_answer_not_clean"] += 1
        return None

    normalized_args = {key: args[key] for key in args}
    normalized_observation = {"status": "success", "data": observation["data"]}
    cleaned_item = {
        "id": raw_id,
        "task_type": "single_tool_call",
        "subtype": subtype,
        "scene": expected_scene,
        "expected_behavior": "should_call_tool",
        "expected_tool_chain": [expected_tool],
        "expected_arguments_subset": normalized_args,
        "quality_tags": _quality_tags(subtype, item.get("quality_tags"), fixed_subset=fixed_subset),
        "conversations": [
            {"from": "system", "value": TRIPAI_TOOL_USE_SYSTEM_PROMPT},
            {"from": "human", "value": question},
            {"from": "function_call", "value": _json_dumps({"name": tool_name, "arguments": normalized_args})},
            {"from": "observation", "value": _json_dumps(normalized_observation)},
            {"from": "gpt", "value": answer},
        ],
    }
    score = _candidate_score(subtype, normalized_args, normalized_observation, answer)
    return Candidate(line_number=line_number, raw_id=raw_id, subtype=subtype, score=score, item=cleaned_item)


def _renumber_selected(candidates_by_subtype: dict[str, list[Candidate]], target_per_subtype: int) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for subtype in SUBTYPE_ORDER:
        selected = candidates_by_subtype[subtype][:target_per_subtype]
        for index, candidate in enumerate(selected, start=1):
            item = dict(candidate.item)
            item["id"] = f"stage2_v2_single_tool_call_{subtype}_{index:06d}"
            cleaned.append(item)
    return cleaned


def _to_source_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = build_amap_tool_schemas()
    source_rows: list[dict[str, Any]] = []
    for row in rows:
        messages_with_answer: list[dict[str, Any]] = []
        call_id = "call_001"
        for message in row["conversations"]:
            role = message["from"]
            value = message["value"]
            if role == "system":
                messages_with_answer.append({"role": "system", "content": value})
            elif role == "human":
                messages_with_answer.append({"role": "user", "content": value})
            elif role == "function_call":
                payload = json.loads(value)
                messages_with_answer.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call_id,
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
                messages_with_answer.append({"role": "tool", "tool_call_id": call_id, "content": value})
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


def clean_single_tool_call(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    target_per_subtype: int = TARGET_PER_SUBTYPE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rejected: Counter[str] = Counter()
    fixes: Counter[str] = Counter()
    candidates_by_subtype: dict[str, list[Candidate]] = {subtype: [] for subtype in SUBTYPE_ORDER}
    seen_questions: set[str] = set()
    seen_function_calls: set[str] = set()

    for line_number, item in rows:
        candidate = _clean_one(line_number, item, rejected, fixes)
        if candidate is None:
            continue
        question = candidate.item["conversations"][1]["value"]
        function_call = candidate.item["conversations"][2]["value"]
        if question in seen_questions:
            rejected["duplicate_question"] += 1
            continue
        if function_call in seen_function_calls:
            rejected["duplicate_function_call"] += 1
            continue
        seen_questions.add(question)
        seen_function_calls.add(function_call)
        candidates_by_subtype[candidate.subtype].append(candidate)

    for subtype, candidates in candidates_by_subtype.items():
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.raw_id, candidate.line_number))
        if len(candidates) < target_per_subtype:
            raise RuntimeError(
                f"{subtype} has only {len(candidates)} clean candidates, "
                f"but target_per_subtype={target_per_subtype}"
            )

    cleaned = _renumber_selected(candidates_by_subtype, target_per_subtype)
    validation_errors = _validate_processed_rows(cleaned)
    if validation_errors:
        raise RuntimeError(f"processed single_tool_call failed validation: {validation_errors[:5]}")

    selected_counts = Counter(item["subtype"] for item in cleaned)
    candidate_counts = {subtype: len(candidates_by_subtype[subtype]) for subtype in SUBTYPE_ORDER}
    answer_lengths = [len(item["conversations"][-1]["value"]) for item in cleaned]
    route_modes = Counter()
    tool_counts = Counter()
    for item in cleaned:
        function_payload = json.loads(item["conversations"][2]["value"])
        tool_counts[function_payload["name"]] += 1
        if function_payload["name"] == "amap_plan_route":
            route_modes[function_payload["arguments"].get("mode")] += 1

    report = {
        "input_records": len(rows),
        "target_per_subtype": target_per_subtype,
        "target_total": target_per_subtype * len(SUBTYPE_ORDER),
        "candidate_counts": candidate_counts,
        "selected_count": len(cleaned),
        "selected_counts": dict(sorted(selected_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "route_mode_counts": dict(sorted(route_modes.items())),
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
    parser = argparse.ArgumentParser(description="Clean raw Stage2 single_tool_call JSONL into processed Stage2 JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-per-subtype", type=int, default=TARGET_PER_SUBTYPE)
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    rows = _read_jsonl(args.input)
    cleaned, report = clean_single_tool_call(rows, target_per_subtype=args.target_per_subtype)
    output_path = _write_jsonl(args.output, cleaned)
    report_path = _write_json(args.report, report)
    print(f"[OK] cleaned single_tool_call written to: {output_path}")
    print(f"[OK] report written to: {report_path}")
    print(f"[OK] selected: {report['selected_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
