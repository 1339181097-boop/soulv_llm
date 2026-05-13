from __future__ import annotations

import argparse
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
    _answer_is_clean,
    _conversation_value,
    _format_meter_distance,
    _geocode_answer,
    _json_dumps,
    _loads_object,
    _normalize_text,
    _ordered_unique,
    _read_jsonl,
    _route_answer,
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
    Path("data/raw_stage2/tool_result_grounded_answer.jsonl"),
    Path("data/raw_stage2/tool_result_grounded_answer_v2.jsonl"),
]
DEFAULT_OUTPUT = Path("data/processed_stage2/tool_result_grounded_answer.jsonl")
DEFAULT_REPORT = Path("data/processed_stage2/tool_result_grounded_answer_report.json")

TARGET_COUNTS = {
    "route_grounded_answer": 180,
    "poi_grounded_answer": 160,
    "geocode_grounded_answer": 60,
}
SUBTYPE_ORDER = ("route_grounded_answer", "poi_grounded_answer", "geocode_grounded_answer")

SUBTYPE_FINAL_TOOL = {
    "route_grounded_answer": "amap_plan_route",
    "poi_grounded_answer": "amap_search_poi",
    "geocode_grounded_answer": "amap_geocode",
}
TOOL_SCENE = {
    "amap_geocode": "amap_geocode",
    "amap_search_poi": "amap_search_poi",
    "amap_plan_route": "amap_plan_route",
}
ALLOWED_TOOL_ARGS = {
    "amap_geocode": {"address", "city"},
    "amap_search_poi": {"keyword", "city", "around_location", "radius_m"},
    "amap_plan_route": {"origin", "destination", "mode", "city"},
}
REQUIRED_TOOL_ARGS = {
    "amap_geocode": {"address"},
    "amap_search_poi": {"keyword"},
    "amap_plan_route": {"origin", "destination"},
}
ARG_ORDER = {
    "amap_geocode": ("address", "city"),
    "amap_search_poi": ("keyword", "city", "around_location", "radius_m"),
    "amap_plan_route": ("origin", "destination", "mode", "city"),
}
ROUTE_MODES = {"transit", "driving", "walking", "bicycling"}
ALLOWED_CHAINS = {
    ("amap_geocode",),
    ("amap_search_poi",),
    ("amap_plan_route",),
    ("amap_geocode", "amap_search_poi"),
    ("amap_geocode", "amap_plan_route"),
}
CUSTOMER_SERVICE_RE = re.compile(r"(您好|亲|马上安排|小助手)")


@dataclass(frozen=True)
class Candidate:
    source: str
    line_number: int
    raw_id: str
    subtype: str
    final_tool: str
    score: int
    item: dict[str, Any]


def _read_jsonl_inputs(paths: Iterable[Path]) -> list[tuple[str, int, dict[str, Any]]]:
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for path in paths:
        for line_number, item in _read_jsonl(path):
            rows.append((str(path), line_number, item))
    return rows


def _parse_conversations(item: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None, "missing_conversations"
    if len(conversations) not in {5, 7}:
        return None, "conversation_length_not_5_or_7"
    if not all(isinstance(message, dict) for message in conversations):
        return None, "conversation_message_not_object"

    roles = tuple(message.get("from") for message in conversations)
    if roles not in {
        ("system", "human", "function_call", "observation", "gpt"),
        ("system", "human", "function_call", "observation", "function_call", "observation", "gpt"),
    }:
        return None, "bad_role_sequence"
    if any(not _normalize_text(message.get("value")) for message in conversations):
        return None, "empty_conversation_value"
    return conversations, None


def _function_calls_and_observations(
    conversations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for message in conversations:
        if message.get("from") == "function_call":
            payload = _loads_object(_normalize_text(message.get("value")))
            if payload is None:
                raise ValueError("bad_function_call_json")
            calls.append(payload)
        elif message.get("from") == "observation":
            payload = _loads_object(_normalize_text(message.get("value")))
            if payload is None:
                raise ValueError("bad_observation_json")
            observations.append(payload)
    return calls, observations


def _tool_chain(calls: list[dict[str, Any]]) -> tuple[str, ...]:
    chain: list[str] = []
    for call in calls:
        name = call.get("name")
        if isinstance(name, str):
            chain.append(name)
    return tuple(chain)


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

    for key in REQUIRED_TOOL_ARGS[tool_name]:
        if key not in normalized:
            return None
    if tool_name == "amap_plan_route" and normalized.get("mode") not in ROUTE_MODES:
        return None
    return normalized


def _normalize_success_observation(raw_observation: dict[str, Any]) -> dict[str, Any] | None:
    if raw_observation.get("status") != "success":
        return None
    data = raw_observation.get("data")
    if not isinstance(data, dict):
        return None
    return {"status": "success", "data": data}


def _normalize_city(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    for token in ("省", "市", "自治区", "特别行政区"):
        cleaned = cleaned.replace(token, "")
    return cleaned


def _geocode_city_matches_argument(args: dict[str, Any], observation: dict[str, Any]) -> bool:
    arg_city = _normalize_city(args.get("city"))
    obs_city = _normalize_city(observation.get("data", {}).get("city"))
    if not arg_city or not obs_city:
        return True
    return arg_city in obs_city


def _validate_two_step_link(
    chain: tuple[str, ...],
    calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> bool:
    if len(chain) != 2:
        return True
    first_location = observations[0].get("data", {}).get("location")
    if not isinstance(first_location, str) or not first_location.strip():
        return False
    second_args = calls[1].get("arguments")
    if not isinstance(second_args, dict):
        return False
    if chain[1] == "amap_search_poi":
        return second_args.get("around_location") == first_location
    if chain[1] == "amap_plan_route":
        return first_location in {second_args.get("origin"), second_args.get("destination")}
    return False


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
    return distance_km > 30 or duration_min > 480


def _normalize_expected_subset(
    raw_subset: Any,
    normalized_calls: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(raw_subset, dict):
        return None, False

    fixed = False
    if len(normalized_calls) == 1:
        tool_name = normalized_calls[0]["name"]
        expected = raw_subset.get(tool_name) if set(raw_subset) == {tool_name} else raw_subset
        actual_args = normalized_calls[0]["arguments"]
        if not isinstance(expected, dict):
            return dict(actual_args), True
        fixed = expected != actual_args or expected is not raw_subset
        return dict(actual_args), fixed

    expected_by_tool = raw_subset if all(key in ALLOWED_TOOL_ARGS for key in raw_subset) else {}
    normalized_subset: dict[str, Any] = {}
    for call in normalized_calls:
        tool_name = call["name"]
        actual_args = call["arguments"]
        expected = expected_by_tool.get(tool_name)
        if expected != actual_args:
            fixed = True
        normalized_subset[tool_name] = dict(actual_args)
    if not expected_by_tool:
        fixed = True
    return normalized_subset, fixed


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


def _poi_grounded_answer(args: dict[str, Any], observation: dict[str, Any]) -> str | None:
    data = observation.get("data")
    if not isinstance(data, dict):
        return None
    keyword = _normalize_text(data.get("keyword") or args.get("keyword"))
    city = _normalize_text(data.get("city") or args.get("city"))
    around = _normalize_text(args.get("around_location") or data.get("around_location"))
    pois = data.get("pois")
    if not keyword or not city or not isinstance(pois, list):
        return None
    valid_pois = [poi for poi in pois if isinstance(poi, dict) and _normalize_text(poi.get("name"))]
    if len(valid_pois) < 2:
        return None

    selected = valid_pois[: min(5, len(valid_pois))]
    result_text = "；".join(_poi_result_text(poi) for poi in selected)
    if not result_text:
        return None
    if around and not re.match(r"^\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*$", around):
        place = around if around.startswith(city) else f"{city}{around}"
        prefix = f"{place}附近可以优先看这些{keyword}："
    else:
        prefix = f"{city}可以优先看这些{keyword}："
    return prefix + result_text + "。建议优先选距离更近、地址更匹配行程的一家，出发前再确认营业状态。"


def _final_answer(
    subtype: str,
    question: str,
    normalized_calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> str | None:
    final_tool = normalized_calls[-1]["name"]
    args = normalized_calls[-1]["arguments"]
    observation = observations[-1]
    data = observation.get("data")
    if not isinstance(data, dict):
        return None

    if subtype == "route_grounded_answer" and final_tool == "amap_plan_route":
        return _route_answer(args, data)
    if subtype == "poi_grounded_answer" and final_tool == "amap_search_poi":
        return _poi_grounded_answer(args, observation)
    if subtype == "geocode_grounded_answer" and final_tool == "amap_geocode":
        return _geocode_answer(question, args, data)
    return None


def _quality_tags(
    subtype: str,
    raw_tags: Any,
    *,
    chain: tuple[str, ...],
    fixed_subset: bool,
    fixed_scene: bool,
) -> list[str]:
    tags = [
        "tool_result_grounded_answer",
        subtype,
        "success_observation",
        "grounded_answer",
        "human_readable_units",
    ]
    tags.append("two_step" if len(chain) == 2 else "single_tool")
    if fixed_subset:
        tags.append("fixed_expected_arguments_subset")
    if fixed_scene:
        tags.append("fixed_scene")
    if isinstance(raw_tags, list):
        tags.extend(_normalize_text(tag) for tag in raw_tags if isinstance(tag, str) and _normalize_text(tag))
    return _ordered_unique(tags)


def _strict_answer_is_clean(answer: str) -> bool:
    if not _answer_is_clean(answer):
        return False
    if CUSTOMER_SERVICE_RE.search(answer):
        return False
    if any(token in answer for token in ("大约约", "需要约", "步行约。")):
        return False
    if RAW_SECONDS_RE.search(answer) or RAW_LONG_METERS_RE.search(answer):
        return False
    return True


def _candidate_score(subtype: str, final_tool: str, args: dict[str, Any], observation: dict[str, Any], answer: str) -> int:
    score = 100
    data = observation.get("data") if isinstance(observation.get("data"), dict) else {}
    if subtype == "route_grounded_answer":
        duration = data.get("duration_min")
        distance = data.get("distance_km")
        if isinstance(duration, (int, float)) and duration > 0:
            score += 10
        if isinstance(distance, (int, float)) and distance > 0:
            score += 10
        if args.get("mode") == "transit":
            score += 6
        elif args.get("mode") in {"walking", "bicycling"}:
            score += 4
        steps = data.get("main_steps") or data.get("route_steps")
        if isinstance(steps, list):
            score += min(len(steps), 5)
    elif subtype == "poi_grounded_answer":
        pois = data.get("pois")
        if isinstance(pois, list):
            score += min(len(pois), 5) * 4
            if any(_normalize_text(poi.get("distance")) for poi in pois if isinstance(poi, dict)):
                score += 8
        if args.get("around_location"):
            score += 6
        if args.get("radius_m"):
            score += 4
    elif subtype == "geocode_grounded_answer":
        if data.get("district"):
            score += 8
        if data.get("formatted_address"):
            score += 8
        if data.get("location"):
            score += 4
    if 45 <= len(answer) <= 260:
        score += 5
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
    if item.get("task_type") != "tool_result_grounded_answer":
        rejected["bad_task_type"] += 1
        return None
    if subtype not in SUBTYPE_FINAL_TOOL:
        rejected["bad_subtype"] += 1
        return None
    if item.get("expected_behavior") != "should_call_tool":
        rejected["bad_expected_behavior"] += 1
        return None

    conversations, conversation_error = _parse_conversations(item)
    if conversations is None:
        rejected[conversation_error or "bad_conversations"] += 1
        return None
    question = _conversation_value(conversations, "human")
    if not question or EMOJI_RE.search(question):
        rejected["bad_question"] += 1
        return None

    try:
        raw_calls, raw_observations = _function_calls_and_observations(conversations)
    except ValueError as exc:
        rejected[str(exc)] += 1
        return None

    chain = _tool_chain(raw_calls)
    if chain not in ALLOWED_CHAINS:
        rejected["bad_tool_chain"] += 1
        return None
    if item.get("expected_tool_chain") != list(chain):
        rejected["expected_tool_chain_mismatch"] += 1
        return None
    if len(raw_calls) != len(raw_observations):
        rejected["tool_call_observation_count_mismatch"] += 1
        return None
    if not _validate_two_step_link(chain, raw_calls, raw_observations):
        rejected["two_step_link_invalid"] += 1
        return None

    final_tool = chain[-1]
    if final_tool != SUBTYPE_FINAL_TOOL[subtype]:
        rejected["subtype_final_tool_mismatch"] += 1
        return None

    normalized_calls: list[dict[str, Any]] = []
    normalized_observations: list[dict[str, Any]] = []
    for raw_call, raw_observation in zip(raw_calls, raw_observations):
        tool_name = raw_call.get("name")
        if tool_name not in ALLOWED_TOOL_ARGS:
            rejected["unknown_tool_name"] += 1
            return None
        normalized_args = _normalize_arguments(tool_name, raw_call.get("arguments"))
        if normalized_args is None:
            rejected["bad_tool_arguments"] += 1
            return None
        normalized_observation = _normalize_success_observation(raw_observation)
        if normalized_observation is None:
            rejected["observation_not_success"] += 1
            return None
        if tool_name == "amap_geocode" and not _geocode_city_matches_argument(normalized_args, normalized_observation):
            rejected["geocode_city_mismatch"] += 1
            return None
        if tool_name == "amap_plan_route" and _is_implausible_walking_route(normalized_args, normalized_observation):
            rejected["implausible_walking_route"] += 1
            return None
        normalized_calls.append({"name": tool_name, "arguments": normalized_args})
        normalized_observations.append(normalized_observation)

    expected_subset, fixed_subset = _normalize_expected_subset(item.get("expected_arguments_subset"), normalized_calls)
    if expected_subset is None:
        rejected["bad_expected_arguments_subset"] += 1
        return None
    if fixed_subset:
        fixes["fixed_expected_arguments_subset"] += 1

    answer = _final_answer(subtype, question, normalized_calls, normalized_observations)
    if answer is None:
        rejected["answer_rewrite_failed"] += 1
        return None
    answer = _normalize_text(answer)
    if not _strict_answer_is_clean(answer):
        rejected["rewritten_answer_not_clean"] += 1
        return None

    expected_scene = "two_step_chain" if len(chain) == 2 else TOOL_SCENE[final_tool]
    fixed_scene = item.get("scene") != expected_scene
    if fixed_scene:
        fixes["fixed_scene"] += 1

    cleaned_conversations: list[dict[str, str]] = [
        {"from": "system", "value": TRIPAI_TOOL_USE_SYSTEM_PROMPT},
        {"from": "human", "value": question},
    ]
    for call, observation in zip(normalized_calls, normalized_observations):
        cleaned_conversations.append({"from": "function_call", "value": _json_dumps(call)})
        cleaned_conversations.append({"from": "observation", "value": _json_dumps(observation)})
    cleaned_conversations.append({"from": "gpt", "value": answer})

    cleaned_item = {
        "id": raw_id,
        "task_type": "tool_result_grounded_answer",
        "subtype": subtype,
        "scene": expected_scene,
        "expected_behavior": "should_call_tool",
        "expected_tool_chain": list(chain),
        "expected_arguments_subset": expected_subset,
        "quality_tags": _quality_tags(
            subtype,
            item.get("quality_tags"),
            chain=chain,
            fixed_subset=fixed_subset,
            fixed_scene=fixed_scene,
        ),
        "conversations": cleaned_conversations,
    }
    score = _candidate_score(subtype, final_tool, normalized_calls[-1]["arguments"], normalized_observations[-1], answer)
    return Candidate(
        source=source,
        line_number=line_number,
        raw_id=raw_id,
        subtype=subtype,
        final_tool=final_tool,
        score=score,
        item=cleaned_item,
    )


def _select_candidates(
    candidates: list[Candidate],
    *,
    target_count: int,
    used_raw_ids: set[str],
    used_questions: set[str],
    used_function_calls: set[str],
    rejected: Counter[str],
) -> list[dict[str, Any]]:
    selected: list[Candidate] = []
    for candidate in candidates:
        if len(selected) >= target_count:
            break
        if candidate.raw_id in used_raw_ids:
            rejected["duplicate_raw_id"] += 1
            continue
        question = candidate.item["conversations"][1]["value"]
        function_calls = tuple(
            message["value"] for message in candidate.item["conversations"] if message["from"] == "function_call"
        )
        function_key = "\n".join(function_calls)
        if question in used_questions:
            rejected["duplicate_question"] += 1
            continue
        if function_key in used_function_calls:
            rejected["duplicate_function_call"] += 1
            continue
        selected.append(candidate)
        used_raw_ids.add(candidate.raw_id)
        used_questions.add(question)
        used_function_calls.add(function_key)

    if len(selected) < target_count:
        raise RuntimeError(f"expected {target_count} selected records, got {len(selected)}")
    return [candidate.item for candidate in selected]


def _renumber_selected(rows_by_subtype: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for subtype in SUBTYPE_ORDER:
        for index, row in enumerate(rows_by_subtype[subtype], start=1):
            item = dict(row)
            item["id"] = f"stage2_v2_tool_result_grounded_answer_{subtype}_{index:06d}"
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


def clean_tool_result_grounded_answer(
    rows: list[tuple[str, int, dict[str, Any]]] | list[tuple[int, dict[str, Any]]],
    *,
    target_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = dict(target_counts or TARGET_COUNTS)
    if set(targets) != set(SUBTYPE_ORDER):
        raise ValueError(f"target_counts must contain exactly: {SUBTYPE_ORDER!r}")

    rejected: Counter[str] = Counter()
    fixes: Counter[str] = Counter()
    candidates_by_subtype: dict[str, list[Candidate]] = {subtype: [] for subtype in SUBTYPE_ORDER}
    input_sources: Counter[str] = Counter()

    for row in rows:
        if len(row) == 2:
            line_number, item = row  # type: ignore[misc]
            source = "<memory>"
        else:
            source, line_number, item = row  # type: ignore[misc]
        input_sources[str(source)] += 1
        candidate = _clean_one(str(source), int(line_number), item, rejected, fixes)
        if candidate is None:
            continue
        candidates_by_subtype[candidate.subtype].append(candidate)

    for subtype, candidates in candidates_by_subtype.items():
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.source.endswith("_v2.jsonl") is False, candidate.raw_id, candidate.line_number))
        if len(candidates) < targets[subtype]:
            raise RuntimeError(f"{subtype} has only {len(candidates)} clean candidates, but target={targets[subtype]}")

    used_raw_ids: set[str] = set()
    used_questions: set[str] = set()
    used_function_calls: set[str] = set()
    selected_by_subtype: dict[str, list[dict[str, Any]]] = {}
    for subtype in SUBTYPE_ORDER:
        selected_by_subtype[subtype] = _select_candidates(
            candidates_by_subtype[subtype],
            target_count=targets[subtype],
            used_raw_ids=used_raw_ids,
            used_questions=used_questions,
            used_function_calls=used_function_calls,
            rejected=rejected,
        )

    cleaned = _renumber_selected(selected_by_subtype)
    validation_errors = _validate_processed_rows(cleaned)
    if validation_errors:
        raise RuntimeError(f"processed tool_result_grounded_answer failed validation: {validation_errors[:5]}")

    selected_counts = Counter(item["subtype"] for item in cleaned)
    candidate_counts = {subtype: len(candidates_by_subtype[subtype]) for subtype in SUBTYPE_ORDER}
    answer_lengths = [len(item["conversations"][-1]["value"]) for item in cleaned]
    tool_counts = Counter()
    route_modes = Counter()
    city_counts = Counter()
    chain_counts = Counter()
    function_calls = Counter()
    questions = Counter()
    raw_seconds_answers = 0
    raw_long_meters_answers = 0
    for item in cleaned:
        questions[item["conversations"][1]["value"]] += 1
        chain: list[str] = []
        for message in item["conversations"]:
            if message["from"] != "function_call":
                continue
            function_calls[message["value"]] += 1
            payload = json.loads(message["value"])
            tool_name = payload["name"]
            args = payload["arguments"]
            chain.append(tool_name)
            tool_counts[tool_name] += 1
            if isinstance(args.get("city"), str):
                city_counts[args["city"]] += 1
            if tool_name == "amap_plan_route":
                route_modes[args.get("mode")] += 1
        chain_counts[" -> ".join(chain)] += 1
        answer = item["conversations"][-1]["value"]
        raw_seconds_answers += int(bool(RAW_SECONDS_RE.search(answer)))
        raw_long_meters_answers += int(bool(RAW_LONG_METERS_RE.search(answer)))

    report = {
        "input_records": sum(input_sources.values()),
        "input_sources": dict(sorted(input_sources.items())),
        "target_counts": targets,
        "target_total": sum(targets.values()),
        "candidate_counts": candidate_counts,
        "selected_count": len(cleaned),
        "selected_counts": dict(sorted(selected_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "tool_chain_counts": dict(sorted(chain_counts.items())),
        "route_mode_counts": dict(sorted(route_modes.items())),
        "city_count": len(city_counts),
        "top_cities": [{"city": city, "count": count} for city, count in city_counts.most_common(20)],
        "selected_duplicate_questions": sum(count - 1 for count in questions.values() if count > 1),
        "selected_duplicate_function_calls": sum(count - 1 for count in function_calls.values() if count > 1),
        "selected_raw_seconds_answers": raw_seconds_answers,
        "selected_raw_long_meters_answers": raw_long_meters_answers,
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
    parser = argparse.ArgumentParser(
        description="Clean raw Stage2 tool_result_grounded_answer JSONL into processed Stage2 JSONL."
    )
    parser.add_argument("--input", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--route-target", type=int, default=TARGET_COUNTS["route_grounded_answer"])
    parser.add_argument("--poi-target", type=int, default=TARGET_COUNTS["poi_grounded_answer"])
    parser.add_argument("--geocode-target", type=int, default=TARGET_COUNTS["geocode_grounded_answer"])
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    rows = _read_jsonl_inputs(args.input)
    cleaned, report = clean_tool_result_grounded_answer(
        rows,
        target_counts={
            "route_grounded_answer": args.route_target,
            "poi_grounded_answer": args.poi_target,
            "geocode_grounded_answer": args.geocode_target,
        },
    )
    output_path = _write_jsonl(args.output, cleaned)
    report_path = _write_json(args.report, report)
    print(f"[OK] cleaned tool_result_grounded_answer written to: {output_path}")
    print(f"[OK] report written to: {report_path}")
    print(f"[OK] selected: {report['selected_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
