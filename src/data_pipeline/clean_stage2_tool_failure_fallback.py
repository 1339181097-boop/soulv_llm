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
    EMOJI_RE,
    _flatten_expected_subset,
    _json_dumps,
    _loads_object,
    _normalize_text,
    _ordered_unique,
    _parse_conversations,
    _read_jsonl,
    _to_source_dataset,
    _write_json,
    _write_jsonl,
)
from src.data_pipeline.data_utils import configure_console_output
from src.tool_use.datasets import export_tool_use_dataset_to_sharegpt, validate_sharegpt_tool_dataset, validate_tool_use_source_dataset
from src.tool_use.protocol import TRIPAI_TOOL_USE_SYSTEM_PROMPT

DEFAULT_INPUT = Path("data/raw_stage2/tool_failure_fallback.jsonl")
DEFAULT_OUTPUT = Path("data/processed_stage2/tool_failure_fallback.jsonl")
DEFAULT_REPORT = Path("data/processed_stage2/tool_failure_fallback_report.json")

TARGET_COUNTS = {
    "route_error_or_empty": 60,
    "poi_error_or_empty": 60,
    "geocode_error_or_empty": 40,
}
SUBTYPE_ORDER = ("route_error_or_empty", "poi_error_or_empty", "geocode_error_or_empty")
SUBTYPE_TO_TOOL = {
    "route_error_or_empty": ("amap_plan_route", "amap_plan_route"),
    "poi_error_or_empty": ("amap_search_poi", "amap_search_poi"),
    "geocode_error_or_empty": ("amap_geocode", "amap_geocode"),
}

ARG_ORDER = {
    "amap_geocode": ("address", "city"),
    "amap_search_poi": ("keyword", "city", "around_location", "radius_m"),
    "amap_plan_route": ("origin", "destination", "mode", "city"),
}
ALLOWED_TOOL_ARGS = {tool: set(args) for tool, args in ARG_ORDER.items()}
REQUIRED_TOOL_ARGS = {
    "amap_geocode": {"address"},
    "amap_search_poi": {"keyword"},
    "amap_plan_route": {"origin", "destination", "mode"},
}
ROUTE_MODES = {"transit", "driving", "walking", "bicycling"}

COORD_RE = re.compile(r"-?\d+(?:\.\d+)?\s*[,，]\s*-?\d+(?:\.\d+)?")
CONCRETE_ROUTE_RE = re.compile(r"(?:乘坐|搭乘|换乘).{0,20}(?:号线|公交|巴士|专线|路)")
CONCRETE_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:米|公里|km|KM|分钟|小时|秒)")


@dataclass(frozen=True)
class Candidate:
    line_number: int
    raw_id: str
    subtype: str
    status: str
    score: int
    item: dict[str, Any]


def _normalizable_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _normalize_text(value)


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
        text = _normalizable_string(value)
        if not text:
            return None
        normalized[key] = text

    for key in REQUIRED_TOOL_ARGS[tool_name]:
        if key not in normalized:
            return None
    if tool_name == "amap_plan_route" and normalized.get("mode") not in ROUTE_MODES:
        return None
    return normalized


def _normalize_failure_observation(raw_observation: dict[str, Any]) -> dict[str, Any] | None:
    status = raw_observation.get("status")
    if status not in {"empty", "error"}:
        return None
    data = raw_observation.get("data")
    if data not in ({}, None):
        return None
    if status == "empty":
        return {"status": "empty", "reason": "no_result"}
    return {"status": "error", "reason": "amap_request_failed", "retryable": False}


def _fallback_answer(tool_name: str, status: str) -> str:
    if tool_name == "amap_plan_route":
        if status == "error":
            return (
                "刚才路线工具请求失败，我暂时拿不到可靠路线结果，先不硬编。"
                "你可以补充更具体的起终点、出行方式或出发时间，稍后再试，或用地图 App 实时核对。"
            )
        return (
            "刚才路线工具没有返回可靠结果，我暂时无法确认可用路线，先不硬编。"
            "你可以补充更具体的起终点、出行方式或出发时间，稍后再试，或用地图 App 实时核对。"
        )
    if tool_name == "amap_search_poi":
        if status == "error":
            return (
                "刚才周边检索工具请求失败，我暂时拿不到可靠地点结果，先不硬编。"
                "你可以补充更具体的城市、地址、附近地标或换个关键词，稍后再试，或用地图 App 实时核对。"
            )
        return (
            "刚才周边检索工具没有返回可靠结果，我暂时无法确认附近可用地点，先不硬编。"
            "你可以补充更具体的城市、地址、附近地标或换个关键词，稍后再试，或用地图 App 实时核对。"
        )
    if status == "error":
        return (
            "刚才定位工具请求失败，我暂时拿不到可靠位置结果，先不硬编。"
            "你可以补充更完整的城市、详细地址或附近地标，稍后再试，或用地图 App 实时核对。"
        )
    return (
        "刚才定位工具没有返回可靠结果，我暂时无法确认准确位置，先不硬编。"
        "你可以补充更完整的城市、详细地址或附近地标，稍后再试，或用地图 App 实时核对。"
    )


def _answer_is_safe_fallback(answer: str) -> bool:
    if not answer or "\n" in answer:
        return False
    if EMOJI_RE.search(answer):
        return False
    if any(token in answer for token in ("<think>", "</think>", "reasoning_content", "```", "**")):
        return False
    if any(token in answer for token in ("tool_call", "observation", "formatted_address", "route_steps", '"status"')):
        return False
    if COORD_RE.search(answer) or CONCRETE_ROUTE_RE.search(answer) or CONCRETE_UNIT_RE.search(answer):
        return False
    return all(token in answer for token in ("暂时", "可靠", "不硬编"))


def _quality_tags(subtype: str, status: str, raw_tags: Any, *, fixed_subset: bool) -> list[str]:
    tags = [
        "tool_failure_fallback",
        subtype,
        f"observation_{status}",
        "single_tool",
        "explicit_fallback",
        "no_hallucination",
        "no_route_coordinate_poi_output",
    ]
    if fixed_subset:
        tags.append("fixed_expected_arguments_subset")
    if isinstance(raw_tags, list):
        tags.extend(str(tag).strip() for tag in raw_tags if isinstance(tag, str) and tag.strip())
    return _ordered_unique(tags)


def _candidate_score(subtype: str, status: str, args: dict[str, Any], raw_answer: str) -> int:
    score = 100
    if status == "error":
        score += 4
    if subtype == "route_error_or_empty":
        if args.get("mode") in {"transit", "walking", "bicycling"}:
            score += 5
        if args.get("city"):
            score += 4
    elif subtype == "poi_error_or_empty":
        if args.get("around_location"):
            score += 8
        if args.get("radius_m"):
            score += 4
        if args.get("city"):
            score += 4
    elif subtype == "geocode_error_or_empty" and args.get("city"):
        score += 4
    if 35 <= len(raw_answer) <= 220:
        score += 3
    return score


def _clean_one(line_number: int, item: dict[str, Any], rejected: Counter[str], fixes: Counter[str]) -> Candidate | None:
    raw_id = _normalize_text(item.get("id"))
    subtype = _normalize_text(item.get("subtype"))
    if item.get("task_type") != "tool_failure_fallback":
        rejected["bad_task_type"] += 1
        return None
    if subtype not in SUBTYPE_TO_TOOL:
        rejected["bad_subtype"] += 1
        return None

    expected_scene, expected_tool = SUBTYPE_TO_TOOL[subtype]
    if item.get("scene") != expected_scene:
        rejected["scene_subtype_mismatch"] += 1
        return None
    if item.get("expected_behavior") != "should_fallback":
        rejected["bad_expected_behavior"] += 1
        return None
    if item.get("expected_tool_chain") != [expected_tool]:
        rejected["bad_expected_tool_chain"] += 1
        return None

    conversations, conversation_error = _parse_conversations(item)
    if conversations is None:
        rejected[conversation_error or "bad_conversations"] += 1
        return None

    question = _normalizable_string(next(message["value"] for message in conversations if message["from"] == "human"))
    if not question or EMOJI_RE.search(question):
        rejected["bad_question"] += 1
        return None

    function_payload = _loads_object(next(message["value"] for message in conversations if message["from"] == "function_call"))
    raw_observation = _loads_object(next(message["value"] for message in conversations if message["from"] == "observation"))
    if not function_payload:
        rejected["bad_function_call_json"] += 1
        return None
    if not raw_observation:
        rejected["bad_observation_json"] += 1
        return None

    tool_name = function_payload.get("name")
    if tool_name != expected_tool:
        rejected["function_tool_mismatch"] += 1
        return None
    normalized_args = _normalize_arguments(tool_name, function_payload.get("arguments"))
    if normalized_args is None:
        rejected["bad_tool_arguments"] += 1
        return None

    expected_subset, fixed_subset = _flatten_expected_subset(item.get("expected_arguments_subset"), tool_name)
    if expected_subset is None:
        rejected["bad_expected_arguments_subset"] += 1
        return None
    for key, value in expected_subset.items():
        if normalized_args.get(key) != value:
            rejected["expected_arguments_subset_mismatch"] += 1
            return None
    if fixed_subset:
        fixes["nested_expected_arguments_subset"] += 1

    observation = _normalize_failure_observation(raw_observation)
    if observation is None:
        rejected["observation_not_error_or_empty"] += 1
        return None
    status = observation["status"]

    answer = _fallback_answer(tool_name, status)
    if not _answer_is_safe_fallback(answer):
        rejected["rewritten_answer_not_safe"] += 1
        return None

    cleaned_item = {
        "id": raw_id,
        "task_type": "tool_failure_fallback",
        "subtype": subtype,
        "scene": expected_scene,
        "expected_behavior": "should_fallback",
        "expected_tool_chain": [expected_tool],
        "expected_arguments_subset": normalized_args,
        "quality_tags": _quality_tags(subtype, status, item.get("quality_tags"), fixed_subset=fixed_subset),
        "conversations": [
            {"from": "system", "value": TRIPAI_TOOL_USE_SYSTEM_PROMPT},
            {"from": "human", "value": question},
            {"from": "function_call", "value": _json_dumps({"name": tool_name, "arguments": normalized_args})},
            {"from": "observation", "value": _json_dumps(observation)},
            {"from": "gpt", "value": answer},
        ],
    }
    raw_answer = _normalizable_string(next(message["value"] for message in conversations if message["from"] == "gpt"))
    score = _candidate_score(subtype, status, normalized_args, raw_answer)
    return Candidate(line_number=line_number, raw_id=raw_id, subtype=subtype, status=status, score=score, item=cleaned_item)


def _interleave_by_status(candidates: list[Candidate]) -> list[Candidate]:
    buckets = {
        "error": [candidate for candidate in candidates if candidate.status == "error"],
        "empty": [candidate for candidate in candidates if candidate.status == "empty"],
    }
    ordered: list[Candidate] = []
    next_status = "error" if len(buckets["error"]) <= len(buckets["empty"]) else "empty"
    while buckets["error"] or buckets["empty"]:
        other = "empty" if next_status == "error" else "error"
        if buckets[next_status]:
            ordered.append(buckets[next_status].pop(0))
        elif buckets[other]:
            ordered.append(buckets[other].pop(0))
        next_status = other
    return ordered


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
    selected_keys: set[tuple[int, str]] = set()
    ordered = _interleave_by_status(candidates)

    def add_phase(*, require_unique_question: bool, require_unique_function_call: bool) -> None:
        if len(selected) >= target_count:
            return
        for candidate in ordered:
            if len(selected) >= target_count:
                return
            key = (candidate.line_number, candidate.raw_id)
            if key in selected_keys:
                continue
            if candidate.raw_id in used_raw_ids:
                rejected["duplicate_raw_id"] += 1
                continue
            question = candidate.item["conversations"][1]["value"]
            function_call = candidate.item["conversations"][2]["value"]
            if require_unique_question and question in used_questions:
                continue
            if require_unique_function_call and function_call in used_function_calls:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            used_raw_ids.add(candidate.raw_id)
            used_questions.add(question)
            used_function_calls.add(function_call)

    add_phase(require_unique_question=True, require_unique_function_call=True)
    add_phase(require_unique_question=True, require_unique_function_call=False)
    add_phase(require_unique_question=False, require_unique_function_call=False)

    if len(selected) < target_count:
        raise RuntimeError(f"expected {target_count} selected records, got {len(selected)}")
    return [candidate.item for candidate in selected]


def _renumber_selected(rows_by_subtype: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for subtype in SUBTYPE_ORDER:
        for index, row in enumerate(rows_by_subtype[subtype], start=1):
            item = dict(row)
            item["id"] = f"stage2_v2_tool_failure_fallback_{subtype}_{index:06d}"
            cleaned.append(item)
    return cleaned


def _validate_processed_rows(rows: list[dict[str, Any]]) -> list[str]:
    source_dataset = _to_source_dataset(rows)
    errors = validate_tool_use_source_dataset(source_dataset)
    if errors:
        return errors
    sharegpt_dataset = export_tool_use_dataset_to_sharegpt(source_dataset)
    return validate_sharegpt_tool_dataset(sharegpt_dataset)


def clean_tool_failure_fallback(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    target_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = dict(target_counts or TARGET_COUNTS)
    if set(targets) != set(SUBTYPE_ORDER):
        raise ValueError(f"target_counts must contain exactly: {SUBTYPE_ORDER!r}")

    rejected: Counter[str] = Counter()
    fixes: Counter[str] = Counter()
    candidates_by_subtype: dict[str, list[Candidate]] = {subtype: [] for subtype in SUBTYPE_ORDER}

    for line_number, item in rows:
        candidate = _clean_one(line_number, item, rejected, fixes)
        if candidate is None:
            continue
        candidates_by_subtype[candidate.subtype].append(candidate)

    for subtype, candidates in candidates_by_subtype.items():
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.raw_id, candidate.line_number))
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
        raise RuntimeError(f"processed tool_failure_fallback failed validation: {validation_errors[:5]}")

    selected_counts = Counter(item["subtype"] for item in cleaned)
    candidate_counts = {subtype: len(candidates_by_subtype[subtype]) for subtype in SUBTYPE_ORDER}
    answer_lengths = [len(item["conversations"][-1]["value"]) for item in cleaned]
    tool_counts = Counter()
    status_counts = Counter()
    route_modes = Counter()
    city_counts = Counter()
    function_calls = Counter()
    questions = Counter()
    for item in cleaned:
        questions[item["conversations"][1]["value"]] += 1
        function_payload = json.loads(item["conversations"][2]["value"])
        observation = json.loads(item["conversations"][3]["value"])
        function_calls[item["conversations"][2]["value"]] += 1
        tool_name = function_payload["name"]
        args = function_payload["arguments"]
        tool_counts[tool_name] += 1
        status_counts[observation["status"]] += 1
        if isinstance(args.get("city"), str):
            city_counts[args["city"]] += 1
        if tool_name == "amap_plan_route":
            route_modes[args.get("mode")] += 1

    report = {
        "input_records": len(rows),
        "target_counts": targets,
        "target_total": sum(targets.values()),
        "candidate_counts": candidate_counts,
        "selected_count": len(cleaned),
        "selected_counts": dict(sorted(selected_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "observation_status_counts": dict(sorted(status_counts.items())),
        "route_mode_counts": dict(sorted(route_modes.items())),
        "city_count": len(city_counts),
        "top_cities": city_counts.most_common(20),
        "selected_duplicate_questions": sum(count - 1 for count in questions.values() if count > 1),
        "selected_duplicate_function_calls": sum(count - 1 for count in function_calls.values() if count > 1),
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
    parser = argparse.ArgumentParser(description="Clean raw Stage2 tool_failure_fallback JSONL into processed Stage2 JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--route-target", type=int, default=TARGET_COUNTS["route_error_or_empty"])
    parser.add_argument("--poi-target", type=int, default=TARGET_COUNTS["poi_error_or_empty"])
    parser.add_argument("--geocode-target", type=int, default=TARGET_COUNTS["geocode_error_or_empty"])
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    target_counts = {
        "route_error_or_empty": args.route_target,
        "poi_error_or_empty": args.poi_target,
        "geocode_error_or_empty": args.geocode_target,
    }
    rows = _read_jsonl(args.input)
    cleaned, report = clean_tool_failure_fallback(rows, target_counts=target_counts)
    output_path = _write_jsonl(args.output, cleaned)
    report_path = _write_json(args.report, report)
    print(f"[OK] cleaned tool_failure_fallback written to: {output_path}")
    print(f"[OK] report written to: {report_path}")
    print(f"[OK] selected: {report['selected_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
