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
    _flatten_expected_subset,
    _format_meter_distance,
    _json_dumps,
    _loads_object,
    _normalize_text,
    _parse_conversations,
    _read_jsonl,
    _route_answer,
    _to_source_dataset,
    _write_json,
    _write_jsonl,
)
from src.data_pipeline.data_utils import configure_console_output
from src.tool_use.datasets import export_tool_use_dataset_to_sharegpt, validate_sharegpt_tool_dataset, validate_tool_use_source_dataset
from src.tool_use.protocol import TRIPAI_TOOL_USE_SYSTEM_PROMPT

DEFAULT_INPUT = Path("data/raw_stage2/slot_filling_tool_call.jsonl")
DEFAULT_OUTPUT = Path("data/processed_stage2/slot_filling_tool_call.jsonl")
DEFAULT_REPORT = Path("data/processed_stage2/slot_filling_tool_call_report.json")
TARGET_PER_SUBTYPE = 120

SUBTYPE_TO_TOOL = {
    "route_slot_filling": ("amap_plan_route", "amap_plan_route"),
    "poi_slot_filling": ("amap_search_poi", "amap_search_poi"),
}
SUBTYPE_ORDER = ("route_slot_filling", "poi_slot_filling")

ARG_ORDER = {
    "amap_plan_route": ("origin", "destination", "mode", "city"),
    "amap_search_poi": ("keyword", "city", "around_location", "radius_m"),
}
ALLOWED_TOOL_ARGS = {tool: set(args) for tool, args in ARG_ORDER.items()}
REQUIRED_TOOL_ARGS = {tool: set(args) for tool, args in ARG_ORDER.items()}
ROUTE_MODES = {"transit", "driving", "walking", "bicycling"}
ROUTE_MODE_PHRASES = {
    "transit": ("坐地铁公交", "用公共交通", "坐公交地铁"),
    "driving": ("开车", "自驾", "驾车"),
    "walking": ("步行", "走路", "走过去"),
    "bicycling": ("骑车", "骑行", "骑自行车"),
}

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
COORD_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*$")


@dataclass(frozen=True)
class Candidate:
    line_number: int
    raw_id: str
    subtype: str
    score: int
    item: dict[str, Any]


def _is_coordinate(value: Any) -> bool:
    return isinstance(value, str) and bool(COORD_RE.match(value))


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
    if not keyword or not city or not isinstance(pois, list):
        return None

    valid_pois = [poi for poi in pois if isinstance(poi, dict) and str(poi.get("name") or "").strip()]
    if not valid_pois:
        return None
    selected = valid_pois[: min(5, len(valid_pois))]
    result_text = "；".join(_poi_result_text(poi) for poi in selected)
    if not result_text:
        return None

    radius = args.get("radius_m")
    radius_text = f"{radius}米内" if isinstance(radius, int) and radius > 0 else ""
    if around and not _is_coordinate(around):
        place = around if around.startswith(city) else f"{city}{around}"
        prefix = f"{place}附近{radius_text}可以优先看这些{keyword}："
    else:
        prefix = f"{city}{radius_text}有这些{keyword}可以优先看："

    if len(selected) == 1:
        suffix = "目前工具只返回这一处结果，建议出发前再用地图确认营业状态和排队情况。"
    else:
        suffix = "建议优先选距离更近、地址更匹配行程的一家。"
    return prefix + result_text + "。" + suffix


def _rewrite_answer(tool_name: str, args: dict[str, Any], observation: dict[str, Any]) -> str | None:
    if observation.get("status") != "success":
        return None
    data = observation.get("data")
    if not isinstance(data, dict):
        return None
    if tool_name == "amap_plan_route":
        return _route_answer(args, data)
    if tool_name == "amap_search_poi":
        return _poi_answer(args, data)
    return None


def _quality_tags(subtype: str, raw_tags: Any, *, fixed_subset: bool, single_poi_result: bool) -> list[str]:
    tags = [
        "slot_filling",
        "single_tool",
        subtype,
        "success_observation",
        "strict_arguments",
        "grounded_answer",
        "human_readable_units",
    ]
    if fixed_subset:
        tags.append("fixed_expected_arguments_subset")
    if single_poi_result:
        tags.append("single_poi_result_grounded")
    if isinstance(raw_tags, list):
        tags.extend(str(tag).strip() for tag in raw_tags if isinstance(tag, str) and tag.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def _candidate_score(subtype: str, args: dict[str, Any], observation: dict[str, Any], answer: str) -> int:
    score = 100
    data = observation.get("data") if isinstance(observation.get("data"), dict) else {}
    if subtype == "route_slot_filling":
        duration = data.get("duration_min")
        distance = data.get("distance_km")
        if isinstance(duration, (int, float)) and duration > 0:
            score += 8
        if isinstance(distance, (int, float)) and distance > 0:
            score += 8
        if args.get("mode") in {"transit", "walking", "bicycling"}:
            score += 4
        if args.get("mode") == "walking" and isinstance(distance, (int, float)) and distance > 30:
            score -= 40
        if isinstance(distance, (int, float)) and distance > 150:
            score -= 20
    elif subtype == "poi_slot_filling":
        pois = data.get("pois")
        if isinstance(pois, list):
            score += min(len(pois), 5) * 4
            if len(pois) == 1:
                score -= 12
            if any(str(poi.get("distance") or "").strip() for poi in pois if isinstance(poi, dict)):
                score += 8
        if args.get("around_location"):
            score += 8
        if isinstance(args.get("radius_m"), int):
            score += 8
    if 40 <= len(answer) <= 260:
        score += 5
    return score


def _clean_one(line_number: int, item: dict[str, Any], rejected: Counter[str], fixes: Counter[str]) -> Candidate | None:
    raw_id = _normalize_text(item.get("id"))
    subtype = _normalize_text(item.get("subtype"))
    if item.get("task_type") != "slot_filling_tool_call":
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
        if arg_name == "radius_m":
            if not isinstance(value, int) or value <= 0:
                rejected["missing_required_argument_radius_m"] += 1
                return None
            continue
        if not isinstance(value, str) or not value.strip():
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

    if observation.get("status") != "success":
        rejected["observation_not_success"] += 1
        return None
    data = observation.get("data")
    if not isinstance(data, dict):
        rejected["observation_data_not_object"] += 1
        return None
    single_poi_result = False
    if tool_name == "amap_search_poi":
        pois = data.get("pois")
        if not isinstance(pois, list) or not pois:
            rejected["poi_observation_missing_pois"] += 1
            return None
        single_poi_result = len([poi for poi in pois if isinstance(poi, dict) and poi.get("name")]) == 1

    answer = _rewrite_answer(tool_name, args, observation)
    if answer is None:
        rejected["answer_rewrite_failed"] += 1
        return None
    answer = _normalize_text(answer)
    if not _answer_is_clean(answer):
        rejected["rewritten_answer_not_clean"] += 1
        return None

    normalized_args = {key: args[key] for key in ARG_ORDER[tool_name]}
    normalized_observation = {"status": "success", "data": data}
    cleaned_item = {
        "id": raw_id,
        "task_type": "slot_filling_tool_call",
        "subtype": subtype,
        "scene": expected_scene,
        "expected_behavior": "should_call_tool",
        "expected_tool_chain": [expected_tool],
        "expected_arguments_subset": normalized_args,
        "quality_tags": _quality_tags(
            subtype,
            item.get("quality_tags"),
            fixed_subset=fixed_subset,
            single_poi_result=single_poi_result,
        ),
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


def _variant_question(subtype: str, args: dict[str, Any], variant_index: int) -> str:
    if subtype == "route_slot_filling":
        mode = str(args.get("mode") or "transit")
        phrases = ROUTE_MODE_PHRASES.get(mode, ("出行",))
        phrase = phrases[variant_index % len(phrases)]
        origin = str(args.get("origin") or "").strip()
        destination = str(args.get("destination") or "").strip()
        city = str(args.get("city") or "").strip()
        templates = (
            "我在{city}，想从{origin}{phrase}去{destination}，帮我看下路线。",
            "从{origin}到{destination}，我打算{phrase}，请查一下大概怎么走。",
            "{city}这边，从{origin}{phrase}到{destination}大概要多久？",
            "帮我查查{city}{origin}去{destination}的{phrase}方案。",
            "我准备{phrase}，从{origin}出发到{destination}，路线怎么安排？",
        )
        return templates[variant_index % len(templates)].format(
            city=city,
            origin=origin,
            destination=destination,
            phrase=phrase,
        )

    keyword = str(args.get("keyword") or "").strip()
    city = str(args.get("city") or "").strip()
    around = str(args.get("around_location") or "").strip()
    radius = args.get("radius_m")
    radius_text = f"{radius}米内" if isinstance(radius, int) else "附近"
    templates = (
        "我在{city}，想找{around}附近{radius_text}的{keyword}，帮我查一下。",
        "帮我查查{city}{around}周边{radius_text}有哪些{keyword}。",
        "{city}{around}附近{radius_text}有没有合适的{keyword}？",
        "想在{around}旁边找{keyword}，范围先按{radius_text}看。",
        "给我看一下{city}{around}附近{radius_text}的{keyword}结果。",
    )
    return templates[variant_index % len(templates)].format(
        city=city,
        around=around,
        radius_text=radius_text,
        keyword=keyword,
    )


def _make_unique_question(
    candidate: Candidate,
    used_questions: set[str],
    variant_counts: Counter[str],
    fixes: Counter[str],
) -> tuple[dict[str, Any], bool]:
    item = json.loads(json.dumps(candidate.item, ensure_ascii=False))
    question = item["conversations"][1]["value"]
    if question not in used_questions:
        return item, False

    function_call = item["conversations"][2]["value"]
    payload = json.loads(function_call)
    args = payload["arguments"]
    variant_key = f"{candidate.subtype}:{function_call}"
    for _ in range(20):
        variant_index = variant_counts[variant_key]
        variant_counts[variant_key] += 1
        rewritten = _variant_question(candidate.subtype, args, variant_index)
        if rewritten and rewritten not in used_questions:
            item["conversations"][1]["value"] = rewritten
            item["quality_tags"] = [*item["quality_tags"], "rewritten_duplicate_question"]
            fixes["rewritten_duplicate_question"] += 1
            return item, True
    raise RuntimeError(f"could not create unique question for {candidate.raw_id}")


def _select_candidates(
    candidates: list[Candidate],
    *,
    target_count: int,
    used_questions: set[str],
    fixes: Counter[str],
) -> list[dict[str, Any]]:
    selected: list[Candidate] = []
    selected_keys: set[tuple[int, str]] = set()
    seen_function_calls: set[str] = set()

    def add_phase(*, require_unique_question: bool, require_unique_function_call: bool) -> None:
        if len(selected) >= target_count:
            return
        for candidate in candidates:
            if len(selected) >= target_count:
                return
            key = (candidate.line_number, candidate.raw_id)
            if key in selected_keys:
                continue
            question = candidate.item["conversations"][1]["value"]
            function_call = candidate.item["conversations"][2]["value"]
            if require_unique_question and question in used_questions:
                continue
            if require_unique_function_call and function_call in seen_function_calls:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            used_questions.add(question)
            seen_function_calls.add(function_call)

    add_phase(require_unique_question=True, require_unique_function_call=True)
    add_phase(require_unique_question=True, require_unique_function_call=False)

    variant_counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    rewritten_indices: set[int] = set()
    for index, candidate in enumerate(selected):
        output.append(candidate.item)
        rewritten_indices.add(index)

    if len(selected) < target_count:
        for candidate in candidates:
            if len(selected) >= target_count:
                break
            key = (candidate.line_number, candidate.raw_id)
            if key in selected_keys:
                continue
            item, _changed = _make_unique_question(candidate, used_questions, variant_counts, fixes)
            question = item["conversations"][1]["value"]
            selected.append(candidate)
            selected_keys.add(key)
            used_questions.add(question)
            seen_function_calls.add(item["conversations"][2]["value"])
            output.append(item)

    if len(output) != target_count:
        raise RuntimeError(f"expected {target_count} selected records, got {len(output)}")
    return output


def _renumber_selected(rows_by_subtype: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for subtype in SUBTYPE_ORDER:
        for index, row in enumerate(rows_by_subtype[subtype], start=1):
            item = dict(row)
            item["id"] = f"stage2_v2_slot_filling_tool_call_{subtype}_{index:06d}"
            cleaned.append(item)
    return cleaned


def _validate_processed_rows(rows: list[dict[str, Any]]) -> list[str]:
    source_dataset = _to_source_dataset(rows)
    errors = validate_tool_use_source_dataset(source_dataset)
    if errors:
        return errors
    sharegpt_dataset = export_tool_use_dataset_to_sharegpt(source_dataset)
    return validate_sharegpt_tool_dataset(sharegpt_dataset)


def clean_slot_filling_tool_call(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    target_per_subtype: int = TARGET_PER_SUBTYPE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        if len(candidates) < target_per_subtype:
            raise RuntimeError(
                f"{subtype} has only {len(candidates)} clean candidates, "
                f"but target_per_subtype={target_per_subtype}"
            )

    used_questions: set[str] = set()
    selected_by_subtype: dict[str, list[dict[str, Any]]] = {}
    for subtype in SUBTYPE_ORDER:
        selected_by_subtype[subtype] = _select_candidates(
            candidates_by_subtype[subtype],
            target_count=target_per_subtype,
            used_questions=used_questions,
            fixes=fixes,
        )

    cleaned = _renumber_selected(selected_by_subtype)
    validation_errors = _validate_processed_rows(cleaned)
    if validation_errors:
        raise RuntimeError(f"processed slot_filling_tool_call failed validation: {validation_errors[:5]}")

    selected_counts = Counter(item["subtype"] for item in cleaned)
    candidate_counts = {subtype: len(candidates_by_subtype[subtype]) for subtype in SUBTYPE_ORDER}
    answer_lengths = [len(item["conversations"][-1]["value"]) for item in cleaned]
    tool_counts = Counter()
    route_modes = Counter()
    poi_keywords = Counter()
    city_counts = Counter()
    function_calls = Counter()
    questions = Counter()
    single_poi_result_count = 0
    for item in cleaned:
        questions[item["conversations"][1]["value"]] += 1
        function_payload = json.loads(item["conversations"][2]["value"])
        function_calls[item["conversations"][2]["value"]] += 1
        tool_name = function_payload["name"]
        args = function_payload["arguments"]
        tool_counts[tool_name] += 1
        if isinstance(args.get("city"), str):
            city_counts[args["city"]] += 1
        if tool_name == "amap_plan_route":
            route_modes[args.get("mode")] += 1
        elif tool_name == "amap_search_poi":
            poi_keywords[args.get("keyword")] += 1
            observation = json.loads(item["conversations"][3]["value"])
            pois = observation.get("data", {}).get("pois")
            if isinstance(pois, list) and len(pois) == 1:
                single_poi_result_count += 1

    report = {
        "input_records": len(rows),
        "target_per_subtype": target_per_subtype,
        "target_total": target_per_subtype * len(SUBTYPE_ORDER),
        "candidate_counts": candidate_counts,
        "selected_count": len(cleaned),
        "selected_counts": dict(sorted(selected_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "route_mode_counts": dict(sorted(route_modes.items())),
        "poi_keyword_counts": dict(sorted(poi_keywords.items())),
        "city_count": len(city_counts),
        "top_cities": city_counts.most_common(20),
        "selected_duplicate_questions": sum(count - 1 for count in questions.values() if count > 1),
        "selected_duplicate_function_calls": sum(count - 1 for count in function_calls.values() if count > 1),
        "single_poi_result_count": single_poi_result_count,
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
    parser = argparse.ArgumentParser(description="Clean raw Stage2 slot_filling_tool_call JSONL into processed Stage2 JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-per-subtype", type=int, default=TARGET_PER_SUBTYPE)
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    rows = _read_jsonl(args.input)
    cleaned, report = clean_slot_filling_tool_call(rows, target_per_subtype=args.target_per_subtype)
    output_path = _write_jsonl(args.output, cleaned)
    report_path = _write_json(args.report, report)
    print(f"[OK] cleaned slot_filling_tool_call written to: {output_path}")
    print(f"[OK] report written to: {report_path}")
    print(f"[OK] selected: {report['selected_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
