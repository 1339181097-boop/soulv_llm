from __future__ import annotations

import json

from src.data_pipeline.clean_stage2_tool_failure_fallback import clean_tool_failure_fallback


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _raw_item(
    *,
    sample_id: str,
    human: str,
    subtype: str,
    scene: str,
    tool_name: str,
    arguments: dict,
    observation: dict,
    expected_subset: dict,
    raw_answer: str = "虽然工具失败了，但你可以乘坐1号线到达。",
) -> dict:
    return {
        "id": sample_id,
        "task_type": "tool_failure_fallback",
        "subtype": subtype,
        "scene": scene,
        "expected_behavior": "should_fallback",
        "expected_tool_chain": [tool_name],
        "expected_arguments_subset": expected_subset,
        "quality_tags": ["fallback"],
        "conversations": [
            {"from": "system", "value": "TripAI tool rules."},
            {"from": "human", "value": human},
            {"from": "function_call", "value": _json({"name": tool_name, "arguments": arguments})},
            {"from": "observation", "value": _json(observation)},
            {"from": "gpt", "value": raw_answer},
        ],
    }


def test_clean_tool_failure_fallback_balances_subtypes_normalizes_and_rewrites_answers() -> None:
    rows = [
        (
            1,
            _raw_item(
                sample_id="raw_route",
                human="从甲地坐公交去乙地怎么走？",
                subtype="route_error_or_empty",
                scene="amap_plan_route",
                tool_name="amap_plan_route",
                arguments={"origin": "甲地", "destination": "乙地", "mode": "transit", "city": "杭州"},
                observation={"status": "error", "message": "timeout", "data": {}},
                expected_subset={"origin": "甲地", "destination": "乙地", "mode": "transit", "city": "杭州"},
            ),
        ),
        (
            2,
            _raw_item(
                sample_id="raw_poi",
                human="杭州西湖附近有没有咖啡店？",
                subtype="poi_error_or_empty",
                scene="amap_search_poi",
                tool_name="amap_search_poi",
                arguments={"keyword": "咖啡店", "city": "杭州", "around_location": "西湖", "radius_m": 1000},
                observation={"status": "empty", "reason": "no_result"},
                expected_subset={
                    "amap_search_poi": {
                        "keyword": "咖啡店",
                        "city": "杭州",
                        "around_location": "西湖",
                        "radius_m": 1000,
                    }
                },
            ),
        ),
        (
            3,
            _raw_item(
                sample_id="raw_geocode",
                human="帮我确认雷峰塔的位置。",
                subtype="geocode_error_or_empty",
                scene="amap_geocode",
                tool_name="amap_geocode",
                arguments={"address": "雷峰塔", "city": "杭州"},
                observation={"status": "empty", "data": {}, "message": "empty"},
                expected_subset={"address": "雷峰塔", "city": "杭州"},
            ),
        ),
        (
            4,
            _raw_item(
                sample_id="bad_scene",
                human="北京附近找酒店。",
                subtype="poi_error_or_empty",
                scene="amap_geocode",
                tool_name="amap_search_poi",
                arguments={"keyword": "酒店", "city": "北京"},
                observation={"status": "empty"},
                expected_subset={"keyword": "酒店", "city": "北京"},
            ),
        ),
    ]

    cleaned, report = clean_tool_failure_fallback(
        rows,
        target_counts={"route_error_or_empty": 1, "poi_error_or_empty": 1, "geocode_error_or_empty": 1},
    )

    assert len(cleaned) == 3
    assert report["selected_counts"] == {
        "geocode_error_or_empty": 1,
        "poi_error_or_empty": 1,
        "route_error_or_empty": 1,
    }
    assert report["fixes"] == {"nested_expected_arguments_subset": 1}
    assert report["rejected"]["scene_subtype_mismatch"] == 1
    assert all(item["conversations"][-1]["value"] != "虽然工具失败了，但你可以乘坐1号线到达。" for item in cleaned)
    assert all("乘坐1号线" not in item["conversations"][-1]["value"] for item in cleaned)
    assert all(tuple(message["from"] for message in item["conversations"]) == ("system", "human", "function_call", "observation", "gpt") for item in cleaned)

    observations = [json.loads(item["conversations"][3]["value"]) for item in cleaned]
    assert observations == [
        {"status": "error", "reason": "amap_request_failed", "retryable": False},
        {"status": "empty", "reason": "no_result"},
        {"status": "empty", "reason": "no_result"},
    ]
