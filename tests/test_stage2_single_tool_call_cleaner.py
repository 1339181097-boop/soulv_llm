from __future__ import annotations

import json

from src.data_pipeline.clean_stage2_single_tool_call import clean_single_tool_call


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
    observation_data: dict,
    expected_subset: dict,
) -> dict:
    return {
        "id": sample_id,
        "task_type": "single_tool_call",
        "subtype": subtype,
        "scene": scene,
        "expected_behavior": "should_call_tool",
        "expected_tool_chain": [tool_name],
        "expected_arguments_subset": expected_subset,
        "quality_tags": ["single_tool"],
        "conversations": [
            {"from": "system", "value": "TripAI tool rules."},
            {"from": "human", "value": human},
            {"from": "function_call", "value": _json({"name": tool_name, "arguments": arguments})},
            {"from": "observation", "value": _json({"status": "success", "data": observation_data})},
            {"from": "gpt", "value": "raw answer"},
        ],
    }


def test_clean_single_tool_call_balances_subtypes_and_rewrites_answers() -> None:
    rows = [
        (
            1,
            _raw_item(
                sample_id="raw_route",
                human="从甲地步行去乙地要多久？",
                subtype="direct_route",
                scene="amap_plan_route",
                tool_name="amap_plan_route",
                arguments={"origin": "甲地", "destination": "乙地", "mode": "walking", "city": "杭州"},
                observation_data={
                    "mode": "walking",
                    "origin": "甲地",
                    "destination": "乙地",
                    "duration_min": 20,
                    "distance_km": 1.6,
                    "main_steps": ["沿测试路向东步行100米右转", "沿样例街步行200米到达"],
                },
                expected_subset={"origin": "甲地", "destination": "乙地", "mode": "walking", "city": "杭州"},
            ),
        ),
        (
            2,
            _raw_item(
                sample_id="raw_geocode",
                human="雷峰塔的坐标是多少？",
                subtype="direct_geocode",
                scene="amap_geocode",
                tool_name="amap_geocode",
                arguments={"address": "雷峰塔", "city": "杭州"},
                observation_data={
                    "query": "雷峰塔",
                    "city": "杭州市",
                    "district": "西湖区",
                    "formatted_address": "浙江省杭州市西湖区雷峰塔",
                    "location": "120.148000,30.236000",
                },
                expected_subset={"address": "雷峰塔", "city": "杭州"},
            ),
        ),
        (
            3,
            _raw_item(
                sample_id="raw_poi",
                human="杭州西湖附近有药店吗？",
                subtype="direct_poi",
                scene="amap_search_poi",
                tool_name="amap_search_poi",
                arguments={"keyword": "药店", "city": "杭州", "around_location": "杭州西湖"},
                observation_data={
                    "keyword": "药店",
                    "city": "杭州",
                    "around_location": "120.1,30.2",
                    "count": 2,
                    "pois": [
                        {"name": "甲药店", "address": "测试路1号", "distance": "230"},
                        {"name": "乙药店", "address": "样例街2号", "distance": "560"},
                    ],
                },
                expected_subset={"amap_search_poi": {"keyword": "药店", "city": "杭州", "around_location": "杭州西湖"}},
            ),
        ),
    ]

    cleaned, report = clean_single_tool_call(rows, target_per_subtype=1)

    assert len(cleaned) == 3
    assert report["selected_counts"] == {"direct_geocode": 1, "direct_poi": 1, "direct_route": 1}
    assert report["fixes"] == {"nested_expected_arguments_subset": 1}
    assert all(item["conversations"][-1]["value"] != "raw answer" for item in cleaned)
    assert all(tuple(message["from"] for message in item["conversations"]) == ("system", "human", "function_call", "observation", "gpt") for item in cleaned)
