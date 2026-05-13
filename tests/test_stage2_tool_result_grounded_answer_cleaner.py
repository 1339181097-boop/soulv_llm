from __future__ import annotations

import json

from src.data_pipeline.clean_stage2_tool_result_grounded_answer import clean_tool_result_grounded_answer


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
    raw_answer: str = "raw answer",
) -> dict:
    return {
        "id": sample_id,
        "task_type": "tool_result_grounded_answer",
        "subtype": subtype,
        "scene": scene,
        "expected_behavior": "should_call_tool",
        "expected_tool_chain": [tool_name],
        "expected_arguments_subset": expected_subset,
        "quality_tags": ["grounded"],
        "conversations": [
            {"from": "system", "value": "TripAI tool rules."},
            {"from": "human", "value": human},
            {"from": "function_call", "value": _json({"name": tool_name, "arguments": arguments})},
            {"from": "observation", "value": _json({"status": "success", "data": observation_data})},
            {"from": "gpt", "value": raw_answer},
        ],
    }


def test_clean_tool_result_grounded_answer_rewrites_and_balances() -> None:
    route_args = {"origin": "甲地", "destination": "乙地", "mode": "transit", "city": "杭州"}
    poi_args = {"keyword": "咖啡馆", "city": "杭州", "around_location": "西湖", "radius_m": 800}
    geocode_args = {"address": "雷峰塔", "city": "杭州"}
    rows = [
        (
            1,
            _raw_item(
                sample_id="raw_route",
                human="从甲地坐公交到乙地怎么走？",
                subtype="route_grounded_answer",
                scene="amap_plan_route",
                tool_name="amap_plan_route",
                arguments=route_args,
                observation_data={
                    "mode": "transit",
                    "origin": "甲地",
                    "destination": "乙地",
                    "duration_min": 42,
                    "distance_km": 8.8,
                    "main_steps": ["甲地乘坐地铁1号线到中转站", "换乘公交28路到乙地"],
                },
                expected_subset={"amap_plan_route": {"origin": "甲地", "destination": "乙地", "mode": "bus"}},
                raw_answer="预计耗时2520秒，距离8800米。",
            ),
        ),
        (
            2,
            _raw_item(
                sample_id="raw_poi",
                human="西湖附近找咖啡馆。",
                subtype="poi_grounded_answer",
                scene="amap_search_poi",
                tool_name="amap_search_poi",
                arguments=poi_args,
                observation_data={
                    "keyword": "咖啡馆",
                    "city": "杭州",
                    "around_location": "120.1,30.2",
                    "count": 2,
                    "pois": [
                        {"name": "甲咖啡", "address": "测试路1号", "distance": "230"},
                        {"name": "乙咖啡", "address": "样例街2号", "distance": "1560"},
                    ],
                },
                expected_subset=poi_args,
                raw_answer="有咖啡馆。\n自己看。",
            ),
        ),
        (
            3,
            _raw_item(
                sample_id="raw_geocode",
                human="雷峰塔在哪个区？",
                subtype="geocode_grounded_answer",
                scene="amap_search_poi",
                tool_name="amap_geocode",
                arguments=geocode_args,
                observation_data={
                    "query": "雷峰塔",
                    "city": "杭州市",
                    "district": "西湖区",
                    "formatted_address": "浙江省杭州市西湖区雷峰塔",
                    "location": "120.148000,30.236000",
                },
                expected_subset=geocode_args,
                raw_answer="雷峰塔在别的地方。",
            ),
        ),
        (
            4,
            _raw_item(
                sample_id="bad_city",
                human="重庆解放碑在哪？",
                subtype="geocode_grounded_answer",
                scene="amap_geocode",
                tool_name="amap_geocode",
                arguments={"address": "解放碑", "city": "重庆"},
                observation_data={
                    "query": "解放碑",
                    "city": "广元市",
                    "district": "旺苍县",
                    "formatted_address": "四川省广元市旺苍县解放碑",
                    "location": "106.288878,32.225434",
                },
                expected_subset={"address": "解放碑", "city": "重庆"},
            ),
        ),
        (
            5,
            _raw_item(
                sample_id="bad_walk",
                human="从甲地走到乙地要多久？",
                subtype="route_grounded_answer",
                scene="amap_plan_route",
                tool_name="amap_plan_route",
                arguments={"origin": "甲地", "destination": "乙地", "mode": "walking", "city": "杭州"},
                observation_data={
                    "mode": "walking",
                    "origin": "甲地",
                    "destination": "乙地",
                    "duration_min": 600,
                    "distance_km": 45.0,
                    "main_steps": ["沿测试路步行"],
                },
                expected_subset={"origin": "甲地", "destination": "乙地", "mode": "walking", "city": "杭州"},
            ),
        ),
    ]

    cleaned, report = clean_tool_result_grounded_answer(
        rows,
        target_counts={"route_grounded_answer": 1, "poi_grounded_answer": 1, "geocode_grounded_answer": 1},
    )

    assert len(cleaned) == 3
    assert report["selected_counts"] == {
        "geocode_grounded_answer": 1,
        "poi_grounded_answer": 1,
        "route_grounded_answer": 1,
    }
    assert report["rejected"]["geocode_city_mismatch"] == 1
    assert report["rejected"]["implausible_walking_route"] == 1
    assert report["fixes"]["fixed_expected_arguments_subset"] == 1
    assert report["fixes"]["fixed_scene"] == 1
    assert report["selected_raw_seconds_answers"] == 0
    assert report["selected_raw_long_meters_answers"] == 0
    assert all(item["conversations"][-1]["value"] != "raw answer" for item in cleaned)
    assert all("\n" not in item["conversations"][-1]["value"] for item in cleaned)
    assert cleaned[-1]["scene"] == "amap_geocode"
