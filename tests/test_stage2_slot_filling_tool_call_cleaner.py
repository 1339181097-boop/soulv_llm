from __future__ import annotations

import json

from src.data_pipeline.clean_stage2_slot_filling_tool_call import clean_slot_filling_tool_call


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
) -> dict:
    return {
        "id": sample_id,
        "task_type": "slot_filling_tool_call",
        "subtype": subtype,
        "scene": scene,
        "expected_behavior": "should_call_tool",
        "expected_tool_chain": [tool_name],
        "expected_arguments_subset": arguments,
        "quality_tags": ["natural_language"],
        "conversations": [
            {"from": "system", "value": "TripAI tool rules."},
            {"from": "human", "value": human},
            {"from": "function_call", "value": _json({"name": tool_name, "arguments": arguments})},
            {"from": "observation", "value": _json({"status": "success", "data": observation_data})},
            {"from": "gpt", "value": "raw answer"},
        ],
    }


def test_clean_slot_filling_balances_subtypes_rewrites_answers_and_duplicate_questions() -> None:
    route_args = {"origin": "重庆北站", "destination": "解放碑", "mode": "driving", "city": "重庆"}
    poi_args_one = {"keyword": "咖啡馆", "city": "北京", "around_location": "三里屯", "radius_m": 500}
    poi_args_two = {"keyword": "书店", "city": "上海", "around_location": "人民广场", "radius_m": 500}
    rows = [
        (
            1,
            _raw_item(
                sample_id="raw_route_1",
                human="我在重庆北站，要去解放碑，开车过去怎么走？",
                subtype="route_slot_filling",
                scene="amap_plan_route",
                tool_name="amap_plan_route",
                arguments=route_args,
                observation_data={
                    "mode": "driving",
                    "origin": "重庆北站",
                    "destination": "解放碑",
                    "duration_min": 125,
                    "distance_km": 12.5,
                    "main_steps": ["沿测试路向东行驶100米", "沿样例路继续行驶200米"],
                },
            ),
        ),
        (
            2,
            _raw_item(
                sample_id="raw_route_2",
                human="我在重庆北站，要去解放碑，开车过去怎么走？",
                subtype="route_slot_filling",
                scene="amap_plan_route",
                tool_name="amap_plan_route",
                arguments=route_args,
                observation_data={
                    "mode": "driving",
                    "origin": "重庆北站",
                    "destination": "解放碑",
                    "duration_min": 126,
                    "distance_km": 12.6,
                    "main_steps": ["沿测试路向东行驶100米", "沿样例路继续行驶200米"],
                },
            ),
        ),
        (
            3,
            _raw_item(
                sample_id="raw_poi_1",
                human="北京三里屯附近500米内有啥咖啡馆？",
                subtype="poi_slot_filling",
                scene="amap_search_poi",
                tool_name="amap_search_poi",
                arguments=poi_args_one,
                observation_data={
                    "keyword": "咖啡馆",
                    "city": "北京",
                    "around_location": "116.45,39.93",
                    "count": 1,
                    "pois": [{"name": "甲咖啡", "address": "三里屯路1号", "distance": "57"}],
                },
            ),
        ),
        (
            4,
            _raw_item(
                sample_id="raw_poi_2",
                human="上海人民广场附近500米内有没有书店？",
                subtype="poi_slot_filling",
                scene="amap_search_poi",
                tool_name="amap_search_poi",
                arguments=poi_args_two,
                observation_data={
                    "keyword": "书店",
                    "city": "上海",
                    "around_location": "121.47,31.23",
                    "count": 2,
                    "pois": [
                        {"name": "甲书店", "address": "人民大道1号", "distance": "120"},
                        {"name": "乙书店", "address": "人民大道2号", "distance": "260"},
                    ],
                },
            ),
        ),
    ]

    cleaned, report = clean_slot_filling_tool_call(rows, target_per_subtype=2)

    assert len(cleaned) == 4
    assert report["selected_counts"] == {"poi_slot_filling": 2, "route_slot_filling": 2}
    assert report["fixes"]["rewritten_duplicate_question"] == 1
    assert report["selected_duplicate_questions"] == 0
    assert all(item["conversations"][-1]["value"] != "raw answer" for item in cleaned)
    assert "约2小时5分钟" in cleaned[0]["conversations"][-1]["value"]
    assert all(tuple(message["from"] for message in item["conversations"]) == ("system", "human", "function_call", "observation", "gpt") for item in cleaned)
