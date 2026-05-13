from __future__ import annotations

import json

from src.data_pipeline.clean_stage2_clarify_then_call import SUBTYPE_ORDER, clean_clarify_then_call, RawRow


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _targets(**overrides: int) -> dict[str, int]:
    targets = {subtype: 0 for subtype in SUBTYPE_ORDER}
    targets.update(overrides)
    return targets


def _raw_item(
    *,
    sample_id: str,
    human: str,
    clarify: str,
    followup: str,
    subtype: str,
    scene: str,
    tool_name: str,
    arguments: dict,
    observation_data: dict,
    expected_subset: dict,
) -> dict:
    return {
        "id": sample_id,
        "task_type": "clarify_then_call",
        "subtype": subtype,
        "scene": scene,
        "expected_behavior": "should_clarify",
        "expected_tool_chain": [tool_name],
        "expected_arguments_subset": expected_subset,
        "quality_tags": ["clarify_first"],
        "conversations": [
            {"from": "system", "value": "TripAI tool rules."},
            {"from": "human", "value": human},
            {"from": "gpt", "value": clarify},
            {"from": "human", "value": followup},
            {"from": "function_call", "value": _json({"name": tool_name, "arguments": arguments})},
            {"from": "observation", "value": _json({"status": "success", "data": observation_data})},
            {"from": "gpt", "value": "raw answer"},
        ],
    }


def test_clean_clarify_then_call_rewrites_answers_and_rebuilds_subsets() -> None:
    route_args = {"origin": "北京南站", "destination": "颐和园", "mode": "transit", "city": "北京"}
    poi_args = {"keyword": "咖啡馆", "city": "北京", "around_location": "三里屯", "radius_m": 500}
    rows = [
        RawRow(
            source_file="raw.jsonl",
            line_number=1,
            item=_raw_item(
                sample_id="raw_route",
                human="从北京南站到颐和园怎么走？",
                clarify="请问你想用哪种出行方式，比如公交地铁、开车、步行还是骑行？",
                followup="坐地铁公交。",
                subtype="route_missing_mode",
                scene="amap_plan_route",
                tool_name="amap_plan_route",
                arguments=route_args,
                observation_data={
                    "mode": "transit",
                    "origin": "北京南站",
                    "destination": "颐和园",
                    "duration_min": 42,
                    "distance_km": 18.8,
                    "main_steps": ["乘坐地铁4号线到北宫门站", "出站后步行500米"],
                },
                expected_subset={"mode": "transit | driving | walking | bicycling"},
            ),
        ),
        RawRow(
            source_file="raw_v2.jsonl",
            line_number=2,
            item=_raw_item(
                sample_id="raw_poi",
                human="我想找附近的咖啡馆。",
                clarify="请问你现在在哪个位置或附近哪个参考点？",
                followup="在北京三里屯附近，500米内。",
                subtype="poi_missing_anchor",
                scene="amap_search_poi",
                tool_name="amap_search_poi",
                arguments=poi_args,
                observation_data={
                    "keyword": "咖啡馆",
                    "city": "北京",
                    "around_location": "116.454118,39.935589",
                    "count": 3,
                    "pois": [
                        {"name": "甲咖啡", "address": "三里屯路1号", "distance": "57"},
                        {"name": "乙咖啡", "address": "三里屯路2号", "distance": "180"},
                        {"name": "丙咖啡", "address": "三里屯路3号", "distance": "260"},
                    ],
                },
                expected_subset={"keyword": "必填"},
            ),
        ),
    ]

    cleaned, report = clean_clarify_then_call(
        rows,
        target_counts=_targets(route_missing_mode=1, poi_missing_anchor=1),
    )

    assert len(cleaned) == 2
    assert report["selected_counts"] == {"poi_missing_anchor": 1, "route_missing_mode": 1}
    assert report["fixes"]["rebuilt_expected_arguments_subset"] == 2
    assert all(tuple(message["from"] for message in item["conversations"]) == ("system", "human", "gpt", "human", "function_call", "observation", "gpt") for item in cleaned)
    assert cleaned[0]["expected_arguments_subset"] == route_args
    assert "约42分钟" in cleaned[0]["conversations"][-1]["value"]
    assert "甲咖啡" in cleaned[1]["conversations"][-1]["value"]
