from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_success, write_json
from src.tool_use.protocol import TRIPAI_TOOL_USE_SYSTEM_PROMPT

DEFAULT_OUTPUT_PATH = "src/tool_eval/datasets/stage2_amap_golden.json"
DEFAULT_THINKING_CANARY_OUTPUT_PATH = "src/tool_eval/datasets/stage2_amap_thinking_canary.json"


def _messages(user: str, system: str = TRIPAI_TOOL_USE_SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _case(
    case_id: str,
    task_type: str,
    expected_behavior: str,
    user: str,
    *,
    expected_tool_chain: list[str] | None = None,
    expected_arguments_subset: dict[str, Any] | None = None,
    must_include: list[str] | None = None,
    tool_test_mode: dict[str, Any] | None = None,
    system: str = TRIPAI_TOOL_USE_SYSTEM_PROMPT,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": case_id,
        "task_type": task_type,
        "expected_behavior": expected_behavior,
        "expected_tool_chain": expected_tool_chain or [],
        "messages": _messages(user, system),
    }
    if expected_arguments_subset:
        item["expected_arguments_subset"] = expected_arguments_subset
    if must_include:
        item["must_include"] = must_include
    if tool_test_mode:
        item["tool_test_mode"] = tool_test_mode
    return item


def build_golden_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    route_cases = [
        ("route_transit_001", "北京南站", "颐和园", "transit", "我从北京南站出发去颐和园，优先公共交通，帮我规划路线。"),
        ("route_transit_002", "上海虹桥站", "豫园", "transit", "上海虹桥站到豫园怎么坐地铁公交更省心？"),
        ("route_transit_003", "杭州东站", "灵隐寺", "transit", "我在杭州东站，想坐公共交通去灵隐寺，帮我看下路线。"),
        ("route_driving_001", "成都东站", "宽窄巷子", "driving", "从成都东站开车去宽窄巷子，帮我规划一下驾驶路线。"),
        ("route_driving_002", "深圳宝安机场", "观澜湖", "driving", "深圳宝安机场到观澜湖如果自驾，路线怎么走？"),
        ("route_walking_001", "西湖音乐喷泉", "断桥残雪", "walking", "从西湖音乐喷泉步行到断桥残雪，帮我看下路线。"),
        ("route_walking_002", "南京夫子庙", "老门东", "walking", "夫子庙到老门东想走过去，大概怎么走？"),
        ("route_bicycling_001", "苏州博物馆", "平江路", "bicycling", "我想从苏州博物馆骑车去平江路，帮我规划路线。"),
        ("route_bicycling_002", "厦门大学", "曾厝垵", "bicycling", "厦门大学到曾厝垵骑行怎么走比较顺？"),
        ("route_transit_004", "天津站", "水上公园", "transit", "天津站出发去水上公园，优先地铁公交，帮我规划。"),
    ]
    for case_id, origin, destination, mode, user in route_cases:
        cases.append(
            _case(
                f"golden_{case_id}",
                "single_tool_call",
                "should_call_tool",
                user,
                expected_tool_chain=["amap_plan_route"],
                expected_arguments_subset={"origin": origin, "destination": destination, "mode": mode},
                must_include=["路线"],
            )
        )

    geocode_cases = [
        ("geocode_001", "雷峰塔", "杭州", "雷峰塔具体位置在哪里？"),
        ("geocode_002", "广州塔", "广州", "广州塔在哪个位置？"),
        ("geocode_003", "洪崖洞", "重庆", "帮我确认一下洪崖洞的位置。"),
        ("geocode_004", "鼓浪屿码头", "厦门", "鼓浪屿码头具体在哪？"),
        ("geocode_005", "大雁塔", "西安", "大雁塔在西安哪里？"),
        ("geocode_006", "星海广场", "大连", "星海广场的具体位置帮我查一下。"),
    ]
    for case_id, address, city, user in geocode_cases:
        cases.append(
            _case(
                f"golden_{case_id}",
                "single_tool_call",
                "should_call_tool",
                user,
                expected_tool_chain=["amap_geocode"],
                expected_arguments_subset={"address": address, "city": city},
                must_include=["位置"],
            )
        )

    poi_cases = [
        ("poi_hotel_001", "酒店", "杭州", "西湖", "帮我找一下西湖附近适合中转住一晚的酒店。"),
        ("poi_restaurant_001", "餐厅", "南京", "夫子庙", "夫子庙附近有什么餐厅可以顺路吃饭？"),
        ("poi_subway_001", "地铁站", "上海", "豫园", "豫园附近最近的地铁站帮我查一下。"),
        ("poi_parking_001", "停车场", "成都", "宽窄巷子", "宽窄巷子周边哪里方便停车？"),
        ("poi_mall_001", "商场", "深圳", "世界之窗", "世界之窗附近有适合顺路逛逛的商场吗？"),
        ("poi_spot_001", "景点", "苏州", "平江路", "平江路附近还有什么景点可以一起逛？"),
    ]
    for case_id, keyword, city, around_location, user in poi_cases:
        cases.append(
            _case(
                f"golden_{case_id}",
                "single_tool_call",
                "should_call_tool",
                user,
                expected_tool_chain=["amap_search_poi"],
                expected_arguments_subset={"keyword": keyword, "city": city},
                must_include=[keyword],
            )
        )

    clarify_cases = [
        ("clarify_route_origin_001", "帮我规划一下去机场的公共交通路线。"),
        ("clarify_route_destination_001", "我从北京南站出发，帮我看一下怎么坐车。"),
        ("clarify_route_city_001", "从火车站去人民公园，优先地铁公交，帮我规划路线。"),
        ("clarify_route_mode_001", "从西湖到灵隐寺怎么走？"),
        ("clarify_poi_city_001", "帮我找一下万达广场附近的酒店。"),
        ("clarify_poi_anchor_001", "我想找附近可以吃饭的地方。"),
        ("clarify_poi_keyword_001", "西湖附近有什么方便一点的地方？"),
        ("clarify_geocode_city_001", "人民公园具体在哪里？"),
    ]
    for case_id, user in clarify_cases:
        cases.append(
            _case(
                f"golden_{case_id}",
                "clarify_then_call",
                "should_clarify",
                user,
                expected_tool_chain=[],
            )
        )

    no_tool_cases = [
        ("no_tool_001", "西湖适合带老人慢慢逛吗？"),
        ("no_tool_002", "第一次去南京，两天一晚节奏怎么安排比较轻松？"),
        ("no_tool_003", "亲子去上海迪士尼，出发前需要注意什么？"),
        ("no_tool_004", "杭州有什么适合雨天的室内体验？"),
        ("no_tool_005", "去成都旅行，住春熙路附近有什么优缺点？"),
        ("no_tool_006", "老人去苏州园林游玩，路线节奏上有什么建议？"),
        ("no_tool_007", "冬天去哈尔滨需要准备哪些衣物？"),
        ("no_tool_008", "第一次去厦门，鼓浪屿和环岛路怎么取舍？"),
    ]
    for case_id, user in no_tool_cases:
        cases.append(
            _case(
                f"golden_{case_id}",
                "no_tool_needed",
                "should_answer_directly",
                user,
                expected_tool_chain=[],
            )
        )

    fallback_cases = [
        ("fallback_route_001", "帮我规划从虹桥机场到外滩的最快路线。", "amap_plan_route", ["amap_plan_route"]),
        ("fallback_route_002", "从杭州东站到灵隐寺怎么走最快？", "amap_plan_route", ["amap_plan_route"]),
        ("fallback_geocode_001", "帮我确认一下一个叫蓝鲸小院的具体位置。", "amap_geocode", ["amap_geocode"]),
        ("fallback_geocode_002", "查一下星河湾码头在哪。", "amap_geocode", ["amap_geocode"]),
        ("fallback_poi_001", "帮我找一下西湖附近适合停车的地方。", "amap_search_poi", ["amap_search_poi"]),
        ("fallback_poi_002", "南京夫子庙附近有什么适合老人吃饭的餐厅？", "amap_search_poi", ["amap_search_poi"]),
    ]
    for case_id, user, force_error_on, chain in fallback_cases:
        cases.append(
            _case(
                f"golden_{case_id}",
                "tool_failure_fallback",
                "should_fallback",
                user,
                expected_tool_chain=chain,
                tool_test_mode={"force_error_on": force_error_on},
            )
        )

    chain_poi_cases = [
        ("chain_poi_001", "鼓浪屿码头", "厦门", "酒店", "帮我找一下鼓浪屿码头附近步行方便的酒店。"),
        ("chain_poi_002", "大雁塔北广场", "西安", "餐厅", "大雁塔北广场附近有什么餐厅适合顺路吃饭？"),
        ("chain_poi_003", "平江路历史街区", "苏州", "地铁站", "平江路历史街区附近最近的地铁站在哪里？"),
    ]
    for case_id, address, city, keyword, user in chain_poi_cases:
        cases.append(
            _case(
                f"golden_{case_id}",
                "tool_result_grounded_answer",
                "should_call_tool",
                user,
                expected_tool_chain=["amap_geocode", "amap_search_poi"],
                must_include=[keyword],
            )
        )

    chain_route_cases = [
        ("chain_route_001", "西湖音乐喷泉附近", "灵隐寺", "杭州", "我在西湖音乐喷泉附近，想去灵隐寺，帮我规划公共交通路线。"),
        ("chain_route_002", "春熙路太古里", "宽窄巷子", "成都", "我在春熙路太古里，想去宽窄巷子，帮我看一下路线。"),
        ("chain_route_003", "夫子庙景区", "南京博物院", "南京", "从夫子庙景区去南京博物院，公共交通怎么走？"),
    ]
    for case_id, origin, destination, city, user in chain_route_cases:
        cases.append(
            _case(
                f"golden_{case_id}",
                "tool_result_grounded_answer",
                "should_call_tool",
                user,
                expected_tool_chain=["amap_geocode", "amap_plan_route"],
                must_include=["路线"],
            )
        )

    if len(cases) != 50:
        raise AssertionError(f"Expected 50 golden cases, got {len(cases)}")
    return cases


def build_thinking_canary_cases() -> list[dict[str, Any]]:
    golden = build_golden_cases()
    wanted_ids = {
        "golden_route_transit_001",
        "golden_route_driving_001",
        "golden_poi_hotel_001",
        "golden_geocode_001",
        "golden_clarify_route_origin_001",
        "golden_no_tool_001",
        "golden_fallback_route_001",
        "golden_chain_route_001",
    }
    return [case for case in golden if case["id"] in wanted_ids]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the 50-case stage2 AMap golden eval dataset.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Golden eval dataset output path.")
    parser.add_argument(
        "--thinking-canary-output",
        default=DEFAULT_THINKING_CANARY_OUTPUT_PATH,
        help="Small thinking-mode canary dataset output path.",
    )
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    golden_path = write_json(args.output, build_golden_cases())
    canary_path = write_json(args.thinking_canary_output, build_thinking_canary_cases())
    log_success(f"Stage2 golden eval dataset written to: {golden_path}")
    log_success(f"Stage2 thinking canary dataset written to: {canary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
