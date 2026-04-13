from __future__ import annotations

from copy import deepcopy
from typing import Any

TRIPAI_TOOL_USE_SYSTEM_PROMPT = (
    "你是 TripAI 旅行助手。"
    "当用户问题需要实时路线、位置、周边 POI 等信息时，优先使用工具。"
    "如果参数缺失，先澄清再调用。"
    "如果不需要工具，直接自然回答。"
    "如果工具失败或结果为空，明确说明不确定性并给出稳妥建议，不要编造。"
)

EXPECTED_BEHAVIORS = {
    "should_call_tool",
    "should_clarify",
    "should_answer_directly",
    "should_fallback",
}

AMAP_GEOCODE_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_geocode",
        "description": "将景点、商圈、酒店、地标或结构化地址解析成高德位置结果与经纬度。",
        "parameters": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "需要解析的地址、景点名、酒店名或商圈名。",
                },
                "city": {
                    "type": "string",
                    "description": "可选的城市名，用来缩小搜索范围。",
                },
            },
            "required": ["address"],
        },
    },
}

AMAP_SEARCH_POI_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_search_poi",
        "description": "搜索酒店、景点、地铁站、商圈等 POI，可按城市检索，也可围绕某个位置查找。",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "POI 搜索词，例如 酒店、景点、地铁站、西湖 酒店。",
                },
                "city": {
                    "type": "string",
                    "description": "可选的城市名。",
                },
                "around_location": {
                    "type": "string",
                    "description": "可选的位置中心，支持“经度,纬度”或可解析的位置名称。",
                },
                "radius_m": {
                    "type": "integer",
                    "description": "可选的搜索半径，单位米。默认 3000。",
                },
            },
            "required": ["keyword"],
        },
    },
}

AMAP_PLAN_ROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "amap_plan_route",
        "description": (
            "规划从出发地到目的地的路线。"
            "origin 和 destination 可以是地点名称、结构化地址，或“经度,纬度”。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "出发地，支持地址、地标名或经纬度。",
                },
                "destination": {
                    "type": "string",
                    "description": "目的地，支持地址、地标名或经纬度。",
                },
                "mode": {
                    "type": "string",
                    "enum": ["transit", "driving", "walking", "bicycling"],
                    "description": "路线模式，默认 transit。",
                },
                "city": {
                    "type": "string",
                    "description": "公交/地铁规划时建议提供城市名。",
                },
            },
            "required": ["origin", "destination"],
        },
    },
}

AMAP_TOOL_SCHEMAS = [AMAP_GEOCODE_TOOL, AMAP_SEARCH_POI_TOOL, AMAP_PLAN_ROUTE_TOOL]
AMAP_TOOL_NAMES = {tool["function"]["name"] for tool in AMAP_TOOL_SCHEMAS}

ALLOWED_TWO_STEP_CHAINS = {
    ("amap_geocode", "amap_plan_route"),
    ("amap_geocode", "amap_search_poi"),
}


def build_amap_tool_schemas() -> list[dict[str, Any]]:
    return deepcopy(AMAP_TOOL_SCHEMAS)


def build_tool_success(data: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "data": data}


def build_tool_empty(reason: str = "no_result") -> dict[str, Any]:
    return {"status": "empty", "reason": reason}


def build_tool_error(reason: str, *, retryable: bool = False) -> dict[str, Any]:
    return {"status": "error", "reason": reason, "retryable": retryable}
