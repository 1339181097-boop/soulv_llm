from __future__ import annotations

from src.tool_use.datasets import (
    export_tool_use_dataset_to_sharegpt,
    validate_sharegpt_tool_dataset,
    validate_tool_use_source_dataset,
)
from src.tool_use.protocol import build_amap_tool_schemas


def _sample_source_item() -> dict:
    tools = build_amap_tool_schemas()
    return {
        "id": "tool_sample_001",
        "task_type": "single_tool_call",
        "scene": "amap_geocode",
        "expected_behavior": "should_call_tool",
        "tools": tools,
        "messages": [
            {"role": "system", "content": "你是 TripAI 旅行助手。"},
            {"role": "user", "content": "雷峰塔在哪？"},
        ],
        "messages_with_answer": [
            {"role": "system", "content": "你是 TripAI 旅行助手。"},
            {"role": "user", "content": "雷峰塔在哪？"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {"name": "amap_geocode", "arguments": "{\"address\":\"雷峰塔\",\"city\":\"杭州\"}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "{\"status\":\"success\",\"data\":{\"location\":\"120.147376,30.236977\"}}",
            },
            {"role": "assistant", "content": "我已经帮你定位到雷峰塔的大致位置了。"},
        ],
    }


def test_tool_use_source_dataset_validation_passes_for_valid_sample() -> None:
    dataset = [_sample_source_item()]
    assert validate_tool_use_source_dataset(dataset) == []


def test_tool_use_export_produces_valid_sharegpt_dataset() -> None:
    dataset = [_sample_source_item()]
    exported = export_tool_use_dataset_to_sharegpt(dataset)

    assert validate_sharegpt_tool_dataset(exported) == []
    assert exported[0]["conversations"][2]["from"] == "function_call"
    assert exported[0]["conversations"][3]["from"] == "observation"
    assert exported[0]["conversations"][4]["from"] == "gpt"


def test_tool_use_source_dataset_rejects_disallowed_tool_chain() -> None:
    item = _sample_source_item()
    item["messages_with_answer"].insert(
        3,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_002",
                    "type": "function",
                    "function": {"name": "amap_geocode", "arguments": "{\"address\":\"西湖\",\"city\":\"杭州\"}"},
                }
            ],
        },
    )
    item["messages_with_answer"].insert(
        4,
        {
            "role": "tool",
            "tool_call_id": "call_002",
            "content": "{\"status\":\"success\",\"data\":{\"location\":\"120.147376,30.236977\"}}",
        },
    )

    errors = validate_tool_use_source_dataset([item])
    assert any("disallowed tool chain" in error for error in errors)
