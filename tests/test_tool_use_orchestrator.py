from __future__ import annotations

from src.tool_use.orchestrator import ToolCallingOrchestrator


class FakeChatClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses

    def complete(self, messages, *, model, tools=None, max_tokens=512, temperature=0.0, top_p=1.0):  # noqa: ANN001
        assert model == "fake-model"
        return self.responses.pop(0)


class FakeAmapClient:
    def geocode(self, address: str, city: str | None = None) -> dict:
        return {
            "status": "success",
            "data": {"query": address, "city": city, "location": "120.100000,30.200000"},
        }

    def search_poi(self, keyword: str, *, city: str | None = None, around_location=None, radius_m=None) -> dict:  # noqa: ANN001
        return {"status": "success", "data": {"keyword": keyword, "city": city, "pois": []}}

    def plan_route(self, origin: str, destination: str, *, mode: str = "transit", city: str | None = None) -> dict:
        return {
            "status": "success",
            "data": {"origin": origin, "destination": destination, "mode": mode, "city": city},
        }


def _tool_call_response(tool_name: str, arguments: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }


def _content_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_orchestrator_executes_tool_and_returns_final_answer() -> None:
    chat_client = FakeChatClient(
        [
            _tool_call_response("amap_geocode", "{\"address\":\"雷峰塔\",\"city\":\"杭州\"}"),
            _content_response("我已经帮你定位到雷峰塔的大致位置了。"),
        ]
    )
    orchestrator = ToolCallingOrchestrator(
        chat_client=chat_client,
        model="fake-model",
        amap_client=FakeAmapClient(),
    )

    result = orchestrator.run(
        [
            {"role": "system", "content": "你是 TripAI 旅行助手。"},
            {"role": "user", "content": "雷峰塔在哪？"},
        ]
    )

    assert result["status"] == "completed"
    assert result["tool_sequence"] == ["amap_geocode"]
    assert "定位" in result["final_answer"]


def test_orchestrator_marks_disallowed_second_tool_chain_as_error() -> None:
    chat_client = FakeChatClient(
        [
            _tool_call_response("amap_geocode", "{\"address\":\"西湖\",\"city\":\"杭州\"}"),
            _tool_call_response("amap_geocode", "{\"address\":\"雷峰塔\",\"city\":\"杭州\"}"),
            _content_response("我需要你再提供更明确的地点范围。"),
        ]
    )
    orchestrator = ToolCallingOrchestrator(
        chat_client=chat_client,
        model="fake-model",
        amap_client=FakeAmapClient(),
    )

    result = orchestrator.run(
        [
            {"role": "system", "content": "你是 TripAI 旅行助手。"},
            {"role": "user", "content": "帮我找一下附近酒店。"},
        ]
    )

    assert result["tool_sequence"] == ["amap_geocode", "amap_geocode"]
    executed_calls = result["executed_calls"]
    assert executed_calls[1]["result"]["status"] == "error"
    assert executed_calls[1]["result"]["reason"] == "disallowed_tool_chain"
