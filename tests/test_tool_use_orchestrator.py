from __future__ import annotations

import io
import json
from urllib import error

import src.tool_use.orchestrator as orchestrator_module
from src.tool_use.orchestrator import ToolCallingOrchestrator
from src.tool_use.orchestrator import OpenAICompatibleChatClient


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


def test_orchestrator_executes_textual_tool_call_fallback() -> None:
    chat_client = FakeChatClient(
        [
            _content_response('<tool_call>\n{"name":"amap_geocode","arguments":{"address":"西湖","city":"杭州"}}\n</tool_call>'),
            _content_response("我已经帮你查到西湖的大致位置了。"),
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
            {"role": "user", "content": "西湖在哪里？"},
        ]
    )

    assert result["status"] == "completed"
    assert result["tool_sequence"] == ["amap_geocode"]
    assert result["executed_calls"][0]["arguments"]["address"] == "西湖"


def test_orchestrator_tolerates_textual_tool_call_with_extra_trailing_brace() -> None:
    chat_client = FakeChatClient(
        [
            _content_response(
                '<tool_call>\n'
                '{"name":"amap_search_poi","arguments":{"keyword":"酒店","city":"杭州","around_location":"西湖","radius_m":3000}}}\n'
                '</tool_call>'
            ),
            _content_response("我已经整理了附近酒店。"),
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
            {"role": "user", "content": "帮我找西湖附近酒店。"},
        ]
    )

    assert result["status"] == "completed"
    assert result["tool_sequence"] == ["amap_search_poi"]
    assert result["executed_calls"][0]["arguments"]["keyword"] == "酒店"


class _DummyHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_DummyHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def test_openai_client_retries_with_smaller_max_tokens_on_context_limit(monkeypatch) -> None:  # noqa: ANN001
    attempted_max_tokens: list[int] = []
    context_error_payload = {
        "error": {
            "message": (
                "You passed 7681 input tokens and requested 512 output tokens. "
                "However, the model's context length is only 8192 tokens, resulting in a maximum input length "
                "of 7680 tokens. Please reduce the length of the input prompt. "
                "(parameter=input_tokens, value=7681)"
            )
        }
    }

    def fake_urlopen(req, timeout):  # noqa: ANN001
        payload = json.loads(req.data.decode("utf-8"))
        attempted_max_tokens.append(payload["max_tokens"])
        if len(attempted_max_tokens) == 1:
            body = json.dumps(context_error_payload, ensure_ascii=False).encode("utf-8")
            raise error.HTTPError(req.full_url, 400, "Bad Request", hdrs=None, fp=io.BytesIO(body))
        return _DummyHTTPResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(orchestrator_module.request, "urlopen", fake_urlopen)

    client = OpenAICompatibleChatClient(base_url="http://example.com/v1", api_key="EMPTY", retry_count=0)
    response = client.complete(
        [{"role": "user", "content": "hello"}],
        model="fake-model",
        max_tokens=512,
    )

    assert attempted_max_tokens == [512, 511]
    assert response["choices"][0]["message"]["content"] == "ok"
