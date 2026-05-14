from __future__ import annotations

import json
import threading
from functools import partial
from urllib import request

import src.deploy.frontend_server as frontend_server
from src.deploy.frontend_server import (
    BACKEND_SYSTEM_PROMPT,
    DEFAULT_FRONTEND_DIR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_P,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    FrontendHTTPServer,
    FrontendRequestHandler,
    ServerConfig,
    _build_upstream_request_headers,
    _normalize_upstream_base_url,
    _sanitize_chat_messages,
    _should_use_amap,
)


def test_normalize_upstream_base_url_strips_v1_suffix() -> None:
    assert _normalize_upstream_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert _normalize_upstream_base_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000"
    assert _normalize_upstream_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"


def test_build_upstream_request_headers_prefers_server_side_api_key() -> None:
    headers = _build_upstream_request_headers(
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer client-token",
            "Connection": "keep-alive",
        },
        server_side_api_key="server-token",
    )

    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer server-token"
    assert "Connection" not in headers


def test_backend_system_prompt_is_fixed_prompt_text() -> None:
    assert "TripAI 的 AI 旅行助手“小奇”" in BACKEND_SYSTEM_PROMPT
    assert "```" not in BACKEND_SYSTEM_PROMPT



def test_sanitize_chat_messages_ignores_frontend_system_prompt() -> None:
    messages = _sanitize_chat_messages(
        [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "西湖适合老人逛吗？"},
        ]
    )

    assert messages == [{"role": "user", "content": "西湖适合老人逛吗？"}]


def test_should_use_amap_detects_route_prompt() -> None:
    assert _should_use_amap([{"role": "user", "content": "从北京南站到颐和园怎么走？"}])
    assert not _should_use_amap([{"role": "user", "content": "西湖适合带老人慢慢逛吗？"}])


class _FakeChatClient:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.init_kwargs = kwargs

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.__class__.calls.append({"messages": messages, "kwargs": kwargs, "init_kwargs": self.init_kwargs})
        return {"choices": [{"message": {"role": "assistant", "content": "plain answer"}}]}


class _FakeOrchestrator:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.init_kwargs = kwargs

    def run(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.__class__.calls.append({"messages": messages, "kwargs": kwargs, "init_kwargs": self.init_kwargs})
        return {"final_answer": "amap answer", "tool_sequence": ["amap_plan_route"]}


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        if hasattr(exc, "code") and hasattr(exc, "read"):
            return exc.code, json.loads(exc.read().decode("utf-8"))  # type: ignore[attr-defined]
        raise


def _start_test_server():
    app_config = ServerConfig(
        frontend_dir=DEFAULT_FRONTEND_DIR,
        upstream_base_url="http://127.0.0.1:9999",
        upstream_api_key="server-key",
        default_model="fake-model",
        request_timeout_seconds=5,
    )
    handler_class = partial(FrontendRequestHandler, directory=str(DEFAULT_FRONTEND_DIR))
    server = FrontendHTTPServer(("127.0.0.1", 0), handler_class, app_config=app_config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_api_chat_plain_uses_backend_prompt_and_default_sampling(monkeypatch) -> None:  # noqa: ANN001
    _FakeChatClient.calls = []
    monkeypatch.setattr(frontend_server, "OpenAICompatibleChatClient", _FakeChatClient)

    server, base_url = _start_test_server()
    try:
        status, payload = _post_json(
            f"{base_url}/api/chat",
            {
                "messages": [
                    {"role": "system", "content": "FRONTEND SYSTEM"},
                    {"role": "user", "content": "西湖适合老人慢慢逛吗？"},
                ]
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert payload["mode"] == "chat"
    assert payload["final_answer"] == "plain answer"
    call = _FakeChatClient.calls[0]
    assert call["messages"][0] == {"role": "system", "content": BACKEND_SYSTEM_PROMPT}
    assert all(message.get("content") != "FRONTEND SYSTEM" for message in call["messages"])
    assert call["kwargs"]["model"] == "fake-model"
    assert call["kwargs"]["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call["kwargs"]["temperature"] == DEFAULT_TEMPERATURE
    assert call["kwargs"]["top_p"] == DEFAULT_TOP_P
    assert call["kwargs"]["top_k"] == DEFAULT_TOP_K
    assert call["kwargs"]["min_p"] == DEFAULT_MIN_P


def test_api_chat_amap_prompt_uses_orchestrator_with_default_sampling(monkeypatch) -> None:  # noqa: ANN001
    _FakeOrchestrator.calls = []
    monkeypatch.setattr(frontend_server, "OpenAICompatibleChatClient", _FakeChatClient)
    monkeypatch.setattr(frontend_server, "ToolCallingOrchestrator", _FakeOrchestrator)

    server, base_url = _start_test_server()
    try:
        status, payload = _post_json(
            f"{base_url}/api/chat",
            {"messages": [{"role": "user", "content": "从北京南站到颐和园怎么走？"}]},
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert payload["mode"] == "amap"
    assert payload["final_answer"] == "amap answer"
    assert payload["tool_sequence"] == ["amap_plan_route"]
    call = _FakeOrchestrator.calls[0]
    assert call["messages"][0] == {"role": "system", "content": BACKEND_SYSTEM_PROMPT}
    assert call["kwargs"]["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call["kwargs"]["temperature"] == DEFAULT_TEMPERATURE
    assert call["kwargs"]["top_p"] == DEFAULT_TOP_P
    assert call["kwargs"]["top_k"] == DEFAULT_TOP_K
    assert call["kwargs"]["min_p"] == DEFAULT_MIN_P


def test_api_chat_rejects_empty_messages() -> None:
    server, base_url = _start_test_server()
    try:
        status, payload = _post_json(f"{base_url}/api/chat", {"messages": []})
    finally:
        server.shutdown()
        server.server_close()

    assert status == 400
    assert "messages" in payload["error"]
