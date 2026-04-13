from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request

from .amap_client import AmapClient
from .protocol import ALLOWED_TWO_STEP_CHAINS, build_amap_tool_schemas, build_tool_error

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _resolve_chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _coerce_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        if not isinstance(function.get("arguments"), str):
            arguments = json.dumps(function.get("arguments", {}), ensure_ascii=False)
            function = {**function, "arguments": arguments}
        normalized.append(
            {
                "id": tool_call.get("id") or f"call_{len(normalized) + 1}",
                "type": tool_call.get("type") or "function",
                "function": function,
            }
        )
    return normalized


def _extract_assistant_message(response_payload: dict[str, Any]) -> dict[str, Any]:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No choices returned from chat completion endpoint.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("First choice is not an object.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Response choice does not contain message object.")

    normalized_message = {
        "role": "assistant",
        "content": _coerce_response_text(message.get("content")),
    }
    tool_calls = _normalize_tool_calls(message.get("tool_calls"))
    if tool_calls:
        normalized_message["tool_calls"] = tool_calls
    return normalized_message


class ChatCompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict[str, Any]:
        ...


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int = 60,
        retry_count: int = 2,
        disable_thinking: bool = True,
    ) -> None:
        self.request_url = _resolve_chat_completions_url(base_url)
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.disable_thinking = disable_thinking

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            req = request.Request(self.request_url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {error_body[:800]}")
                if exc.code not in RETRYABLE_STATUS_CODES or attempt >= self.retry_count:
                    raise last_error
            except error.URLError as exc:
                last_error = RuntimeError(f"Request failed: {exc}")
                if attempt >= self.retry_count:
                    raise last_error
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Endpoint returned invalid JSON: {exc}") from exc

            time.sleep(min(2**attempt, 8))

        if last_error is None:
            raise RuntimeError("Chat completion failed without captured exception.")
        raise last_error


@dataclass
class ExecutedToolCall:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class ToolCallingOrchestrator:
    def __init__(
        self,
        *,
        chat_client: ChatCompletionClient,
        model: str,
        amap_client: AmapClient,
        tool_schemas: list[dict[str, Any]] | None = None,
        max_tool_rounds: int = 2,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.amap_client = amap_client
        self.tool_schemas = tool_schemas or build_amap_tool_schemas()
        self.max_tool_rounds = max_tool_rounds

    def run(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tool_test_mode: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        transcript = [dict(message) for message in messages]
        raw_responses: list[dict[str, Any]] = []
        executed_calls: list[ExecutedToolCall] = []
        tool_sequence: list[str] = []

        for _ in range(self.max_tool_rounds + 1):
            response_payload = self.chat_client.complete(
                transcript,
                model=self.model,
                tools=self.tool_schemas,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw_responses.append(response_payload)
            assistant_message = _extract_assistant_message(response_payload)

            if not assistant_message.get("tool_calls"):
                transcript.append(assistant_message)
                return {
                    "status": "completed",
                    "messages": transcript,
                    "final_answer": assistant_message.get("content", ""),
                    "tool_sequence": tool_sequence,
                    "executed_calls": [call.__dict__ for call in executed_calls],
                    "raw_responses": raw_responses,
                }

            transcript.append(assistant_message)
            for tool_call in assistant_message["tool_calls"]:
                function = tool_call["function"]
                tool_name = function["name"]
                arguments = json.loads(function["arguments"])
                tool_sequence.append(tool_name)

                if len(tool_sequence) > self.max_tool_rounds:
                    tool_result = build_tool_error("tool_round_limit_exceeded")
                elif len(tool_sequence) == 2 and tuple(tool_sequence) not in ALLOWED_TWO_STEP_CHAINS:
                    tool_result = build_tool_error("disallowed_tool_chain")
                else:
                    tool_result = self._execute_tool(tool_name, arguments, tool_test_mode=tool_test_mode)

                executed_calls.append(
                    ExecutedToolCall(tool_name=tool_name, arguments=arguments, result=tool_result)
                )
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        fallback_message = {
            "role": "assistant",
            "content": "我已经尝试调用工具，但本轮工具链超过了 MVP 允许的上限。建议先明确地点后重试。",
        }
        transcript.append(fallback_message)
        return {
            "status": "tool_round_limit_reached",
            "messages": transcript,
            "final_answer": fallback_message["content"],
            "tool_sequence": tool_sequence,
            "executed_calls": [call.__dict__ for call in executed_calls],
            "raw_responses": raw_responses,
        }

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        tool_test_mode: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        forced_error_tool = (tool_test_mode or {}).get("force_error_on")
        if forced_error_tool == tool_name:
            return build_tool_error("forced_tool_error_for_eval")

        if tool_name == "amap_geocode":
            return self.amap_client.geocode(arguments["address"], city=arguments.get("city"))
        if tool_name == "amap_search_poi":
            return self.amap_client.search_poi(
                arguments["keyword"],
                city=arguments.get("city"),
                around_location=arguments.get("around_location"),
                radius_m=arguments.get("radius_m"),
            )
        if tool_name == "amap_plan_route":
            return self.amap_client.plan_route(
                arguments["origin"],
                arguments["destination"],
                mode=arguments.get("mode", "transit"),
                city=arguments.get("city"),
            )
        return build_tool_error(f"unknown_tool:{tool_name}")
