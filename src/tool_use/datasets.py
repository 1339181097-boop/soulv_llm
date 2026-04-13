from __future__ import annotations

import json
from typing import Any

from .protocol import ALLOWED_TWO_STEP_CHAINS, AMAP_TOOL_NAMES, EXPECTED_BEHAVIORS

SOURCE_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
SHAREGPT_ALLOWED_ROLES = {"human", "gpt", "function_call", "observation", "system"}


def _validate_message_list(messages: Any, *, item_index: int, field_name: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(messages, list) or not messages:
        return [f"{field_name} of item {item_index} must be a non-empty list."]

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{field_name}[{message_index}] of item {item_index} must be an object.")
            continue

        role = message.get("role")
        if role not in SOURCE_ALLOWED_ROLES:
            errors.append(f"{field_name}[{message_index}] of item {item_index} has invalid role: {role!r}.")

        if role == "assistant":
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            has_content = isinstance(content, str) and bool(content.strip())
            has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
            if not has_content and not has_tool_calls:
                errors.append(
                    f"{field_name}[{message_index}] of item {item_index} assistant message needs content or tool_calls."
                )
            if has_tool_calls:
                errors.extend(_validate_tool_calls(tool_calls, item_index=item_index, field_name=field_name))
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            content = message.get("content")
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                errors.append(f"{field_name}[{message_index}] of item {item_index} tool message needs tool_call_id.")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"{field_name}[{message_index}] of item {item_index} tool message needs content.")
            else:
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    errors.append(
                        f"{field_name}[{message_index}] of item {item_index} tool content must be valid JSON string."
                    )
        else:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"{field_name}[{message_index}] of item {item_index} needs non-empty content.")

    return errors


def _validate_tool_calls(tool_calls: list[Any], *, item_index: int, field_name: str) -> list[str]:
    errors: list[str] = []
    for tool_call_index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            errors.append(f"{field_name} tool_call {tool_call_index} of item {item_index} must be an object.")
            continue

        call_id = tool_call.get("id")
        call_type = tool_call.get("type")
        function = tool_call.get("function")
        if not isinstance(call_id, str) or not call_id.strip():
            errors.append(f"{field_name} tool_call {tool_call_index} of item {item_index} needs id.")
        if call_type != "function":
            errors.append(f"{field_name} tool_call {tool_call_index} of item {item_index} must be type=function.")
        if not isinstance(function, dict):
            errors.append(f"{field_name} tool_call {tool_call_index} of item {item_index} needs function object.")
            continue

        name = function.get("name")
        arguments = function.get("arguments")
        if name not in AMAP_TOOL_NAMES:
            errors.append(
                f"{field_name} tool_call {tool_call_index} of item {item_index} uses unknown tool {name!r}."
            )
        if not isinstance(arguments, str) or not arguments.strip():
            errors.append(
                f"{field_name} tool_call {tool_call_index} of item {item_index} needs JSON-string arguments."
            )
            continue

        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            errors.append(
                f"{field_name} tool_call {tool_call_index} of item {item_index} arguments must be valid JSON."
            )
            continue

        if not isinstance(parsed_arguments, dict):
            errors.append(
                f"{field_name} tool_call {tool_call_index} of item {item_index} arguments must decode to object."
            )

    return errors


def _validate_tools(tools: Any, *, item_index: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(tools, list) or not tools:
        return [f"tools of item {item_index} must be a non-empty list."]

    seen_names: set[str] = set()
    for tool_index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            errors.append(f"tools[{tool_index}] of item {item_index} must be an object.")
            continue

        if tool.get("type") != "function":
            errors.append(f"tools[{tool_index}] of item {item_index} must be type=function.")
            continue

        function = tool.get("function")
        if not isinstance(function, dict):
            errors.append(f"tools[{tool_index}] of item {item_index} needs function object.")
            continue

        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if name not in AMAP_TOOL_NAMES:
            errors.append(f"tools[{tool_index}] of item {item_index} uses unknown tool {name!r}.")
        if name in seen_names:
            errors.append(f"tools[{tool_index}] of item {item_index} duplicates tool name {name!r}.")
        seen_names.add(name)
        if not isinstance(description, str) or not description.strip():
            errors.append(f"tools[{tool_index}] of item {item_index} needs description.")
        if not isinstance(parameters, dict):
            errors.append(f"tools[{tool_index}] of item {item_index} needs JSON schema parameters.")

    return errors


def _validate_tool_chain(messages_with_answer: list[dict[str, Any]], *, item_index: int) -> list[str]:
    tool_sequence: list[str] = []
    for message in messages_with_answer:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                tool_sequence.append(function["name"])

    if len(tool_sequence) > 2:
        return [f"item {item_index} exceeds max tool rounds: {tool_sequence!r}"]
    if len(tool_sequence) == 2 and tuple(tool_sequence) not in ALLOWED_TWO_STEP_CHAINS:
        return [f"item {item_index} uses disallowed tool chain: {tool_sequence!r}"]
    return []


def validate_tool_use_source_dataset(dataset: Any) -> list[str]:
    if not isinstance(dataset, list):
        return ["Tool-use source dataset must be a JSON array."]

    errors: list[str] = []
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            errors.append(f"item {index} must be an object.")
            continue

        if item.get("expected_behavior") not in EXPECTED_BEHAVIORS:
            errors.append(f"item {index} has invalid expected_behavior {item.get('expected_behavior')!r}.")

        for field_name in ("id", "task_type", "scene"):
            value = item.get(field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"item {index} needs non-empty {field_name}.")

        errors.extend(_validate_tools(item.get("tools"), item_index=index))
        errors.extend(_validate_message_list(item.get("messages"), item_index=index, field_name="messages"))
        errors.extend(
            _validate_message_list(
                item.get("messages_with_answer"),
                item_index=index,
                field_name="messages_with_answer",
            )
        )

        messages_with_answer = item.get("messages_with_answer")
        if isinstance(messages_with_answer, list):
            errors.extend(_validate_tool_chain(messages_with_answer, item_index=index))

    return errors


def _assistant_to_sharegpt_messages(message: dict[str, Any]) -> list[dict[str, str]]:
    sharegpt_messages: list[dict[str, str]] = []
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        function = tool_call["function"]
        arguments = json.loads(function["arguments"])
        payload = {"name": function["name"], "arguments": arguments}
        sharegpt_messages.append({"from": "function_call", "value": json.dumps(payload, ensure_ascii=False)})

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        sharegpt_messages.append({"from": "gpt", "value": content})
    return sharegpt_messages


def export_tool_use_dataset_to_sharegpt(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for item in dataset:
        conversations: list[dict[str, str]] = []
        for message in item["messages_with_answer"]:
            role = message["role"]
            if role == "system":
                conversations.append({"from": "system", "value": message["content"]})
            elif role == "user":
                conversations.append({"from": "human", "value": message["content"]})
            elif role == "assistant":
                conversations.extend(_assistant_to_sharegpt_messages(message))
            elif role == "tool":
                conversations.append({"from": "observation", "value": message["content"]})

        exported.append(
            {
                "id": item["id"],
                "task_type": item["task_type"],
                "scene": item["scene"],
                "expected_behavior": item["expected_behavior"],
                "tools": json.dumps(item["tools"], ensure_ascii=False),
                "conversations": conversations,
            }
        )

    return exported


def validate_sharegpt_tool_dataset(dataset: Any) -> list[str]:
    if not isinstance(dataset, list):
        return ["Exported sharegpt tool dataset must be a JSON array."]

    errors: list[str] = []
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            errors.append(f"item {index} must be an object.")
            continue

        tools = item.get("tools")
        conversations = item.get("conversations")
        if not isinstance(tools, str) or not tools.strip():
            errors.append(f"item {index} tools must be a JSON string.")
        else:
            try:
                parsed_tools = json.loads(tools)
            except json.JSONDecodeError:
                errors.append(f"item {index} tools must decode to JSON.")
            else:
                errors.extend(_validate_tools(parsed_tools, item_index=index))

        if not isinstance(conversations, list) or not conversations:
            errors.append(f"item {index} conversations must be a non-empty list.")
            continue

        for conversation_index, message in enumerate(conversations):
            if not isinstance(message, dict):
                errors.append(f"item {index} conversations[{conversation_index}] must be an object.")
                continue
            role = message.get("from")
            value = message.get("value")
            if role not in SHAREGPT_ALLOWED_ROLES:
                errors.append(
                    f"item {index} conversations[{conversation_index}] has invalid sharegpt role {role!r}."
                )
            if not isinstance(value, str) or not value.strip():
                errors.append(f"item {index} conversations[{conversation_index}] needs non-empty value.")
                continue
            if role in {"function_call", "observation"}:
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    errors.append(
                        f"item {index} conversations[{conversation_index}] must be valid JSON for {role}."
                    )
                    continue
                if role == "function_call" and (
                    not isinstance(parsed_value, dict) or "name" not in parsed_value or "arguments" not in parsed_value
                ):
                    errors.append(
                        f"item {index} conversations[{conversation_index}] function_call needs name and arguments."
                    )

    return errors
