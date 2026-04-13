from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_info, log_success, read_json, write_json
from src.tool_use.orchestrator import OpenAICompatibleChatClient
from src.tool_use.protocol import build_amap_tool_schemas

DEFAULT_DATASET_PATH = "src/tool_eval/datasets/native_tool_baseline.json"
DEFAULT_OUTPUT_PATH = "src/tool_eval/reports/native_tool_baseline_outputs.json"


def _extract_tool_calls(response_payload: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return []
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    return tool_calls if isinstance(tool_calls, list) else []


def _extract_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native Qwen tool-call baseline against a remote OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", required=True, help="Endpoint root URL.")
    parser.add_argument("--api-key", default="EMPTY", help="Bearer token for the endpoint.")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Native baseline dataset path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top_p.")
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()

    dataset = read_json(args.dataset)
    client = OpenAICompatibleChatClient(base_url=args.base_url, api_key=args.api_key)
    tools = build_amap_tool_schemas()

    outputs: list[dict[str, Any]] = []
    for index, case in enumerate(dataset, start=1):
        log_info(f"Running native baseline case {index}/{len(dataset)}: {case['id']}")
        response_payload = client.complete(
            case["messages"],
            model=args.model,
            tools=tools,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        outputs.append(
            {
                "id": case["id"],
                "task_type": case["task_type"],
                "expected_behavior": case["expected_behavior"],
                "expected_tool_chain": case.get("expected_tool_chain", []),
                "messages": case["messages"],
                "tool_calls": _extract_tool_calls(response_payload),
                "content": _extract_content(response_payload),
                "raw_response": response_payload,
            }
        )

    output_path = write_json(args.output, outputs)
    log_success(f"Native baseline outputs written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
