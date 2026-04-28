from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_success, read_json, write_json

DEFAULT_INPUT_PATH = "src/tool_eval/reports/native_tool_baseline_outputs.json"
DEFAULT_OUTPUT_PATH = "src/tool_eval/reports/native_tool_baseline_summary.json"


def _argument_json_stats(records: list[dict[str, Any]]) -> tuple[int, int]:
    total = 0
    valid = 0
    for record in records:
        for tool_call in record.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            arguments = function.get("arguments")
            total += 1
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                valid += 1
    return total, valid


def _normalize_tool_chain(record: dict[str, Any]) -> list[str]:
    chain: list[str] = []
    for tool_call in record.get("tool_calls", []) or []:
        function = tool_call.get("function", {})
        tool_name = function.get("name")
        if isinstance(tool_name, str):
            chain.append(tool_name)
    return chain


def _arguments_are_valid(record: dict[str, Any]) -> bool:
    for tool_call in record.get("tool_calls", []) or []:
        function = tool_call.get("function", {})
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return False
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
    return True


def _summarize_case(record: dict[str, Any]) -> dict[str, Any]:
    predicted_chain = _normalize_tool_chain(record)
    expected_chain = record.get("expected_tool_chain", [])
    expected_behavior = record.get("expected_behavior")
    tool_selection_correct = predicted_chain == expected_chain
    no_tool_correct = True
    if expected_behavior == "should_answer_directly":
        no_tool_correct = not predicted_chain and bool(str(record.get("content") or "").strip())
    clarify_correct = True
    if expected_behavior == "should_clarify":
        clarify_correct = not predicted_chain and bool(str(record.get("content") or "").strip())
    return {
        "id": record.get("id"),
        "task_type": record.get("task_type"),
        "expected_behavior": expected_behavior,
        "expected_tool_chain": expected_chain,
        "predicted_tool_chain": predicted_chain,
        "tool_selection_correct": tool_selection_correct,
        "arguments_valid": _arguments_are_valid(record),
        "no_tool_correct": no_tool_correct,
        "clarify_correct": clarify_correct,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize native tool-call baseline outputs.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Native baseline output JSON path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Summary output JSON path.")
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    records: list[dict[str, Any]] = read_json(args.input)

    total_cases = len(records)
    tool_call_cases = sum(1 for record in records if record.get("tool_calls"))
    no_tool_cases = sum(1 for record in records if not record.get("tool_calls"))
    argument_total, argument_valid = _argument_json_stats(records)
    per_case = [_summarize_case(record) for record in records]
    total_for_rate = len(per_case) or 1

    tool_name_counts: dict[str, int] = {}
    for record in records:
        for tool_call in record.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            if isinstance(tool_name, str):
                tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1

    summary: dict[str, Any] = {
        "total_cases": total_cases,
        "tool_call_cases": tool_call_cases,
        "no_tool_cases": no_tool_cases,
        "argument_json_total": argument_total,
        "argument_json_valid": argument_valid,
        "tool_selection_accuracy": round(sum(item["tool_selection_correct"] for item in per_case) / total_for_rate, 4),
        "argument_json_valid_rate": round(argument_valid / (argument_total or 1), 4),
        "no_tool_accuracy": round(sum(item["no_tool_correct"] for item in per_case) / total_for_rate, 4),
        "clarify_accuracy": round(sum(item["clarify_correct"] for item in per_case) / total_for_rate, 4),
        "tool_name_counts": tool_name_counts,
        "per_case": per_case,
        "alignment_notes": [
            "若 tool_call_cases > 0，说明原生模型已经具备可利用的 OpenAI-compatible tool-calling 基础。",
            "若 argument_json_valid == argument_json_total，说明 arguments 可以直接沿用 JSON string 规范。",
            "若 no_tool_cases 里能自然直答，TripAI stage2 应该优先做协议对齐，而不是从零发明工具习惯。",
        ],
    }

    output_path = write_json(args.output, summary)
    log_success(f"Native baseline summary written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
