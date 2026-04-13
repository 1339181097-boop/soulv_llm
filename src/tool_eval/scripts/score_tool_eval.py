from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_success, read_json, write_json

DEFAULT_INPUT_PATH = "src/tool_eval/reports/stage2_amap_tool_eval_outputs.json"
DEFAULT_OUTPUT_PATH = "src/tool_eval/reports/stage2_amap_tool_eval_summary.json"

FALLBACK_HINTS = ("暂时", "重试", "再查", "核对", "高德", "稍后")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _first_assistant_after_prompt(messages: list[dict[str, Any]], prompt_length: int) -> dict[str, Any] | None:
    for message in messages[prompt_length:]:
        if message.get("role") == "assistant":
            return message
    return None


def _argument_subset_match(expected: dict[str, Any], predicted_calls: list[dict[str, Any]]) -> bool:
    if not expected:
        return True
    if not predicted_calls:
        return False
    first_call = predicted_calls[0]
    arguments = first_call.get("arguments", {})
    return all(_normalize_text(arguments.get(key)) == _normalize_text(value) for key, value in expected.items())


def _must_include_match(must_include: list[str], final_answer: str) -> bool:
    if not must_include:
        return True
    lowered_answer = final_answer.lower()
    return all(token.lower() in lowered_answer for token in must_include)


def _summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    result = record["result"]
    executed_calls = result.get("executed_calls", [])
    predicted_calls = [
        {"tool_name": item.get("tool_name"), "arguments": item.get("arguments", {}), "result": item.get("result", {})}
        for item in executed_calls
    ]
    predicted_chain = [item["tool_name"] for item in predicted_calls]
    expected_chain = record.get("expected_tool_chain", [])
    final_answer = result.get("final_answer", "")
    prompt_length = len(record.get("messages", []))
    first_assistant = _first_assistant_after_prompt(result.get("messages", []), prompt_length) or {}

    tool_selection_correct = predicted_chain == expected_chain
    arguments_correct = _argument_subset_match(record.get("expected_arguments_subset", {}), predicted_calls)
    clarify_correct = True
    if record["expected_behavior"] == "should_clarify":
        clarify_correct = bool(first_assistant) and not first_assistant.get("tool_calls") and bool(predicted_chain)

    no_tool_correct = True
    if record["expected_behavior"] == "should_answer_directly":
        no_tool_correct = not predicted_chain and bool(final_answer.strip())

    fallback_correct = True
    if record["expected_behavior"] == "should_fallback":
        fallback_correct = any(token in final_answer for token in FALLBACK_HINTS)

    final_answer_grounded = _must_include_match(record.get("must_include", []), final_answer)
    execution_success = all(item.get("result", {}).get("status") == "success" for item in predicted_calls)
    if record["expected_behavior"] == "should_fallback":
        execution_success = any(item.get("result", {}).get("status") == "error" for item in predicted_calls)

    overall_pass = all(
        [
            tool_selection_correct or record["expected_behavior"] == "should_answer_directly",
            arguments_correct,
            clarify_correct,
            no_tool_correct,
            fallback_correct,
            final_answer_grounded,
        ]
    )

    return {
        "id": record["id"],
        "task_type": record["task_type"],
        "expected_behavior": record["expected_behavior"],
        "predicted_tool_chain": predicted_chain,
        "tool_selection_correct": tool_selection_correct,
        "arguments_correct": arguments_correct,
        "clarify_correct": clarify_correct,
        "no_tool_correct": no_tool_correct,
        "fallback_correct": fallback_correct,
        "execution_success": execution_success,
        "final_answer_grounded": final_answer_grounded,
        "overall_pass": overall_pass,
        "final_answer": final_answer,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score stage2 AMap tool-use eval outputs.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Tool eval output JSON path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Summary output JSON path.")
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()

    records: list[dict[str, Any]] = read_json(args.input)
    per_case = [_summarize_record(record) for record in records]
    total = len(per_case) or 1
    summary = {
        "total_cases": len(per_case),
        "tool_selection_accuracy": round(sum(item["tool_selection_correct"] for item in per_case) / total, 4),
        "argument_accuracy": round(sum(item["arguments_correct"] for item in per_case) / total, 4),
        "clarify_accuracy": round(sum(item["clarify_correct"] for item in per_case) / total, 4),
        "no_tool_accuracy": round(sum(item["no_tool_correct"] for item in per_case) / total, 4),
        "fallback_accuracy": round(sum(item["fallback_correct"] for item in per_case) / total, 4),
        "execution_success_rate": round(sum(item["execution_success"] for item in per_case) / total, 4),
        "final_answer_grounded_rate": round(sum(item["final_answer_grounded"] for item in per_case) / total, 4),
        "overall_pass_rate": round(sum(item["overall_pass"] for item in per_case) / total, 4),
        "release_gate": {
            "tool_selection_accuracy_gte_0_85": round(sum(item["tool_selection_correct"] for item in per_case) / total, 4)
            >= 0.85,
            "argument_accuracy_gte_0_80": round(sum(item["arguments_correct"] for item in per_case) / total, 4)
            >= 0.80,
            "execution_success_rate_gte_0_90": round(sum(item["execution_success"] for item in per_case) / total, 4)
            >= 0.90,
        },
        "per_case": per_case,
    }

    output_path = write_json(args.output, summary)
    log_success(f"Stage2 tool eval summary written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
