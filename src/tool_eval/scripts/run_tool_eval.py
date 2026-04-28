from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_error, log_info, log_success, read_json, write_json
from src.tool_use import AmapClient, OpenAICompatibleChatClient, ToolCallingOrchestrator

DEFAULT_DATASET_PATH = "src/tool_eval/datasets/stage2_amap_golden.json"
DEFAULT_OUTPUT_PATH = "src/tool_eval/reports/stage2_amap_tool_eval_outputs.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage2 AMap tool-use eval against a remote OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", required=True, help="Endpoint root URL.")
    parser.add_argument("--api-key", default="EMPTY", help="Bearer token for the endpoint.")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Tool eval dataset JSON path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top_p.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Run Qwen3 with thinking enabled as a canary path. The default release path disables thinking.",
    )
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()

    dataset: list[dict[str, Any]] = read_json(args.dataset)
    chat_client = OpenAICompatibleChatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        disable_thinking=not args.enable_thinking,
    )
    amap_client = AmapClient()
    orchestrator = ToolCallingOrchestrator(chat_client=chat_client, model=args.model, amap_client=amap_client)

    outputs: list[dict[str, Any]] = []
    output_path = write_json(args.output, outputs)
    log_info(f"Writing incremental tool eval outputs to: {output_path}")

    for index, case in enumerate(dataset, start=1):
        log_info(f"Running tool eval case {index}/{len(dataset)}: {case['id']}")
        try:
            result = orchestrator.run(
                case["messages"],
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                tool_test_mode=case.get("tool_test_mode"),
            )
        except Exception as exc:
            write_json(args.output, outputs)
            log_error(f"Tool eval interrupted at case {case['id']}: {exc}")
            raise

        outputs.append(
            {
                "id": case["id"],
                "task_type": case["task_type"],
                "expected_behavior": case["expected_behavior"],
                "expected_tool_chain": case.get("expected_tool_chain", []),
                "expected_arguments_subset": case.get("expected_arguments_subset", {}),
                "must_include": case.get("must_include", []),
                "messages": case["messages"],
                "result": result,
            }
        )
        write_json(args.output, outputs)

    output_path = write_json(args.output, outputs)
    log_success(f"Stage2 tool eval outputs written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
