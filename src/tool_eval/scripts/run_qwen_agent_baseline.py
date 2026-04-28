from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_info, log_success, read_json, write_json
from src.tool_use import AmapClient
from src.tool_use.protocol import build_tool_error

DEFAULT_DATASET_PATH = "src/tool_eval/datasets/native_tool_baseline.json"
DEFAULT_OUTPUT_PATH = "src/tool_eval/reports/qwen_agent_baseline_outputs.json"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _load_qwen_agent_classes() -> tuple[Any, Any, Any]:
    try:
        from qwen_agent.agents import Assistant
        from qwen_agent.tools.base import BaseTool, register_tool
    except ImportError as exc:
        raise RuntimeError(
            "qwen-agent is required for this baseline. Install it in the eval environment, "
            "for example: pip install qwen-agent"
        ) from exc
    return Assistant, BaseTool, register_tool


def _register_amap_tools(amap_client: AmapClient) -> list[str]:
    _, BaseTool, register_tool = _load_qwen_agent_classes()

    @register_tool("amap_geocode")
    class AmapGeocodeTool(BaseTool):  # type: ignore[misc, valid-type]
        description = "Resolve a place name or address to an AMap location."
        parameters = [
            {"name": "address", "type": "string", "description": "Place name or address.", "required": True},
            {"name": "city", "type": "string", "description": "Optional city name.", "required": False},
        ]

        def call(self, params: str, **kwargs: Any) -> str:
            arguments = json.loads(params)
            if not isinstance(arguments, dict):
                return _json_dumps(build_tool_error("invalid_arguments"))
            return _json_dumps(amap_client.geocode(arguments["address"], city=arguments.get("city")))

    @register_tool("amap_search_poi")
    class AmapSearchPoiTool(BaseTool):  # type: ignore[misc, valid-type]
        description = "Search AMap POIs by keyword, city, optional around location, and radius."
        parameters = [
            {"name": "keyword", "type": "string", "description": "POI search keyword.", "required": True},
            {"name": "city", "type": "string", "description": "Optional city name.", "required": False},
            {
                "name": "around_location",
                "type": "string",
                "description": "Optional location center, either lng,lat or place name.",
                "required": False,
            },
            {"name": "radius_m", "type": "integer", "description": "Optional search radius in meters.", "required": False},
        ]

        def call(self, params: str, **kwargs: Any) -> str:
            arguments = json.loads(params)
            if not isinstance(arguments, dict):
                return _json_dumps(build_tool_error("invalid_arguments"))
            return _json_dumps(
                amap_client.search_poi(
                    arguments["keyword"],
                    city=arguments.get("city"),
                    around_location=arguments.get("around_location"),
                    radius_m=arguments.get("radius_m"),
                )
            )

    @register_tool("amap_plan_route")
    class AmapPlanRouteTool(BaseTool):  # type: ignore[misc, valid-type]
        description = "Plan an AMap route between origin and destination."
        parameters = [
            {"name": "origin", "type": "string", "description": "Route origin.", "required": True},
            {"name": "destination", "type": "string", "description": "Route destination.", "required": True},
            {
                "name": "mode",
                "type": "string",
                "description": "Route mode: transit, driving, walking, or bicycling.",
                "required": False,
            },
            {"name": "city", "type": "string", "description": "Optional city name.", "required": False},
        ]

        def call(self, params: str, **kwargs: Any) -> str:
            arguments = json.loads(params)
            if not isinstance(arguments, dict):
                return _json_dumps(build_tool_error("invalid_arguments"))
            return _json_dumps(
                amap_client.plan_route(
                    arguments["origin"],
                    arguments["destination"],
                    mode=arguments.get("mode", "transit"),
                    city=arguments.get("city"),
                )
            )

    return ["amap_geocode", "amap_search_poi", "amap_plan_route"]


def _extract_agent_messages(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [message for message in response if isinstance(message, dict)]
    if isinstance(response, dict):
        return [response]
    return [{"role": "assistant", "content": str(response)}]


def _extract_tool_chain(messages: list[dict[str, Any]]) -> list[str]:
    chain: list[str] = []
    for message in messages:
        function_call = message.get("function_call")
        if isinstance(function_call, dict) and isinstance(function_call.get("name"), str):
            chain.append(function_call["name"])
            continue
        role = message.get("role")
        name = message.get("name")
        if role in {"function", "tool"} and isinstance(name, str):
            chain.append(name)
    return chain


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Qwen-Agent baseline for TripAI AMap tool use.")
    parser.add_argument("--model", required=True, help="Served model name or local model identifier for Qwen-Agent.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint root URL.")
    parser.add_argument("--api-key", default="EMPTY", help="Bearer token for the endpoint.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Baseline dataset path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top_p.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Run Qwen3 with thinking enabled. The default Qwen-Agent baseline disables thinking.",
    )
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    Assistant, _, _ = _load_qwen_agent_classes()

    dataset: list[dict[str, Any]] = read_json(args.dataset)
    function_list = _register_amap_tools(AmapClient())
    llm_cfg = {
        "model": args.model,
        "model_server": args.base_url,
        "api_key": args.api_key,
        "generate_cfg": {
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
        },
    }
    agent = Assistant(llm=llm_cfg, function_list=function_list)

    outputs: list[dict[str, Any]] = []
    output_path = write_json(args.output, outputs)
    log_info(f"Writing incremental Qwen-Agent baseline outputs to: {output_path}")

    for index, case in enumerate(dataset, start=1):
        log_info(f"Running Qwen-Agent baseline case {index}/{len(dataset)}: {case['id']}")
        response: Any = None
        for response in agent.run(messages=case["messages"]):
            pass
        agent_messages = _extract_agent_messages(response)
        outputs.append(
            {
                "id": case["id"],
                "task_type": case["task_type"],
                "expected_behavior": case["expected_behavior"],
                "expected_tool_chain": case.get("expected_tool_chain", []),
                "messages": case["messages"],
                "predicted_tool_chain": _extract_tool_chain(agent_messages),
                "agent_messages": agent_messages,
            }
        )
        write_json(args.output, outputs)

    output_path = write_json(args.output, outputs)
    log_success(f"Qwen-Agent baseline outputs written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
