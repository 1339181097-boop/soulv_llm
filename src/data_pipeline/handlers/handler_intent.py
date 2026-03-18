from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    load_records,
    log_error,
    log_info,
    log_success,
    log_warn,
    resolve_path,
    write_json,
)
from src.data_pipeline.global_cleaner import clean_text, normalize_text

DEFAULT_INPUT_PATH = "data/raw/intent.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_intent.json"
DEFAULT_SYSTEM_PROMPT = (
    "ن½ وک¯ن¸€ن¸ھه¤ڑهٹںèƒ½و™؛èƒ½و—…è،Œهٹ©و‰‹ï¼Œèƒ½ه¤ںç²¾ه‡†è¯†هˆ«ç”¨وˆ·و„ڈه›¾ï¼Œهڈھèƒ½ه‡؛çژ°ن¸€ç§چç”¨وˆ·و„ڈه›¾ï¼Œ"
    'jsonو ¼ه¼ڈè¾“ه‡؛ï¼ڑ{"intentionName":"FUNCTION_FLIGHTS_SEARCH_STRATEGY"}'
)

# ه…¬هڈ¸ه½“ه‰چو­£هœ¨ن½؟ç”¨çڑ„و„ڈه›¾è§„èŒƒم€‚هگژç»­و‰€وœ‰و¸…و´—ه’Œè®­ç»ƒéƒ½ن»¥è؟™ن»½و‍ڑن¸¾ن¸؛ه‡†م€‚
CANONICAL_INTENT_NAMES = {
    "FUNCTION_FLIGHTS_SEARCH_STRATEGY",
    "FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY",
    "FUNCTION_FLIGHTS_PASSENGER_STRATEGY",
    "FUNCTION_HOTELS_STRATEGY",
    "TRAVEL_STRATEGY",
    "TRAVEL_LOCATION_STRATEGY",
    "FUNCTION_TICKETS_STRATEGY",
    "FUNCTION_CAR_RENTAL_STRATEGY",
    "FUNCTION_VISA_STRATEGY",
    "DEFAULT_STRATEGY",
}


def _normalize_intent_response(record: dict[str, Any]) -> str:
    assistant_response = record.get("assistant_response") or record.get("tool_call") or record.get("tool_json")
    if assistant_response:
        if isinstance(assistant_response, str):
            return clean_text(assistant_response, max_length=4000, mask_sensitive=False)
        return json.dumps(assistant_response, ensure_ascii=False)

    intention_name = normalize_text(record.get("intentionName"))
    if intention_name:
        return json.dumps({"intentionName": intention_name}, ensure_ascii=False)

    return ""


def build_intent_sample(record: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    user_query = clean_text(record.get("user_query") or record.get("query"), max_length=2000)
    assistant_content = _normalize_intent_response(record)

    if not user_query or not assistant_content:
        return None

    system_prompt = clean_text(
        record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        max_length=3000,
        mask_sensitive=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def process_intent_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    log_info(f"ه¼€ه§‹ه¤„çگ†و„ڈه›¾و•°وچ®: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"وœھو‰¾هˆ°و„ڈه›¾هژںه§‹و•°وچ®ï¼Œه…ˆè·³è؟‡: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, list[dict[str, str]]]] = []
    skipped = 0
    non_canonical = 0

    for record in raw_records:
        intention_name = normalize_text(record.get("intentionName"))
        if intention_name and intention_name not in CANONICAL_INTENT_NAMES:
            non_canonical += 1
            log_warn(f"هڈ‘çژ°وœھو”¶ه½•çڑ„و„ڈه›¾هگچ: {intention_name}")

        sample = build_intent_sample(record)
        if sample is None:
            skipped += 1
            continue
        processed.append(sample)

    output_path = write_json(output_json_path, processed)
    log_success(
        f"و„ڈه›¾و•°وچ®ه¤„çگ†ه®Œوˆگï¼Œè¾“ه‡؛ {len(processed)} و‌،ï¼Œè·³è؟‡ {skipped} و‌،م€‚"
        f"وœھو”¶ه½•و„ڈه›¾هگچ {non_canonical} و‌،م€‚"
    )
    log_info(f"è¾“ه‡؛و–‡ن»¶: {output_path}")
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ه°†و„ڈه›¾è¯†هˆ«و•°وچ®è½¬وچ¢ن¸؛ ChatMLم€‚")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="هژںه§‹و„ڈه›¾و•°وچ®è·¯ه¾„ï¼Œو”¯وŒپ JSON/JSONLم€‚")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="è¾“ه‡؛ ChatML JSON è·¯ه¾„م€‚")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_intent_data(args.input, args.output)


if __name__ == "__main__":
    main()

