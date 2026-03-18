from __future__ import annotations

import argparse
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
from src.data_pipeline.global_cleaner import clean_text

DEFAULT_INPUT_PATH = "data/raw/multiturn.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_multiturn.json"
DEFAULT_SYSTEM_PROMPT = "ن½ وک¯وگœو—… AI و—…è،Œç®،ه®¶â€œه°ڈه¥‡â€‌ï¼Œè¯·ç»“هگˆن¸ٹن¸‹و–‡è؟‍ç»­ه›‍ç­”ç”¨وˆ·é—®é¢کم€‚"
ALLOWED_ROLES = {"system", "user", "assistant", "tool", "observation"}


def _normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages") or record.get("conversation")
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip()
        content = clean_text(item.get("content"), max_length=6000)
        if not role or role not in ALLOWED_ROLES or not content:
            continue
        normalized.append({"role": role, "content": content})

    return normalized


def build_multiturn_sample(record: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    messages = _normalize_messages(record)
    if not messages:
        return None

    roles = [message["role"] for message in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    return {"messages": messages}


def process_multiturn_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    log_info(f"ه¼€ه§‹ه¤„çگ†ه¤ڑè½®ه¯¹è¯‌و•°وچ®: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"وœھو‰¾هˆ°ه¤ڑè½®ه¯¹è¯‌هژںه§‹و•°وچ®ï¼Œه…ˆè·³è؟‡: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, list[dict[str, str]]]] = []
    skipped = 0
    for record in raw_records:
        sample = build_multiturn_sample(record)
        if sample is None:
            skipped += 1
            continue
        processed.append(sample)

    output_path = write_json(output_json_path, processed)
    log_success(f"ه¤ڑè½®ه¯¹è¯‌و•°وچ®ه¤„çگ†ه®Œوˆگï¼Œè¾“ه‡؛ {len(processed)} و‌،ï¼Œè·³è؟‡ {skipped} و‌،م€‚")
    log_info(f"è¾“ه‡؛و–‡ن»¶: {output_path}")
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ه°†ه¤ڑè½®ه¯¹è¯‌و•°وچ®è½¬وچ¢ن¸؛ ChatMLم€‚")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="هژںه§‹ه¤ڑè½®ه¯¹è¯‌و•°وچ®è·¯ه¾„ï¼Œو”¯وŒپ JSON/JSONLم€‚")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="è¾“ه‡؛ ChatML JSON è·¯ه¾„م€‚")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_multiturn_data(args.input, args.output)


if __name__ == "__main__":
    main()

