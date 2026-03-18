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

DEFAULT_INPUT_PATH = "data/raw/basic_qa.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_basic_qa.json"
DEFAULT_SYSTEM_PROMPT = "ن½ وک¯وگœو—…و™¯ç‚¹ç™¾ç§‘هٹ©و‰‹ï¼Œè¯·هں؛ن؛ژن؛‹ه®‍ه›‍ç­”ç”¨وˆ·çڑ„و—…è،Œé—®ç­”م€‚"


def build_basic_qa_sample(record: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    question = clean_text(record.get("question") or record.get("user_query"), max_length=2000)
    answer = clean_text(record.get("answer") or record.get("assistant_response"), max_length=5000)
    if not question or not answer:
        return None

    context = clean_text(record.get("context"), max_length=2000)
    if context:
        question = f"هڈ‚è€ƒن؟،وپ¯ï¼ڑ{context}\n\nç”¨وˆ·é—®é¢کï¼ڑ{question}"

    system_prompt = clean_text(
        record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        max_length=1000,
        mask_sensitive=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def process_basic_qa_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    log_info(f"ه¼€ه§‹ه¤„çگ†هں؛ç،€ QA و•°وچ®: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"وœھو‰¾هˆ°هں؛ç،€ QA هژںه§‹و•°وچ®ï¼Œه…ˆè·³è؟‡: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, list[dict[str, str]]]] = []
    skipped = 0
    for record in raw_records:
        sample = build_basic_qa_sample(record)
        if sample is None:
            skipped += 1
            continue
        processed.append(sample)

    output_path = write_json(output_json_path, processed)
    log_success(f"هں؛ç،€ QA و•°وچ®ه¤„çگ†ه®Œوˆگï¼Œè¾“ه‡؛ {len(processed)} و‌،ï¼Œè·³è؟‡ {skipped} و‌،م€‚")
    log_info(f"è¾“ه‡؛و–‡ن»¶: {output_path}")
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ه°†هں؛ç،€ QA و•°وچ®è½¬وچ¢ن¸؛ ChatMLم€‚")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="هژںه§‹ QA و•°وچ®è·¯ه¾„ï¼Œو”¯وŒپ JSON/JSONLم€‚")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="è¾“ه‡؛ ChatML JSON è·¯ه¾„م€‚")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_basic_qa_data(args.input, args.output)


if __name__ == "__main__":
    main()

