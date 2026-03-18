from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    iter_jsonl,
    log_error,
    log_info,
    log_success,
    resolve_path,
    write_json,
)
from src.data_pipeline.global_cleaner import clean_text, normalize_text

DEFAULT_INPUT_PATH = "data/raw/travel_guide.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_itinerary.json"
DEFAULT_SEED = 42
MIN_CONTENT_LENGTH = 100

SYSTEM_PROMPTS = (
    "ن½ وک¯وگœو—…و™؛و…§ç§‘وٹ€çڑ„ن¸“ه±‍ AI و—…è،Œç®،ه®¶â€œه°ڈه¥‡â€‌م€‚è¯·ن½؟ç”¨ç»“و‍„و¸…و™°م€پن؟،وپ¯ه®Œو•´م€په¯Œوœ‰ن؛²ه’Œهٹ›çڑ„ن¸­و–‡ï¼Œن¸؛و—…è،Œè€…ç”ںوˆگهڈ¯و‰§è،Œçڑ„و”»ç•¥م€‚",
    "ن½ وک¯ tripAI çڑ„çژ‹ç‰Œو—…è،Œé،¾é—®â€œه°ڈه¥‡â€‌م€‚ه›‍ه¤چو—¶è¦پن؟‌وŒپن؛؛è®¾ç¨³ه®ڑï¼Œو“…é•؟ن½؟ç”¨هˆ†و®µم€پو ‡é¢که’Œé€‚é‡ڈ Emojiï¼Œè¾“ه‡؛ه®Œو•´è،Œç¨‹è§„هˆ’م€‚",
    "ن½ وک¯و—…و¸¸è§„هˆ’ن¸“ه®¶â€œه°ڈه¥‡â€‌م€‚è¯·é’ˆه¯¹ç”¨وˆ·ç›®çڑ„هœ°ه’Œه‡؛è،Œه¤©و•°ï¼Œç»™ه‡؛و¸…و™°çڑ„ Day1م€پDay2 ه¼ڈè،Œç¨‹ه®‰وژ’ï¼Œه¹¶è،¥ه……ن؛¤é€ڑم€پن½ڈه®؟ه’Œè´´ه£«م€‚",
)

USER_QUERY_TEMPLATES = (
    "ه¸®وˆ‘è§„هˆ’ن¸€ن»½ {destination} {days} ه¤©çڑ„و—…و¸¸و”»ç•¥ï¼Œه°½é‡ڈè¯¦ç»†ن¸€ç‚¹م€‚",
    "وˆ‘و‰“ç®—هژ» {destination} çژ© {days} ه¤©ï¼Œè¯·ن½ ç»™وˆ‘ن¸€ن¸ھه®Œو•´è،Œç¨‹ه®‰وژ’م€‚",
    "tripAIï¼Œه¸®وˆ‘هپڑن¸€ن»½ {destination} {days} ه¤©è‡ھç”±è،Œو”»ç•¥ï¼Œè¦پوœ‰و¯ڈه¤©çڑ„é‡چç‚¹م€‚",
    "ه‡†ه¤‡ه’Œه®¶ن؛؛هژ» {destination} و—…è،Œ {days} ه¤©ï¼Œç»™وˆ‘ن¸€ن»½çœپه؟ƒهڈˆو¸…و™°çڑ„è·¯ç؛؟م€‚",
    "وƒ³هژ» {destination} çژ© {days} ه¤©ï¼Œé؛»çƒ¦ن½ وŒ‰ه¤©ه¸®وˆ‘وژ’ن¸€ن¸‹هگƒن½ڈè،Œم€‚",
)


def synthesize_user_query(destination: str, days: str, rng: random.Random) -> str:
    template = rng.choice(USER_QUERY_TEMPLATES)
    return template.format(destination=destination, days=days)


def build_itinerary_sample(record: dict[str, Any], rng: random.Random) -> dict[str, list[dict[str, str]]] | None:
    destination = normalize_text(record.get("destination") or "وœھçں¥ç›®çڑ„هœ°")
    days = normalize_text(record.get("days") or "3")
    content = clean_text(record.get("itinerary_content"), max_length=20000)

    if len(content) < MIN_CONTENT_LENGTH:
        return None

    return {
        "messages": [
            {"role": "system", "content": rng.choice(SYSTEM_PROMPTS)},
            {"role": "user", "content": synthesize_user_query(destination, days, rng)},
            {"role": "assistant", "content": content},
        ]
    }


def process_itinerary_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    rng = random.Random(seed)
    processed_data: list[dict[str, list[dict[str, str]]]] = []
    skipped_short = 0
    total_records = 0

    log_info(f"ه¼€ه§‹ه¤„çگ†و”»ç•¥و•°وچ®: {resolve_path(input_file_path)}")

    try:
        for _, record in iter_jsonl(input_file_path):
            total_records += 1
            sample = build_itinerary_sample(record, rng)
            if sample is None:
                skipped_short += 1
                continue
            processed_data.append(sample)
    except FileNotFoundError:
        log_error(f"وœھو‰¾هˆ°è¾“ه…¥و–‡ن»¶: {resolve_path(input_file_path)}")
        return []

    output_path = write_json(output_json_path, processed_data)
    log_success(
        "و”»ç•¥و•°وچ®ه¤„çگ†ه®Œوˆگم€‚"
        f"è¯»هڈ– {total_records} و‌،ï¼Œè¾“ه‡؛ {len(processed_data)} و‌،ï¼Œ"
        f"è·³è؟‡ {skipped_short} و‌،è؟‡çں­وˆ–ç©؛ه†…ه®¹م€‚"
    )
    log_info(f"è¾“ه‡؛و–‡ن»¶: {output_path}")
    return processed_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ه°†و”»ç•¥هژںه§‹ JSONL è½¬وچ¢ن¸؛ ChatML è®­ç»ƒو•°وچ®م€‚")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="هژںه§‹و”»ç•¥ JSONL è·¯ه¾„م€‚")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="è¾“ه‡؛ ChatML JSON è·¯ه¾„م€‚")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="éڑڈوœ؛ç§چه­گï¼Œن؟‌è¯پç»“و‍œهڈ¯ه¤چçژ°م€‚")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_itinerary_data(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()

