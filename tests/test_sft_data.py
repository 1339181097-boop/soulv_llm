from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    log_error,
    log_info,
    log_success,
    read_json,
    resolve_path,
    validate_chatml_dataset,
)

DEFAULT_TARGET_FILE = "data/processed/sft_roleplay_safety.json"


def preview_sample(dataset: list[dict], seed: int) -> None:
    rng = random.Random(seed)
    sample = rng.choice(dataset)
    print("=" * 80)
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print("=" * 80)


def test_chatml_format(file_path: str = DEFAULT_TARGET_FILE, *, preview: bool = True, seed: int = 42) -> bool:
    configure_console_output()
    resolved = resolve_path(file_path)
    log_info(f"ه¼€ه§‹و ،éھŒ SFT و•°وچ®و–‡ن»¶: {resolved}")

    if not resolved.exists():
        log_error(f"وœھو‰¾هˆ°ç›®و ‡و–‡ن»¶: {resolved}")
        return False

    try:
        dataset = read_json(resolved)
    except json.JSONDecodeError as exc:
        log_error(f"JSON è§£و‍گه¤±è´¥: {exc}")
        return False

    errors = validate_chatml_dataset(dataset)
    if errors:
        log_error(f"و ،éھŒه¤±è´¥ï¼Œه…±هڈ‘çژ° {len(errors)} ن¸ھé—®é¢کم€‚")
        for error in errors[:10]:
            log_error(error)
        return False

    sample_count = len(dataset)
    log_success(f"ChatML ç»“و‍„و ،éھŒé€ڑè؟‡ï¼Œه…± {sample_count} و‌،و ·وœ¬م€‚")
    if preview and dataset:
        log_info("éڑڈوœ؛وٹ½و ·ن¸€و‌،و•°وچ®ç”¨ن؛ژن؛؛ه·¥و£€وں¥م€‚")
        preview_sample(dataset, seed)

    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="و ،éھŒوگœو—… SFT و•°وچ®وک¯هگ¦ç¬¦هگˆ ChatML و ¼ه¼ڈم€‚")
    parser.add_argument("--file", default=DEFAULT_TARGET_FILE, help="ه¾…و ،éھŒçڑ„ JSON و•°وچ®è·¯ه¾„م€‚")
    parser.add_argument("--no-preview", action="store_true", help="ه…³é—­éڑڈوœ؛و ·وœ¬é¢„è§ˆم€‚")
    parser.add_argument("--seed", type=int, default=42, help="éڑڈوœ؛وٹ½و ·ç§چه­گم€‚")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    ok = test_chatml_format(args.file, preview=not args.no_preview, seed=args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

