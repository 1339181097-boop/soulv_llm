from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    read_json,
    resolve_path,
    validate_chatml_dataset,
    write_json,
    log_info,
    log_success,
    log_warn,
)

DEFAULT_OUTPUT_PATH = "data/final/soulv_mixed_sft.json"
DEFAULT_SEED = 42
# 默认纳入当前已经清洗完成的四类 SFT 数据。
# 在不指定 total_samples 时会全量合并并打乱；
# 当需要按目标样本量重采样时，再使用下方权重控制混料比例。
DEFAULT_SPECS = {
    "sft_itinerary.json": 0.40,
    "sft_intent.json": 0.20,
    "sft_dialogue.json": 0.25,
    "sft_roleplay_safety.json": 0.15,
}


@dataclass
class DatasetBucket:
    filename: str
    weight: float
    records: list[dict]


def _parse_specs(raw_specs: list[str]) -> dict[str, float]:
    specs: dict[str, float] = {}
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"و— و•ˆè§„و ¼: {raw_spec}ï¼Œه؛”ن¸؛ filename=weight")
        filename, raw_weight = raw_spec.split("=", 1)
        weight = float(raw_weight)
        if weight <= 0:
            raise ValueError(f"و‌ƒé‡چه؟…é،»ه¤§ن؛ژ 0: {raw_spec}")
        specs[filename.strip()] = weight
    return specs


def _load_bucket(filename: str, weight: float) -> DatasetBucket | None:
    path = resolve_path(f"data/processed/{filename}")
    if not path.exists():
        log_warn(f"ç¼؛ه°‘و•°وچ®é›†ï¼Œè·³è؟‡و··هگˆ: {path}")
        return None

    dataset = read_json(path)
    errors = validate_chatml_dataset(dataset)
    if errors:
        log_warn(f"و•°وچ®é›†و ¼ه¼ڈن¸چهگˆو³•ï¼Œè·³è؟‡و··هگˆ: {path}")
        for error in errors[:5]:
            log_warn(error)
        return None

    return DatasetBucket(filename=filename, weight=weight, records=list(dataset))


def _target_counts(buckets: list[DatasetBucket], total_samples: int) -> dict[str, int]:
    total_weight = sum(bucket.weight for bucket in buckets)
    requested: dict[str, int] = {}
    remaining = total_samples

    for index, bucket in enumerate(buckets):
        if index == len(buckets) - 1:
            requested[bucket.filename] = remaining
            continue
        count = round(total_samples * (bucket.weight / total_weight))
        requested[bucket.filename] = count
        remaining -= count

    return requested


def mix_datasets(
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    seed: int = DEFAULT_SEED,
    total_samples: int | None = None,
    specs: dict[str, float] | None = None,
) -> list[dict]:
    configure_console_output()
    rng = random.Random(seed)
    effective_specs = specs or DEFAULT_SPECS
    buckets: list[DatasetBucket] = []

    for filename, weight in effective_specs.items():
        bucket = _load_bucket(filename, weight)
        if bucket is not None and bucket.records:
            buckets.append(bucket)

    if not buckets:
        log_warn("و²،وœ‰و‰¾هˆ°هڈ¯و··هگˆçڑ„ processed و•°وچ®é›†م€‚")
        return []

    mixed: list[dict] = []
    if total_samples is None:
        for bucket in buckets:
            mixed.extend(bucket.records)
    else:
        counts = _target_counts(buckets, total_samples)
        for bucket in buckets:
            target = min(counts[bucket.filename], len(bucket.records))
            if target < counts[bucket.filename]:
                log_warn(
                    f"{bucket.filename} هڈ¯ç”¨و ·وœ¬ن¸چè¶³ï¼Œç›®و ‡ {counts[bucket.filename]} و‌،ï¼Œه®‍é™…ن»…وٹ½هڈ– {target} و‌،م€‚"
                )
            mixed.extend(rng.sample(bucket.records, target))

    rng.shuffle(mixed)
    output_path = write_json(output_json_path, mixed)
    log_success(f"و•°وچ®و··هگˆه®Œوˆگï¼Œè¾“ه‡؛ {len(mixed)} و‌،و ·وœ¬م€‚")
    log_info(f"è¾“ه‡؛و–‡ن»¶: {output_path}")
    return mixed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="وŒ‰و¯”ن¾‹و··هگˆ processed ç›®ه½•ن¸‹çڑ„ ChatML و•°وچ®م€‚")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="و··هگˆè¾“ه‡؛ JSON è·¯ه¾„م€‚")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="éڑڈوœ؛ç§چه­گم€‚")
    parser.add_argument("--total-samples", type=int, default=None, help="ç›®و ‡و··هگˆو ·وœ¬و•°ï¼Œé»کè®¤هڈ–ه…¨éƒ¨م€‚")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="و•°وچ®é›†é…چو¯”ï¼Œو ¼ه¼ڈ filename=weightï¼Œهڈ¯é‡چه¤چوŒ‡ه®ڑم€‚",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    specs = _parse_specs(args.spec) if args.spec else None
    mix_datasets(args.output, seed=args.seed, total_samples=args.total_samples, specs=specs)


if __name__ == "__main__":
    main()



