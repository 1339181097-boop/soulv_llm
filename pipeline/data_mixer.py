from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_utils import (
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
DEFAULT_SPECS = {
    "sft_itinerary.json": 0.4,
    "sft_multiturn.json": 0.25,
    "sft_intent.json": 0.15,
    "sft_roleplay_safety.json": 0.1,
    "sft_basic_qa.json": 0.1,
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
            raise ValueError(f"无效规格: {raw_spec}，应为 filename=weight")
        filename, raw_weight = raw_spec.split("=", 1)
        weight = float(raw_weight)
        if weight <= 0:
            raise ValueError(f"权重必须大于 0: {raw_spec}")
        specs[filename.strip()] = weight
    return specs


def _load_bucket(filename: str, weight: float) -> DatasetBucket | None:
    path = resolve_path(f"data/processed/{filename}")
    if not path.exists():
        log_warn(f"缺少数据集，跳过混合: {path}")
        return None

    dataset = read_json(path)
    errors = validate_chatml_dataset(dataset)
    if errors:
        log_warn(f"数据集格式不合法，跳过混合: {path}")
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
        log_warn("没有找到可混合的 processed 数据集。")
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
                    f"{bucket.filename} 可用样本不足，目标 {counts[bucket.filename]} 条，实际仅抽取 {target} 条。"
                )
            mixed.extend(rng.sample(bucket.records, target))

    rng.shuffle(mixed)
    output_path = write_json(output_json_path, mixed)
    log_success(f"数据混合完成，输出 {len(mixed)} 条样本。")
    log_info(f"输出文件: {output_path}")
    return mixed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按比例混合 processed 目录下的 ChatML 数据。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="混合输出 JSON 路径。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子。")
    parser.add_argument("--total-samples", type=int, default=None, help="目标混合样本数，默认取全部。")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="数据集配比，格式 filename=weight，可重复指定。",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    specs = _parse_specs(args.spec) if args.spec else None
    mix_datasets(args.output, seed=args.seed, total_samples=args.total_samples, specs=specs)


if __name__ == "__main__":
    main()
