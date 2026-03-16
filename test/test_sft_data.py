from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_utils import (
    configure_console_output,
    log_error,
    log_info,
    log_success,
    read_json,
    resolve_path,
    validate_chatml_dataset,
)

DEFAULT_TARGET_FILE = "data/processed/sft_intent.json"


def preview_sample(dataset: list[dict], seed: int) -> None:
    rng = random.Random(seed)
    sample = rng.choice(dataset)
    print("=" * 80)
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print("=" * 80)


def test_chatml_format(file_path: str = DEFAULT_TARGET_FILE, *, preview: bool = True, seed: int = 42) -> bool:
    configure_console_output()
    resolved = resolve_path(file_path)
    log_info(f"开始校验 SFT 数据文件: {resolved}")

    if not resolved.exists():
        log_error(f"未找到目标文件: {resolved}")
        return False

    try:
        dataset = read_json(resolved)
    except json.JSONDecodeError as exc:
        log_error(f"JSON 解析失败: {exc}")
        return False

    errors = validate_chatml_dataset(dataset)
    if errors:
        log_error(f"校验失败，共发现 {len(errors)} 个问题。")
        for error in errors[:10]:
            log_error(error)
        return False

    sample_count = len(dataset)
    log_success(f"ChatML 结构校验通过，共 {sample_count} 条样本。")
    if preview and dataset:
        log_info("随机抽样一条数据用于人工检查。")
        preview_sample(dataset, seed)

    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校验搜旅 SFT 数据是否符合 ChatML 格式。")
    parser.add_argument("--file", default=DEFAULT_TARGET_FILE, help="待校验的 JSON 数据路径。")
    parser.add_argument("--no-preview", action="store_true", help="关闭随机样本预览。")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样种子。")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    ok = test_chatml_format(args.file, preview=not args.no_preview, seed=args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
