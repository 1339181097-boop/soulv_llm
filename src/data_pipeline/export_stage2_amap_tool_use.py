from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    log_error,
    log_info,
    log_success,
    read_json,
    write_json,
)
from src.tool_use.datasets import export_tool_use_dataset_to_sharegpt, validate_sharegpt_tool_dataset

DEFAULT_SOURCE_PATH = "data/tool_use/stage2_amap_tool_use_source.json"
DEFAULT_OUTPUT_PATH = "data/final/stage2_amap_tool_use_sft.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export stage2 tool-use source dataset into LLaMA-Factory sharegpt format.")
    parser.add_argument("--source", default=DEFAULT_SOURCE_PATH, help="Stage2 source dataset JSON path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Exported sharegpt JSON path.")
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()

    dataset = read_json(args.source)
    exported = export_tool_use_dataset_to_sharegpt(dataset)
    errors = validate_sharegpt_tool_dataset(exported)
    if errors:
        log_error(f"Exported dataset validation failed with {len(errors)} errors.")
        for error in errors[:10]:
            log_error(error)
        return 1

    output_path = write_json(args.output, exported)
    log_info(f"Loaded source dataset: {args.source}")
    log_success(f"Exported sharegpt dataset written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
