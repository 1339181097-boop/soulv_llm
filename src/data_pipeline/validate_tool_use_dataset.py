from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import configure_console_output, log_error, log_info, log_success, read_json
from src.tool_use.datasets import validate_sharegpt_tool_dataset, validate_tool_use_source_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate stage2 tool-use dataset.")
    parser.add_argument("--file", required=True, help="Dataset JSON path.")
    parser.add_argument(
        "--format",
        choices=("source", "sharegpt"),
        default="source",
        help="Dataset format to validate.",
    )
    parser.add_argument("--preview", action="store_true", help="Preview first sample when validation succeeds.")
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()

    dataset = read_json(args.file)
    validator = validate_tool_use_source_dataset if args.format == "source" else validate_sharegpt_tool_dataset
    errors = validator(dataset)
    if errors:
        log_error(f"Validation failed with {len(errors)} errors.")
        for error in errors[:10]:
            log_error(error)
        return 1

    log_success(f"{args.format} dataset validation passed: {args.file}")
    if args.preview and isinstance(dataset, list) and dataset:
        log_info(json.dumps(dataset[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
