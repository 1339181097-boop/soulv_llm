from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _copy_if_present(source: Path, target_dir: Path) -> None:
    if not source.exists():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    print(f"[OK] copied dataset file: {source} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge repo dataset registrations into a remote LLaMA-Factory dataset_info.json file."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target dataset_info.json path under the remote LLaMA-Factory data directory.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source JSON files to merge into the target dataset_info.json.",
    )
    parser.add_argument(
        "--copy-dataset",
        action="append",
        default=[],
        help="Optional dataset JSON files to copy into the target data directory.",
    )
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, Any] = {}
    if target.exists():
        merged.update(_load_json(target))

    for source_arg in args.source:
        source = Path(source_arg).expanduser().resolve()
        payload = _load_json(source)
        merged.update(payload)
        print(f"[OK] merged registration: {source}")

    with target.open("w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"[OK] wrote dataset registry: {target}")

    for dataset_arg in args.copy_dataset:
        _copy_if_present(Path(dataset_arg).expanduser().resolve(), target.parent)


if __name__ == "__main__":
    main()
