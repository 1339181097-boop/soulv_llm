from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_mixer import (
    DEFAULT_CUTOFF_LEN,
    DEFAULT_MAX_CONSECUTIVE_TASK,
    DEFAULT_SEED,
    DEFAULT_TOKENIZER_PATH,
    _build_quality_report,
    _build_token_report,
    _evaluate_token_gates,
    _load_token_counter,
    _max_consecutive_task,
)
from src.data_pipeline.data_utils import configure_console_output, log_info, log_success, read_json, resolve_path, write_json

DEFAULT_BASE_PATH = "data/final/stage1_general_sft.json"
DEFAULT_REPAIR_PATH = "data/final/stage1_badcase_repair_sft.json"
DEFAULT_OUTPUT_PATH = "data/final/stage1_general_sft_with_badcase_repair_15pct.json"
DEFAULT_REPORT_PATH = "data/final/stage1_general_sft_with_badcase_repair_15pct_report.json"
REPAIR_MIX_SOURCE = "stage1_badcase_repair_sft_weighted_15pct"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def _target_repair_count(base_count: int, repair_share_cap: float) -> int:
    if not 0 < repair_share_cap < 1:
        raise ValueError("--repair-share-cap must be between 0 and 1")
    return math.floor(base_count * repair_share_cap / (1 - repair_share_cap))


def _sample_repair_records(
    repair_records: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_count <= 0:
        return [], {"target_count": target_count, "full_cycles": 0, "remainder": 0, "oversample_count": 0}
    if not repair_records:
        raise ValueError("Repair dataset is empty")

    sampled: list[dict[str, Any]] = []
    full_cycles, remainder = divmod(target_count, len(repair_records))
    for _ in range(full_cycles):
        sampled.extend(rng.sample(repair_records, len(repair_records)))
    if remainder:
        sampled.extend(rng.sample(repair_records, remainder))
    return sampled, {
        "target_count": target_count,
        "available_unique_count": len(repair_records),
        "full_cycles": full_cycles,
        "remainder": remainder,
        "oversample_count": max(0, target_count - len(repair_records)),
    }


def _remap_repair_record(record: dict[str, Any], sequence_index: int) -> dict[str, Any]:
    remapped = copy.deepcopy(record)
    original_id = str(record.get("id") or f"repair_{sequence_index:06d}")
    digest = _hash_text(f"{original_id}:{sequence_index}")
    remapped["id"] = f"badcase_repair_mix_{sequence_index:06d}_{digest}"
    remapped["record_id"] = f"badcase_repair_mix_{sequence_index:06d}"
    remapped["source_repair_id"] = original_id
    remapped["mix_source"] = REPAIR_MIX_SOURCE
    remapped["mix_repeat_index"] = sequence_index
    return remapped


def _shuffle_with_task_cap(
    records: list[dict[str, Any]],
    *,
    rng: random.Random,
    max_consecutive_task: int,
    attempts: int = 300,
) -> tuple[list[dict[str, Any]], str]:
    for _ in range(attempts):
        candidate = list(records)
        rng.shuffle(candidate)
        if _max_consecutive_task(candidate)["count"] <= max_consecutive_task:
            return candidate, "global_shuffle"
    candidate = sorted(records, key=lambda sample: (str(sample.get("task_type")), rng.random()))
    return candidate, "task_sorted_fallback"


def _count_by_source(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        if record.get("mix_source") == REPAIR_MIX_SOURCE:
            counter["badcase_repair"] += 1
        else:
            counter["stage1_base"] += 1
    return dict(counter)


def build_mixed_dataset(
    *,
    base_path: str | Path = DEFAULT_BASE_PATH,
    repair_path: str | Path = DEFAULT_REPAIR_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    repair_share_cap: float = 0.15,
    seed: int = DEFAULT_SEED,
    max_consecutive_task: int = DEFAULT_MAX_CONSECUTIVE_TASK,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
) -> dict[str, Any]:
    configure_console_output()
    resolved_base = resolve_path(base_path)
    resolved_repair = resolve_path(repair_path)
    resolved_output = resolve_path(output_path)
    resolved_report = resolve_path(report_path)

    base_records = read_json(resolved_base)
    repair_records = read_json(resolved_repair)
    if not isinstance(base_records, list) or not isinstance(repair_records, list):
        raise ValueError("Base and repair inputs must be JSON arrays")

    rng = random.Random(seed)
    target_repair_count = _target_repair_count(len(base_records), repair_share_cap)
    sampled_repair, sampling_report = _sample_repair_records(repair_records, target_repair_count, rng)
    remapped_repair = [_remap_repair_record(record, index) for index, record in enumerate(sampled_repair, start=1)]
    mixed, order_mode = _shuffle_with_task_cap(
        [copy.deepcopy(record) for record in base_records] + remapped_repair,
        rng=rng,
        max_consecutive_task=max_consecutive_task,
    )

    task_counts = dict(Counter(str(record.get("task_type")) for record in mixed))
    quality_report = _build_quality_report(
        mixed,
        expected_task_counts=task_counts,
        max_consecutive_task=max_consecutive_task,
    )

    token_counter, tokenizer_error = _load_token_counter(tokenizer_path)
    token_report: dict[str, Any] | None = None
    token_gates: dict[str, Any] = {
        "passed": False,
        "reason": "tokenizer_unavailable",
        "tokenizer_error": tokenizer_error,
    }
    if token_counter is not None:
        token_report = _build_token_report(mixed, token_counter, cutoff_len=DEFAULT_CUTOFF_LEN)
        token_gates = _evaluate_token_gates(token_report)

    repair_count = len(remapped_repair)
    mixed_count = len(mixed)
    repair_share = repair_count / mixed_count if mixed_count else 0.0
    passed = quality_report["passed"] and repair_share <= repair_share_cap + 1e-12 and (
        token_counter is None or token_gates["passed"]
    )
    report = {
        "name": "stage1_general_sft_with_badcase_repair_15pct",
        "status": "passed" if passed else "failed",
        "output_path": str(resolved_output),
        "output_written": False,
        "base_path": str(resolved_base),
        "repair_path": str(resolved_repair),
        "seed": seed,
        "sample_count": mixed_count,
        "base_count": len(base_records),
        "repair_count": repair_count,
        "repair_share_cap": repair_share_cap,
        "repair_share_actual": round(repair_share, 6),
        "repair_sampling": sampling_report,
        "source_counts": _count_by_source(mixed),
        "task_counts": task_counts,
        "order_mode": order_mode,
        "quality": quality_report,
        "tokenizer_path": str(resolve_path(tokenizer_path)),
        "tokenizer_error": tokenizer_error,
        "token_report": token_report,
        "token_gates": token_gates,
    }

    if not passed:
        write_json(resolved_report, report)
        raise ValueError(f"Mixed dataset failed strict gates. See report: {resolved_report}")

    write_json(resolved_output, mixed)
    report["output_written"] = True
    write_json(resolved_report, report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix Stage1 base SFT with weighted badcase repair data capped by share.")
    parser.add_argument("--base", default=DEFAULT_BASE_PATH, help="Base stage1_general_sft JSON.")
    parser.add_argument("--repair", default=DEFAULT_REPAIR_PATH, help="Prepared badcase repair SFT JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Mixed output JSON.")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Mixed output report JSON.")
    parser.add_argument("--repair-share-cap", type=float, default=0.15, help="Maximum repair share in final mixed dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-consecutive-task", type=int, default=DEFAULT_MAX_CONSECUTIVE_TASK)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    log_info("Building Stage1 badcase repair mix")
    report = build_mixed_dataset(
        base_path=args.base,
        repair_path=args.repair,
        output_path=args.output,
        report_path=args.report,
        repair_share_cap=args.repair_share_cap,
        seed=args.seed,
        max_consecutive_task=args.max_consecutive_task,
        tokenizer_path=args.tokenizer_path,
    )
    log_success(f"Wrote mixed dataset: {report['output_path']}")
    log_success(
        f"Samples={report['sample_count']}, repair={report['repair_count']} "
        f"({report['repair_share_actual']:.4%})"
    )
    log_success(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
