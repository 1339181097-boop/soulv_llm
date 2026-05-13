from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline import clean_stage2_clarify_then_call as cleaner
from src.data_pipeline.data_utils import configure_console_output


DEFAULT_SEED = Path("data/processed_stage2/clarify_then_call_strict.jsonl")
DEFAULT_INPUTS = [
    Path("data/raw_stage2/clarify_then_call.jsonl"),
    Path("data/raw_stage2/clarify_then_call_v2.jsonl"),
    Path("data/raw_stage2/clarify_then_call_v3.jsonl"),
    Path("data/raw_stage2/clarify_then_call_v4.jsonl"),
]
DEFAULT_OUTPUT = Path("data/processed_stage2/clarify_then_call_strict_merged.jsonl")
DEFAULT_REPORT = Path("data/processed_stage2/clarify_then_call_strict_merged_report.json")


def _read_processed_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object.")
        rows.append(payload)
    return rows


def _duplicate_excess(counter: Counter[str]) -> int:
    return sum(count - 1 for count in counter.values() if count > 1)


def _selection_report(
    *,
    seed_file: Path,
    cleaned: list[dict[str, Any]],
    seed_items: list[dict[str, Any]],
    raw_rows: list[cleaner.RawRow],
    candidate_counts: dict[str, int],
    rejected: Counter[str],
    fixes: Counter[str],
    duplicate_budget: int,
) -> dict[str, Any]:
    selected_counts = Counter(item["subtype"] for item in cleaned)
    seed_counts = Counter(item["subtype"] for item in seed_items)
    tool_counts: Counter[str] = Counter()
    route_modes: Counter[str] = Counter()
    observation_statuses: Counter[str] = Counter()
    questions: Counter[str] = Counter()
    dialogue_keys: Counter[str] = Counter()
    function_calls: Counter[str] = Counter()
    argument_signatures: Counter[str] = Counter()
    answer_lengths = [len(item["conversations"][-1]["value"]) for item in cleaned]

    for item in cleaned:
        questions[item["conversations"][1]["value"]] += 1
        dialogue_keys[cleaner._dialogue_key(item)] += 1
        function_payload = json.loads(item["conversations"][4]["value"])
        observation = json.loads(item["conversations"][5]["value"])
        tool_name = function_payload["name"]
        args = function_payload["arguments"]
        tool_counts[tool_name] += 1
        observation_statuses[observation.get("status")] += 1
        function_calls[item["conversations"][4]["value"]] += 1
        argument_signatures[json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False, sort_keys=True)] += 1
        if tool_name == "amap_plan_route":
            route_modes[args.get("mode")] += 1

    source_counts = Counter(row.source_file for row in raw_rows)
    raw_counts = Counter(row.item.get("subtype") for row in raw_rows)
    shortage_counts = {
        subtype: max(0, cleaner.TARGET_COUNTS[subtype] - selected_counts.get(subtype, 0))
        for subtype in cleaner.SUBTYPE_ORDER
    }
    duplicate_questions = _duplicate_excess(questions)
    duplicate_dialogue_keys = _duplicate_excess(dialogue_keys)
    duplicate_function_calls = _duplicate_excess(function_calls)
    duplicate_argument_signatures = _duplicate_excess(argument_signatures)

    return {
        "merge_strategy": "seed_first_strict_incremental_zero_duplicate_function_call",
        "seed_file": str(seed_file),
        "seed_count": len(seed_items),
        "seed_counts": dict(sorted(seed_counts.items())),
        "added_count": max(0, len(cleaned) - len(seed_items)),
        "input_records": len(raw_rows),
        "input_files": dict(sorted(source_counts.items())),
        "target_counts": dict(cleaner.TARGET_COUNTS),
        "target_total": sum(cleaner.TARGET_COUNTS.values()),
        "raw_subtype_counts": dict(sorted(raw_counts.items())),
        "candidate_counts": {subtype: candidate_counts.get(subtype, 0) for subtype in cleaner.SUBTYPE_ORDER},
        "selected_count": len(cleaned),
        "selected_counts": dict(sorted(selected_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "route_mode_counts": dict(sorted(route_modes.items())),
        "observation_status_counts": dict(sorted(observation_statuses.items())),
        "selected_duplicate_questions": duplicate_questions,
        "selected_duplicate_dialogue_keys": duplicate_dialogue_keys,
        "selected_duplicate_dialogue_key_rate": round(duplicate_dialogue_keys / len(cleaned), 4) if cleaned else 0,
        "selected_duplicate_function_calls": duplicate_function_calls,
        "selected_duplicate_function_call_rate": round(duplicate_function_calls / len(cleaned), 4) if cleaned else 0,
        "selected_duplicate_argument_signatures": duplicate_argument_signatures,
        "selected_duplicate_argument_signature_rate": round(duplicate_argument_signatures / len(cleaned), 4) if cleaned else 0,
        "duplicate_function_call_budget": duplicate_budget,
        "shortage_counts": {subtype: count for subtype, count in shortage_counts.items() if count},
        "shortage_total": sum(shortage_counts.values()),
        "rejected": dict(sorted(rejected.items())),
        "fixes": dict(sorted(fixes.items())),
        "answer_length": {
            "min": min(answer_lengths, default=0),
            "max": max(answer_lengths, default=0),
            "avg": round(sum(answer_lengths) / len(cleaned), 1) if cleaned else 0,
        },
        "validation": {
            "source_validator_errors": 0,
            "sharegpt_validator_errors": 0,
        },
        "output_format": "processed_stage2_jsonl_with_conversations",
    }


def merge_clarify_then_call_strict(
    *,
    seed_file: Path,
    seed_items: list[dict[str, Any]],
    raw_rows: list[cleaner.RawRow],
    allow_duplicate_function_calls: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_errors = cleaner._validate_processed_rows(seed_items)
    if seed_errors:
        raise RuntimeError(f"seed clarify_then_call failed validation: {seed_errors[:5]}")

    rejected: Counter[str] = Counter()
    fixes: Counter[str] = Counter()
    candidates_by_subtype: dict[str, list[cleaner.Candidate]] = {subtype: [] for subtype in cleaner.SUBTYPE_ORDER}
    for row in raw_rows:
        candidate = cleaner._clean_one(row, rejected, fixes)
        if candidate is not None:
            candidates_by_subtype[candidate.subtype].append(candidate)

    for subtype, candidates in candidates_by_subtype.items():
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.argument_signature,
                candidate.raw_id,
                candidate.source_file,
                candidate.line_number,
            )
        )

    selected_by_subtype: dict[str, list[dict[str, Any]]] = {subtype: [] for subtype in cleaner.SUBTYPE_ORDER}
    used_questions: set[str] = set()
    used_dialogue_keys: set[str] = set()
    function_call_counts: Counter[str] = Counter()
    for item in seed_items:
        subtype = item.get("subtype")
        if subtype not in cleaner.TARGET_COUNTS:
            raise RuntimeError(f"seed item {item.get('id')} has unsupported subtype={subtype!r}")
        selected_by_subtype[str(subtype)].append(item)
        used_questions.add(item["conversations"][1]["value"])
        used_dialogue_keys.add(cleaner._dialogue_key(item))
        function_call_counts[item["conversations"][4]["value"]] += 1

    for subtype in cleaner.SUBTYPE_ORDER:
        seed_count = len(selected_by_subtype[subtype])
        target_count = cleaner.TARGET_COUNTS[subtype]
        if seed_count > target_count:
            raise RuntimeError(f"seed subtype {subtype} has {seed_count} rows, target={target_count}")

    duplicate_budget = 0
    if allow_duplicate_function_calls:
        duplicate_budget = int(sum(cleaner.TARGET_COUNTS.values()) * cleaner.MAX_DUPLICATE_RATE)

    for subtype in cleaner.SUBTYPE_ORDER:
        remaining = cleaner.TARGET_COUNTS[subtype] - len(selected_by_subtype[subtype])
        if remaining <= 0:
            continue
        selected_by_subtype[subtype].extend(
            cleaner._select_for_subtype(
                subtype,
                candidates_by_subtype[subtype],
                remaining,
                used_questions=used_questions,
                used_dialogue_keys=used_dialogue_keys,
                function_call_counts=function_call_counts,
                duplicate_budget=duplicate_budget,
                allow_partial=True,
                fixes=fixes,
            )
        )

    cleaned = cleaner._renumber_selected(selected_by_subtype)
    validation_errors = cleaner._validate_processed_rows(cleaned)
    if validation_errors:
        raise RuntimeError(f"merged clarify_then_call failed validation: {validation_errors[:5]}")

    candidate_counts = {subtype: len(candidates_by_subtype[subtype]) for subtype in cleaner.SUBTYPE_ORDER}
    report = _selection_report(
        cleaned=cleaned,
        seed_file=seed_file,
        seed_items=seed_items,
        raw_rows=raw_rows,
        candidate_counts=candidate_counts,
        rejected=rejected,
        fixes=fixes,
        duplicate_budget=duplicate_budget,
    )
    return cleaned, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge strict clarify_then_call seed data with strict raw candidates.")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--input", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--allow-duplicate-function-calls", action="store_true")
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    seed_items = _read_processed_jsonl(args.seed)
    raw_rows = cleaner._read_jsonl_files(args.input)
    cleaned, report = merge_clarify_then_call_strict(
        seed_file=args.seed,
        seed_items=seed_items,
        raw_rows=raw_rows,
        allow_duplicate_function_calls=args.allow_duplicate_function_calls,
    )
    cleaner._write_jsonl(args.output, cleaned)
    cleaner._write_json(args.report, report)
    print(f"[OK] merged clarify_then_call written to: {args.output}")
    print(f"[OK] report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
