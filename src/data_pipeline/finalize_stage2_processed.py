from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    load_records,
    log_error,
    log_info,
    log_success,
    resolve_path,
    write_json,
)
from src.tool_use.datasets import validate_sharegpt_tool_dataset
from src.tool_use.protocol import ALLOWED_TWO_STEP_CHAINS, AMAP_TOOL_NAMES, build_amap_tool_schemas

DEFAULT_PROCESSED_DIR = "data/processed_stage2"
DEFAULT_OUTPUT_PATH = "data/final/stage2_amap_tool_use_sft.json"
DEFAULT_REPORT_PATH = "data/final/stage2_amap_tool_use_report.json"
DEFAULT_SEED = 42
DEFAULT_MAX_CONSECUTIVE_TASK = 4
MIX_PROFILE = "qwen3_32b_stage2_processed_stage2_quality_first"


@dataclass(frozen=True)
class Stage2ProcessedInput:
    filename: str
    task_type: str
    target_count: int


DEFAULT_INPUTS: tuple[Stage2ProcessedInput, ...] = (
    Stage2ProcessedInput("clarify_then_call_strict_merged.jsonl", "clarify_then_call", 1200),
    Stage2ProcessedInput("two_step_chain.jsonl", "two_step_chain", 800),
    Stage2ProcessedInput("tool_result_grounded_answer.jsonl", "tool_result_grounded_answer", 400),
    Stage2ProcessedInput("single_tool_call.jsonl", "single_tool_call", 240),
    Stage2ProcessedInput("slot_filling_tool_call.jsonl", "slot_filling_tool_call", 240),
    Stage2ProcessedInput("tool_failure_fallback.jsonl", "tool_failure_fallback", 160),
    Stage2ProcessedInput("no_tool_needed.jsonl", "no_tool_needed", 160),
)


class Stage2FinalizationError(ValueError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        errors = report.get("validation", {}).get("sample_errors", [])
        preview = "; ".join(str(error) for error in errors[:3])
        super().__init__(preview or "stage2 processed finalization failed")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _target_counts(inputs: tuple[Stage2ProcessedInput, ...]) -> dict[str, int]:
    return {stage_input.task_type: stage_input.target_count for stage_input in inputs}


def _load_processed_records(
    processed_dir: str | Path,
    inputs: tuple[Stage2ProcessedInput, ...],
) -> dict[str, list[dict[str, Any]]]:
    base = resolve_path(processed_dir)
    records_by_task: dict[str, list[dict[str, Any]]] = {}
    for stage_input in inputs:
        records_by_task[stage_input.task_type] = load_records(base / stage_input.filename)
    return records_by_task


def _prepare_record(
    record: dict[str, Any],
    *,
    expected_task_type: str,
    tools_json: str,
    source_name: str,
    record_index: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    for field_name in ("id", "task_type", "scene", "expected_behavior"):
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{source_name}[{record_index}] needs non-empty {field_name}.")

    task_type = record.get("task_type")
    if isinstance(task_type, str) and task_type != expected_task_type:
        errors.append(
            f"{source_name}[{record_index}] has task_type {task_type!r}, expected {expected_task_type!r}."
        )

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        errors.append(f"{source_name}[{record_index}] needs non-empty conversations.")
        conversations = []

    item = {
        "id": str(record.get("id", "")),
        "task_type": str(record.get("task_type", "")),
        "scene": str(record.get("scene", "")),
        "expected_behavior": str(record.get("expected_behavior", "")),
        "tools": tools_json,
        "conversations": conversations,
    }
    return item, errors


def _parse_function_call(value: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        return None, None, f"invalid function_call JSON: {exc}"
    if not isinstance(payload, dict):
        return None, None, "function_call payload must be an object"
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str):
        return None, None, "function_call payload needs string name"
    if not isinstance(arguments, dict):
        return name, None, "function_call payload needs object arguments"
    return name, arguments, None


def _tool_chain(item: dict[str, Any]) -> list[str]:
    chain: list[str] = []
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return chain
    for message in conversations:
        if not isinstance(message, dict) or message.get("from") != "function_call":
            continue
        name, _, _ = _parse_function_call(str(message.get("value", "")))
        if name:
            chain.append(name)
    return chain


def _tool_chain_key(item: dict[str, Any]) -> str:
    chain = _tool_chain(item)
    return " -> ".join(chain) if chain else "<none>"


def _observation_status(value: str) -> str | None:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def _has_unconverted_long_unit(text: str) -> bool:
    index = 0
    while index < len(text):
        if not text[index].isdigit():
            index += 1
            continue

        start = index
        while index < len(text) and text[index].isdigit():
            index += 1
        raw_value = text[start:index]
        cursor = index
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            continue

        number = int(raw_value)
        unit = text[cursor]
        if unit == "\u79d2" and number >= 60:
            return True
        if unit == "\u7c73" and number >= 1000:
            context = text[max(0, start - 20) : min(len(text), cursor + 10)]
            if "\u6d77\u62d4" in context or "\u9ad8\u539f" in context:
                continue
            return True
    return False


def _semantic_validation_errors(dataset: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(dataset):
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"item {index} needs non-empty id.")
        elif sample_id in seen_ids:
            errors.append(f"item {index} duplicates id {sample_id!r}.")
        else:
            seen_ids.add(sample_id)

        conversations = item.get("conversations")
        if not isinstance(conversations, list):
            errors.append(f"item {index} conversations must be a list.")
            continue

        chain = _tool_chain(item)
        for tool_name in chain:
            if tool_name not in AMAP_TOOL_NAMES:
                errors.append(f"item {index} uses unknown tool {tool_name!r}.")
        if len(chain) > 2:
            errors.append(f"item {index} exceeds max tool rounds: {chain!r}.")
        if len(chain) == 2 and tuple(chain) not in ALLOWED_TWO_STEP_CHAINS:
            errors.append(f"item {index} uses disallowed tool chain: {chain!r}.")

        task_type = item.get("task_type")
        roles = [message.get("from") for message in conversations if isinstance(message, dict)]
        if task_type == "no_tool_needed" and any(role in {"function_call", "observation"} for role in roles):
            errors.append(f"item {index} no_tool_needed contains tool trace.")

        if task_type == "clarify_then_call":
            first_gpt = next((i for i, role in enumerate(roles) if role == "gpt"), None)
            first_function = next((i for i, role in enumerate(roles) if role == "function_call"), None)
            if first_gpt is None or first_function is None:
                errors.append(f"item {index} clarify_then_call needs clarify answer and function_call.")
            elif first_function <= first_gpt:
                errors.append(f"item {index} clarify_then_call calls tool before clarification.")
            elif "human" not in roles[first_gpt + 1 : first_function]:
                errors.append(f"item {index} clarify_then_call needs user supplement before function_call.")

        if task_type == "tool_failure_fallback":
            for message in conversations:
                if not isinstance(message, dict) or message.get("from") != "observation":
                    continue
                status = _observation_status(str(message.get("value", "")))
                if status not in {"error", "empty"}:
                    errors.append(f"item {index} fallback observation status must be error/empty, got {status!r}.")

        for message in conversations:
            if not isinstance(message, dict) or message.get("from") != "gpt":
                continue
            value = message.get("value")
            if isinstance(value, str) and _has_unconverted_long_unit(value):
                errors.append(f"item {index} contains unconverted long seconds/meters in assistant answer.")
                break

    return errors


def _max_consecutive_task(records: list[dict[str, Any]]) -> dict[str, Any]:
    max_task: str | None = None
    max_run = 0
    current_task: str | None = None
    current_run = 0
    for sample in records:
        task_type = sample.get("task_type")
        if task_type == current_task:
            current_run += 1
        else:
            current_task = str(task_type)
            current_run = 1
        if current_run > max_run:
            max_run = current_run
            max_task = current_task
    return {"task_type": max_task, "count": max_run}


def _weighted_interleave_tasks(
    records_by_task: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    *,
    max_consecutive_task: int,
) -> list[dict[str, Any]]:
    queues: dict[str, deque[dict[str, Any]]] = {}
    for task_type, records in records_by_task.items():
        shuffled = list(records)
        rng.shuffle(shuffled)
        queues[task_type] = deque(shuffled)

    mixed: list[dict[str, Any]] = []
    current_task: str | None = None
    current_run = 0
    remaining_total = sum(len(queue) for queue in queues.values())

    while remaining_total:
        candidates = [
            task_type
            for task_type, queue in queues.items()
            if queue and not (task_type == current_task and current_run >= max_consecutive_task)
        ]
        if not candidates:
            raise ValueError("Unable to satisfy max consecutive task constraint.")

        candidate_total = sum(len(queues[task_type]) for task_type in candidates)
        pick = rng.uniform(0, candidate_total)
        cursor = 0.0
        chosen_task = candidates[-1]
        for task_type in candidates:
            cursor += len(queues[task_type])
            if pick <= cursor:
                chosen_task = task_type
                break

        mixed.append(queues[chosen_task].popleft())
        remaining_total -= 1
        if chosen_task == current_task:
            current_run += 1
        else:
            current_task = chosen_task
            current_run = 1

    return mixed


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _nested_counter_dict(counter: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: _counter_dict(value) for key, value in sorted(counter.items())}


def _build_report(
    *,
    dataset: list[dict[str, Any]],
    inputs: tuple[Stage2ProcessedInput, ...],
    processed_dir: str | Path,
    seed: int,
    max_consecutive_task: int,
    preparation_errors: list[str],
    sharegpt_errors: list[str],
    semantic_errors: list[str],
    loaded_counts: dict[str, int],
    subtype_counts: dict[str, Counter[str]],
    output_path: str | Path,
) -> dict[str, Any]:
    target_counts = _target_counts(inputs)
    task_counts = Counter(str(item.get("task_type")) for item in dataset)
    scene_counts = Counter(str(item.get("scene")) for item in dataset)
    behavior_counts = Counter(str(item.get("expected_behavior")) for item in dataset)
    tool_chain_counts = Counter(_tool_chain_key(item) for item in dataset)
    role_sequence_counts = Counter(
        " -> ".join(str(message.get("from")) for message in item.get("conversations", []) if isinstance(message, dict))
        for item in dataset
    )
    shortfall = {
        task_type: max(0, target_counts.get(task_type, 0) - task_counts.get(task_type, 0))
        for task_type in target_counts
    }
    sample_errors = preparation_errors + sharegpt_errors + semantic_errors
    validation_passed = not sample_errors

    input_files: dict[str, dict[str, Any]] = {}
    base = resolve_path(processed_dir)
    for stage_input in inputs:
        actual_count = task_counts.get(stage_input.task_type, 0)
        input_files[stage_input.task_type] = {
            "file": str(base / stage_input.filename),
            "target_count": stage_input.target_count,
            "loaded_count": loaded_counts.get(stage_input.task_type, 0),
            "final_count": actual_count,
            "shortfall": max(0, stage_input.target_count - actual_count),
        }

    return {
        "mix_profile": MIX_PROFILE,
        "mix_strategy": "quality_first_no_oversampling_from_processed_stage2",
        "output_format": "llamafactory_sharegpt_tools",
        "output_path": str(resolve_path(output_path)),
        "seed": seed,
        "max_consecutive_task_allowed": max_consecutive_task,
        "total_samples": len(dataset),
        "target_total": sum(target_counts.values()),
        "shortfall_total": sum(shortfall.values()),
        "targets": target_counts,
        "actual_counts": _counter_dict(task_counts),
        "shortfall_by_task": shortfall,
        "input_files": input_files,
        "distributions": {
            "task_type": _counter_dict(task_counts),
            "scene": _counter_dict(scene_counts),
            "expected_behavior": _counter_dict(behavior_counts),
            "subtype_by_task": _nested_counter_dict(subtype_counts),
            "tool_chain": _counter_dict(tool_chain_counts),
            "role_sequence": _counter_dict(role_sequence_counts),
        },
        "quality": {
            "status": "passed" if validation_passed else "failed",
            "max_consecutive_task": _max_consecutive_task(dataset),
            "unique_id_count": len({item.get("id") for item in dataset}),
            "duplicate_id_count": len(dataset) - len({item.get("id") for item in dataset}),
        },
        "validation": {
            "passed": validation_passed,
            "preparation_error_count": len(preparation_errors),
            "sharegpt_validator_error_count": len(sharegpt_errors),
            "semantic_error_count": len(semantic_errors),
            "sample_errors": sample_errors[:50],
        },
        "status": "passed" if validation_passed else "failed",
        "output_written": False,
    }


def build_final_dataset_from_records(
    records_by_task: dict[str, list[dict[str, Any]]],
    *,
    inputs: tuple[Stage2ProcessedInput, ...] = DEFAULT_INPUTS,
    seed: int = DEFAULT_SEED,
    max_consecutive_task: int = DEFAULT_MAX_CONSECUTIVE_TASK,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    tools_json = _json_dumps(build_amap_tool_schemas())
    prepared_by_task: dict[str, list[dict[str, Any]]] = {}
    preparation_errors: list[str] = []
    loaded_counts: dict[str, int] = {}
    subtype_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for stage_input in inputs:
        raw_records = records_by_task.get(stage_input.task_type, [])
        loaded_counts[stage_input.task_type] = len(raw_records)
        prepared_records: list[dict[str, Any]] = []
        for record_index, record in enumerate(raw_records):
            subtype = record.get("subtype")
            subtype_counts[stage_input.task_type][str(subtype) if isinstance(subtype, str) else "<missing>"] += 1
            item, errors = _prepare_record(
                record,
                expected_task_type=stage_input.task_type,
                tools_json=tools_json,
                source_name=stage_input.filename,
                record_index=record_index,
            )
            preparation_errors.extend(errors)
            prepared_records.append(item)
        prepared_by_task[stage_input.task_type] = prepared_records

    try:
        dataset = _weighted_interleave_tasks(
            prepared_by_task,
            rng,
            max_consecutive_task=max_consecutive_task,
        )
    except ValueError as exc:
        dataset = [item for records in prepared_by_task.values() for item in records]
        preparation_errors.append(str(exc))

    sharegpt_errors = validate_sharegpt_tool_dataset(dataset)
    semantic_errors = _semantic_validation_errors(dataset)
    report = _build_report(
        dataset=dataset,
        inputs=inputs,
        processed_dir=processed_dir,
        seed=seed,
        max_consecutive_task=max_consecutive_task,
        preparation_errors=preparation_errors,
        sharegpt_errors=sharegpt_errors,
        semantic_errors=semantic_errors,
        loaded_counts=loaded_counts,
        subtype_counts=subtype_counts,
        output_path=output_path,
    )
    return dataset, report


def finalize_stage2_processed(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    inputs: tuple[Stage2ProcessedInput, ...] = DEFAULT_INPUTS,
    seed: int = DEFAULT_SEED,
    max_consecutive_task: int = DEFAULT_MAX_CONSECUTIVE_TASK,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_by_task = _load_processed_records(processed_dir, inputs)
    dataset, report = build_final_dataset_from_records(
        records_by_task,
        inputs=inputs,
        seed=seed,
        max_consecutive_task=max_consecutive_task,
        processed_dir=processed_dir,
        output_path=output_path,
    )
    if report["status"] != "passed":
        raise Stage2FinalizationError(report)

    output_file = write_json(output_path, dataset)
    report["output_written"] = True
    report["output_path"] = str(output_file)
    report_file = write_json(report_path, report)
    log_success(f"Stage2 final dataset written: {output_file}")
    log_success(f"Stage2 final report written: {report_file}")
    return dataset, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize cleaned processed_stage2 tool-use data.")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR, help="Directory with processed_stage2 JSONL files.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Final LLaMA-Factory sharegpt output path.")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Final report output path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic shuffle seed.")
    parser.add_argument(
        "--max-consecutive-task",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_TASK,
        help="Maximum consecutive samples from the same task type.",
    )
    return parser


def main() -> int:
    configure_console_output()
    args = build_arg_parser().parse_args()
    log_info(f"Finalizing stage2 processed data from: {resolve_path(args.processed_dir)}")
    try:
        dataset, report = finalize_stage2_processed(
            processed_dir=args.processed_dir,
            output_path=args.output,
            report_path=args.report,
            seed=args.seed,
            max_consecutive_task=args.max_consecutive_task,
        )
    except Stage2FinalizationError as exc:
        report = exc.report
        log_error("Stage2 finalization failed; final files were not written.")
        for error in report.get("validation", {}).get("sample_errors", [])[:10]:
            log_error(str(error))
        return 1

    log_success(
        "Stage2 finalization passed: "
        f"{len(dataset)} samples, shortfall {report['shortfall_total']} versus target {report['target_total']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
