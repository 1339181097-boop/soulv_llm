from __future__ import annotations

import argparse
import bisect
import math
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    load_records,
    log_info,
    log_success,
    log_warn,
    resolve_path,
    validate_chatml_dataset,
    write_json,
)

DEFAULT_OUTPUT_PATH = "data/final/stage1_general_sft.json"
DEFAULT_REPORT_PATH = "data/final/stage_mix_report.json"
DEFAULT_TOKENIZER_PATH = "models/tokenizers/Qwen3-32B"
DEFAULT_SEED = 42
DEFAULT_MAX_CONSECUTIVE_TASK = 8
DEFAULT_CUTOFF_LEN = 2048

ALLOWED_STAGE1_ROLES = {"system", "user", "assistant"}
FORBIDDEN_TOP_LEVEL_KEYS = {
    "tool_calls",
    "function_call",
    "function_calls",
    "tools",
    "expected_behavior",
    "messages_with_answer",
}
FORBIDDEN_MESSAGE_KEYS = {"tool_calls", "function_call", "function_calls"}

GUIDE_TASK = "guide_generation"
TRAVEL_QA_TASK = "travel_qa"
HOTEL_TASK = "hotel_recommendation"
TRAFFIC_TASK = "traffic_planning"
PERSONA_TASK = "persona_understanding"
MULTI_TURN_TASK = "multi_turn_dialogue"
LONG_FORM_TASKS = {GUIDE_TASK, MULTI_TURN_TASK}
STRUCTURED_REASONING_TASKS = {HOTEL_TASK, TRAFFIC_TASK, PERSONA_TASK}


@dataclass(frozen=True)
class DatasetBucket:
    filename: str
    weight: float
    records: list[dict[str, Any]]
    task_type: str | None = None


@dataclass(frozen=True)
class StageInput:
    filename: str
    task_type: str
    target_count: int
    stratify_by_tokens: bool = False


@dataclass(frozen=True)
class StageRecipe:
    name: str
    output_path: str
    seed: int
    inputs: tuple[StageInput, ...]
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
    max_consecutive_task: int = DEFAULT_MAX_CONSECUTIVE_TASK
    cutoff_len: int = DEFAULT_CUTOFF_LEN
    require_tokenizer: bool = True


class TokenCounter(Protocol):
    def count_text_tokens(self, text: str) -> int:
        ...

    def count_chat_tokens(self, messages: list[dict[str, Any]]) -> int:
        ...


class TransformersTokenCounter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count_text_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(self.tokenizer.encode(text))

    def count_chat_tokens(self, messages: list[dict[str, Any]]) -> int:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                return len(rendered)
            except Exception:
                pass

        total = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str):
                total += self.count_text_tokens(role)
            if isinstance(content, str):
                total += self.count_text_tokens(content)
        return total


STAGE1_CURRENT_INPUTS = (
    StageInput("sft_guide_generation_2026_04_24_strict.jsonl", GUIDE_TASK, 650),
    StageInput("sft_travel_qa_2026_04_22_strict.jsonl", TRAVEL_QA_TASK, 3250),
    StageInput("sft_hotel_recommendation_0423_strict.jsonl", HOTEL_TASK, 1900),
    StageInput("sft_traffic_planning_strict_round2_final.jsonl", TRAFFIC_TASK, 1919),
    StageInput("sft_persona_understanding_strict_round2.jsonl", PERSONA_TASK, 1300),
    StageInput(
        "sft_multi_turn_dialogue_2026_4_22_strict_round2.jsonl",
        MULTI_TURN_TASK,
        900,
        stratify_by_tokens=True,
    ),
)

STAGE1_CURRENT_TARGET_COUNTS = {
    stage_input.filename: stage_input.target_count for stage_input in STAGE1_CURRENT_INPUTS
}
STAGE1_CURRENT_TASK_TARGET_COUNTS = {
    stage_input.task_type: stage_input.target_count for stage_input in STAGE1_CURRENT_INPUTS
}

DEFAULT_STAGE_RECIPES = (
    StageRecipe(
        name="stage1_general_sft",
        output_path=DEFAULT_OUTPUT_PATH,
        seed=DEFAULT_SEED,
        inputs=STAGE1_CURRENT_INPUTS,
    ),
)

# Weighted mixing is still available for ad-hoc debug runs. The formal stage1
# path above uses explicit bucket counts and hard quality gates instead.
DEFAULT_SPECS = {
    "sft_guide_generation_2026_04_24_strict.jsonl": 0.0655,
    "sft_travel_qa_2026_04_22_strict.jsonl": 0.3277,
    "sft_hotel_recommendation_0423_strict.jsonl": 0.1916,
    "sft_traffic_planning_strict_round2_final.jsonl": 0.1935,
    "sft_persona_understanding_strict_round2.jsonl": 0.1311,
    "sft_multi_turn_dialogue_2026_4_22_strict_round2.jsonl": 0.0907,
}
DEFAULT_TARGET_COUNTS = STAGE1_CURRENT_TARGET_COUNTS

TOKEN_GATE_THRESHOLDS = {
    "guide_assistant_share_max": 0.33,
    "multi_turn_assistant_share_max": 0.25,
    "long_form_assistant_share_max": 0.56,
    "long_form_total_share_max": 0.48,
    "structured_reasoning_assistant_share_min": 0.33,
    "travel_qa_assistant_share_min": 0.09,
}
TOKEN_GATE_EPSILON = 0.001


def _parse_specs(raw_specs: list[str]) -> dict[str, float]:
    specs: dict[str, float] = {}
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"Invalid spec: {raw_spec}. Expected filename=weight")
        filename, raw_weight = raw_spec.split("=", 1)
        weight = float(raw_weight)
        if weight <= 0:
            raise ValueError(f"Weight must be > 0: {raw_spec}")
        specs[filename.strip()] = weight
    return specs


def _parse_counts(raw_counts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_count in raw_counts:
        if "=" not in raw_count:
            raise ValueError(f"Invalid count: {raw_count}. Expected filename=count")
        filename, raw_value = raw_count.split("=", 1)
        count = int(raw_value)
        if count < 0:
            raise ValueError(f"Count must be >= 0: {raw_count}")
        counts[filename.strip()] = count
    return counts


def _role_total_length(sample: dict[str, Any], role: str) -> int:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def _sample_total_length(sample: dict[str, Any]) -> int:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def _percentile(lengths: list[int], fraction: float) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min": 0, "avg": 0.0, "p50": 0, "p90": 0, "max": 0}
    return {
        "min": min(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
        "p50": _percentile(lengths, 0.5),
        "p90": _percentile(lengths, 0.9),
        "max": max(lengths),
    }


def _load_dataset_file(path: str | Path) -> list[dict[str, Any]]:
    return load_records(path)


def _load_token_counter(tokenizer_path: str | Path) -> tuple[TokenCounter | None, str | None]:
    resolved = resolve_path(tokenizer_path)
    if not resolved.exists():
        return None, f"Tokenizer path does not exist: {resolved}"

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        return None, f"transformers is not installed: {exc}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(resolved), trust_remote_code=True)
    except Exception as exc:
        return None, f"Failed to load tokenizer from {resolved}: {exc}"

    return TransformersTokenCounter(tokenizer), None


def _resolve_token_counter(
    *,
    tokenizer: TokenCounter | None,
    tokenizer_path: str | Path | None,
) -> tuple[TokenCounter | None, str | None]:
    if tokenizer is not None:
        return tokenizer, None
    if tokenizer_path is None:
        return None, "Tokenizer path was not provided."
    return _load_token_counter(tokenizer_path)


def _task_type_for_records(records: list[dict[str, Any]], filename: str) -> str:
    for sample in records:
        task_type = sample.get("task_type")
        if isinstance(task_type, str) and task_type:
            return task_type
    return filename.removeprefix("sft_").removesuffix(".json").removesuffix(".jsonl")


def _load_bucket(filename: str, weight: float) -> DatasetBucket | None:
    path = resolve_path(f"data/processed/{filename}")
    if not path.exists():
        log_warn(f"Missing processed dataset, skipped: {path}")
        return None

    dataset = _load_dataset_file(path)
    errors = validate_chatml_dataset(dataset)
    if errors:
        log_warn(f"Invalid ChatML dataset, skipped: {path}")
        for error in errors[:5]:
            log_warn(error)
        return None

    return DatasetBucket(
        filename=filename,
        weight=weight,
        records=list(dataset),
        task_type=_task_type_for_records(dataset, filename),
    )


def _resolve_target_counts(buckets: list[DatasetBucket], total_samples: int) -> dict[str, int]:
    total_weight = sum(bucket.weight for bucket in buckets)
    if total_weight <= 0:
        raise ValueError("Total weight must be > 0")

    exact_counts = []
    allocated = 0
    for index, bucket in enumerate(buckets):
        exact = total_samples * (bucket.weight / total_weight)
        base = math.floor(exact)
        exact_counts.append((index, bucket.filename, base, exact - base))
        allocated += base

    remainder = total_samples - allocated
    exact_counts.sort(key=lambda item: (-item[3], item[0]))

    requested: dict[str, int] = {filename: base for _, filename, base, _ in exact_counts}
    for _, filename, _, _ in exact_counts[:remainder]:
        requested[filename] += 1
    return requested


def _sample_records(records: list[dict[str, Any]], target: int, rng: random.Random) -> tuple[list[dict[str, Any]], int]:
    if target <= 0:
        return [], 0
    if not records:
        return [], 0
    if target <= len(records):
        return rng.sample(records, target), 0

    sampled: list[dict[str, Any]] = []
    full_cycles, remainder = divmod(target, len(records))
    for _ in range(full_cycles):
        sampled.extend(rng.sample(records, len(records)))
    if remainder:
        sampled.extend(rng.sample(records, remainder))
    duplicates = target - len(records)
    return sampled, duplicates


def _sample_without_oversampling(
    records: list[dict[str, Any]],
    target: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    if target <= 0:
        return [], 0
    if target > len(records):
        return rng.sample(records, len(records)), target - len(records)
    return rng.sample(records, target), 0


def _allocate_by_fraction(total: int, sizes: list[int]) -> list[int]:
    size_sum = sum(sizes)
    if total <= 0 or size_sum <= 0:
        return [0 for _ in sizes]

    exact_counts: list[tuple[int, int, float]] = []
    allocated = 0
    for index, size in enumerate(sizes):
        exact = total * (size / size_sum)
        base = min(size, math.floor(exact))
        exact_counts.append((index, base, exact - base))
        allocated += base

    counts = [base for _, base, _ in exact_counts]
    remainder = total - allocated
    exact_counts.sort(key=lambda item: (-item[2], item[0]))
    for index, _, _ in exact_counts:
        if remainder <= 0:
            break
        if counts[index] >= sizes[index]:
            continue
        counts[index] += 1
        remainder -= 1
    return counts


def _stratified_sample_by_metric(
    records: list[dict[str, Any]],
    target: int,
    rng: random.Random,
    metric_fn: Any,
    *,
    strata_count: int = 5,
) -> tuple[list[dict[str, Any]], int]:
    if target <= 0:
        return [], 0
    if target > len(records):
        return rng.sample(records, len(records)), target - len(records)
    if target == len(records):
        return rng.sample(records, len(records)), 0

    ordered = sorted(records, key=metric_fn)
    strata: list[list[dict[str, Any]]] = []
    chunk_size = math.ceil(len(ordered) / strata_count)
    for start in range(0, len(ordered), chunk_size):
        strata.append(ordered[start : start + chunk_size])

    counts = _allocate_by_fraction(target, [len(stratum) for stratum in strata])
    sampled: list[dict[str, Any]] = []
    for stratum, count in zip(strata, counts):
        if count:
            sampled.extend(rng.sample(stratum, count))

    if len(sampled) < target:
        sampled_ids = {id(sample) for sample in sampled}
        remaining = [sample for sample in records if id(sample) not in sampled_ids]
        sampled.extend(rng.sample(remaining, target - len(sampled)))
    elif len(sampled) > target:
        sampled = rng.sample(sampled, target)

    rng.shuffle(sampled)
    return sampled, 0


def _assistant_token_count(sample: dict[str, Any], token_counter: TokenCounter) -> int:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += token_counter.count_text_tokens(content)
    return total


def _total_chat_token_count(sample: dict[str, Any], token_counter: TokenCounter) -> int:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return 0
    return token_counter.count_chat_tokens(messages)


def _select_stage_records(
    *,
    recipe: StageRecipe,
    records_by_filename: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    token_counter: TokenCounter | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, Any] = {}
    shortfalls: dict[str, int] = {}

    for stage_input in recipe.inputs:
        records = records_by_filename.get(stage_input.filename, [])
        if stage_input.stratify_by_tokens and token_counter is not None:
            sampled, shortfall = _stratified_sample_by_metric(
                records,
                stage_input.target_count,
                rng,
                lambda sample: _total_chat_token_count(sample, token_counter),
            )
            selection_mode = "token_length_stratified"
        else:
            sampled, shortfall = _sample_without_oversampling(records, stage_input.target_count, rng)
            selection_mode = "deterministic_random_no_oversample"

        selected_by_task[stage_input.task_type] = sampled
        if shortfall:
            shortfalls[stage_input.filename] = shortfall
        files[stage_input.filename] = {
            "task_type": stage_input.task_type,
            "available_count": len(records),
            "target_count": stage_input.target_count,
            "selected_count": len(sampled),
            "shortfall": shortfall,
            "oversample_count": 0,
            "selection_mode": selection_mode,
        }

    return selected_by_task, {
        "files": files,
        "shortfalls": shortfalls,
        "has_shortfall": bool(shortfalls),
    }


def _assistant_token_sum(records: list[dict[str, Any]], token_counter: TokenCounter) -> int:
    return sum(_assistant_token_count(sample, token_counter) for sample in records)


def _compute_multi_turn_assistant_budget(
    selected_by_task: dict[str, list[dict[str, Any]]],
    token_counter: TokenCounter,
) -> dict[str, Any]:
    fixed_tasks = [task for task in selected_by_task if task != MULTI_TURN_TASK]
    fixed_assistant_tokens = sum(
        _assistant_token_sum(selected_by_task[task], token_counter) for task in fixed_tasks
    )
    guide_tokens = _assistant_token_sum(selected_by_task.get(GUIDE_TASK, []), token_counter)
    structured_tokens = sum(
        _assistant_token_sum(selected_by_task.get(task, []), token_counter)
        for task in STRUCTURED_REASONING_TASKS
    )
    travel_qa_tokens = _assistant_token_sum(selected_by_task.get(TRAVEL_QA_TASK, []), token_counter)

    lower_bounds = {
        "guide_assistant_share_max": guide_tokens / TOKEN_GATE_THRESHOLDS["guide_assistant_share_max"]
        - fixed_assistant_tokens
        if TOKEN_GATE_THRESHOLDS["guide_assistant_share_max"] > 0
        else 0.0,
    }
    upper_bounds = {
        "multi_turn_assistant_share_max": (
            TOKEN_GATE_THRESHOLDS["multi_turn_assistant_share_max"] * fixed_assistant_tokens
        )
        / (1 - TOKEN_GATE_THRESHOLDS["multi_turn_assistant_share_max"]),
        "long_form_assistant_share_max": (
            TOKEN_GATE_THRESHOLDS["long_form_assistant_share_max"] * fixed_assistant_tokens
            - guide_tokens
        )
        / (1 - TOKEN_GATE_THRESHOLDS["long_form_assistant_share_max"]),
        "structured_reasoning_assistant_share_min": structured_tokens
        / TOKEN_GATE_THRESHOLDS["structured_reasoning_assistant_share_min"]
        - fixed_assistant_tokens
        if TOKEN_GATE_THRESHOLDS["structured_reasoning_assistant_share_min"] > 0
        else 0.0,
        "travel_qa_assistant_share_min": travel_qa_tokens
        / TOKEN_GATE_THRESHOLDS["travel_qa_assistant_share_min"]
        - fixed_assistant_tokens
        if TOKEN_GATE_THRESHOLDS["travel_qa_assistant_share_min"] > 0
        else 0.0,
    }

    lower = max(0.0, *lower_bounds.values())
    upper = min(upper_bounds.values())
    feasible = lower <= upper
    desired = (lower + upper) / 2 if not feasible else (lower + upper) / 2
    return {
        "fixed_assistant_tokens": fixed_assistant_tokens,
        "guide_assistant_tokens": guide_tokens,
        "structured_reasoning_assistant_tokens": structured_tokens,
        "travel_qa_assistant_tokens": travel_qa_tokens,
        "lower_bounds": {key: round(value, 2) for key, value in lower_bounds.items()},
        "upper_bounds": {key: round(value, 2) for key, value in upper_bounds.items()},
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "feasible": feasible,
        "desired": round(desired, 2),
    }


def _select_by_assistant_token_budget(
    records: list[dict[str, Any]],
    target: int,
    rng: random.Random,
    token_counter: TokenCounter,
    desired_token_sum: float,
) -> list[dict[str, Any]]:
    if target >= len(records):
        return rng.sample(records, len(records))

    keyed_records = [
        (_assistant_token_count(sample, token_counter), rng.random(), index, sample)
        for index, sample in enumerate(records)
    ]
    keyed_records.sort(key=lambda item: (item[0], item[1]))
    token_values = [item[0] for item in keyed_records]

    selected_positions = set(range(target))
    unselected_positions = set(range(target, len(keyed_records)))
    current_sum = sum(token_values[:target])

    def best_distance(value: float) -> float:
        return abs(value - desired_token_sum)

    improved = True
    iterations = 0
    while improved and iterations < target:
        iterations += 1
        improved = False
        current_distance = best_distance(current_sum)
        best_swap: tuple[int, int, int] | None = None
        selected_sorted = sorted(selected_positions, key=lambda pos: token_values[pos])
        unselected_sorted = sorted(unselected_positions, key=lambda pos: token_values[pos])
        unselected_tokens = [token_values[pos] for pos in unselected_sorted]

        for selected_pos in selected_sorted:
            needed_token = token_values[selected_pos] + (desired_token_sum - current_sum)
            candidate_offsets = []
            insertion_point = bisect.bisect_left(unselected_tokens, needed_token)
            for offset in (insertion_point - 1, insertion_point, insertion_point + 1):
                if 0 <= offset < len(unselected_sorted):
                    candidate_offsets.append(offset)

            for offset in candidate_offsets:
                unselected_pos = unselected_sorted[offset]
                new_sum = current_sum - token_values[selected_pos] + token_values[unselected_pos]
                new_distance = best_distance(new_sum)
                if new_distance < current_distance:
                    best_swap = (selected_pos, unselected_pos, new_sum)
                    current_distance = new_distance

        if best_swap is not None:
            selected_pos, unselected_pos, new_sum = best_swap
            selected_positions.remove(selected_pos)
            selected_positions.add(unselected_pos)
            unselected_positions.remove(unselected_pos)
            unselected_positions.add(selected_pos)
            current_sum = new_sum
            improved = True

    selected = [keyed_records[position][3] for position in selected_positions]
    rng.shuffle(selected)
    return selected


def _retune_multi_turn_selection(
    *,
    recipe: StageRecipe,
    selected_by_task: dict[str, list[dict[str, Any]]],
    records_by_filename: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    token_counter: TokenCounter,
    selection_report: dict[str, Any],
) -> None:
    multi_input = next(
        (stage_input for stage_input in recipe.inputs if stage_input.task_type == MULTI_TURN_TASK),
        None,
    )
    if multi_input is None or not multi_input.stratify_by_tokens:
        return

    candidates = records_by_filename.get(multi_input.filename, [])
    if len(candidates) < multi_input.target_count:
        return

    budget = _compute_multi_turn_assistant_budget(selected_by_task, token_counter)
    selected = _select_by_assistant_token_budget(
        candidates,
        multi_input.target_count,
        rng,
        token_counter,
        budget["desired"],
    )
    selected_by_task[MULTI_TURN_TASK] = selected
    actual = _assistant_token_sum(selected, token_counter)
    budget["actual"] = actual
    budget["actual_distance_from_desired"] = round(abs(actual - budget["desired"]), 2)
    selection_report["multi_turn_token_budget"] = budget
    selection_report["files"][multi_input.filename]["selection_mode"] = "assistant_token_budget"


def _max_consecutive_task(records: list[dict[str, Any]]) -> dict[str, Any]:
    max_task = None
    max_run = 0
    current_task = None
    current_run = 0
    for sample in records:
        task_type = sample.get("task_type")
        if task_type == current_task:
            current_run += 1
        else:
            current_task = task_type
            current_run = 1
        if current_run > max_run:
            max_run = current_run
            max_task = task_type
    return {"task_type": max_task, "count": max_run}


def _weighted_interleave_tasks(
    selected_by_task: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    *,
    max_consecutive_task: int,
) -> list[dict[str, Any]]:
    queues: dict[str, deque[dict[str, Any]]] = {}
    for task_type, records in selected_by_task.items():
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


def _constrained_global_shuffle(
    selected_by_task: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    *,
    max_consecutive_task: int,
    attempts: int = 200,
) -> tuple[list[dict[str, Any]], str]:
    flat = [sample for records in selected_by_task.values() for sample in records]
    for _ in range(attempts):
        candidate = list(flat)
        rng.shuffle(candidate)
        if _max_consecutive_task(candidate)["count"] <= max_consecutive_task:
            return candidate, "global_shuffle"
    return (
        _weighted_interleave_tasks(
            selected_by_task,
            rng,
            max_consecutive_task=max_consecutive_task,
        ),
        "weighted_interleave_fallback",
    )


def _strict_stage1_chatml_errors(dataset: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, sample in enumerate(dataset):
        forbidden_top = sorted(FORBIDDEN_TOP_LEVEL_KEYS.intersection(sample.keys()))
        if forbidden_top:
            errors.append(f"sample {index} has forbidden top-level keys: {forbidden_top}")

        messages = sample.get("messages")
        if not isinstance(messages, list):
            continue

        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in ALLOWED_STAGE1_ROLES:
                errors.append(f"sample {index} message {message_index} has invalid role: {role}")

            forbidden_message = sorted(FORBIDDEN_MESSAGE_KEYS.intersection(message.keys()))
            if forbidden_message:
                errors.append(
                    f"sample {index} message {message_index} has forbidden message keys: {forbidden_message}"
                )

            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"sample {index} message {message_index} has empty content")
    return errors


def _primary_sample_id(sample: dict[str, Any]) -> str | None:
    for field_name in ("id", "record_id"):
        value = sample.get(field_name)
        if isinstance(value, str) and value.strip():
            return f"{field_name}:{value.strip()}"
    return None


def _source_record_id(sample: dict[str, Any]) -> str | None:
    value = sample.get("record_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_quality_report(
    records: list[dict[str, Any]],
    *,
    expected_task_counts: dict[str, int],
    max_consecutive_task: int,
) -> dict[str, Any]:
    chatml_errors = validate_chatml_dataset(records)
    strict_errors = _strict_stage1_chatml_errors(records)
    task_counts = dict(Counter(str(sample.get("task_type")) for sample in records))
    task_count_mismatches = {
        task_type: {
            "expected": expected,
            "actual": task_counts.get(task_type, 0),
        }
        for task_type, expected in expected_task_counts.items()
        if task_counts.get(task_type, 0) != expected
    }

    duplicate_primary_ids: list[str] = []
    seen_primary_ids: set[str] = set()
    missing_primary_id_count = 0
    for sample in records:
        sample_id = _primary_sample_id(sample)
        if sample_id is None:
            missing_primary_id_count += 1
            continue
        if sample_id in seen_primary_ids:
            duplicate_primary_ids.append(sample_id)
        else:
            seen_primary_ids.add(sample_id)

    source_record_ids = [
        source_record_id for sample in records if (source_record_id := _source_record_id(sample)) is not None
    ]
    duplicate_source_record_id_count = len(source_record_ids) - len(set(source_record_ids))
    max_run = _max_consecutive_task(records)
    passed = (
        not chatml_errors
        and not strict_errors
        and not task_count_mismatches
        and not duplicate_primary_ids
        and missing_primary_id_count == 0
        and max_run["count"] <= max_consecutive_task
    )

    return {
        "passed": passed,
        "chatml_error_count": len(chatml_errors),
        "chatml_errors": chatml_errors[:20],
        "strict_error_count": len(strict_errors),
        "strict_errors": strict_errors[:20],
        "task_type_counts": task_counts,
        "expected_task_counts": expected_task_counts,
        "task_count_mismatches": task_count_mismatches,
        "duplicate_primary_id_count": len(duplicate_primary_ids),
        "duplicate_primary_ids": duplicate_primary_ids[:20],
        "missing_primary_id_count": missing_primary_id_count,
        "duplicate_source_record_id_count": duplicate_source_record_id_count,
        "duplicate_source_record_id_note": (
            "record_id is treated as source metadata when id is present; primary sample identity uses id first."
        ),
        "max_consecutive_task": max_run,
        "max_consecutive_task_allowed": max_consecutive_task,
    }


def _build_token_report(
    records: list[dict[str, Any]],
    token_counter: TokenCounter,
    *,
    cutoff_len: int,
) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    total_assistant_tokens = 0
    total_chat_tokens = 0

    for task_type in sorted({str(sample.get("task_type")) for sample in records}):
        task_records = [sample for sample in records if sample.get("task_type") == task_type]
        assistant_tokens = [_assistant_token_count(sample, token_counter) for sample in task_records]
        chat_tokens = [_total_chat_token_count(sample, token_counter) for sample in task_records]
        task_assistant_sum = sum(assistant_tokens)
        task_total_sum = sum(chat_tokens)
        total_assistant_tokens += task_assistant_sum
        total_chat_tokens += task_total_sum
        by_task[task_type] = {
            "sample_count": len(task_records),
            "assistant_tokens": task_assistant_sum,
            "total_chat_tokens": task_total_sum,
            "assistant_token_length": _length_summary(assistant_tokens),
            "total_chat_token_length": _length_summary(chat_tokens),
            "cutoff_len": cutoff_len,
            "cutoff_exceeded_count": sum(1 for value in chat_tokens if value > cutoff_len),
        }

    for task_report in by_task.values():
        task_report["assistant_token_share"] = (
            round(task_report["assistant_tokens"] / total_assistant_tokens, 6)
            if total_assistant_tokens
            else 0.0
        )
        task_report["total_chat_token_share"] = (
            round(task_report["total_chat_tokens"] / total_chat_tokens, 6)
            if total_chat_tokens
            else 0.0
        )

    return {
        "total_assistant_tokens": total_assistant_tokens,
        "total_chat_tokens": total_chat_tokens,
        "by_task": by_task,
    }


def _sum_share(token_report: dict[str, Any], tasks: set[str], share_key: str) -> float:
    by_task = token_report.get("by_task", {})
    return round(sum(by_task.get(task, {}).get(share_key, 0.0) for task in tasks), 6)


def _evaluate_token_gates(token_report: dict[str, Any]) -> dict[str, Any]:
    by_task = token_report.get("by_task", {})
    guide_assistant_share = by_task.get(GUIDE_TASK, {}).get("assistant_token_share", 0.0)
    multi_assistant_share = by_task.get(MULTI_TURN_TASK, {}).get("assistant_token_share", 0.0)
    long_form_assistant_share = _sum_share(token_report, LONG_FORM_TASKS, "assistant_token_share")
    long_form_total_share = _sum_share(token_report, LONG_FORM_TASKS, "total_chat_token_share")
    structured_assistant_share = _sum_share(
        token_report,
        STRUCTURED_REASONING_TASKS,
        "assistant_token_share",
    )
    travel_qa_assistant_share = by_task.get(TRAVEL_QA_TASK, {}).get("assistant_token_share", 0.0)

    checks = {
        "guide_assistant_share": {
            "actual": guide_assistant_share,
            "max": TOKEN_GATE_THRESHOLDS["guide_assistant_share_max"],
            "passed": guide_assistant_share
            <= TOKEN_GATE_THRESHOLDS["guide_assistant_share_max"] + TOKEN_GATE_EPSILON,
        },
        "multi_turn_assistant_share": {
            "actual": multi_assistant_share,
            "max": TOKEN_GATE_THRESHOLDS["multi_turn_assistant_share_max"],
            "passed": multi_assistant_share
            <= TOKEN_GATE_THRESHOLDS["multi_turn_assistant_share_max"] + TOKEN_GATE_EPSILON,
        },
        "long_form_assistant_share": {
            "actual": long_form_assistant_share,
            "max": TOKEN_GATE_THRESHOLDS["long_form_assistant_share_max"],
            "passed": long_form_assistant_share
            <= TOKEN_GATE_THRESHOLDS["long_form_assistant_share_max"] + TOKEN_GATE_EPSILON,
        },
        "long_form_total_share": {
            "actual": long_form_total_share,
            "max": TOKEN_GATE_THRESHOLDS["long_form_total_share_max"],
            "passed": long_form_total_share
            <= TOKEN_GATE_THRESHOLDS["long_form_total_share_max"] + TOKEN_GATE_EPSILON,
        },
        "structured_reasoning_assistant_share": {
            "actual": structured_assistant_share,
            "min": TOKEN_GATE_THRESHOLDS["structured_reasoning_assistant_share_min"],
            "passed": structured_assistant_share + TOKEN_GATE_EPSILON
            >= TOKEN_GATE_THRESHOLDS["structured_reasoning_assistant_share_min"],
        },
        "travel_qa_assistant_share": {
            "actual": travel_qa_assistant_share,
            "min": TOKEN_GATE_THRESHOLDS["travel_qa_assistant_share_min"],
            "passed": travel_qa_assistant_share + TOKEN_GATE_EPSILON
            >= TOKEN_GATE_THRESHOLDS["travel_qa_assistant_share_min"],
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "thresholds": TOKEN_GATE_THRESHOLDS,
        "epsilon": TOKEN_GATE_EPSILON,
        "checks": checks,
    }


def _build_bucket_audit(
    buckets: list[DatasetBucket],
    requested_counts: dict[str, int],
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    projected_total_length_all = 0.0
    projected_assistant_length_all = 0.0

    for bucket in buckets:
        user_lengths = [_role_total_length(sample, "user") for sample in bucket.records]
        assistant_lengths = [_role_total_length(sample, "assistant") for sample in bucket.records]
        total_lengths = [_sample_total_length(sample) for sample in bucket.records]
        length_report = {
            "user_length": _length_summary(user_lengths),
            "assistant_length": _length_summary(assistant_lengths),
            "total_length": _length_summary(total_lengths),
            "avg_total_length": round(sum(total_lengths) / len(total_lengths), 2) if total_lengths else 0.0,
            "avg_assistant_length": round(sum(assistant_lengths) / len(assistant_lengths), 2)
            if assistant_lengths
            else 0.0,
        }
        target_count = requested_counts.get(bucket.filename, 0)
        projected_total_length = round(length_report["avg_total_length"] * target_count, 2)
        projected_assistant_length = round(length_report["avg_assistant_length"] * target_count, 2)
        projected_total_length_all += projected_total_length
        projected_assistant_length_all += projected_assistant_length
        audits[bucket.filename] = {
            "task_type": bucket.task_type or _task_type_for_records(bucket.records, bucket.filename),
            "source_sample_count": len(bucket.records),
            "target_sample_count": target_count,
            "length_stats": length_report,
            "projected_total_length": projected_total_length,
            "projected_assistant_length": projected_assistant_length,
        }

    for audit in audits.values():
        audit["projected_total_length_share"] = (
            round(audit["projected_total_length"] / projected_total_length_all, 4)
            if projected_total_length_all > 0
            else 0.0
        )
        audit["projected_assistant_share"] = (
            round(audit["projected_assistant_length"] / projected_assistant_length_all, 4)
            if projected_assistant_length_all > 0
            else 0.0
        )

    return audits


def _load_requested_buckets(
    *,
    target_counts: dict[str, int] | None,
    specs: dict[str, float] | None,
    total_samples: int | None,
) -> tuple[list[DatasetBucket], dict[str, int], list[str]]:
    buckets: list[DatasetBucket] = []
    missing_requested: list[str] = []

    if target_counts:
        requested_counts = {filename: count for filename, count in target_counts.items() if count > 0}
        for filename in requested_counts:
            bucket = _load_bucket(filename, 1.0)
            if bucket is None or not bucket.records:
                missing_requested.append(filename)
                continue
            buckets.append(bucket)
        return buckets, requested_counts, missing_requested

    effective_specs = specs or DEFAULT_SPECS
    for filename, weight in effective_specs.items():
        bucket = _load_bucket(filename, weight)
        if bucket is not None and bucket.records:
            buckets.append(bucket)

    if total_samples is not None:
        requested_counts = _resolve_target_counts(buckets, total_samples)
    else:
        requested_counts = {bucket.filename: len(bucket.records) for bucket in buckets}
    return buckets, requested_counts, missing_requested


def build_mixed_dataset(
    output_json_path: str,
    *,
    seed: int,
    total_samples: int | None,
    specs: dict[str, float] | None = None,
    target_counts: dict[str, int] | None = None,
    strategy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_console_output()
    rng = random.Random(seed)
    buckets, requested_counts, missing_requested = _load_requested_buckets(
        target_counts=target_counts,
        specs=specs,
        total_samples=total_samples,
    )

    if not buckets:
        log_warn("No processed datasets available for mixing.")
        return [], {"sample_count": 0, "dataset_counts": {}, "missing_requested": missing_requested}

    mixed: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    dataset_sizes: dict[str, int] = {}
    oversample_counts: dict[str, int] = {}
    for bucket in buckets:
        dataset_sizes[bucket.filename] = len(bucket.records)
        target = requested_counts.get(bucket.filename, 0)
        sampled, duplicates = _sample_records(bucket.records, target, rng)
        mixed.extend(sampled)
        dataset_counts[bucket.filename] = len(sampled)
        oversample_counts[bucket.filename] = duplicates
        if duplicates > 0:
            log_warn(
                f"Oversampled {bucket.filename}: requested {target}, unique {len(bucket.records)}, duplicated {duplicates}"
            )

    rng.shuffle(mixed)
    output_path = write_json(output_json_path, mixed)
    task_type_counts = dict(Counter(str(sample.get("task_type")) for sample in mixed))
    report = {
        "output_path": str(output_path),
        "seed": seed,
        "target_total_samples": total_samples,
        "requested_counts": requested_counts,
        "sample_count": len(mixed),
        "dataset_counts": dataset_counts,
        "task_type_counts": task_type_counts,
        "dataset_sizes": dataset_sizes,
        "oversample_counts": oversample_counts,
        "missing_requested": missing_requested,
        "bucket_audits": _build_bucket_audit(buckets, requested_counts),
        "strategy": strategy,
    }

    log_success(f"Mixed dataset written: {output_path} ({len(mixed)} samples)")
    log_info(f"Dataset counts: {dataset_counts}")
    return mixed, report


def mix_datasets(
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    seed: int = DEFAULT_SEED,
    total_samples: int | None = None,
    specs: dict[str, float] | None = None,
    target_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    effective_target_counts = target_counts
    effective_specs = specs
    if effective_target_counts is None and effective_specs is None and total_samples is None:
        effective_target_counts = DEFAULT_TARGET_COUNTS

    mixed, _ = build_mixed_dataset(
        output_json_path,
        seed=seed,
        total_samples=total_samples,
        specs=effective_specs,
        target_counts=effective_target_counts,
    )
    return mixed


def _load_stage_records(recipe: StageRecipe) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    records_by_filename: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for stage_input in recipe.inputs:
        path = resolve_path("data/processed") / stage_input.filename
        if not path.exists():
            missing.append(stage_input.filename)
            records_by_filename[stage_input.filename] = []
            continue
        records_by_filename[stage_input.filename] = _load_dataset_file(path)
    return records_by_filename, missing


def build_stage_dataset_from_records(
    *,
    recipe: StageRecipe,
    records_by_filename: dict[str, list[dict[str, Any]]],
    output_json_path: str | Path,
    report_path: str | Path | None = None,
    tokenizer: TokenCounter | None = None,
    tokenizer_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_console_output()
    rng = random.Random(recipe.seed)
    effective_tokenizer_path = tokenizer_path if tokenizer_path is not None else recipe.tokenizer_path
    token_counter, tokenizer_error = _resolve_token_counter(
        tokenizer=tokenizer,
        tokenizer_path=effective_tokenizer_path,
    )

    selected_by_task, selection_report = _select_stage_records(
        recipe=recipe,
        records_by_filename=records_by_filename,
        rng=rng,
        token_counter=token_counter,
    )
    if token_counter is not None:
        _retune_multi_turn_selection(
            recipe=recipe,
            selected_by_task=selected_by_task,
            records_by_filename=records_by_filename,
            rng=rng,
            token_counter=token_counter,
            selection_report=selection_report,
        )
    mixed, order_mode = _constrained_global_shuffle(
        selected_by_task,
        rng,
        max_consecutive_task=recipe.max_consecutive_task,
    )

    expected_task_counts = {stage_input.task_type: stage_input.target_count for stage_input in recipe.inputs}
    quality_report = _build_quality_report(
        mixed,
        expected_task_counts=expected_task_counts,
        max_consecutive_task=recipe.max_consecutive_task,
    )

    token_report: dict[str, Any] | None = None
    token_gates = {
        "passed": False,
        "reason": "tokenizer_unavailable",
        "tokenizer_error": tokenizer_error,
    }
    if token_counter is not None:
        token_report = _build_token_report(mixed, token_counter, cutoff_len=recipe.cutoff_len)
        token_gates = _evaluate_token_gates(token_report)

    passed = (
        (token_counter is not None or not recipe.require_tokenizer)
        and not selection_report["has_shortfall"]
        and quality_report["passed"]
        and token_gates["passed"]
    )

    output_path = resolve_path(output_json_path)
    if passed:
        written_output_path = write_json(output_path, mixed)
        output_written = True
        log_success(f"Stage dataset written: {written_output_path} ({len(mixed)} samples)")
    else:
        output_written = False
        log_warn("Stage dataset failed gates; formal output was not written.")

    report = {
        "name": recipe.name,
        "status": "passed" if passed else "failed",
        "output_path": str(output_path),
        "output_written": output_written,
        "seed": recipe.seed,
        "sample_count": len(mixed),
        "order_mode": order_mode,
        "tokenizer_path": str(resolve_path(effective_tokenizer_path)) if effective_tokenizer_path else None,
        "tokenizer_error": tokenizer_error,
        "selection": selection_report,
        "quality": quality_report,
        "token_report": token_report,
        "token_gates": token_gates,
        "target_counts": {
            stage_input.filename: stage_input.target_count for stage_input in recipe.inputs
        },
        "task_target_counts": expected_task_counts,
    }

    if report_path is not None:
        write_json(report_path, report)
        log_info(f"Stage report written: {resolve_path(report_path)}")

    return mixed, report


def _build_stage_dataset(
    recipe: StageRecipe,
    *,
    report_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    records_by_filename, missing = _load_stage_records(recipe)
    _, report = build_stage_dataset_from_records(
        recipe=recipe,
        records_by_filename=records_by_filename,
        output_json_path=recipe.output_path,
        report_path=None,
        tokenizer_path=tokenizer_path,
    )
    if missing:
        report["status"] = "failed"
        report["missing_requested"] = missing
        report["output_written"] = False
    if report_path is not None:
        write_json(report_path, report)
    return report


def build_stage_datasets(
    recipes: tuple[StageRecipe, ...] = DEFAULT_STAGE_RECIPES,
    *,
    report_path: str = DEFAULT_REPORT_PATH,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    configure_console_output()
    stage_reports: dict[str, Any] = {}
    for recipe in recipes:
        stage_reports[recipe.name] = _build_stage_dataset(recipe, tokenizer_path=tokenizer_path)

    payload = {"stages": stage_reports}
    write_json(report_path, payload)
    log_success(f"Stage mix report written: {resolve_path(report_path)}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix processed ChatML datasets.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path for one-off mixing.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--total-samples", type=int, default=None, help="Target sample count for weighted mixing.")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Dataset spec in filename=weight format. May be passed multiple times.",
    )
    parser.add_argument(
        "--count",
        action="append",
        default=[],
        help="Dataset target in filename=count format. May be passed multiple times.",
    )
    parser.add_argument(
        "--stage",
        choices=("none", "current"),
        default="none",
        help="Build the current gated stage dataset instead of a one-off mixed dataset.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help="Stage report output path when --stage current is used.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer path used for formal stage token gates.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.stage == "current":
        build_stage_datasets(report_path=args.report, tokenizer_path=args.tokenizer_path)
        return

    if args.spec and args.count:
        raise ValueError("Use either --spec or --count in one run, not both.")

    specs = _parse_specs(args.spec) if args.spec else None
    target_counts = _parse_counts(args.count) if args.count else None
    mix_datasets(
        args.output,
        seed=args.seed,
        total_samples=args.total_samples,
        specs=specs,
        target_counts=target_counts,
    )


if __name__ == "__main__":
    main()
