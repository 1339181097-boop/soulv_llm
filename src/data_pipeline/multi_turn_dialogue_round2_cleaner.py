from __future__ import annotations

import argparse
import copy
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

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
    write_jsonl,
)
from src.data_pipeline.global_cleaner import clean_text

DEFAULT_INPUT_PATH = "data/processed/sft_multi_turn_dialogue_2026_4_22_strict.json"
DEFAULT_OUTPUT_PATH = "data/processed/sft_multi_turn_dialogue_2026_4_22_strict_round2.jsonl"
DEFAULT_JSON_OUTPUT_PATH = "data/processed/sft_multi_turn_dialogue_2026_4_22_strict_round2.json"
DEFAULT_REPORT_PATH = "data/reports/multi_turn_dialogue_round2_report.json"

STAGE1_MULTI_TURN_FINAL_TARGET = 900
STAGE1_MULTI_TURN_CANDIDATE_MIN = 1800
STAGE1_MULTI_TURN_CANDIDATE_MAX = 2500

REQUIRED_FIELDS = ("record_id", "task_type", "source", "source_id", "updated_at", "messages")
ALLOWED_ROLES = {"system", "user", "assistant"}

EXACT_GENERIC_USER_TURNS = {
    "请调整行程",
    "再调整一下",
    "最后调整一次",
    "调整一下",
    "请优化行程",
    "继续调整",
}
GENERIC_ACTION_SUFFIXES = (
    "请调整行程",
    "再调整一下",
    "最后调整一次",
    "请优化行程",
    "能重新规划一下吗",
    "能重新安排一下吗",
)
GENERIC_SLOT_PATTERNS = (
    "人文/自然/美食/购物等",
    "增加或变更同行人员",
    "增加同行人员",
    "增加或减少景点",
    "增加或减少天数",
    "变更交通方式",
    "变更住宿地点或标准",
    "预算调整（增加或减少）",
    "预算调整(增加或减少)",
    "增加或减少",
)
GENERIC_LABEL_REPLACEMENTS = (
    (re.compile(r"^（?更注重人文/自然/美食/购物等）?[，,：:]?"), ""),
    (re.compile(r"^人群调整（(?:增加或变更同行人员|增加同行人员)）[，,：:]?"), "人群调整："),
    (re.compile(r"^（(?:增加或变更同行人员|增加同行人员)）[，,：:]?"), "人群调整："),
    (re.compile(r"^行程调整（增加或减少景点）[，,：:]?"), "行程调整："),
    (re.compile(r"^时间调整（增加或减少天数）[，,：:]?"), "时间调整："),
    (re.compile(r"^交通调整（变更交通方式）[，,：:]?"), "交通调整："),
    (re.compile(r"^住宿调整（变更住宿地点或标准）[，,：:]?"), "住宿调整："),
    (re.compile(r"^预算调整[（(]增加或减少[）)][，,：:]?"), "预算调整："),
    (re.compile(r"^（变更住宿地点或标准）[，,：:]?"), "住宿调整："),
)
GENERIC_SLOT_PATTERNS_RE = tuple(re.compile(re.escape(slot)) for slot in GENERIC_SLOT_PATTERNS)
GENERIC_WRAPPER_PATTERNS = (
    re.compile(r"(?:补充约束|约束更新)[：:]", re.IGNORECASE),
    re.compile(r"(?:住宿需求调整为|同行人员有变化|时间安排调整为|交通方式调整为|偏好调整|请调整行程|预算也调整一下)[：:]", re.IGNORECASE),
    re.compile(r"(?:请调整行程|再调整一下|最后调整一次|调整一下|请优化行程|继续调整)", re.IGNORECASE),
    re.compile(r"(?:人群调整|行程调整|时间调整|交通调整|住宿调整|预算调整)[：:]?", re.IGNORECASE),
)
DETAIL_TRIM_CHARS = " \t\r\n，,。；;：:()（）[]【】—-"

PSEUDO_MULTI_TURN_PATTERN = re.compile(r"其余不变|同上续写|后续同上|同上|续写")
TOOL_OR_ROUTE_PATTERN = re.compile(
    r"tool_calls?|function_call|路由结果|意图识别|intent|```json|\{\"intent\"|\"tool\"",
    re.IGNORECASE,
)
REALTIME_OR_TRANSACTION_PATTERN = re.compile(
    r"实时|余票|库存|下单|支付|订单|预订成功|订票|购票|售票|票务|票价|门票|二维码|扫码"
)
SCHEDULE_OR_HOURS_PATTERN = re.compile(r"开放时间|营业时间|开放时段|闭馆|班次|车次|航班|首末班|末班车")
MOJIBAKE_PATTERN = re.compile(r"�|(?:ç|è|é|å|æ|ä|ï|¼|½|œ|‰|¤|»){4,}")

MAX_ASSISTANT_CHARS = 900


def _messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    messages = sample.get("messages")
    return messages if isinstance(messages, list) else []


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _core_roles(messages: list[dict[str, Any]]) -> list[str]:
    roles = [message.get("role") for message in messages]
    return roles[1:] if roles[:1] == ["system"] else roles


def _has_meaningful_detail(text: str) -> bool:
    stripped = text.strip(DETAIL_TRIM_CHARS)
    if not stripped:
        return False
    if stripped in EXACT_GENERIC_USER_TURNS:
        return False
    detail = _extract_meaningful_detail(stripped)
    if not detail:
        return False
    return not any(detail == suffix for suffix in GENERIC_ACTION_SUFFIXES)


def _extract_meaningful_detail(text: str) -> str:
    detail = text.strip()
    previous = None
    while detail != previous:
        previous = detail
        for pattern in GENERIC_WRAPPER_PATTERNS:
            detail = pattern.sub("", detail)
        for pattern in GENERIC_SLOT_PATTERNS_RE:
            detail = pattern.sub("", detail)
        detail = detail.strip(DETAIL_TRIM_CHARS)
    if detail.startswith("为") and len(detail) > 1:
        detail = detail[1:].strip(DETAIL_TRIM_CHARS)
    return detail


def _constraint_detail(constraint: str) -> str:
    text = constraint.strip()
    for prefix in ("补充约束：", "补充约束:", "约束更新：", "约束更新:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    for separator in ("——", "：", ":"):
        if separator in text:
            head, detail = text.split(separator, 1)
            if detail.strip():
                return detail.strip()
            return head.strip()
    return text


def _is_placeholder_constraint(constraint: str) -> bool:
    return not _has_meaningful_detail(_constraint_detail(constraint))


def _constraint_to_user_turn(constraint: str, generic_turn: str) -> str | None:
    if _is_placeholder_constraint(constraint):
        return None

    detail = _extract_meaningful_detail(_constraint_detail(constraint)).rstrip("。")
    if not _has_meaningful_detail(detail):
        return None

    if "预算" in constraint:
        return f"预算也调整一下：{detail}。"
    if "住宿" in constraint:
        return f"住宿需求调整为：{detail}。"
    if "交通" in constraint:
        return f"交通方式调整为：{detail}。"
    if "人群" in constraint or "同行" in constraint or "老人" in detail or "儿童" in detail or "孩子" in detail:
        return f"同行人员有变化：{detail}。"
    if "时间" in constraint or "天数" in constraint:
        return f"时间安排调整为：{detail}。"
    if "行程" in constraint or "景点" in constraint:
        return f"请调整行程：{detail}。"
    if "偏好" in constraint or "更注重" in constraint or "人文" in constraint:
        return f"偏好调整：{detail}。"
    if generic_turn in {"最后调整一次", "再调整一下"}:
        return f"再根据这个变化调整一下：{detail}。"
    return f"请根据这个新约束调整：{detail}。"


def _strip_generic_labels(user_turn: str) -> str:
    cleaned = user_turn.strip()
    for pattern, replacement in GENERIC_LABEL_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned).strip()

    cleaned = re.sub(r"^(?:人群调整|行程调整|时间调整|交通调整|住宿调整|预算调整)：(?:请调整行程|再调整一下|最后调整一次)[，,：:]?", "", cleaned).strip()
    cleaned = re.sub(r"^(?:请调整行程|再调整一下|最后调整一次)[，,：:]\s*", "", cleaned).strip()
    return cleaned


def _needs_constraint_rewrite(user_turn: str) -> bool:
    stripped = user_turn.strip()
    if stripped in EXACT_GENERIC_USER_TURNS:
        return True
    return any(slot in stripped for slot in GENERIC_SLOT_PATTERNS)


def _normalize_user_turn_candidate(user_turn: str) -> str:
    candidate = user_turn.strip()
    for pattern, replacement in GENERIC_LABEL_REPLACEMENTS:
        candidate = pattern.sub(replacement, candidate).strip()
    candidate = re.sub(r"^[（(](?:增加或变更同行人员|增加同行人员|增加或减少景点|增加或减少天数|变更交通方式|变更住宿地点或标准|更注重人文/自然/美食/购物等|预算调整（增加或减少）)[）)][，,：:—-]*", "", candidate).strip()
    candidate = re.sub(r"^(?:请调整行程|再调整一下|最后调整一次|调整一下|请优化行程|继续调整)[，,：:—-]+\s*", "", candidate).strip()
    candidate = re.sub(r"[，,。；; ]*(?:请调整行程|再调整一下|最后调整一次|调整一下|请优化行程|继续调整)\s*$", "", candidate).strip()

    detail = _extract_meaningful_detail(candidate)
    original = user_turn.strip()
    if "变更交通方式" in original and detail:
        return f"交通方式调整为：{detail}。"
    if "变更住宿地点或标准" in original and detail:
        return f"住宿需求调整为：{detail}。"
    if ("增加或变更同行人员" in original or "增加同行人员" in original) and detail:
        return f"同行人员有变化：{detail}。"
    if "增加或减少景点" in original and detail:
        return f"请调整行程：{detail}。"
    if "增加或减少天数" in original and detail:
        return f"时间安排调整为：{detail}。"
    if "人文/自然/美食/购物等" in original and detail:
        return f"偏好调整：{detail}。"
    return candidate.strip()


def _repair_user_turn(user_turn: str, constraint: str | None) -> tuple[str | None, bool]:
    stripped = user_turn.strip()
    if _needs_constraint_rewrite(stripped):
        candidate = _normalize_user_turn_candidate(stripped)
        if _has_meaningful_detail(candidate):
            if not candidate.endswith(("。", "？", "！")):
                candidate += "。"
            return candidate, candidate != stripped
        if constraint is None:
            return None, False
        return _constraint_to_user_turn(constraint, stripped), True

    if any(slot in stripped for slot in GENERIC_SLOT_PATTERNS):
        cleaned = _strip_generic_labels(stripped)
        if _has_meaningful_detail(cleaned):
            return cleaned, cleaned != stripped
        if constraint is None:
            return None, False
        return _constraint_to_user_turn(constraint, stripped), True

    return stripped, False


def _rebuild_constraint_changes(cleaned: dict[str, Any]) -> None:
    user_turns = [
        _message_content(message).rstrip("。")
        for message in _messages(cleaned)
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_turns:
        cleaned["constraint_changes"] = []
        return

    original_constraints = cleaned.get("constraint_changes")
    initial_constraint = ""
    if isinstance(original_constraints, list) and original_constraints:
        initial_constraint = clean_text(original_constraints[0], max_length=1200, mask_sensitive=True).rstrip("。")
    if not _has_meaningful_detail(initial_constraint):
        initial_constraint = f"初始需求：{user_turns[0]}"

    rebuilt = [initial_constraint]
    rebuilt.extend(turn for turn in user_turns[1:] if _has_meaningful_detail(turn))
    cleaned["constraint_changes"] = rebuilt


def _validate_multi_turn_structure(sample: dict[str, Any]) -> str | None:
    if sample.get("task_type") != "multi_turn_dialogue":
        return "wrong_task_type"

    for field in REQUIRED_FIELDS:
        if sample.get(field) in (None, ""):
            return "missing_required_field"

    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return "invalid_messages"

    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    if any(role not in ALLOWED_ROLES for role in roles):
        return "bad_roles"

    system_positions = [index for index, role in enumerate(roles) if role == "system"]
    if system_positions and system_positions != [0]:
        return "bad_system_position"

    core = roles[1:] if roles[:1] == ["system"] else roles
    expected = ["user" if index % 2 == 0 else "assistant" for index in range(len(core))]
    if core != expected:
        return "bad_alternation"

    pairs = sum(1 for index in range(0, len(core) - 1, 2) if core[index : index + 2] == ["user", "assistant"])
    if pairs < 3:
        return "too_few_turns"
    if not core or core[-1] != "assistant":
        return "not_end_assistant"

    for message in messages:
        if not isinstance(message, dict):
            return "invalid_message_item"
        if not isinstance(message.get("content"), str) or not message.get("content", "").strip():
            return "empty_content"

    return None


def _classify_content(sample: dict[str, Any]) -> str | None:
    constraints = sample.get("constraint_changes")
    if not isinstance(constraints, list) or not constraints:
        return "missing_constraint_changes"
    if any(not isinstance(item, str) or not item.strip() for item in constraints):
        return "invalid_constraint_changes"
    if any(_is_placeholder_constraint(item) for item in constraints[1:]):
        return "placeholder_constraint_change"

    for message in _messages(sample):
        content = _message_content(message)
        if MOJIBAKE_PATTERN.search(content):
            return "mojibake_content"
        if TOOL_OR_ROUTE_PATTERN.search(content):
            return "tool_or_route_content"
        if PSEUDO_MULTI_TURN_PATTERN.search(content):
            return "pseudo_multi_turn_phrase"
        if REALTIME_OR_TRANSACTION_PATTERN.search(content):
            return "realtime_ticket_or_transaction"
        if SCHEDULE_OR_HOURS_PATTERN.search(content):
            return "schedule_or_hours_content"
        if message.get("role") == "assistant" and len(content) >= MAX_ASSISTANT_CHARS:
            return "overlong_assistant_turn"

    user_turns = [_message_content(message) for message in _messages(sample) if message.get("role") == "user"]
    for user_turn in user_turns[1:]:
        if _needs_constraint_rewrite(user_turn):
            return "unrepaired_pseudo_user_turn"

    return None


def clean_round2_sample(sample: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    structural_reason = _validate_multi_turn_structure(sample)
    if structural_reason is not None:
        return None, structural_reason, []

    cleaned = copy.deepcopy(sample)
    edits: list[str] = []

    for message in cleaned["messages"]:
        normalized = clean_text(message.get("content"), max_length=4000, mask_sensitive=True)
        if normalized != message.get("content"):
            message["content"] = normalized
            edits.append("normalized_message_text")

    constraints = cleaned.get("constraint_changes")
    if isinstance(constraints, list):
        cleaned_constraints = [clean_text(item, max_length=1200, mask_sensitive=True) for item in constraints]
        if cleaned_constraints != constraints:
            cleaned["constraint_changes"] = cleaned_constraints
            edits.append("normalized_constraint_changes")
    else:
        cleaned_constraints = []

    user_message_indexes = [
        index for index, message in enumerate(cleaned["messages"]) if isinstance(message, dict) and message.get("role") == "user"
    ]
    for user_turn_position, message_index in enumerate(user_message_indexes):
        if user_turn_position == 0:
            continue

        message = cleaned["messages"][message_index]
        constraint = cleaned_constraints[user_turn_position] if user_turn_position < len(cleaned_constraints) else None
        repaired, changed = _repair_user_turn(message["content"], constraint)
        if repaired is None:
            return None, "unrepairable_pseudo_user_turn", edits
        if changed:
            message["content"] = repaired
            edits.append("repaired_user_constraint_turn")

    _rebuild_constraint_changes(cleaned)
    edits.append("rebuilt_constraint_changes")

    content_reason = _classify_content(cleaned)
    if content_reason is not None:
        return None, content_reason, edits

    return cleaned, None, edits


def classify_round2_filter_reason(sample: dict[str, Any]) -> str | None:
    _, reason, _ = clean_round2_sample(sample)
    return reason


def _balanced_cap(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(samples) <= limit:
        return samples

    buckets: dict[str, deque[tuple[int, dict[str, Any]]]] = defaultdict(deque)
    for index, sample in enumerate(samples):
        key = str(sample.get("city") or sample.get("destination") or sample.get("topic") or "unknown")
        buckets[key].append((index, sample))

    selected: list[tuple[int, dict[str, Any]]] = []
    keys = sorted(buckets, key=lambda key: (-len(buckets[key]), key))
    while len(selected) < limit and keys:
        next_keys: list[str] = []
        for key in keys:
            if len(selected) >= limit:
                break
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.popleft())
            if bucket:
                next_keys.append(key)
        keys = next_keys

    selected.sort(key=lambda item: item[0])
    return [sample for _, sample in selected]


def filter_round2_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]], Counter[str]]:
    filtered: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    edit_counts: Counter[str] = Counter()
    removed_examples: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        cleaned, reason, edits = clean_round2_sample(sample)
        edit_counts.update(edits)
        if cleaned is not None and reason is None:
            filtered.append(cleaned)
            continue

        reasons[reason or "unknown"] += 1
        if len(removed_examples) < 60:
            removed_examples.append(
                {
                    "index": index,
                    "record_id": sample.get("record_id"),
                    "reason": reason,
                    "constraint_changes": sample.get("constraint_changes"),
                    "user_turns": [
                        message.get("content")
                        for message in _messages(sample)
                        if isinstance(message, dict) and message.get("role") == "user"
                    ],
                }
            )

    if len(filtered) > STAGE1_MULTI_TURN_CANDIDATE_MAX:
        before_cap = len(filtered)
        filtered = _balanced_cap(filtered, STAGE1_MULTI_TURN_CANDIDATE_MAX)
        reasons["balanced_cap_to_candidate_max"] = before_cap - len(filtered)

    return filtered, reasons, removed_examples, edit_counts


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    role_patterns = Counter(tuple(message.get("role") for message in _messages(sample)) for sample in samples)
    pair_counts = Counter(
        sum(1 for index in range(0, len(_core_roles(_messages(sample))) - 1, 2) if _core_roles(_messages(sample))[index : index + 2] == ["user", "assistant"])
        for sample in samples
    )
    assistant_lengths = [
        len(_message_content(message))
        for sample in samples
        for message in _messages(sample)
        if message.get("role") == "assistant"
    ]

    return {
        "count": len(samples),
        "task_type_counts": dict(Counter(sample.get("task_type") for sample in samples)),
        "source_counts": dict(Counter(sample.get("source") for sample in samples)),
        "scene_counts": dict(Counter(sample.get("scene") for sample in samples)),
        "city_count": len({sample.get("city") for sample in samples if sample.get("city")}),
        "top_cities": dict(Counter(sample.get("city") for sample in samples).most_common(20)),
        "pair_counts": dict(sorted(pair_counts.items())),
        "role_patterns_top": {str(pattern): count for pattern, count in role_patterns.most_common(5)},
        "assistant_length_max": max(assistant_lengths, default=0),
        "assistant_length_p95": sorted(assistant_lengths)[int((len(assistant_lengths) - 1) * 0.95)] if assistant_lengths else 0,
    }


def run_round2_cleaning(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    json_output_path: str = DEFAULT_JSON_OUTPUT_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_console_output()
    resolved_input = resolve_path(input_path)
    log_info(f"开始 multi_turn_dialogue 二轮清洗: {resolved_input}")

    samples = load_records(input_path)

    chatml_errors = validate_chatml_dataset(samples)
    if chatml_errors:
        raise ValueError(f"输入数据 ChatML 校验失败，前 3 条错误: {chatml_errors[:3]}")

    filtered, reasons, removed_examples, edit_counts = filter_round2_samples(samples)
    output_file = write_jsonl(output_path, filtered)
    json_output_file = write_json(json_output_path, filtered)

    report = {
        "input_path": str(resolved_input),
        "output_path": str(output_file),
        "json_output_path": str(json_output_file),
        "final_target_count": STAGE1_MULTI_TURN_FINAL_TARGET,
        "candidate_count_range": [STAGE1_MULTI_TURN_CANDIDATE_MIN, STAGE1_MULTI_TURN_CANDIDATE_MAX],
        "meets_final_target": len(filtered) >= STAGE1_MULTI_TURN_FINAL_TARGET,
        "meets_candidate_range": STAGE1_MULTI_TURN_CANDIDATE_MIN <= len(filtered) <= STAGE1_MULTI_TURN_CANDIDATE_MAX,
        "input_summary": _summarize(samples),
        "output_summary": _summarize(filtered),
        "removed_count": len(samples) - len(filtered),
        "removed_reasons": dict(reasons),
        "edit_counts": dict(edit_counts),
        "removed_examples": removed_examples,
    }
    report_file = write_json(report_path, report)

    log_success(f"二轮清洗完成，保留 {len(filtered)} / {len(samples)} 条。")
    if len(filtered) < STAGE1_MULTI_TURN_CANDIDATE_MIN:
        log_warn(f"保留量低于 multi_turn_dialogue 原始候选建议下限 {STAGE1_MULTI_TURN_CANDIDATE_MIN}，需要补数。")
    if len(filtered) > STAGE1_MULTI_TURN_CANDIDATE_MAX:
        log_warn(f"保留量高于 multi_turn_dialogue 原始候选建议上限 {STAGE1_MULTI_TURN_CANDIDATE_MAX}。")
    log_info(f"二轮 JSONL 产物: {output_file}")
    log_info(f"兼容 JSON 产物: {json_output_file}")
    log_info(f"二轮报告: {report_file}")
    return filtered, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="multi_turn_dialogue 32B stage1 二轮严格清洗。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="上一轮 strict ChatML JSON。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="二轮 JSONL 输出。")
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT_PATH, help="兼容 JSON 数组输出。")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="二轮清洗报告输出。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_round2_cleaning(args.input, args.output, args.json_output, args.report)


if __name__ == "__main__":
    main()
