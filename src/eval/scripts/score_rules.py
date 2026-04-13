from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_error, log_info, log_success, resolve_path, write_json

SEVERITY_ORDER = {"minor": 1, "major": 2, "blocker": 3}
SEVERITY_PENALTY = {"minor": 10, "major": 35, "blocker": 100}

TASK_MIN_LENGTH = {
    "guide_generation": 80,
    "travel_qa": 20,
    "hotel_recommendation": 40,
    "traffic_planning": 40,
    "persona_understanding": 40,
    "multi_turn_dialogue": 35,
}

STRUCTURED_OUTPUT_TOKENS = (
    "tool_calls",
    "function_call",
    "observation",
    '"arguments"',
    '"name"',
    "<tool_call",
    "</tool_call",
    "intentionname",
)

REASONING_TRACE_TOKENS = ("<think>", "</think>", "<reasoning>", "</reasoning>")

TIME_SENSITIVE_KEYWORDS = (
    "票价",
    "余票",
    "库存",
    "班次",
    "车次",
    "发车",
    "发班",
    "开放时间",
    "营业时间",
    "几点开",
    "几点关",
    "门票",
)

MULTI_TURN_UPDATE_HINTS = (
    "别整体重写",
    "别整篇",
    "别从头",
    "按新条件更新",
    "按新条件调整",
    "更新一下",
    "改一下",
    "少半天",
    "缩短",
    "加一个条件",
    "更放松",
    "不要重写",
)

EXACT_PRICE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:元|块|人民币|rmb)")
EXACT_TIME_RE = re.compile(r"(?:[01]?\d|2[0-3])[:：点时]\d{0,2}")
ROUTE_ENUM_RE = re.compile(r"\d{1,3}路")
JSON_LIKE_RE = re.compile(r"^\s*[\[{].*[\]}]\s*$", re.DOTALL)


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    severity: str
    message: str
    evidence: str
    matched_must_not_do: list[str]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())


def _meaningful_length(text: str) -> int:
    return len(_normalize_text(text))


def _sequence_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _last_message(messages: list[dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _previous_assistant_message(messages: list[dict[str, Any]]) -> str:
    assistants = [message.get("content", "") for message in messages if message.get("role") == "assistant"]
    if not assistants:
        return ""
    return assistants[-1]


def _contains_json_like_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if JSON_LIKE_RE.match(stripped):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, (dict, list))
    return False


def _contains_structured_trace(text: str) -> str:
    lowered = text.lower()
    if _contains_json_like_payload(text):
        return "json_payload"
    for token in STRUCTURED_OUTPUT_TOKENS:
        if token in lowered:
            return token
    return ""


def _contains_reasoning_trace(text: str) -> str:
    lowered = text.lower()
    for token in REASONING_TRACE_TOKENS:
        if token in lowered:
            return token
    return ""


def _find_consecutive_repeat(text: str) -> str:
    normalized = _normalize_text(text)
    if len(normalized) < 24:
        return ""
    for size in range(4, 17):
        limit = len(normalized) - size * 6 + 1
        for index in range(max(0, limit)):
            chunk = normalized[index : index + size]
            if len(set(chunk)) == 1:
                continue
            if normalized[index : index + size * 6] == chunk * 6:
                return chunk
    return ""


def _has_excessive_route_enumeration(text: str) -> str:
    matches = ROUTE_ENUM_RE.findall(text)
    if len(matches) >= 8:
        return ", ".join(matches[:8])
    return ""


def _triggered_must_not_do(sample: dict[str, Any], categories: set[str]) -> list[str]:
    raw_items = sample.get("must_not_do", [])
    if not isinstance(raw_items, list):
        return []

    category_tokens = {
        "structured_output": ("json", "tool", "轨迹", "结构化", "trace"),
        "reasoning_trace": ("think", "trace", "轨迹"),
        "too_short": ("过短", "too short"),
        "question_echo": ("只重复", "只复述", "repeat labels"),
        "time_sensitive_fact": ("票价", "库存", "开放", "班次", "车次", "booking", "价格"),
        "exact_price": ("价格", "票价", "price"),
        "repetition": ("模板", "重复", "重写"),
        "multi_turn_update": ("重写", "最新约束", "ignore", "更新"),
    }

    matched: list[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            continue
        lowered = item.lower()
        for category in categories:
            for token in category_tokens.get(category, ()):
                if token in lowered:
                    matched.append(item)
                    break
            else:
                continue
            break
    return matched


def _system_instruction_text(sample: dict[str, Any]) -> str:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    for message in messages:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            parts.append(message["content"])
    return "\n".join(parts)


def _matched_policy_text(sample: dict[str, Any], categories: set[str]) -> list[str]:
    matched = list(_triggered_must_not_do(sample, categories))
    system_text = _system_instruction_text(sample).lower()

    category_tokens = {
        "time_sensitive_fact": ("票价", "库存", "开放", "班次", "车次", "booking", "价格", "live prices", "live schedules"),
        "exact_price": ("价格", "票价", "price", "fare", "fares"),
    }

    for category in categories:
        for token in category_tokens.get(category, ()):
            if token in system_text:
                matched.append(f"[system] {token}")
                break

    deduped: list[str] = []
    seen: set[str] = set()
    for item in matched:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _policy_forbids(sample: dict[str, Any], categories: set[str]) -> bool:
    return bool(_matched_policy_text(sample, categories))


def _make_hit(sample: dict[str, Any], *, rule_id: str, severity: str, message: str, evidence: str, categories: set[str]) -> RuleHit:
    return RuleHit(
        rule_id=rule_id,
        severity=severity,
        message=message,
        evidence=evidence,
        matched_must_not_do=_matched_policy_text(sample, categories),
    )


def _check_too_short(sample: dict[str, Any], prediction: str) -> RuleHit | None:
    task_type = sample.get("task_type", "")
    min_length = TASK_MIN_LENGTH.get(task_type, 30)
    length = _meaningful_length(prediction)
    if length == 0:
        return None
    if length < max(8, min_length // 2):
        return _make_hit(
            sample,
            rule_id="too_short",
            severity="blocker",
            message=f"Response is far too short for {task_type or 'this task'}",
            evidence=f"meaningful_length={length}, expected>={min_length}",
            categories={"too_short"},
        )
    if length < min_length:
        return _make_hit(
            sample,
            rule_id="too_short",
            severity="major",
            message=f"Response is shorter than the minimum heuristic for {task_type or 'this task'}",
            evidence=f"meaningful_length={length}, expected>={min_length}",
            categories={"too_short"},
        )
    return None


def _check_question_echo(sample: dict[str, Any], prediction: str) -> RuleHit | None:
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return None

    user_text = _last_message(messages, "user")
    if not user_text:
        return None

    normalized_prediction = _normalize_text(prediction)
    normalized_user = _normalize_text(user_text)
    if not normalized_prediction or not normalized_user:
        return None

    ratio = _sequence_ratio(normalized_prediction[:600], normalized_user[:600])
    same_length = abs(len(normalized_prediction) - len(normalized_user)) <= max(12, len(normalized_user) // 4)
    if normalized_prediction == normalized_user or (ratio >= 0.94 and same_length):
        return _make_hit(
            sample,
            rule_id="question_echo",
            severity="blocker",
            message="Prediction mostly repeats the user question instead of answering it",
            evidence=f"similarity={ratio:.2f}",
            categories={"question_echo"},
        )
    return None


def _check_repetition(sample: dict[str, Any], prediction: str) -> RuleHit | None:
    repeated_chunk = _find_consecutive_repeat(prediction)
    if repeated_chunk:
        return _make_hit(
            sample,
            rule_id="excessive_repetition",
            severity="blocker",
            message="Prediction contains excessive repeated content",
            evidence=f"repeated_chunk={repeated_chunk[:20]}",
            categories={"repetition"},
        )

    route_evidence = _has_excessive_route_enumeration(prediction)
    if route_evidence:
        return _make_hit(
            sample,
            rule_id="enumeration_overflow",
            severity="blocker",
            message="Prediction contains an unrealistic transport enumeration instead of actionable advice",
            evidence=route_evidence,
            categories={"repetition", "time_sensitive_fact"},
        )
    return None


def _check_time_sensitive_facts(sample: dict[str, Any], prediction: str) -> list[RuleHit]:
    hits: list[RuleHit] = []
    latest_user = _last_message(sample.get("messages", []), "user")

    lowered_prediction = prediction.lower()
    if any(keyword in prediction for keyword in TIME_SENSITIVE_KEYWORDS):
        if EXACT_TIME_RE.search(prediction) or EXACT_PRICE_RE.search(prediction):
            hits.append(
                _make_hit(
                    sample,
                    rule_id="time_sensitive_fact",
                    severity="blocker",
                    message="Prediction states time-sensitive operational facts in a concrete way",
                    evidence="matched time-sensitive keywords with concrete number/time",
                    categories={"time_sensitive_fact"},
                )
            )

    if EXACT_PRICE_RE.search(prediction) and not EXACT_PRICE_RE.search(latest_user):
        hits.append(
            _make_hit(
                sample,
                rule_id="exact_price",
                severity="blocker",
                message="Prediction gives a concrete price that was not provided by the user",
                evidence=EXACT_PRICE_RE.search(prediction).group(0),
                categories={"exact_price", "time_sensitive_fact"},
            )
        )

    if "余票" in lowered_prediction or "库存" in lowered_prediction:
        hits.append(
            _make_hit(
                sample,
                rule_id="inventory_claim",
                severity="blocker",
                message="Prediction mentions inventory or seat availability as if it were certain",
                evidence="余票/库存",
                categories={"time_sensitive_fact"},
            )
        )

    return hits


def _check_time_sensitive_facts_strict_by_policy(sample: dict[str, Any], prediction: str) -> list[RuleHit]:
    hits: list[RuleHit] = []
    latest_user = _last_message(sample.get("messages", []), "user")

    lowered_prediction = prediction.lower()
    forbids_time_sensitive = _policy_forbids(sample, {"time_sensitive_fact"})
    forbids_exact_price = _policy_forbids(sample, {"exact_price", "time_sensitive_fact"})

    has_time_sensitive_keyword = any(keyword in prediction for keyword in TIME_SENSITIVE_KEYWORDS)
    has_concrete_operational_time = bool(EXACT_TIME_RE.search(prediction)) and any(
        keyword in prediction for keyword in ("开放", "开门", "关门", "班次", "车次", "发车", "末班", "首班", "时刻", "余票", "库存")
    )

    if forbids_time_sensitive and has_time_sensitive_keyword and (has_concrete_operational_time or EXACT_PRICE_RE.search(prediction)):
        hits.append(
            _make_hit(
                sample,
                rule_id="time_sensitive_fact",
                severity="blocker",
                message="Prediction states time-sensitive operational facts in a concrete way",
                evidence="matched constrained live-fact keywords with concrete number/time",
                categories={"time_sensitive_fact"},
            )
        )

    price_match = EXACT_PRICE_RE.search(prediction)
    if forbids_exact_price and price_match and not EXACT_PRICE_RE.search(latest_user):
        hits.append(
            _make_hit(
                sample,
                rule_id="exact_price",
                severity="blocker",
                message="Prediction gives a concrete price that was not provided by the user",
                evidence=price_match.group(0),
                categories={"exact_price", "time_sensitive_fact"},
            )
        )

    if forbids_time_sensitive and ("余票" in lowered_prediction or "库存" in lowered_prediction):
        hits.append(
            _make_hit(
                sample,
                rule_id="inventory_claim",
                severity="blocker",
                message="Prediction mentions inventory or seat availability as if it were certain",
                evidence="余票/库存",
                categories={"time_sensitive_fact"},
            )
        )

    return hits


def _check_multi_turn_update(sample: dict[str, Any], prediction: str) -> list[RuleHit]:
    if sample.get("task_type") != "multi_turn_dialogue":
        return []

    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return []

    last_user = _last_message(messages, "user")
    previous_assistant = _previous_assistant_message(messages)
    if not last_user or not previous_assistant:
        return []

    hits: list[RuleHit] = []
    normalized_prediction = _normalize_text(prediction)
    normalized_previous = _normalize_text(previous_assistant)
    similarity = _sequence_ratio(normalized_prediction[:1200], normalized_previous[:1200])
    asks_targeted_update = any(token in last_user for token in MULTI_TURN_UPDATE_HINTS)

    if similarity >= 0.92:
        hits.append(
            _make_hit(
                sample,
                rule_id="multi_turn_ignored_update",
                severity="blocker",
                message="Prediction is almost identical to the previous assistant turn",
                evidence=f"similarity_to_previous_assistant={similarity:.2f}",
                categories={"multi_turn_update"},
            )
        )
        return hits

    if asks_targeted_update and similarity >= 0.78 and len(prediction) >= int(len(previous_assistant) * 0.75):
        hits.append(
            _make_hit(
                sample,
                rule_id="multi_turn_full_rewrite_risk",
                severity="major",
                message="Prediction likely rewrites the whole plan instead of only updating the affected part",
                evidence=f"similarity_to_previous_assistant={similarity:.2f}",
                categories={"multi_turn_update"},
            )
        )

    return hits


def evaluate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    prediction = sample.get("prediction", "")
    if not isinstance(prediction, str):
        prediction = str(prediction)

    hits: list[RuleHit] = []
    status = sample.get("status", "")
    error_message = sample.get("error", "")

    if status != "ok" or error_message:
        hits.append(
            _make_hit(
                sample,
                rule_id="inference_error",
                severity="blocker",
                message="Sample inference did not complete successfully",
                evidence=str(error_message or status),
                categories=set(),
            )
        )

    if not prediction.strip():
        hits.append(
            _make_hit(
                sample,
                rule_id="empty_response",
                severity="blocker",
                message="Prediction is empty",
                evidence="prediction=''",
                categories={"too_short"},
            )
        )

    trace_token = _contains_reasoning_trace(prediction)
    if trace_token:
        hits.append(
            _make_hit(
                sample,
                rule_id="reasoning_trace",
                severity="blocker",
                message="Prediction leaked reasoning / think tags",
                evidence=trace_token,
                categories={"reasoning_trace", "structured_output"},
            )
        )

    structured_token = _contains_structured_trace(prediction)
    if structured_token:
        hits.append(
            _make_hit(
                sample,
                rule_id="structured_output",
                severity="blocker",
                message="Prediction contains structured output / trace content not allowed in this eval stage",
                evidence=structured_token,
                categories={"structured_output"},
            )
        )

    too_short = _check_too_short(sample, prediction)
    if too_short is not None:
        hits.append(too_short)

    question_echo = _check_question_echo(sample, prediction)
    if question_echo is not None:
        hits.append(question_echo)

    repetition = _check_repetition(sample, prediction)
    if repetition is not None:
        hits.append(repetition)

    hits.extend(_check_time_sensitive_facts_strict_by_policy(sample, prediction))
    hits.extend(_check_multi_turn_update(sample, prediction))

    max_severity = "none"
    for hit in hits:
        if SEVERITY_ORDER[hit.severity] > SEVERITY_ORDER.get(max_severity, 0):
            max_severity = hit.severity

    penalty = sum(SEVERITY_PENALTY[hit.severity] for hit in hits)
    rule_score = max(0, 100 - penalty)
    rule_pass = not any(hit.severity in {"blocker", "major"} for hit in hits)

    return {
        "id": sample.get("id", ""),
        "task_type": sample.get("task_type", ""),
        "status": status,
        "rule_pass": rule_pass,
        "rule_score": rule_score,
        "max_severity": max_severity,
        "hit_count": len(hits),
        "triggered_rule_ids": [hit.rule_id for hit in hits],
        "rule_hits": [asdict(hit) for hit in hits],
        "prediction_preview": prediction[:400],
        "prediction_length": _meaningful_length(prediction),
    }


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing eval summary: {summary_path}")
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Run summary must be a JSON object: {summary_path}")
    return payload


def _resolve_task_output_files(run_dir: Path, summary: dict[str, Any], tasks: list[str] | None) -> list[tuple[str, Path]]:
    task_filter = set(tasks or [])
    task_entries = summary.get("tasks", {})
    resolved: list[tuple[str, Path]] = []

    if isinstance(task_entries, dict):
        for task_name, metadata in task_entries.items():
            if task_filter and task_name not in task_filter:
                continue
            if not isinstance(metadata, dict):
                continue
            output_file = metadata.get("output_file")
            if not isinstance(output_file, str):
                continue
            resolved.append((task_name, Path(output_file)))

    if resolved:
        return resolved

    for path in sorted(run_dir.glob("raw_outputs_*.json")):
        task_name = path.stem.removeprefix("raw_outputs_")
        if task_filter and task_name not in task_filter:
            continue
        resolved.append((task_name, path))
    return resolved


def _summarize_task(results: list[dict[str, Any]]) -> dict[str, Any]:
    rule_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    fail_ids: list[str] = []
    for result in results:
        if not result["rule_pass"]:
            fail_ids.append(result["id"])
        rule_counter.update(result["triggered_rule_ids"])
        if result["max_severity"] != "none":
            severity_counter[result["max_severity"]] += 1

    sample_count = len(results)
    pass_count = sum(1 for result in results if result["rule_pass"])
    avg_rule_score = round(sum(result["rule_score"] for result in results) / sample_count, 2) if sample_count else 0.0
    return {
        "sample_count": sample_count,
        "pass_count": pass_count,
        "fail_count": sample_count - pass_count,
        "pass_rate": round(pass_count / sample_count, 4) if sample_count else 0.0,
        "avg_rule_score": avg_rule_score,
        "max_severity_counts": dict(severity_counter),
        "triggered_rule_counts": dict(rule_counter),
        "fail_sample_ids": fail_ids,
    }


def score_rules(args: argparse.Namespace) -> Path:
    configure_console_output()

    run_dir = resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    summary = _load_run_summary(run_dir)
    task_output_files = _resolve_task_output_files(run_dir, summary, args.task or None)
    if not task_output_files:
        raise FileNotFoundError(f"No raw eval outputs found under {run_dir}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rule_counter: Counter[str] = Counter()
    task_summaries: dict[str, Any] = {}
    total_samples = 0
    total_pass = 0

    for task_name, output_path in task_output_files:
        payload = _load_json(output_path)
        if not isinstance(payload, list):
            raise ValueError(f"Raw output file must be a JSON array: {output_path}")

        log_info(f"Scoring task {task_name} with {len(payload)} samples")
        task_results = [evaluate_sample(sample) for sample in payload if isinstance(sample, dict)]
        task_summary = _summarize_task(task_results)
        task_summaries[task_name] = {
            **task_summary,
            "raw_output_file": str(output_path),
        }
        overall_rule_counter.update(task_summary["triggered_rule_counts"])
        total_samples += task_summary["sample_count"]
        total_pass += task_summary["pass_count"]

        result_path = output_dir / f"rule_results_{task_name}.json"
        write_json(result_path, task_results)
        log_success(f"Wrote rule results: {result_path}")

    summary_payload = {
        "run_dir": str(run_dir),
        "run_name": summary.get("run_name", run_dir.name),
        "model_name": summary.get("model_name"),
        "request_model": summary.get("request_model"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_mode": True,
        "pass_definition": "Samples fail if any blocker or major rule is triggered",
        "total_sample_count": total_samples,
        "total_pass_count": total_pass,
        "total_fail_count": total_samples - total_pass,
        "overall_pass_rate": round(total_pass / total_samples, 4) if total_samples else 0.0,
        "task_summaries": task_summaries,
        "overall_triggered_rule_counts": dict(overall_rule_counter),
    }

    summary_path = output_dir / "rule_summary.json"
    write_json(summary_path, summary_payload)
    log_success(f"Wrote rule summary: {summary_path}")
    return summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict rule-based checks over eval raw outputs.")
    parser.add_argument("--run-dir", required=True, help="Eval run directory containing summary.json and raw_outputs_*.json.")
    parser.add_argument("--output-dir", default=None, help="Directory to write rule results. Defaults to --run-dir.")
    parser.add_argument("--task", action="append", default=[], help="Optional task filter. May be passed multiple times.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        summary_path = score_rules(args)
        log_success(f"Rule scoring completed: {summary_path}")
    except Exception as exc:  # noqa: BLE001
        log_error(str(exc))
        raise


if __name__ == "__main__":
    main()
