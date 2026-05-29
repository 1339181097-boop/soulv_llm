from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    iter_jsonl,
    log_error,
    log_info,
    log_success,
    log_warn,
    resolve_path,
    write_json,
    write_jsonl,
)
from src.data_pipeline.global_cleaner import normalize_text, truncate_text

DEFAULT_INPUT_PATH = "data/badcase_fix/qwen3_32b_stage1_text_badcase_repair_expanded_v2.dedup.jsonl"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
VERDICTS = {"keep", "review", "drop"}
SCORE_FIELDS = (
    "correctness",
    "instruction_following",
    "completeness",
    "actionability",
    "safety_and_honesty",
    "information_control",
    "language_quality",
    "training_value",
)


@dataclass(frozen=True)
class DashScopeConfig:
    request_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    retry_count: int
    seed: int | None
    reasoning_effort: str | None
    enable_thinking: bool | None
    response_format_json: bool


def _with_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}")
    return path.with_name(f"{path.name}{suffix}")


def _resolve_chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must be non-empty")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(record: dict[str, Any]) -> str:
    payload = copy.deepcopy(record)
    payload.pop("quality_judge", None)
    payload.pop("_dedup", None)
    return hashlib.sha256(_compact_json(payload).encode("utf-8")).hexdigest()


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                    text = "\n".join(parts).strip()
                    if text:
                        return text
            text = first_choice.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    raise ValueError(f"Unable to extract response text: {json.dumps(response_payload, ensure_ascii=False)[:1000]}")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    if "```json" in stripped:
        candidates.insert(0, stripped.split("```json", 1)[1].split("```", 1)[0].strip())
    if "```" in stripped:
        candidates.append(stripped.split("```", 1)[1].split("```", 1)[0].strip())
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left != -1 and right != -1 and left < right:
        candidates.append(stripped[left : right + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Judge response is not a JSON object: {text[:1000]}")


def _sanitize_score(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        numeric = int(round(float(value)))
    elif isinstance(value, str):
        numeric = int(round(float(value.strip())))
    else:
        numeric = 3
    return max(1, min(5, numeric))


def _sanitize_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.5
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return round(max(0.0, min(1.0, numeric)), 3)


def _sanitize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _derive_verdict(scores: dict[str, int], confidence: float, issue_tags: list[str]) -> str:
    overall = scores["overall_score"]
    critical = min(scores["correctness"], scores["instruction_following"], scores["safety_and_honesty"])
    if overall <= 2 or critical <= 2:
        return "drop"
    if overall < 4 or critical < 4 or confidence < 0.7 or issue_tags:
        return "review"
    return "keep"


def _normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, int] = {}
    for field in SCORE_FIELDS:
        scores[field] = _sanitize_score(payload.get(field, 3))
    scores["overall_score"] = _sanitize_score(payload.get("overall_score", round(sum(scores.values()) / len(scores))))

    confidence = _sanitize_confidence(payload.get("confidence", 0.5))
    issue_tags = _sanitize_string_list(payload.get("issue_tags"))
    strengths = _sanitize_string_list(payload.get("strengths"))
    failure_modes = _sanitize_string_list(payload.get("failure_modes"))

    verdict = payload.get("verdict")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower()
    else:
        verdict = ""
    if verdict not in VERDICTS:
        verdict = _derive_verdict(scores, confidence, issue_tags)

    judge_reason = payload.get("judge_reason", "")
    if not isinstance(judge_reason, str):
        judge_reason = str(judge_reason)

    return {
        "scores": scores,
        "confidence": confidence,
        "verdict": verdict,
        "issue_tags": issue_tags,
        "failure_modes": failure_modes,
        "strengths": strengths,
        "judge_reason": judge_reason.strip(),
    }


def _selected_record_payload(record: dict[str, Any], max_record_chars: int) -> dict[str, Any]:
    payload = {
        "source_badcase_id": record.get("source_badcase_id"),
        "repair_type": record.get("repair_type"),
        "task_type": record.get("task_type"),
        "messages": record.get("messages", []),
        "answer": record.get("answer", ""),
        "fix_tags": record.get("fix_tags", []),
        "notes": record.get("notes", ""),
        "must_include": record.get("must_include", []),
        "must_not_do": record.get("must_not_do", []),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) <= max_record_chars:
        return payload

    compact = copy.deepcopy(payload)
    compact["answer"] = truncate_text(normalize_text(compact.get("answer")), max_length=max(500, max_record_chars // 2))
    compact["notes"] = truncate_text(normalize_text(compact.get("notes")), max_length=500)
    return compact


def _build_judge_messages(record: dict[str, Any], max_record_chars: int) -> list[dict[str, str]]:
    system_prompt = (
        "You are a strict but fair data-quality judge for Chinese travel-assistant SFT repair samples. "
        "Judge whether the sample should be kept for training. Return exactly one JSON object and no extra text."
    )
    schema = {
        "correctness": "integer 1-5",
        "instruction_following": "integer 1-5",
        "completeness": "integer 1-5",
        "actionability": "integer 1-5",
        "safety_and_honesty": "integer 1-5",
        "information_control": "integer 1-5",
        "language_quality": "integer 1-5",
        "training_value": "integer 1-5",
        "overall_score": "integer 1-5",
        "confidence": "number 0.0-1.0",
        "verdict": "keep | review | drop",
        "issue_tags": ["short snake_case tags"],
        "failure_modes": ["short concrete problems"],
        "strengths": ["short concrete strengths"],
        "judge_reason": "brief Chinese reason",
    }
    user_payload = _selected_record_payload(record, max_record_chars)
    instruction = (
        "Evaluate the repaired badcase sample below.\n\n"
        "Keep only if the answer directly solves the user request, preserves user constraints, avoids fabricated live facts "
        "such as exact prices, inventory, opening hours, train numbers, or precise travel times, and is useful as a clean SFT target.\n"
        "Send to review when the quality is uncertain, the answer is too generic, too verbose, weakly grounded, low-value, "
        "or the judgment confidence is low. Drop when it is clearly wrong or unsafe.\n\n"
        "Use scores from 1 to 5. Confidence is your confidence in this quality judgment, from 0.0 to 1.0.\n"
        "Return JSON using this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Sample:\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]


class DashScopeJudgeClient:
    def __init__(self, config: DashScopeConfig) -> None:
        self.config = config

    def judge(self, record: dict[str, Any], *, max_record_chars: int) -> tuple[dict[str, Any], str]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": _build_judge_messages(record, max_record_chars),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if self.config.enable_thinking is not None:
            payload["enable_thinking"] = self.config.enable_thinking
        if self.config.response_format_json:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.config.retry_count + 1):
            req = request.Request(self.config.request_url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    raw_body = response.read().decode("utf-8", errors="replace")
                response_payload = json.loads(raw_body)
                response_text = _extract_response_text(response_payload)
                parsed = _extract_json_object(response_text)
                return _normalize_judge_payload(parsed), response_text
            except error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {error_body[:1000]}")
                if exc.code not in RETRYABLE_STATUS_CODES or attempt >= self.config.retry_count:
                    raise last_error
            except error.URLError as exc:
                last_error = RuntimeError(f"Request failed: {exc}")
                if attempt >= self.config.retry_count:
                    raise last_error
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = RuntimeError(str(exc))
                if attempt >= self.config.retry_count:
                    raise last_error

            sleep_seconds = min(2**attempt, 8)
            log_warn(f"Retrying DashScope judge request after {sleep_seconds}s")
            time.sleep(sleep_seconds)

        if last_error is None:
            raise RuntimeError("DashScope judge request failed")
        raise last_error


def _filter_status(normalized: dict[str, Any], *, min_score: int, min_confidence: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    scores = normalized["scores"]
    if scores["overall_score"] < min_score:
        reasons.append("low_overall_score")
    if normalized["confidence"] < min_confidence:
        reasons.append("low_confidence")
    if normalized["verdict"] != "keep":
        reasons.append(f"verdict_{normalized['verdict']}")
    if min(scores["correctness"], scores["instruction_following"], scores["safety_and_honesty"]) < min_score:
        reasons.append("low_critical_score")
    return ("keep" if not reasons else "review"), reasons


def _attach_judge_result(
    record: dict[str, Any],
    *,
    normalized: dict[str, Any],
    raw_response: str,
    model: str,
    provider: str,
    min_score: int,
    min_confidence: float,
) -> dict[str, Any]:
    status, reasons = _filter_status(normalized, min_score=min_score, min_confidence=min_confidence)
    scored = copy.deepcopy(record)
    scored["quality_judge"] = {
        "provider": provider,
        "model": model,
        "record_hash": _record_hash(record),
        "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        "scores": normalized["scores"],
        "confidence": normalized["confidence"],
        "verdict": normalized["verdict"],
        "issue_tags": normalized["issue_tags"],
        "failure_modes": normalized["failure_modes"],
        "strengths": normalized["strengths"],
        "judge_reason": normalized["judge_reason"],
        "filter_status": status,
        "filter_reasons": reasons,
        "raw_response": raw_response,
    }
    return scored


def _attach_error_result(record: dict[str, Any], *, error_message: str, model: str, provider: str) -> dict[str, Any]:
    scored = copy.deepcopy(record)
    scored["quality_judge"] = {
        "provider": provider,
        "model": model,
        "record_hash": _record_hash(record),
        "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        "scores": {},
        "confidence": 0.0,
        "verdict": "review",
        "issue_tags": ["judge_error"],
        "failure_modes": [error_message[:500]],
        "strengths": [],
        "judge_reason": "Judge request failed; send to manual review.",
        "filter_status": "review",
        "filter_reasons": ["judge_error"],
        "raw_response": "",
    }
    return scored


def _load_existing_scored(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for _, record in iter_jsonl(path):
        judge = record.get("quality_judge")
        if not isinstance(judge, dict):
            continue
        record_hash = judge.get("record_hash")
        if isinstance(record_hash, str) and record_hash:
            existing[record_hash] = record
    return existing


def _write_outputs(
    *,
    scored_path: Path,
    kept_path: Path,
    review_path: Path,
    summary_path: Path,
    scored_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    kept_records = [record for record in scored_records if record.get("quality_judge", {}).get("filter_status") == "keep"]
    review_records = [record for record in scored_records if record.get("quality_judge", {}).get("filter_status") != "keep"]
    reason_counter: Counter[str] = Counter()
    verdict_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    score_values: list[int] = []
    confidence_values: list[float] = []

    for record in scored_records:
        judge = record.get("quality_judge", {})
        if isinstance(judge, dict):
            verdict_counter[str(judge.get("verdict", "unknown"))] += 1
            for reason in judge.get("filter_reasons", []) or []:
                reason_counter[str(reason)] += 1
            scores = judge.get("scores", {})
            if isinstance(scores, dict) and isinstance(scores.get("overall_score"), int):
                score_values.append(scores["overall_score"])
            confidence = judge.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence_values.append(float(confidence))
        task_counter[normalize_text(record.get("task_type")) or "unknown"] += 1

    summary = {
        "input_path": str(resolve_path(args.input)),
        "scored_path": str(scored_path),
        "kept_path": str(kept_path),
        "review_path": str(review_path),
        "summary_path": str(summary_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "dashscope_compatible",
        "model": args.model,
        "base_url": args.base_url,
        "request_url": _resolve_chat_completions_url(args.base_url),
        "min_score": args.min_score,
        "min_confidence": args.min_confidence,
        "input_count": len(scored_records),
        "kept_count": len(kept_records),
        "review_count": len(review_records),
        "verdict_counts": dict(verdict_counter),
        "review_reason_counts": dict(reason_counter),
        "task_counts": dict(task_counter),
        "avg_overall_score": round(sum(score_values) / len(score_values), 3) if score_values else 0.0,
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else 0.0,
    }

    write_jsonl(scored_path, scored_records)
    write_jsonl(kept_path, kept_records)
    write_jsonl(review_path, review_records)
    write_json(summary_path, summary)
    return summary


def _resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key. Set {args.api_key_env} or pass --api-key.")
    return api_key


def run_scoring(args: argparse.Namespace) -> dict[str, Any]:
    configure_console_output()
    input_path = resolve_path(args.input)
    scored_path = resolve_path(args.scored) if args.scored else _with_suffix(input_path, ".scored.jsonl")
    kept_path = resolve_path(args.kept) if args.kept else _with_suffix(input_path, ".kept.jsonl")
    review_path = resolve_path(args.review) if args.review else _with_suffix(input_path, ".review.jsonl")
    summary_path = resolve_path(args.summary) if args.summary else _with_suffix(input_path, ".score_summary.json")

    all_records = [record for _, record in iter_jsonl(input_path)]
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    records = all_records[args.start_index :]
    if args.limit is not None:
        records = records[: args.limit]
    log_info(f"Loaded {len(records)} records from {input_path} starting at index {args.start_index}")

    existing = _load_existing_scored(scored_path) if args.resume else {}
    if existing:
        log_info(f"Loaded {len(existing)} existing scored records for resume")

    api_key = "" if args.dry_run else _resolve_api_key(args)
    enable_thinking: bool | None
    if args.enable_thinking:
        enable_thinking = True
    elif args.disable_thinking:
        enable_thinking = False
    else:
        enable_thinking = None

    config = DashScopeConfig(
        request_url=_resolve_chat_completions_url(args.base_url),
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        retry_count=args.retry_count,
        seed=args.seed,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=enable_thinking,
        response_format_json=args.response_format_json,
    )
    client = DashScopeJudgeClient(config)

    def score_one(index: int, record: dict[str, Any]) -> dict[str, Any]:
        record_hash = _record_hash(record)
        if record_hash in existing:
            existing_record = existing[record_hash]
            existing_judge = existing_record.get("quality_judge")
            existing_reasons = existing_judge.get("filter_reasons", []) if isinstance(existing_judge, dict) else []
            if not args.retry_judge_errors or "judge_error" not in existing_reasons:
                return existing_record

        if args.dry_run:
            normalized = {
                "scores": {field: 3 for field in (*SCORE_FIELDS, "overall_score")},
                "confidence": 0.0,
                "verdict": "review",
                "issue_tags": ["dry_run"],
                "failure_modes": [],
                "strengths": [],
                "judge_reason": "Dry run; no API request was sent.",
            }
            return _attach_judge_result(
                record,
                normalized=normalized,
                raw_response="",
                model=args.model,
                provider="dashscope_compatible",
                min_score=args.min_score,
                min_confidence=args.min_confidence,
            )

        try:
            normalized, raw_response = client.judge(record, max_record_chars=args.max_record_chars)
            return _attach_judge_result(
                record,
                normalized=normalized,
                raw_response=raw_response,
                model=args.model,
                provider="dashscope_compatible",
                min_score=args.min_score,
                min_confidence=args.min_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            log_error(f"Judge failed for record {index}: {exc}")
            return _attach_error_result(record, error_message=str(exc), model=args.model, provider="dashscope_compatible")

    def maybe_checkpoint(done_count: int, scored_so_far: list[dict[str, Any]]) -> None:
        if args.checkpoint_every <= 0 or done_count % args.checkpoint_every != 0:
            return
        _write_outputs(
            scored_path=scored_path,
            kept_path=kept_path,
            review_path=review_path,
            summary_path=summary_path,
            scored_records=scored_so_far,
            args=args,
        )
        log_info(f"Checkpointed {done_count}/{len(records)} scored records")

    scored_records: list[dict[str, Any] | None] = [None] * len(records)
    if args.workers <= 1:
        for index, record in enumerate(records, start=1):
            scored_records[index - 1] = score_one(index, record)
            maybe_checkpoint(index, [item for item in scored_records if item is not None])
            if args.sleep_seconds > 0 and index < len(records):
                time.sleep(args.sleep_seconds)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(score_one, index, record): index for index, record in enumerate(records, start=1)}
            for done_count, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                scored_records[index - 1] = future.result()
                maybe_checkpoint(done_count, [item for item in scored_records if item is not None])
                if args.sleep_seconds > 0 and done_count < len(records):
                    # Light throttle between result handling; requests are already in flight.
                    time.sleep(args.sleep_seconds)

    final_scored_records = [item for item in scored_records if item is not None]
    if len(final_scored_records) != len(records):
        raise RuntimeError(f"Scored {len(final_scored_records)} records, expected {len(records)}")

    output_records = final_scored_records
    if args.resume:
        scored_by_hash = {record.get("quality_judge", {}).get("record_hash"): record for record in existing.values()}
        for record in final_scored_records:
            judge = record.get("quality_judge")
            if isinstance(judge, dict):
                record_hash = judge.get("record_hash")
                if isinstance(record_hash, str) and record_hash:
                    scored_by_hash[record_hash] = record
        output_records = []
        for record in all_records:
            record_hash = _record_hash(record)
            scored = scored_by_hash.get(record_hash)
            if scored is not None:
                output_records.append(scored)

    summary = _write_outputs(
        scored_path=scored_path,
        kept_path=kept_path,
        review_path=review_path,
        summary_path=summary_path,
        scored_records=output_records,
        args=args,
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score badcase repair JSONL records with DashScope DeepSeek and split keep/review outputs.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input JSONL path.")
    parser.add_argument("--scored", default=None, help="All scored records output. Defaults to <input>.scored.jsonl.")
    parser.add_argument("--kept", default=None, help="High-quality records output. Defaults to <input>.kept.jsonl.")
    parser.add_argument("--review", default=None, help="Low-score/low-confidence records output. Defaults to <input>.review.jsonl.")
    parser.add_argument("--summary", default=None, help="Summary JSON output. Defaults to <input>.score_summary.json.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="DashScope compatible-mode base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DashScope model name.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable containing the DashScope API key.")
    parser.add_argument("--api-key", default="", help="DashScope API key. Prefer --api-key-env to avoid shell history leaks.")
    parser.add_argument("--min-score", type=int, default=4, help="Minimum overall and critical scores for automatic keep.")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum judge confidence for automatic keep.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "max"], default="high")
    parser.add_argument("--enable-thinking", action="store_true", help="Send enable_thinking=true.")
    parser.add_argument("--disable-thinking", action="store_true", help="Send enable_thinking=false.")
    parser.add_argument("--response-format-json", action="store_true", help="Send OpenAI response_format=json_object. Do not use with models that do not support it.")
    parser.add_argument("--max-record-chars", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based input offset to score. Useful for batch runs.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", help="Reuse existing records from --scored by record_hash.")
    parser.add_argument("--retry-judge-errors", action="store_true", help="When used with --resume, rescore existing records whose filter reasons include judge_error.")
    parser.add_argument("--dry-run", action="store_true", help="Validate IO and write review placeholders without API requests.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent judge requests.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Write partial scored/kept/review outputs every N completed records. Use 0 to disable.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.enable_thinking and args.disable_thinking:
        parser.error("--enable-thinking and --disable-thinking cannot be used together")
    try:
        summary = run_scoring(args)
    except Exception as exc:  # noqa: BLE001
        log_error(str(exc))
        return 1

    log_success(f"Scored {summary['input_count']} records")
    log_success(f"Kept {summary['kept_count']} records: {summary['kept_path']}")
    log_success(f"Review {summary['review_count']} records: {summary['review_path']}")
    log_success(f"Summary: {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
