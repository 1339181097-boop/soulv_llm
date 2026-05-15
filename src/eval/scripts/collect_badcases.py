from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_error, log_info, log_success, resolve_path, write_json

DEFAULT_OUTPUT_PATH = "data/eval/stage1/stage1_badcases.jsonl"
DEFAULT_SUMMARY_PATH = "data/eval/stage1/stage1_badcases_summary.json"
FAIL_JUDGE_VERDICTS = {"hold", "fail"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval summary: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"summary.json must be an object: {path}")
    return payload


def _resolve_raw_output_files(run_dir: Path, summary: dict[str, Any], tasks: list[str] | None) -> list[tuple[str, Path]]:
    requested = set(tasks or [])
    resolved: list[tuple[str, Path]] = []
    task_entries = summary.get("tasks", {})

    if isinstance(task_entries, dict):
        for task_name, metadata in task_entries.items():
            if requested and task_name not in requested:
                continue
            if not isinstance(metadata, dict):
                continue
            output_file = metadata.get("output_file")
            if isinstance(output_file, str) and output_file.strip():
                resolved.append((task_name, Path(output_file)))

    if resolved:
        return resolved

    for path in sorted(run_dir.glob("raw_outputs_*.json")):
        task_name = path.stem.removeprefix("raw_outputs_")
        if requested and task_name not in requested:
            continue
        resolved.append((task_name, path))
    return resolved


def _load_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array: {path}")

    records: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("id")
        if isinstance(sample_id, str) and sample_id:
            records[sample_id] = item
    return records


def _is_rule_bad(rule_result: dict[str, Any] | None) -> bool:
    if not rule_result:
        return False
    return rule_result.get("rule_pass") is False


def _is_judge_bad(judge_result: dict[str, Any] | None) -> bool:
    if not judge_result:
        return False
    if judge_result.get("pass_or_fail") is False:
        return True
    verdict = judge_result.get("verdict")
    return isinstance(verdict, str) and verdict.lower() in FAIL_JUDGE_VERDICTS


def _is_inference_bad(raw_sample: dict[str, Any]) -> bool:
    return raw_sample.get("status") != "ok" or bool(raw_sample.get("error")) or not str(raw_sample.get("prediction") or "").strip()


def _score_snapshot(judge_result: dict[str, Any] | None) -> dict[str, Any]:
    if not judge_result:
        return {}
    fields = (
        "correctness",
        "instruction_following",
        "completeness",
        "clarity",
        "safety_and_honesty",
        "brand_style",
        "task_specific_score",
        "overall_score",
    )
    return {field: judge_result.get(field) for field in fields if field in judge_result}


def _build_error_summary(
    *,
    raw_sample: dict[str, Any],
    rule_result: dict[str, Any] | None,
    judge_result: dict[str, Any] | None,
    failure_sources: list[str],
) -> str:
    if "inference" in failure_sources:
        error_message = str(raw_sample.get("error") or raw_sample.get("status") or "inference failed")
        return f"推理失败或输出为空：{error_message}"

    if judge_result and _is_judge_bad(judge_result):
        reason = str(judge_result.get("judge_reason") or "").strip()
        if reason:
            return reason
        tags = judge_result.get("issue_tags")
        if isinstance(tags, list) and tags:
            return "Judge 判为 badcase，主要问题标签：" + "、".join(str(tag) for tag in tags)

    if rule_result and _is_rule_bad(rule_result):
        rule_ids = rule_result.get("triggered_rule_ids") or []
        if isinstance(rule_ids, list) and rule_ids:
            return "规则检查未通过，命中规则：" + "、".join(str(rule_id) for rule_id in rule_ids)

    return "该样本被判定为 badcase，需要人工复核。"


def _build_badcase_record(
    *,
    run_dir: Path,
    run_summary: dict[str, Any],
    task_name: str,
    raw_output_file: Path,
    raw_sample: dict[str, Any],
    rule_result: dict[str, Any] | None,
    judge_result: dict[str, Any] | None,
    collected_at_utc: str,
) -> dict[str, Any]:
    sample_id = str(raw_sample.get("id") or "")
    run_name = str(run_summary.get("run_name") or run_dir.name)
    task_type = str(raw_sample.get("task_type") or task_name)
    badcase_id = f"stage1::{run_name}::{task_name}::{sample_id}"

    failure_sources: list[str] = []
    if _is_inference_bad(raw_sample):
        failure_sources.append("inference")
    if _is_rule_bad(rule_result):
        failure_sources.append("rule")
    if _is_judge_bad(judge_result):
        failure_sources.append("judge")

    return {
        "badcase_id": badcase_id,
        "stage": "stage1",
        "split": "regression",
        "source": {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "model_name": run_summary.get("model_name"),
            "request_model": run_summary.get("request_model"),
            "raw_output_file": str(raw_output_file),
            "rule_result_file": str(run_dir / f"rule_results_{task_name}.json"),
            "judge_result_file": str(run_dir / f"judge_results_{task_name}.json"),
        },
        "sample_id": sample_id,
        "task_type": task_type,
        "scene": raw_sample.get("scene"),
        "difficulty": raw_sample.get("difficulty"),
        "tags": raw_sample.get("tags", []),
        "messages": raw_sample.get("messages", []),
        "reference_answer": raw_sample.get("reference_answer", ""),
        "must_include": raw_sample.get("must_include", []),
        "must_not_do": raw_sample.get("must_not_do", []),
        "notes": raw_sample.get("notes", ""),
        "seed_source": raw_sample.get("seed_source"),
        "seed_sample_id": raw_sample.get("seed_sample_id"),
        "model_prediction": raw_sample.get("prediction", ""),
        "inference_status": raw_sample.get("status", ""),
        "inference_error": raw_sample.get("error", ""),
        "gold_answer": "",
        "gold_answer_status": "pending",
        "failure_sources": failure_sources,
        "error_analysis": {
            "summary": _build_error_summary(
                raw_sample=raw_sample,
                rule_result=rule_result,
                judge_result=judge_result,
                failure_sources=failure_sources,
            ),
            "rule": {
                "available": rule_result is not None,
                "rule_pass": rule_result.get("rule_pass") if rule_result else None,
                "rule_score": rule_result.get("rule_score") if rule_result else None,
                "max_severity": rule_result.get("max_severity") if rule_result else None,
                "triggered_rule_ids": rule_result.get("triggered_rule_ids", []) if rule_result else [],
                "rule_hits": rule_result.get("rule_hits", []) if rule_result else [],
            },
            "judge": {
                "available": judge_result is not None,
                "judge_status": judge_result.get("judge_status") if judge_result else None,
                "judge_model": judge_result.get("judge_model") if judge_result else None,
                "pass_or_fail": judge_result.get("pass_or_fail") if judge_result else None,
                "verdict": judge_result.get("verdict") if judge_result else None,
                "scores": _score_snapshot(judge_result),
                "missed_must_include": judge_result.get("missed_must_include", []) if judge_result else [],
                "violated_must_not_do": judge_result.get("violated_must_not_do", []) if judge_result else [],
                "issue_tags": judge_result.get("issue_tags", []) if judge_result else [],
                "judge_reason": judge_result.get("judge_reason", "") if judge_result else "",
            },
        },
        "created_at_utc": collected_at_utc,
        "updated_at_utc": collected_at_utc,
    }


def collect_badcases(run_dir: Path, tasks: list[str] | None = None) -> list[dict[str, Any]]:
    run_summary = _load_run_summary(run_dir)
    task_files = _resolve_raw_output_files(run_dir, run_summary, tasks)
    if not task_files:
        raise FileNotFoundError(f"No raw_outputs_*.json found under {run_dir}")

    collected_at_utc = datetime.now(timezone.utc).isoformat()
    badcases: list[dict[str, Any]] = []
    for task_name, raw_output_file in task_files:
        raw_payload = _load_json(raw_output_file)
        if not isinstance(raw_payload, list):
            raise ValueError(f"Expected JSON array: {raw_output_file}")

        rule_by_id = _load_records_by_id(run_dir / f"rule_results_{task_name}.json")
        judge_by_id = _load_records_by_id(run_dir / f"judge_results_{task_name}.json")
        log_info(
            f"Collecting {task_name}: raw={len(raw_payload)}, rule={len(rule_by_id)}, judge={len(judge_by_id)}"
        )

        for raw_sample in raw_payload:
            if not isinstance(raw_sample, dict):
                continue
            sample_id = raw_sample.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                continue
            rule_result = rule_by_id.get(sample_id)
            judge_result = judge_by_id.get(sample_id)
            if not (_is_inference_bad(raw_sample) or _is_rule_bad(rule_result) or _is_judge_bad(judge_result)):
                continue
            badcases.append(
                _build_badcase_record(
                    run_dir=run_dir,
                    run_summary=run_summary,
                    task_name=task_name,
                    raw_output_file=raw_output_file,
                    raw_sample=raw_sample,
                    rule_result=rule_result,
                    judge_result=judge_result,
                    collected_at_utc=collected_at_utc,
                )
            )

    return sorted(badcases, key=lambda item: (str(item.get("task_type")), str(item.get("sample_id"))))


def _merge_records(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for record in existing:
        badcase_id = record.get("badcase_id")
        if not isinstance(badcase_id, str) or not badcase_id:
            continue
        if badcase_id not in merged_by_id:
            order.append(badcase_id)
        merged_by_id[badcase_id] = record

    for record in incoming:
        badcase_id = record.get("badcase_id")
        if not isinstance(badcase_id, str) or not badcase_id:
            continue

        previous = merged_by_id.get(badcase_id)
        if previous:
            gold_answer = previous.get("gold_answer")
            if isinstance(gold_answer, str) and gold_answer.strip():
                record = {
                    **record,
                    "gold_answer": gold_answer,
                    "gold_answer_status": previous.get("gold_answer_status", "completed"),
                    "gold_answer_meta": previous.get("gold_answer_meta"),
                    "created_at_utc": previous.get("created_at_utc", record.get("created_at_utc")),
                }
            merged_by_id[badcase_id] = record
        else:
            order.append(badcase_id)
            merged_by_id[badcase_id] = record

    return [merged_by_id[badcase_id] for badcase_id in order]


def _build_summary(records: list[dict[str, Any]], *, run_dir: Path, output_path: Path) -> dict[str, Any]:
    task_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    verdict_counter: Counter[str] = Counter()

    for record in records:
        task_counter[str(record.get("task_type") or "unknown")] += 1
        for source in record.get("failure_sources", []) or []:
            source_counter[str(source)] += 1
        judge = record.get("error_analysis", {}).get("judge", {})
        if isinstance(judge, dict):
            verdict = judge.get("verdict")
            if isinstance(verdict, str) and verdict:
                verdict_counter[verdict] += 1
            for tag in judge.get("issue_tags", []) or []:
                issue_counter[str(tag)] += 1

    return {
        "run_dir": str(run_dir),
        "output_path": str(output_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_badcase_count": len(records),
        "task_counts": dict(task_counter),
        "failure_source_counts": dict(source_counter),
        "judge_verdict_counts": dict(verdict_counter),
        "top_issue_tags": dict(issue_counter.most_common(30)),
    }


def run(args: argparse.Namespace) -> Path:
    configure_console_output()
    run_dir = resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)
    incoming = collect_badcases(run_dir, tasks=args.task or None)

    if args.no_merge_existing:
        records = incoming
    else:
        records = _merge_records(_read_jsonl(output_path), incoming)

    _write_jsonl(output_path, records)
    summary = _build_summary(records, run_dir=run_dir, output_path=output_path)
    write_json(summary_path, summary)
    log_success(f"Wrote badcase JSONL: {output_path}")
    log_success(f"Wrote badcase summary: {summary_path}")
    log_info(f"Collected {len(incoming)} badcases from this run; canonical file now has {len(records)} records.")
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect stage1 eval badcases into one canonical JSONL file.")
    parser.add_argument("--run-dir", required=True, help="Eval report directory containing raw/rule/judge outputs.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Canonical badcase JSONL path.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH, help="Summary JSON path.")
    parser.add_argument("--task", action="append", default=[], help="Optional task filter. May be passed multiple times.")
    parser.add_argument(
        "--no-merge-existing",
        action="store_true",
        help="Overwrite output with only the current run's badcases instead of merging with the existing JSONL.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        log_error(str(exc))
        raise


if __name__ == "__main__":
    main()
