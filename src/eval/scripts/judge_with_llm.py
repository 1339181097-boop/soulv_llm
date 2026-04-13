from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, log_error, log_info, log_success, log_warn, resolve_path, write_json

DASHSCOPE_API_KEY = {
    "api_key": "sk-b3c3bf1a6e594ce9869ef3e9a21efee4",
    "model": "qwen-plus",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
SCORE_FIELDS = (
    "correctness",
    "instruction_following",
    "completeness",
    "clarity",
    "safety_and_honesty",
    "brand_style",
    "task_specific_score",
)
VERDICTS = {"pass", "pass_with_risk", "hold", "fail"}
DEFAULT_RUBRICS_DIR = "src/eval/rubrics"
COMMON_RUBRIC_FILE = "common_rubric.md"


@dataclass(frozen=True)
class JudgeConfig:
    request_url: str
    api_key: str
    model: str
    timeout_seconds: int
    retry_count: int
    temperature: float
    max_tokens: int
    seed: int
    enable_thinking: bool


def _default_reports_dir() -> Path:
    return resolve_path("src/eval/reports")


def _resolve_chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must be a non-empty string")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def _coerce_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            nested_content = item.get("content")
            if isinstance(nested_content, str):
                parts.append(nested_content)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = _coerce_response_text(message.get("content"))
                if content:
                    return content
            text = _coerce_response_text(first_choice.get("text"))
            if text:
                return text
    output_text = _coerce_response_text(response_payload.get("output_text"))
    if output_text:
        return output_text
    raise ValueError(f"Unable to extract judge text from response: {json.dumps(response_payload, ensure_ascii=False)[:1000]}")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Judge returned an empty response")

    candidates = [stripped]
    if "```json" in stripped:
        fragment = stripped.split("```json", 1)[1].split("```", 1)[0].strip()
        candidates.insert(0, fragment)
    if "```" in stripped:
        fragment = stripped.split("```", 1)[1].split("```", 1)[0].strip()
        candidates.append(fragment)
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
    raise ValueError(f"Judge response is not valid JSON object: {text[:1000]}")


def _load_rubric_text(rubrics_dir: Path, filename: str) -> str:
    path = rubrics_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing rubric file: {path}")
    return path.read_text(encoding="utf-8").strip()


def _task_rubric_filename(task_type: str) -> str:
    return f"{task_type}.md"


def _sanitize_score(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        numeric = int(round(float(value)))
    elif isinstance(value, str):
        numeric = int(round(float(value.strip())))
    else:
        raise ValueError(f"Invalid score value: {value!r}")
    return max(1, min(5, numeric))


def _sanitize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return items


def _derive_verdict(scores: dict[str, int], violated_must_not_do: list[str], missed_must_include: list[str]) -> str:
    overall = scores["overall_score"]
    critical = min(scores["correctness"], scores["instruction_following"], scores["safety_and_honesty"])
    issue_count = len(violated_must_not_do) + len(missed_must_include)

    if violated_must_not_do or critical <= 2 or overall <= 2:
        return "fail"
    if overall >= 4 and critical >= 4 and issue_count <= 1:
        return "pass"
    if overall >= 4 and critical >= 3:
        return "pass_with_risk"
    if overall >= 3:
        return "hold"
    return "fail"


def _normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, int] = {}
    for field in SCORE_FIELDS:
        scores[field] = _sanitize_score(payload.get(field, 3))

    overall_value = payload.get("overall_score")
    if overall_value is None:
        overall_score = round(sum(scores.values()) / len(scores))
    else:
        overall_score = _sanitize_score(overall_value)
    scores["overall_score"] = overall_score

    covered_must_include = _sanitize_string_list(payload.get("covered_must_include"))
    missed_must_include = _sanitize_string_list(payload.get("missed_must_include"))
    violated_must_not_do = _sanitize_string_list(payload.get("violated_must_not_do"))
    issue_tags = _sanitize_string_list(payload.get("issue_tags"))
    strengths = _sanitize_string_list(payload.get("strengths"))

    verdict = payload.get("verdict")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower()
    else:
        verdict = ""
    if verdict not in VERDICTS:
        verdict = _derive_verdict(scores, violated_must_not_do, missed_must_include)

    judge_reason = payload.get("judge_reason", "")
    if not isinstance(judge_reason, str):
        judge_reason = str(judge_reason)
    judge_reason = judge_reason.strip()

    return {
        **scores,
        "verdict": verdict,
        "pass_or_fail": verdict in {"pass", "pass_with_risk"},
        "covered_must_include": covered_must_include,
        "missed_must_include": missed_must_include,
        "violated_must_not_do": violated_must_not_do,
        "issue_tags": issue_tags,
        "strengths": strengths,
        "judge_reason": judge_reason,
    }


def _fallback_result(sample: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = {
        "correctness": 1,
        "instruction_following": 1,
        "completeness": 1,
        "clarity": 1,
        "safety_and_honesty": 1,
        "brand_style": 1,
        "task_specific_score": 1,
        "overall_score": 1,
        "verdict": "fail",
        "pass_or_fail": False,
        "covered_must_include": [],
        "missed_must_include": _sanitize_string_list(sample.get("must_include")),
        "violated_must_not_do": [],
        "issue_tags": ["inference_error"],
        "strengths": [],
        "judge_reason": reason,
    }
    return payload


def _build_judge_messages(sample: dict[str, Any], common_rubric: str, task_rubric: str) -> list[dict[str, str]]:
    system_prompt = (
        "你是 TripAI stage1 评测中的严格中文评委。"
        "你要根据题目、参考答案、must_include、must_not_do 和 rubric，对模型回答做严格但公平的评分。"
        "禁止输出除 JSON 之外的任何文字。"
        "所有分数必须使用 1 到 5 的整数。"
    )

    user_payload = {
        "task_type": sample.get("task_type", ""),
        "messages": sample.get("messages", []),
        "reference_answer": sample.get("reference_answer", ""),
        "must_include": sample.get("must_include", []),
        "must_not_do": sample.get("must_not_do", []),
        "prediction": sample.get("prediction", ""),
    }

    judge_instruction = (
        "请阅读以下 rubric 与样本，严格评分，并只输出一个 JSON 对象。\n"
        "评分维度必须包含：correctness, instruction_following, completeness, clarity, safety_and_honesty, brand_style, task_specific_score, overall_score。\n"
        "还必须输出：verdict, covered_must_include, missed_must_include, violated_must_not_do, issue_tags, strengths, judge_reason。\n"
        "verdict 只能是 pass / pass_with_risk / hold / fail。\n"
        "如果 must_not_do 被明确违反，或答非所问，或关键约束没有遵守，应倾向于 hold 或 fail。\n"
        "如果回答整体可用但有明显瑕疵，可用 pass_with_risk。\n"
        "请按照 json 格式输出。\n\n"
        f"【Common Rubric】\n{common_rubric}\n\n"
        f"【Task Rubric】\n{task_rubric}\n\n"
        f"【Eval Sample】\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": judge_instruction},
    ]


class DashScopeJudgeClient:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def judge(self, sample: dict[str, Any], common_rubric: str, task_rubric: str) -> tuple[dict[str, Any], str]:
        messages = _build_judge_messages(sample, common_rubric, task_rubric)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "seed": self.config.seed,
            "enable_thinking": self.config.enable_thinking,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(self.config.retry_count + 1):
            req = request.Request(self.config.request_url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    raw_payload = response.read().decode("utf-8", errors="replace")
                response_payload = json.loads(raw_payload)
                response_text = _extract_response_text(response_payload)
                parsed = _extract_json_object(response_text)
                return parsed, response_text
            except error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code} from {self.config.request_url}: {error_body[:1000]}")
                if exc.code not in RETRYABLE_STATUS_CODES or attempt >= self.config.retry_count:
                    raise last_error
            except error.URLError as exc:
                last_error = RuntimeError(f"Request to {self.config.request_url} failed: {exc}")
                if attempt >= self.config.retry_count:
                    raise last_error
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Judge response from {self.config.request_url} is not valid JSON: {exc}") from exc
            except ValueError as exc:
                last_error = RuntimeError(str(exc))
                if attempt >= self.config.retry_count:
                    raise last_error

            sleep_seconds = min(2 ** attempt, 8)
            log_warn(f"Retrying judge request after {sleep_seconds}s due to transient error")
            time.sleep(sleep_seconds)

        if last_error is None:
            raise RuntimeError("Judge request failed without a captured exception")
        raise last_error


def _score_sample(sample: dict[str, Any], client: DashScopeJudgeClient, common_rubric: str, task_rubric: str) -> dict[str, Any]:
    sample_id = sample.get("id", "")
    task_type = sample.get("task_type", "")
    prediction = sample.get("prediction", "")
    status = sample.get("status", "")
    error_message = sample.get("error", "")

    if status != "ok" or error_message or not isinstance(prediction, str) or not prediction.strip():
        fallback = _fallback_result(sample, "Prediction unavailable because eval inference failed or returned empty output.")
        return {
            "id": sample_id,
            "task_type": task_type,
            "judge_status": "skipped_inference_error",
            "judge_model": client.config.model,
            "prediction_length": len(prediction) if isinstance(prediction, str) else 0,
            "raw_judge_response": "",
            **fallback,
        }

    parsed, raw_response = client.judge(sample, common_rubric, task_rubric)
    normalized = _normalize_judge_payload(parsed)
    return {
        "id": sample_id,
        "task_type": task_type,
        "judge_status": "ok",
        "judge_model": client.config.model,
        "prediction_length": len(prediction),
        "raw_judge_response": raw_response,
        **normalized,
    }


def _summarize_task(results: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(results)
    pass_count = sum(1 for result in results if result["pass_or_fail"])
    verdict_counts = Counter(result["verdict"] for result in results)
    issue_counter: Counter[str] = Counter()
    fail_ids: list[str] = []

    averages: dict[str, float] = {}
    for field in (*SCORE_FIELDS, "overall_score"):
        averages[field] = round(sum(float(result[field]) for result in results) / sample_count, 3) if sample_count else 0.0

    for result in results:
        issue_counter.update(result.get("issue_tags", []))
        if not result["pass_or_fail"]:
            fail_ids.append(result["id"])

    return {
        "sample_count": sample_count,
        "pass_count": pass_count,
        "fail_count": sample_count - pass_count,
        "pass_rate": round(pass_count / sample_count, 4) if sample_count else 0.0,
        "avg_scores": averages,
        "verdict_counts": dict(verdict_counts),
        "top_issue_tags": dict(issue_counter.most_common(10)),
        "fail_sample_ids": fail_ids,
    }


def run_judge(args: argparse.Namespace) -> Path:
    configure_console_output()

    run_dir = resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_run_summary(run_dir)
    task_output_files = _resolve_task_output_files(run_dir, summary, args.task or None)
    if not task_output_files:
        raise FileNotFoundError(f"No raw eval outputs found under {run_dir}")

    rubrics_dir = resolve_path(args.rubrics_dir)
    common_rubric = _load_rubric_text(rubrics_dir, COMMON_RUBRIC_FILE)

    request_url = _resolve_chat_completions_url(args.base_url)
    config = JudgeConfig(
        request_url=request_url,
        api_key=args.api_key,
        model=args.judge_model,
        timeout_seconds=args.timeout_seconds,
        retry_count=args.retry_count,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
    )
    client = DashScopeJudgeClient(config)

    overall_issue_counter: Counter[str] = Counter()
    total_samples = 0
    total_pass = 0
    total_errors = 0
    task_summaries: dict[str, Any] = {}

    for task_name, output_path in task_output_files:
        payload = _load_json(output_path)
        if not isinstance(payload, list):
            raise ValueError(f"Raw output file must be a JSON array: {output_path}")
        if args.limit is not None:
            payload = payload[: args.limit]

        task_rubric = _load_rubric_text(rubrics_dir, _task_rubric_filename(task_name))
        log_info(f"Judging task {task_name} with {len(payload)} samples")
        task_results: list[dict[str, Any]] = []
        task_error_count = 0

        for sample in payload:
            if not isinstance(sample, dict):
                continue
            try:
                result = _score_sample(sample, client, common_rubric, task_rubric)
            except Exception as exc:  # noqa: BLE001
                task_error_count += 1
                log_warn(f"{task_name} sample {sample.get('id', '')} judge failed: {exc}")
                fallback = _fallback_result(sample, f"Judge request failed: {exc}")
                result = {
                    "id": sample.get("id", ""),
                    "task_type": sample.get("task_type", task_name),
                    "judge_status": "error",
                    "judge_model": client.config.model,
                    "prediction_length": len(sample.get("prediction", "")) if isinstance(sample.get("prediction"), str) else 0,
                    "raw_judge_response": "",
                    **fallback,
                }
            task_results.append(result)

        result_path = output_dir / f"judge_results_{task_name}.json"
        write_json(result_path, task_results)
        log_success(f"Wrote judge results: {result_path}")

        task_summary = _summarize_task(task_results)
        task_summary["judge_error_count"] = task_error_count
        task_summary["raw_output_file"] = str(output_path)
        task_summaries[task_name] = task_summary
        overall_issue_counter.update(task_summary["top_issue_tags"])
        total_samples += task_summary["sample_count"]
        total_pass += task_summary["pass_count"]
        total_errors += task_error_count

    avg_scores: dict[str, float] = {}
    for field in (*SCORE_FIELDS, "overall_score"):
        values: list[float] = []
        for task_summary in task_summaries.values():
            avg = task_summary.get("avg_scores", {}).get(field)
            if isinstance(avg, (int, float)):
                weight = task_summary.get("sample_count", 0)
                values.extend([float(avg)] * int(weight))
        avg_scores[field] = round(sum(values) / len(values), 3) if values else 0.0

    summary_payload = {
        "run_dir": str(run_dir),
        "run_name": summary.get("run_name", run_dir.name),
        "model_name": summary.get("model_name"),
        "request_model": summary.get("request_model"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_provider": "dashscope_compatible",
        "judge_model": args.judge_model,
        "judge_base_url": args.base_url,
        "judge_request_url": request_url,
        "judge_config": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout_seconds,
            "retry_count": args.retry_count,
            "seed": args.seed,
            "enable_thinking": args.enable_thinking,
            "rubrics_dir": str(rubrics_dir),
        },
        "total_sample_count": total_samples,
        "total_pass_count": total_pass,
        "total_fail_count": total_samples - total_pass,
        "judge_error_count": total_errors,
        "overall_pass_rate": round(total_pass / total_samples, 4) if total_samples else 0.0,
        "overall_avg_scores": avg_scores,
        "task_summaries": task_summaries,
        "overall_top_issue_tags": dict(overall_issue_counter.most_common(20)),
    }

    rule_summary_path = run_dir / "rule_summary.json"
    if rule_summary_path.exists():
        summary_payload["rule_summary_file"] = str(rule_summary_path)

    summary_path = output_dir / "judge_summary.json"
    write_json(summary_path, summary_payload)
    log_success(f"Wrote judge summary: {summary_path}")
    return summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge over eval raw outputs using DashScope compatible chat completions.")
    parser.add_argument("--run-dir", required=True, help="Eval run directory containing summary.json and raw_outputs_*.json.")
    parser.add_argument("--output-dir", default=None, help="Directory to write judge results. Defaults to --run-dir.")
    parser.add_argument("--task", action="append", default=[], help="Optional task filter. May be passed multiple times.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples per task.")
    parser.add_argument("--rubrics-dir", default=DEFAULT_RUBRICS_DIR, help="Directory containing common_rubric.md and <task>.md rubric files.")
    parser.add_argument("--base-url", default=DASHSCOPE_API_KEY["base_url"], help="DashScope compatible-mode base URL.")
    parser.add_argument("--api-key", default=DASHSCOPE_API_KEY["api_key"], help="DashScope API key.")
    parser.add_argument("--judge-model", default=DASHSCOPE_API_KEY["model"], help="Judge model name, such as qwen-plus.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum tokens for judge output.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="HTTP timeout in seconds for each judge request.")
    parser.add_argument("--retry-count", type=int, default=2, help="How many times to retry transient judge errors.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducible judge outputs when supported.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for the judge model. Disabled by default for more stable JSON outputs.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        summary_path = run_judge(args)
        log_success(f"Judge run completed: {summary_path}")
    except Exception as exc:  # noqa: BLE001
        log_error(str(exc))
        raise


if __name__ == "__main__":
    main()
