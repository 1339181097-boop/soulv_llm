from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    log_error,
    log_info,
    log_success,
    log_warn,
    resolve_path,
    write_json,
)

DEFAULT_EVAL_FILES = (
    "eval_guide_generation.json",
    "eval_travel_qa.json",
    "eval_hotel_recommendation.json",
    "eval_traffic_planning.json",
    "eval_persona_understanding.json",
    "eval_multi_turn_dialogue.json",
)

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class EvalTask:
    task_type: str
    eval_path: Path


def _default_reports_dir() -> Path:
    return resolve_path("src/eval/reports")


def _resolve_eval_tasks(eval_dir: Path, tasks: list[str] | None) -> list[EvalTask]:
    requested = set(tasks or [])
    eval_tasks: list[EvalTask] = []
    for filename in DEFAULT_EVAL_FILES:
        path = eval_dir / filename
        if not path.exists():
            log_warn(f"Missing eval file, skipped: {path}")
            continue
        task_type = filename.removeprefix("eval_").removesuffix(".json")
        if requested and task_type not in requested:
            continue
        eval_tasks.append(EvalTask(task_type=task_type, eval_path=path))
    return eval_tasks


def _load_json_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Eval file must be a JSON array: {path}")
    dataset: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Eval item at index {index} is not an object: {path}")
        dataset.append(item)
    return dataset


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("message.role must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message.content must be a non-empty string")
        normalized.append({"role": role, "content": content})
    return normalized


def _resolve_chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must be a non-empty string")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


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


def _extract_prediction(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("response choice is not an object")
        message = first_choice.get("message")
        if isinstance(message, dict):
            prediction = _coerce_response_text(message.get("content"))
            if prediction:
                return prediction
        prediction = _coerce_response_text(first_choice.get("text"))
        if prediction:
            return prediction
    prediction = _coerce_response_text(response_payload.get("output_text"))
    if prediction:
        return prediction
    raise ValueError(
        f"Unable to extract assistant text from response: {json.dumps(response_payload, ensure_ascii=False)[:800]}"
    )


class OpenAICompatibleChatRunner:
    def __init__(
        self,
        request_url: str,
        *,
        api_key: str,
        request_model: str,
        timeout_seconds: int,
        retry_count: int,
        disable_thinking: bool,
    ) -> None:
        self.request_url = request_url
        self.api_key = api_key
        self.request_model = request_model
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.disable_thinking = disable_thinking

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        payload = {
            "model": self.request_model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            req = request.Request(self.request_url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    raw_payload = response.read().decode("utf-8", errors="replace")
                response_payload = json.loads(raw_payload)
                return _extract_prediction(response_payload)
            except error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code} from {self.request_url}: {error_body[:800]}")
                if exc.code not in RETRYABLE_STATUS_CODES or attempt >= self.retry_count:
                    raise last_error
            except error.URLError as exc:
                last_error = RuntimeError(f"Request to {self.request_url} failed: {exc}")
                if attempt >= self.retry_count:
                    raise last_error
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Response from {self.request_url} is not valid JSON: {exc}") from exc

            sleep_seconds = min(2 ** attempt, 8)
            log_warn(f"Retrying request after {sleep_seconds}s due to transient error")
            time.sleep(sleep_seconds)

        if last_error is None:
            raise RuntimeError("Request failed without a captured exception")
        raise last_error


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen stage1 eval against a remote OpenAI-compatible chat endpoint, such as vLLM /v1/chat/completions."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Endpoint root URL. Examples: http://127.0.0.1:8000 or http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="Bearer token sent to the endpoint. Use EMPTY for unsecured local/internal vLLM if needed.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name expected by the remote endpoint.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional display name used in report outputs. Defaults to --model.",
    )
    parser.add_argument(
        "--eval-dir",
        default="src/eval",
        help="Directory containing eval_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for this eval run. Defaults to src/eval/reports/<run_name>.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to <model_name>_<UTC timestamp>.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task type to run. May be passed multiple times. Defaults to all frozen eval files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples per task for smoke tests.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of generated tokens per sample.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling when temperature > 0.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="HTTP timeout in seconds for each inference request.")
    parser.add_argument("--retry-count", type=int, default=2, help="How many times to retry transient HTTP or network errors.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass chat_template_kwargs.enable_thinking=false for Qwen3-style reasoning models on vLLM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser


def _resolve_output_dir(
    reports_dir: Path,
    *,
    run_name: str | None,
    model_name: str,
) -> Path:
    if run_name:
        folder_name = run_name
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder_name = f"{model_name}_{timestamp}"
    return reports_dir / folder_name


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {path}. Pass --overwrite to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)


def _save_task_outputs(path: Path, outputs: list[dict[str, Any]]) -> None:
    write_json(path, outputs)


def run_eval(args: argparse.Namespace) -> Path:
    configure_console_output()

    request_url = _resolve_chat_completions_url(args.base_url)
    eval_dir = resolve_path(args.eval_dir)
    eval_tasks = _resolve_eval_tasks(eval_dir, args.task or None)
    if not eval_tasks:
        raise FileNotFoundError(f"No eval tasks found under {eval_dir}")

    model_name = args.model_name or args.model
    reports_dir = resolve_path(args.output_dir) if args.output_dir else _default_reports_dir()
    output_dir = _resolve_output_dir(reports_dir, run_name=args.run_name, model_name=model_name)
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    log_info(f"Resolved output directory: {output_dir}")
    log_info(f"Using remote endpoint: {request_url}")
    runner = OpenAICompatibleChatRunner(
        request_url,
        api_key=args.api_key,
        request_model=args.model,
        timeout_seconds=args.timeout_seconds,
        retry_count=args.retry_count,
        disable_thinking=args.disable_thinking,
    )

    summary: dict[str, Any] = {
        "provider": "openai_compatible",
        "model_name": model_name,
        "request_model": args.model,
        "base_url": args.base_url,
        "request_url": request_url,
        "run_name": output_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "timeout_seconds": args.timeout_seconds,
            "retry_count": args.retry_count,
            "disable_thinking": args.disable_thinking,
        },
        "tasks": {},
    }

    total_samples = 0
    total_errors = 0

    for task in eval_tasks:
        dataset = _load_json_dataset(task.eval_path)
        if args.limit is not None:
            dataset = dataset[: args.limit]
        log_info(f"Running task {task.task_type} with {len(dataset)} samples")

        task_outputs: list[dict[str, Any]] = []
        task_error_count = 0
        for sample in dataset:
            sample_id = sample.get("id", "")
            try:
                messages = _normalize_messages(sample.get("messages"))
                prediction = runner.generate(
                    messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                status = "ok"
                error_message = ""
            except Exception as exc:  # noqa: BLE001
                prediction = ""
                status = "error"
                error_message = str(exc)
                task_error_count += 1
                log_warn(f"{task.task_type} sample {sample_id} failed: {exc}")

            task_outputs.append(
                {
                    "id": sample_id,
                    "task_type": sample.get("task_type", task.task_type),
                    "scene": sample.get("scene"),
                    "difficulty": sample.get("difficulty"),
                    "tags": sample.get("tags", []),
                    "messages": sample.get("messages", []),
                    "reference_answer": sample.get("reference_answer", ""),
                    "must_include": sample.get("must_include", []),
                    "must_not_do": sample.get("must_not_do", []),
                    "notes": sample.get("notes", ""),
                    "seed_source": sample.get("seed_source"),
                    "seed_sample_id": sample.get("seed_sample_id"),
                    "prediction": prediction,
                    "status": status,
                    "error": error_message,
                    "provider": "openai_compatible",
                    "model_name": model_name,
                    "request_model": args.model,
                    "request_url": request_url,
                }
            )

        output_path = output_dir / f"raw_outputs_{task.task_type}.json"
        _save_task_outputs(output_path, task_outputs)
        log_success(f"Wrote task outputs: {output_path}")

        summary["tasks"][task.task_type] = {
            "eval_file": str(task.eval_path),
            "output_file": str(output_path),
            "sample_count": len(task_outputs),
            "error_count": task_error_count,
        }
        total_samples += len(task_outputs)
        total_errors += task_error_count

    summary["total_sample_count"] = total_samples
    summary["total_error_count"] = total_errors
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    log_success(f"Wrote eval summary: {summary_path}")
    return output_dir


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        output_dir = run_eval(args)
        log_success(f"Eval run completed: {output_dir}")
    except Exception as exc:  # noqa: BLE001
        log_error(str(exc))
        raise


if __name__ == "__main__":
    main()
