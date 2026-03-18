from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

DEFAULT_ENCODING = "utf-8"
ROOT_DIR = Path(__file__).resolve().parents[2]


def configure_console_output() -> None:
    """Force UTF-8 output where the runtime supports stream reconfiguration."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding=DEFAULT_ENCODING, errors="backslashreplace")
            except ValueError:
                continue


def project_path(*parts: str | Path) -> Path:
    path = ROOT_DIR
    for part in parts:
        path = path / Path(part)
    return path


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_path(candidate)


def log(message: str, level: str = "INFO") -> None:
    configure_console_output()
    print(f"[{level}] {message}")


def log_info(message: str) -> None:
    log(message, "INFO")


def log_warn(message: str) -> None:
    log(message, "WARN")


def log_error(message: str) -> None:
    log(message, "ERROR")


def log_success(message: str) -> None:
    log(message, "OK")


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    with resolved.open("r", encoding=DEFAULT_ENCODING) as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                log_warn(f"è·³è؟‡ه‌ڈ JSON è،Œ: {resolved} ç¬¬ {line_number} è،Œ ({exc})")
                continue

            if not isinstance(payload, dict):
                log_warn(f"è·³è؟‡é‌‍ه¯¹è±، JSON è،Œ: {resolved} ç¬¬ {line_number} è،Œ")
                continue

            yield line_number, payload


def load_records(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    if resolved.suffix.lower() == ".jsonl":
        return [record for _, record in iter_jsonl(resolved)]

    with resolved.open("r", encoding=DEFAULT_ENCODING) as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"{resolved} ه؟…é،»وک¯ JSON و•°ç»„وˆ– JSONL و–‡ن»¶م€‚")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if isinstance(item, dict):
            records.append(item)
        else:
            log_warn(f"è·³è؟‡ç¬¬ {index} و‌،é‌‍ه¯¹è±،è®°ه½•: {resolved}")
    return records


def read_json(path: str | Path) -> Any:
    resolved = resolve_path(path)
    with resolved.open("r", encoding=DEFAULT_ENCODING) as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = ensure_parent_dir(path)
    with resolved.open("w", encoding=DEFAULT_ENCODING) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return resolved


def validate_chatml_item(item: Any, item_index: int) -> list[str]:
    errors: list[str] = []

    if not isinstance(item, dict):
        return [f"ç¬¬ {item_index} و‌،و ·وœ¬ن¸چوک¯ JSON ه¯¹è±،م€‚"]

    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        return [f"ç¬¬ {item_index} و‌،و ·وœ¬ç¼؛ه°‘é‌‍ç©؛ messages و•°ç»„م€‚"]

    roles: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"ç¬¬ {item_index} و‌،و ·وœ¬ç¬¬ {message_index} و‌،و¶ˆوپ¯ن¸چوک¯ه¯¹è±،م€‚")
            continue

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str) or not role.strip():
            errors.append(f"ç¬¬ {item_index} و‌،و ·وœ¬ç¬¬ {message_index} و‌،و¶ˆوپ¯ç¼؛ه°‘وœ‰و•ˆ roleم€‚")
        else:
            roles.append(role)

        if not isinstance(content, str) or not content.strip():
            errors.append(f"ç¬¬ {item_index} و‌،و ·وœ¬ç¬¬ {message_index} و‌،و¶ˆوپ¯ç¼؛ه°‘وœ‰و•ˆ contentم€‚")

    if "user" not in roles:
        errors.append(f"ç¬¬ {item_index} و‌،و ·وœ¬ç¼؛ه°‘ user و¶ˆوپ¯م€‚")
    if "assistant" not in roles:
        errors.append(f"ç¬¬ {item_index} و‌،و ·وœ¬ç¼؛ه°‘ assistant و¶ˆوپ¯م€‚")

    return errors


def validate_chatml_dataset(dataset: Any) -> list[str]:
    if not isinstance(dataset, list):
        return ["و•°وچ®é›†ه¤–ه±‚ه؟…é،»وک¯ JSON و•°ç»„م€‚"]

    errors: list[str] = []
    for index, item in enumerate(dataset):
        errors.extend(validate_chatml_item(item, index))
    return errors

