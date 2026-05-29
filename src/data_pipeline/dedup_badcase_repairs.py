from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    iter_jsonl,
    log_info,
    log_success,
    resolve_path,
    write_json,
    write_jsonl,
)
from src.data_pipeline.global_cleaner import normalize_text

DEFAULT_INPUT_PATH = "data/badcase_fix/qwen3_32b_stage1_text_badcase_repair_expanded_v2.jsonl"
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_PROMPT_THRESHOLD = 0.95
DEFAULT_ANSWER_THRESHOLD = 0.95
DEFAULT_NGRAM_SIZE = 3

SPACE_PATTERN = re.compile(r"\s+")
SIMILARITY_SCOPES = {"global", "task", "source"}


@dataclass(frozen=True)
class Fingerprint:
    prompt: frozenset[str]
    answer: frozenset[str]
    combined: frozenset[str]
    scope_key: str


@dataclass(frozen=True)
class SimilarityMatch:
    kept_index: int
    combined_similarity: float
    prompt_similarity: float
    answer_similarity: float
    duplicate_of_hash: str
    duplicate_of_line: int | None


def _with_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}")
    return path.with_name(f"{path.name}{suffix}")


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    return [record for _, record in iter_jsonl(path)]


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _message_contents(record: dict[str, Any], role: str | None = None) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []

    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if role is not None and message.get("role") != role:
            continue
        content = normalize_text(message.get("content"))
        if content:
            contents.append(content)
    return contents


def _answer_text(record: dict[str, Any]) -> str:
    for field in ("answer", "gold_answer", "reference_answer"):
        value = normalize_text(record.get(field))
        if value:
            return value
    assistant_messages = _message_contents(record, "assistant")
    return "\n".join(assistant_messages)


def _hash_payload(record: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in record.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role"))
        if role not in {"system", "user", "assistant"}:
            continue
        content = normalize_text(message.get("content"))
        if content:
            messages.append({"role": role, "content": content})

    return {
        "task_type": normalize_text(record.get("task_type")),
        "messages": messages,
        "answer": _answer_text(record),
    }


def record_content_hash(record: dict[str, Any]) -> str:
    raw = _compact_json(_hash_payload(record)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalize_for_similarity(text: str) -> str:
    text = SPACE_PATTERN.sub("", normalize_text(text).lower())
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in {"P", "S"})


def _char_ngrams(text: str, ngram_size: int) -> frozenset[str]:
    normalized = _normalize_for_similarity(text)
    if not normalized:
        return frozenset()
    if len(normalized) <= ngram_size:
        return frozenset({normalized})
    return frozenset(normalized[index : index + ngram_size] for index in range(len(normalized) - ngram_size + 1))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _scope_key(record: dict[str, Any], similarity_scope: str) -> str:
    if similarity_scope == "global":
        return "global"
    task_type = normalize_text(record.get("task_type")) or "unknown_task"
    if similarity_scope == "source":
        source_badcase_id = normalize_text(record.get("source_badcase_id")) or "unknown_source"
        return f"{task_type}::{source_badcase_id}"
    return task_type


def _fingerprint(record: dict[str, Any], *, ngram_size: int, similarity_scope: str) -> Fingerprint:
    prompt_text = "\n".join(_message_contents(record, "user"))
    answer_text = _answer_text(record)
    combined_text = f"{prompt_text}\n{answer_text}"
    return Fingerprint(
        prompt=_char_ngrams(prompt_text, ngram_size),
        answer=_char_ngrams(answer_text, ngram_size),
        combined=_char_ngrams(combined_text, ngram_size),
        scope_key=_scope_key(record, similarity_scope),
    )


def _find_similar_match(
    fingerprint: Fingerprint,
    kept_fingerprints: list[Fingerprint],
    candidate_indexes: list[int],
    kept_hashes: list[str],
    kept_line_numbers: list[int | None],
    *,
    similarity_threshold: float,
    prompt_threshold: float,
    answer_threshold: float,
) -> SimilarityMatch | None:
    best_match: SimilarityMatch | None = None
    for kept_index in candidate_indexes:
        kept_fingerprint = kept_fingerprints[kept_index]
        combined_similarity = _jaccard(fingerprint.combined, kept_fingerprint.combined)
        prompt_similarity = _jaccard(fingerprint.prompt, kept_fingerprint.prompt)
        answer_similarity = _jaccard(fingerprint.answer, kept_fingerprint.answer)
        is_match = combined_similarity >= similarity_threshold or (
            prompt_similarity >= prompt_threshold and answer_similarity >= answer_threshold
        )
        if not is_match:
            continue

        match = SimilarityMatch(
            kept_index=kept_index,
            combined_similarity=combined_similarity,
            prompt_similarity=prompt_similarity,
            answer_similarity=answer_similarity,
            duplicate_of_hash=kept_hashes[kept_index],
            duplicate_of_line=kept_line_numbers[kept_index],
        )
        if best_match is None or match.combined_similarity > best_match.combined_similarity:
            best_match = match
    return best_match


def _annotate_rejected(
    record: dict[str, Any],
    *,
    reason: str,
    content_hash: str,
    line_number: int | None,
    duplicate_of_hash: str,
    duplicate_of_line: int | None,
    similarity: dict[str, float] | None = None,
) -> dict[str, Any]:
    rejected = copy.deepcopy(record)
    metadata: dict[str, Any] = {
        "reason": reason,
        "content_hash": content_hash,
        "line_number": line_number,
        "duplicate_of_hash": duplicate_of_hash,
        "duplicate_of_line": duplicate_of_line,
    }
    if similarity is not None:
        metadata["similarity"] = similarity
    rejected["_dedup"] = metadata
    return rejected


def deduplicate_records(
    records: list[dict[str, Any]],
    *,
    line_numbers: list[int | None] | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    prompt_threshold: float = DEFAULT_PROMPT_THRESHOLD,
    answer_threshold: float = DEFAULT_ANSWER_THRESHOLD,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    similarity_scope: str = "task",
    skip_similarity: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if similarity_scope not in SIMILARITY_SCOPES:
        raise ValueError(f"Unsupported similarity scope: {similarity_scope}")

    effective_line_numbers = line_numbers or [None] * len(records)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    kept_hashes: list[str] = []
    kept_line_numbers: list[int | None] = []
    kept_fingerprints: list[Fingerprint] = []
    kept_indexes_by_scope: dict[str, list[int]] = defaultdict(list)
    first_hash_seen: dict[str, tuple[int, int | None]] = {}
    rejection_reasons: Counter[str] = Counter()

    for record, line_number in zip(records, effective_line_numbers, strict=True):
        content_hash = record_content_hash(record)
        duplicate_entry = first_hash_seen.get(content_hash)
        if duplicate_entry is not None:
            _, duplicate_line = duplicate_entry
            rejected.append(
                _annotate_rejected(
                    record,
                    reason="exact_hash",
                    content_hash=content_hash,
                    line_number=line_number,
                    duplicate_of_hash=content_hash,
                    duplicate_of_line=duplicate_line,
                )
            )
            rejection_reasons["exact_hash"] += 1
            continue

        fingerprint = _fingerprint(record, ngram_size=ngram_size, similarity_scope=similarity_scope)
        if not skip_similarity:
            match = _find_similar_match(
                fingerprint,
                kept_fingerprints,
                kept_indexes_by_scope[fingerprint.scope_key],
                kept_hashes,
                kept_line_numbers,
                similarity_threshold=similarity_threshold,
                prompt_threshold=prompt_threshold,
                answer_threshold=answer_threshold,
            )
            if match is not None:
                rejected.append(
                    _annotate_rejected(
                        record,
                        reason="similarity",
                        content_hash=content_hash,
                        line_number=line_number,
                        duplicate_of_hash=match.duplicate_of_hash,
                        duplicate_of_line=match.duplicate_of_line,
                        similarity={
                            "combined": round(match.combined_similarity, 6),
                            "prompt": round(match.prompt_similarity, 6),
                            "answer": round(match.answer_similarity, 6),
                        },
                    )
                )
                rejection_reasons["similarity"] += 1
                continue

        kept_index = len(kept)
        kept.append(record)
        kept_hashes.append(content_hash)
        kept_line_numbers.append(line_number)
        kept_fingerprints.append(fingerprint)
        kept_indexes_by_scope[fingerprint.scope_key].append(kept_index)
        first_hash_seen[content_hash] = (kept_index, line_number)

    summary = {
        "input_count": len(records),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "rejection_reasons": dict(rejection_reasons),
        "similarity_config": {
            "similarity_threshold": similarity_threshold,
            "prompt_threshold": prompt_threshold,
            "answer_threshold": answer_threshold,
            "ngram_size": ngram_size,
            "similarity_scope": similarity_scope,
            "skip_similarity": skip_similarity,
        },
        "input_by_task_type": dict(Counter(normalize_text(record.get("task_type")) or "unknown" for record in records)),
        "kept_by_task_type": dict(Counter(normalize_text(record.get("task_type")) or "unknown" for record in kept)),
        "rejected_by_task_type": dict(Counter(normalize_text(record.get("task_type")) or "unknown" for record in rejected)),
        "input_by_repair_type": dict(Counter(normalize_text(record.get("repair_type")) or "unknown" for record in records)),
        "kept_by_repair_type": dict(Counter(normalize_text(record.get("repair_type")) or "unknown" for record in kept)),
    }
    return kept, rejected, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deduplicate badcase repair JSONL records by exact hash and char n-gram similarity.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input badcase repair JSONL path.")
    parser.add_argument("--output", default=None, help="Kept output JSONL path. Defaults to <input>.dedup.jsonl.")
    parser.add_argument("--rejected", default=None, help="Rejected output JSONL path. Defaults to <input>.dedup_rejected.jsonl.")
    parser.add_argument("--summary", default=None, help="Summary JSON path. Defaults to <input>.dedup_summary.json.")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--prompt-threshold", type=float, default=DEFAULT_PROMPT_THRESHOLD)
    parser.add_argument("--answer-threshold", type=float, default=DEFAULT_ANSWER_THRESHOLD)
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    parser.add_argument("--similarity-scope", choices=sorted(SIMILARITY_SCOPES), default="task")
    parser.add_argument("--no-similarity", action="store_true", help="Only run exact hash deduplication.")
    return parser


def main() -> int:
    configure_console_output()
    args = _build_arg_parser().parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output) if args.output else _with_suffix(input_path, ".dedup.jsonl")
    rejected_path = resolve_path(args.rejected) if args.rejected else _with_suffix(input_path, ".dedup_rejected.jsonl")
    summary_path = resolve_path(args.summary) if args.summary else _with_suffix(input_path, ".dedup_summary.json")

    records_with_lines = list(iter_jsonl(input_path))
    records = [record for _, record in records_with_lines]
    line_numbers = [line_number for line_number, _ in records_with_lines]
    log_info(f"Loaded {len(records)} records from {input_path}")

    kept, rejected, summary = deduplicate_records(
        records,
        line_numbers=line_numbers,
        similarity_threshold=args.similarity_threshold,
        prompt_threshold=args.prompt_threshold,
        answer_threshold=args.answer_threshold,
        ngram_size=args.ngram_size,
        similarity_scope=args.similarity_scope,
        skip_similarity=args.no_similarity,
    )
    summary["input_path"] = str(input_path)
    summary["output_path"] = str(output_path)
    summary["rejected_path"] = str(rejected_path)
    summary["summary_path"] = str(summary_path)

    write_jsonl(output_path, kept)
    write_jsonl(rejected_path, rejected)
    write_json(summary_path, summary)

    log_success(f"Kept {len(kept)} records: {output_path}")
    log_success(f"Rejected {len(rejected)} records: {rejected_path}")
    log_success(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
