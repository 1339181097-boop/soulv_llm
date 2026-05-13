from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TRIPAI_SYSTEM_PROMPT = (
    "你是 TripAI 旅行助手。当用户问题需要实时路线、位置、周边 POI 等信息时，优先使用工具。"
    "如果参数缺失，先澄清再调用。如果不需要工具，直接自然回答。"
    "如果工具失败或结果为空，明确说明不确定性并给出稳妥建议，不要编造。"
)

DEFAULT_INPUT = Path("data/raw_stage2/no_tool_needed.jsonl")
DEFAULT_OUTPUT = Path("data/processed_stage2/no_tool_needed.jsonl")
DEFAULT_REPORT = Path("data/processed_stage2/no_tool_needed_report.json")
TARGET_COUNT = 160

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

SPACE_RE = re.compile(r"[ \t\u3000]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")

RAW_STYLE_BANS = (
    "哎呀",
    "哈哈",
    "泪目",
    "随时喊我",
    "喊我",
    "废脚",
    "冒油",
    "拉满",
    "出片",
    "天灵盖",
    "美美",
    "家人们",
    "宝子",
    "超赞",
    "超开心",
    "超有",
    "超美",
    "超棒",
    "超适合",
    "太浪漫了",
    "吃货",
    "绝对不亏",
)

POST_CLEAN_STYLE_BANS = RAW_STYLE_BANS + (
    "啦",
    "哦",
    "特不要",
)

TRAILING_FOLLOWUP_PATTERNS = (
    "如果你想",
    "如果需要",
    "需要我",
    "需要推荐",
    "随时",
    "欢迎",
    "我可以",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object.")
        rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return []
    return [message for message in conversations if isinstance(message, dict)]


def _first_value(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("from") == role and isinstance(message.get("value"), str):
            return str(message["value"]).strip()
    return ""


def _last_value(messages: list[dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("from") == role and isinstance(message.get("value"), str):
            return str(message["value"]).strip()
    return ""


def _contains_tool_trace(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("from") in {"function_call", "observation"} for message in messages)


def _strip_markdown(text: str) -> str:
    cleaned = text.replace("**", "")
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+[.)、]\s*", "", cleaned, flags=re.MULTILINE)
    return cleaned


def _normalize_style(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = EMOJI_RE.sub("", cleaned)
    cleaned = cleaned.replace("～", "。").replace("~", "。")
    cleaned = cleaned.replace("！", "。").replace("!", "。")
    cleaned = cleaned.replace("？", "。").replace("?", "。")
    cleaned = cleaned.replace("挺", "比较")
    cleaned = cleaned.replace("啦", "")
    cleaned = cleaned.replace("哦", "")
    cleaned = cleaned.replace("啥", "什么")
    cleaned = _strip_markdown(cleaned)
    cleaned = SPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = BLANK_LINE_RE.sub("\n\n", cleaned)
    cleaned = re.sub(r"。{2,}", "。", cleaned)
    return cleaned.strip(" \n。") + "。"


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])\s*", text)
    return [part.strip() for part in parts if part.strip()]


def _drop_trailing_followup(sentences: list[str]) -> list[str]:
    while sentences and any(pattern in sentences[-1] for pattern in TRAILING_FOLLOWUP_PATTERNS):
        sentences.pop()
    return sentences


def _truncate_to_window(text: str, *, min_chars: int = 150, max_chars: int = 350) -> str | None:
    sentences = _drop_trailing_followup(_split_sentences(text))
    text = "".join(sentences).strip()
    if not text:
        return None
    if not text.endswith("。"):
        text += "。"

    if len(text) <= max_chars:
        return text if len(text) >= min_chars else None

    selected: list[str] = []
    for sentence in sentences:
        candidate = "".join(selected + [sentence])
        if len(candidate) > max_chars:
            break
        selected.append(sentence)

    truncated = "".join(selected).strip()
    if len(truncated) < min_chars:
        return None
    if not truncated.endswith("。"):
        truncated += "。"
    return truncated


def _has_raw_style_problem(text: str) -> bool:
    return any(token in text for token in RAW_STYLE_BANS) or bool(EMOJI_RE.search(text))


def _has_post_clean_style_problem(text: str) -> bool:
    return any(token in text for token in POST_CLEAN_STYLE_BANS) or bool(EMOJI_RE.search(text))


def _classify_subtype(question: str, answer: str) -> str:
    text = f"{question} {answer}"
    if any(token in question for token in ("公共交通", "交通", "自驾", "打车", "地铁", "公交")):
        return "transport_advice"
    if any(token in text for token in ("预算", "费用", "多少钱", "花费")):
        return "budget_advice"
    if any(token in text for token in ("什么时候", "季节", "几月", "天气", "雨天")):
        return "season_timing"
    if any(token in text for token in ("带孩子", "亲子", "老人", "父母", "家庭")):
        return "family_or_elderly_travel"
    if any(token in text for token in ("准备", "注意", "衣服", "保暖", "防晒", "安全")):
        return "packing_safety"
    if any(token in text for token in ("文化", "体验", "特色", "美食", "小吃")):
        return "culture_food_experience"
    if any(token in text for token in ("住", "住宿", "酒店", "民宿")):
        return "accommodation_advice"
    if any(token in text for token in ("行程", "路线安排", "怎么玩", "几天")):
        return "itinerary_advice"
    if any(token in question for token in ("比", "相比", "哪个", "选择")):
        return "destination_comparison"
    return "travel_advice"


def _score_candidate(question: str, answer: str, subtype: str) -> int:
    score = 100
    length = len(answer)
    if 180 <= length <= 320:
        score += 20
    elif 150 <= length <= 350:
        score += 10
    if subtype in {"budget_advice", "season_timing", "family_or_elderly_travel", "packing_safety"}:
        score += 8
    if any(token in question for token in ("附近", "路线", "怎么走", "在哪")):
        score -= 20
    if "如果" in answer[-30:]:
        score -= 10
    return score


def _build_clean_item(index: int, question: str, answer: str, subtype: str) -> dict[str, Any]:
    return {
        "id": f"stage2_v2_no_tool_needed_{subtype}_{index:06d}",
        "task_type": "no_tool_needed",
        "subtype": subtype,
        "scene": "travel_qa",
        "expected_behavior": "should_answer_directly",
        "expected_tool_chain": [],
        "expected_arguments_subset": {},
        "quality_tags": ["no_tool", "natural_answer", subtype],
        "conversations": [
            {"from": "system", "value": TRIPAI_SYSTEM_PROMPT},
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
    }


def clean_no_tool_needed(rows: list[dict[str, Any]], *, target_count: int = TARGET_COUNT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rejected = Counter()
    candidates: list[tuple[int, str, str, str]] = []
    seen_questions: set[str] = set()

    for raw_index, item in enumerate(rows, start=1):
        messages = _messages(item)
        if not messages:
            rejected["missing_conversations"] += 1
            continue
        if _contains_tool_trace(messages):
            rejected["has_tool_trace"] += 1
            continue

        question = _first_value(messages, "human")
        raw_answer = _last_value(messages, "gpt")
        if not question or not raw_answer:
            rejected["missing_question_or_answer"] += 1
            continue
        if question in seen_questions:
            rejected["duplicate_question"] += 1
            continue
        if _has_raw_style_problem(raw_answer):
            rejected["strong_style_or_emoji"] += 1
            continue

        cleaned_answer = _normalize_style(raw_answer)
        cleaned_answer = _truncate_to_window(cleaned_answer)
        if cleaned_answer is None:
            rejected["length_out_of_range_after_clean"] += 1
            continue
        if _has_post_clean_style_problem(cleaned_answer):
            rejected["style_problem_after_clean"] += 1
            continue

        subtype = _classify_subtype(question, cleaned_answer)
        score = _score_candidate(question, cleaned_answer, subtype)
        candidates.append((score, question, cleaned_answer, subtype))
        seen_questions.add(question)

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:target_count]
    cleaned = [
        _build_clean_item(index=index, question=question, answer=answer, subtype=subtype)
        for index, (_score, question, answer, subtype) in enumerate(selected, start=1)
    ]

    subtype_counts = Counter(item["subtype"] for item in cleaned)
    report = {
        "input_records": len(rows),
        "target_count": target_count,
        "candidate_count": len(candidates),
        "selected_count": len(cleaned),
        "rejected": dict(rejected),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "answer_length": {
            "min": min((len(item["conversations"][-1]["value"]) for item in cleaned), default=0),
            "max": max((len(item["conversations"][-1]["value"]) for item in cleaned), default=0),
            "avg": round(
                sum(len(item["conversations"][-1]["value"]) for item in cleaned) / len(cleaned),
                1,
            )
            if cleaned
            else 0,
        },
    }
    return cleaned, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean raw Stage2 no_tool_needed JSONL into processed Stage2 JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = _read_jsonl(args.input)
    cleaned, report = clean_no_tool_needed(rows, target_count=args.target_count)
    if len(cleaned) != args.target_count:
        raise RuntimeError(f"expected {args.target_count} cleaned records, got {len(cleaned)}")

    output_path = _write_jsonl(args.output, cleaned)
    report_path = _write_json(args.report, report)
    print(f"[OK] cleaned no_tool_needed written to: {output_path}")
    print(f"[OK] report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
