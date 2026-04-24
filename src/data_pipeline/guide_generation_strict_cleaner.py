from __future__ import annotations

import argparse
import hashlib
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
from src.data_pipeline.global_cleaner import clean_text, normalize_text
from src.data_pipeline.system_prompt_loader import load_system_prompt

DEFAULT_INPUT_PATH = "data/raw/guide_generation_raw_0424.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_guide_generation_2026_04_24_strict.jsonl"
DEFAULT_JSON_OUTPUT_PATH = "data/processed/sft_guide_generation_2026_04_24_strict.json"
DEFAULT_REPORT_PATH = "data/reports/guide_generation_2026_04_24_strict_report.json"

STAGE1_GUIDE_FINAL_TARGET = 650
STAGE1_GUIDE_CANDIDATE_MIN = 3000
STAGE1_GUIDE_CANDIDATE_MAX = 5000
DEFAULT_TARGET_COUNT = STAGE1_GUIDE_FINAL_TARGET

MIN_ASSISTANT_CHARS = 520
MAX_ASSISTANT_CHARS = 2600
MAX_USER_CHARS = 260
MAX_PREFIX_CHARS = 220

_DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "你是专业的中文旅行规划助手。请根据用户给出的目的地、天数、人群与偏好，"
    "提供结构清晰、可执行、以规划为核心的旅游攻略。"
    "避免营销口吻，不编造实时票价、营业时间、预约库存或预订结果。"
)
DEFAULT_SYSTEM_PROMPT = load_system_prompt(
    "guide_generation", _DEFAULT_SYSTEM_PROMPT_FALLBACK
)

DAY_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:(?:#{1,6}|[-*])\s*)?(?:Day\s*\d+|DAY\s*\d+|第[一二三四五六七八九十0-9]+天)",
    re.M,
)
DAY_HEADING_CAPTURE_PATTERN = re.compile(
    r"^(?:Day\s*(\d+)|DAY\s*(\d+)|第([一二三四五六七八九十0-9]+)天)\s*[：:]?\s*(.*)$",
    re.I,
)
TIME_OF_DAY_PATTERN = re.compile(r"(清晨|早上|上午|中午|下午|傍晚|晚上|夜里|凌晨)")
TAIL_SECTION_PATTERN = re.compile(
    r"^\s*(?:补充建议|出行建议|实用建议|旅行建议|温馨提示|Tips|TIPS|提示)\s*$",
    re.I,
)
INLINE_TIPS_PATTERN = re.compile(r"^\s*(?:Tips|TIPS|提示)[:：]\s*", re.I)
PRICE_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:元|块|人民币|美元|欧元|日元|港币)|[¥￥$]\s*\d+)"
)
HARD_REALTIME_PATTERN = re.compile(
    r"(?:官网|官方|放票|预约通道|余票|库存|门票|票价|营业时间|开放时间|首班|末班|"
    r"每周[一二三四五六日天]|每月\d+日|扫码|二维码|下载APP|下载App|小程序|"
    r"客服|热线|tool_calls?|function_call|```json|\"\s*tool\s*\"|\{\s*\"(?:intent|intentionName)\")",
    re.I,
)
HARD_SEGMENT_PATTERN = re.compile(
    r"(?:官网|官方|放票|余票|库存|门票|票价|营业时间|开放时间|首班|末班|"
    r"扫码|二维码|下载APP|下载App|App|APP|小程序|客服|热线|tool_calls?|function_call|"
    r"票根|订单|支付|下单|电话确认)"
)
SOFT_BOOKING_PATTERN = re.compile(
    r"(?:建议提前[^，。；]{0,30}(?:预约|预订|安排)|"
    r"需提前[^，。；]{0,30}(?:预约|预订|安排)|"
    r"务必提前[^，。；]{0,30}(?:预约|预订|安排)|"
    r"需单独预约[^，。；]{0,30})"
)
PAREN_RISK_PATTERN = re.compile(
    r"[（(][^()（）]{0,90}(?:官网|官方|放票|预约|预订|门票|票价|开放时间|营业时间|"
    r"首班|末班|班次|场次|每周|每月|电话|热线|\d{1,2}[:：]\d{2})[^()（）]{0,90}[）)]"
)
EXACT_TIME_RANGE_PATTERN = re.compile(r"(?<!\d)\d{1,2}[:：]\d{2}\s*[–—-]\s*\d{1,2}[:：]\d{2}(?!\d)")
EXACT_TIME_POINT_PATTERN = re.compile(r"(?<!\d)\d{1,2}[:：]\d{2}(?!\d)")
CHINESE_TIME_POINT_PATTERN = re.compile(
    r"(清晨|早上|上午|中午|下午|傍晚|晚上|夜里|凌晨)?\s*\d{1,2}\s*点(?:\d{1,2}\s*分)?(?:左右)?"
)
RISKY_CLAUSE_PATTERN = re.compile(
    r"(?:建议提前[^，。；]*|需提前[^，。；]*|务必提前[^，。；]*|需单独预约[^，。；]*|"
    r"每周[^，。；]*|每月[^，。；]*|查当日场次[^，。；]*|查当日公告[^，。；]*|"
    r"门票[^，。；]*|票价[^，。；]*|人均[^，。；]*|官网[^，。；]*|官方[^，。；]*|"
    r"余票[^，。；]*|库存[^，。；]*|首班[^，。；]*|末班[^，。；]*)"
)
POETIC_CLOSING_PATTERN = re.compile(
    r"(?:真正的[^。！？]{0,60}|旅行的意义|这座城市的魅力|早已悄然|物质锚点|永远留在你的记忆中|"
    r"最像.*余韵|最动人的不是|最深的敬意)"
)
MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*")
RESIDUAL_MARKDOWN_PATTERN = re.compile(r"\*\*+|`+")
ASCII_WORD_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ0-9])[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9/&+\-]*(?![A-Za-zÀ-ÿ0-9])")
EMPTY_PAREN_PATTERN = re.compile(r"[（(]\s*[）)]")
BAD_OPEN_PAREN_PATTERN = re.compile(r"[（(]\s*[，。；、:：]")
EMPTY_QUOTE_PATTERN = re.compile(r"[“\"]\s*[”\"]")
LEADING_TEMPLATE_PATTERN = re.compile(
    r"^这份[^。！？]{0,100}(?:行程|攻略)[^。！？]{0,100}(?:专为|适合)[^。！？]*[。！？]?$"
)
RESIDUAL_ARTIFACT_PATTERN = re.compile(
    r"(?:比如或|“\s+[^”]{0,12}”|“商场”商场|现整合为\s+的)"
)
NEAR_DUP_TEXT_PATTERN = re.compile(r"\s+")

_CHINESE_NUMERALS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
    "10": "十",
}


def _safe_text(value: Any, *, max_length: int = 200, mask_sensitive: bool = False) -> str:
    return clean_text(value, max_length=max_length, mask_sensitive=mask_sensitive)


def _nested_query(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("user_query")
    return raw if isinstance(raw, dict) else {}


def _meta_value(record: dict[str, Any], key: str, *, max_length: int = 120) -> str:
    query = _nested_query(record)
    value = query.get(key, record.get(key))
    return _safe_text(value, max_length=max_length, mask_sensitive=False)


def _days_text(days: str) -> str:
    days = normalize_text(days)
    if not days:
        return ""
    return days if days.endswith("天") else f"{days}天"


def _build_user_query(record: dict[str, Any]) -> str:
    destination = _meta_value(record, "destination", max_length=60)
    days = _days_text(_meta_value(record, "days", max_length=10))
    audience = _meta_value(record, "audience", max_length=80)
    budget_level = _meta_value(record, "budget_level", max_length=40)
    travel_style = _meta_value(record, "travel_style", max_length=60)
    theme = _meta_value(record, "theme", max_length=80)
    extra_constraints = _meta_value(record, "extra_constraints", max_length=160)

    context_parts: list[str] = []
    if destination:
        context_parts.append(f"目的地：{destination}")
    if days:
        context_parts.append(f"天数：{days}")
    if audience:
        context_parts.append(f"同行人群：{audience}")
    if budget_level:
        context_parts.append(f"预算：{budget_level}")
    if travel_style:
        context_parts.append(f"旅行风格：{travel_style}")
    if theme:
        context_parts.append(f"主题：{theme}")

    lines: list[str] = []
    if context_parts:
        lines.append("参考信息：" + "；".join(context_parts) + "。")
    if extra_constraints:
        lines.append(f"额外要求：{extra_constraints}")
    lines.append("请给我一份按天安排、重点清晰、可直接参考的旅游攻略。")
    return _safe_text("\n".join(lines), max_length=MAX_USER_CHARS, mask_sensitive=False)


def _replace_english_noise(text: str) -> str:
    replacements = (
        ("nearby", "附近"),
        ("Nearby", "附近"),
        ("APP", "App"),
        ("prefer", "偏好"),
        ("benches", "长椅"),
        ("Benches", "长椅"),
        ("studio", "工作室"),
        ("Studio", "工作室"),
        ("RAW", "原始"),
        ("raw", "原始"),
        ("vlog", "视频记录"),
        ("Vlog", "视频记录"),
        ("sidewalk", "步道"),
        ("Sidewalk", "步道"),
        ("Skywalk", "高空漫步"),
        ("UCCA", "艺术空间"),
        ("visibly", "明显"),
        ("DIY", "手作"),
        ("SPA", "水疗"),
        ("SUV", "越野车"),
        ("ATV", "全地形车"),
        ("LOGO", "标识"),
        ("MALL", "商场"),
        ("MUJI", "简约风"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _strip_markdown_noise(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    for _ in range(3):
        updated = MARKDOWN_BOLD_PATTERN.sub(r"\1", text)
        if updated == text:
            break
        text = updated
    text = RESIDUAL_MARKDOWN_PATTERN.sub("", text)
    text = text.replace("##", "")
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    return normalize_text(text)


def _to_chinese_day_label(value: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return ""
    if normalized in _CHINESE_NUMERALS:
        return _CHINESE_NUMERALS[normalized]
    try:
        numeric = int(normalized)
    except ValueError:
        return normalized
    if 1 <= numeric <= 10:
        return _CHINESE_NUMERALS[str(numeric)]
    return str(numeric)


def _strip_ascii_noise(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.lower() == "day":
            return token
        if re.fullmatch(r"[A-Z]\d{2,4}", token):
            return token
        if re.fullmatch(r"\d+[A-Za-z]", token):
            return token
        return ""

    text = ASCII_WORD_PATTERN.sub(_replace, text)
    text = EMPTY_QUOTE_PATTERN.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def _normalize_punctuation(text: str) -> str:
    replacements = (
        ("（，", "（"),
        ("（。", "（"),
        ("（；", "（"),
        ("(，", "("),
        ("(。", "("),
        ("(；", "("),
        ("，）", "）"),
        ("。）", "）"),
        ("；）", "）"),
        ("，)", ")"),
        ("。)", ")"),
        ("；)", ")"),
        ("——，", "——"),
        ("——。", "。"),
        ("，，", "，"),
        ("。。", "。"),
        ("；；", "；"),
        ("，，", "，"),
        ("，。", "。"),
        ("，！", "！"),
        ("，？", "？"),
        ("。：", "："),
        ("：，", "："),
        ("::", "："),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = BAD_OPEN_PAREN_PATTERN.sub("（", text)
    text = EMPTY_PAREN_PATTERN.sub("", text)
    text = EMPTY_QUOTE_PATTERN.sub("", text)
    text = text.replace("“”", "").replace("\"\"", "")
    text = re.sub(r"[（(]\s*[如若]\s*[）)]", "", text)
    text = re.sub(r"[（(]\s*[如若]\s*[\-—–、，；:：]*\s*", "（", text)
    text = re.sub(r"[，；、]{2,}", "，", text)
    text = re.sub(r"([。！？]){2,}", r"\1", text)
    text = re.sub(r"[（(]\s*[，；、 ]+", "（", text)
    text = re.sub(r"[，；、 ]+[）)]", "）", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize_text(text)


def _drop_template_lead(sentence: str) -> str:
    sentence = normalize_text(sentence)
    if not sentence:
        return ""
    if not LEADING_TEMPLATE_PATTERN.match(sentence):
        return sentence

    marker_positions = [
        sentence.find(marker)
        for marker in ("整体节奏", "全程以", "动线", "住宿建议", "住宿优先", "建议住在", "建议选在", "建议集中住在")
        if marker in sentence
    ]
    marker_positions = [position for position in marker_positions if position >= 0]
    if marker_positions:
        trimmed = sentence[min(marker_positions) :]
        return normalize_text(trimmed.lstrip("，；、 "))
    return ""


def _normalize_line(line: str) -> str:
    line = normalize_text(line)
    line = _strip_markdown_noise(line)
    line = _replace_english_noise(line)
    line = INLINE_TIPS_PATTERN.sub("", line)
    line = re.sub(r"^[•·▪●◦\-*]+\s*", "", line)
    line = _strip_ascii_noise(line)
    line = _normalize_punctuation(line)
    return line.strip()


def _strip_exact_times(text: str) -> str:
    text = EXACT_TIME_RANGE_PATTERN.sub("", text)
    text = EXACT_TIME_POINT_PATTERN.sub("", text)
    text = re.sub(r"提前\s*\d+\s*分钟(?:左右)?", "提前一点", text)

    def _replace_time(match: re.Match[str]) -> str:
        value = match.group(0)
        for marker in ("清晨", "早上", "上午", "中午", "下午", "傍晚", "晚上", "夜里", "凌晨"):
            if marker in value:
                return marker
        return ""

    text = CHINESE_TIME_POINT_PATTERN.sub(_replace_time, text)
    return text


def _clean_sentence(sentence: str) -> str:
    sentence = _normalize_line(sentence)
    if not sentence:
        return ""
    sentence = PAREN_RISK_PATTERN.sub("", sentence)
    sentence = sentence.replace("预约一场", "体验一场").replace("预约一次", "体验一次").replace("预约一节", "体验一节")
    sentence = sentence.replace("需单独预约", "").replace("需提前官网登记", "").replace("需提前登记", "")
    sentence = sentence.replace("建议提前预约", "建议提前安排").replace("建议提前预订", "建议提前安排")
    sentence = sentence.replace("务必提前预订", "建议提前安排").replace("务必提前预约", "建议提前安排")
    sentence = sentence.replace("预约接站车辆", "提前安排接站车辆").replace("预约接送", "提前安排接送")
    sentence = sentence.replace("可请工作人员协助预约", "可请工作人员协助安排")
    sentence = sentence.replace("可预约", "可安排").replace("可预订", "可安排")
    sentence = sentence.replace("预约", "安排").replace("预订", "安排")
    sentence = sentence.replace("免费轮椅租赁", "提供轮椅租赁").replace("免费门票", "").replace("无需门票", "")
    sentence = _strip_exact_times(sentence)
    sentence = SOFT_BOOKING_PATTERN.sub("建议提前安排", sentence)

    raw_segments = [segment for segment in re.split(r"[，；]", sentence) if segment is not None]
    kept_segments: list[str] = []
    for segment in raw_segments:
        cleaned_segment = normalize_text(segment)
        if not cleaned_segment:
            continue
        cleaned_segment = RISKY_CLAUSE_PATTERN.sub("", cleaned_segment)
        cleaned_segment = _strip_exact_times(cleaned_segment)
        cleaned_segment = normalize_text(cleaned_segment.strip("，；、 "))
        if not cleaned_segment:
            continue
        if HARD_SEGMENT_PATTERN.search(cleaned_segment):
            continue
        if PRICE_PATTERN.search(cleaned_segment):
            continue
        kept_segments.append(cleaned_segment)

    sentence = "，".join(kept_segments)
    sentence = sentence.replace("，。", "。").replace("，！", "！").replace("，？", "？")
    sentence = sentence.replace("建议前抵达", "建议早点抵达").replace("建议前到达", "建议早点到达")
    sentence = sentence.replace("需。", "。").replace("无需。", "。")
    sentence = re.sub(r"[，；]{2,}", "，", sentence)
    sentence = re.sub(r"^\W+", "", sentence)
    sentence = _strip_ascii_noise(sentence)
    sentence = _normalize_punctuation(sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip("，；、 ")


def _clean_prefix(prefix: str) -> str:
    normalized = normalize_text(prefix)
    if not normalized:
        return ""

    sentences = [part.strip() for part in re.split(r"(?<=[。！？])", normalized) if part.strip()]
    kept: list[str] = []
    for sentence in sentences:
        sentence = _drop_template_lead(sentence)
        if not sentence:
            continue
        cleaned_sentence = _clean_sentence(sentence)
        if not cleaned_sentence:
            continue
        if POETIC_CLOSING_PATTERN.search(cleaned_sentence):
            continue
        if HARD_REALTIME_PATTERN.search(cleaned_sentence):
            continue
        if len(cleaned_sentence) < 10:
            continue
        if cleaned_sentence.startswith("这份"):
            continue
        if not any(
            marker in cleaned_sentence
            for marker in ("住宿", "节奏", "动线", "适合", "重点", "全程", "行程", "建议", "优先", "避开", "安排", "交通")
        ):
            continue
        kept.append(cleaned_sentence)
        if len("".join(kept)) >= MAX_PREFIX_CHARS or len(kept) >= 2:
            break

    return _normalize_punctuation("".join(kept))


def _split_into_blocks(text: str) -> tuple[str, list[str]]:
    normalized = normalize_text(text)
    matches = list(DAY_MARKER_PATTERN.finditer(normalized))
    if not matches:
        return normalized, []

    prefix = normalized[: matches[0].start()].strip()
    blocks: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        block = normalized[start:end].strip()
        if block:
            blocks.append(block)
    return prefix, blocks


def _normalize_heading(heading: str) -> str:
    heading = _normalize_line(heading)
    match = DAY_HEADING_CAPTURE_PATTERN.match(heading)
    if not match:
        return heading

    day_value = match.group(1) or match.group(2) or match.group(3) or ""
    day_label = _to_chinese_day_label(day_value)
    suffix = normalize_text(match.group(4) or "").strip("：: ")
    normalized_heading = f"第{day_label}天"
    if suffix:
        return f"{normalized_heading}：{suffix}"
    return f"{normalized_heading}："


def _split_heading_line(line: str) -> tuple[str, str]:
    normalized = _normalize_line(line)
    match = DAY_HEADING_CAPTURE_PATTERN.match(normalized)
    if not match:
        return normalized, ""

    day_value = match.group(1) or match.group(2) or match.group(3) or ""
    day_label = _to_chinese_day_label(day_value)
    suffix = normalize_text(match.group(4) or "").strip("：: ")
    normalized_heading = f"第{day_label}天"
    if not suffix:
        return f"{normalized_heading}：", ""
    if len(suffix) <= 24 and not any(marker in suffix for marker in "，。；！？"):
        return f"{normalized_heading}：{suffix}", ""
    return f"{normalized_heading}：", suffix


def _clean_day_block(block: str) -> str:
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return ""

    heading, heading_suffix = _split_heading_line(lines[0])
    body_lines: list[str] = []
    source_lines = ([heading_suffix] if heading_suffix else []) + lines[1:]
    for raw_line in source_lines:
        line = _normalize_line(raw_line)
        if not line:
            continue
        if TAIL_SECTION_PATTERN.match(line):
            break
        if line.startswith(("Tips：", "Tips:", "提示：", "提示:")):
            continue
        cleaned_sentence = _clean_sentence(line)
        if not cleaned_sentence:
            continue
        if POETIC_CLOSING_PATTERN.search(cleaned_sentence):
            continue
        if HARD_REALTIME_PATTERN.search(cleaned_sentence):
            continue
        body_lines.append(cleaned_sentence)

    body = "\n".join(body_lines).strip()
    if not body:
        return ""
    return _normalize_punctuation(f"{heading}\n{body}".strip())


def _clean_unstructured_text(text: str) -> str:
    blocks = [block.strip() for block in normalize_text(text).split("\n\n") if block.strip()]
    kept: list[str] = []
    for block in blocks:
        if TAIL_SECTION_PATTERN.match(block):
            break
        sentences = [part.strip() for part in re.split(r"(?<=[。！？])", block) if part.strip()]
        cleaned_sentences: list[str] = []
        for sentence in sentences:
            cleaned_sentence = _clean_sentence(sentence)
            if not cleaned_sentence:
                continue
            if HARD_REALTIME_PATTERN.search(cleaned_sentence):
                continue
            if POETIC_CLOSING_PATTERN.search(cleaned_sentence):
                continue
            cleaned_sentences.append(cleaned_sentence)
        cleaned_block = normalize_text("".join(cleaned_sentences))
        if cleaned_block:
            kept.append(cleaned_block)
    return _normalize_punctuation("\n\n".join(kept))


def _clean_assistant_content(raw_answer: Any) -> tuple[str, dict[str, int]]:
    text = normalize_text(raw_answer)
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    text = _strip_markdown_noise(text)
    text = _replace_english_noise(text)
    prefix, day_blocks = _split_into_blocks(text)
    stats = {"day_blocks": len(day_blocks)}

    if day_blocks:
        cleaned_prefix = _clean_prefix(prefix)
        cleaned_blocks: list[str] = []
        for block in day_blocks:
            cleaned_block = _clean_day_block(block)
            if cleaned_block:
                cleaned_blocks.append(cleaned_block)
        parts = [part for part in (cleaned_prefix, *cleaned_blocks) if part]
        return _normalize_punctuation("\n\n".join(parts)), stats

    return _clean_unstructured_text(text), stats


def _sample_message(sample: dict[str, Any], role: str) -> str:
    for message in sample.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _answer_fingerprint(text: str) -> str:
    normalized = NEAR_DUP_TEXT_PATTERN.sub("", normalize_text(text))
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _make_id(destination: str, days: str, user_query: str, answer: str) -> str:
    digest = hashlib.md5(
        f"{destination}|{days}|{user_query}|{answer}".encode("utf-8")
    ).hexdigest()[:12]
    return f"guide_generation_{digest}"


def _classify_reason(
    record: dict[str, Any],
    user_query: str,
    answer: str,
    *,
    expected_days: int,
    cleaned_day_blocks: int,
) -> str | None:
    destination = _meta_value(record, "destination", max_length=60)
    if not destination:
        return "missing_destination"
    if not user_query or not answer:
        return "empty_content"
    if len(user_query) > MAX_USER_CHARS:
        return "overlong_user"
    if len(answer) < MIN_ASSISTANT_CHARS:
        return "short_answer"
    if len(answer) > MAX_ASSISTANT_CHARS:
        return "overlong_answer"
    if HARD_REALTIME_PATTERN.search(answer):
        return "residual_realtime_or_tool"
    if PRICE_PATTERN.search(answer):
        return "residual_price"
    if RESIDUAL_ARTIFACT_PATTERN.search(answer):
        return "residual_cleanup_artifact"
    if len(EXACT_TIME_POINT_PATTERN.findall(answer)) >= 1:
        return "residual_exact_time"
    if expected_days >= 2 and cleaned_day_blocks == 0:
        return "missing_day_structure"
    if expected_days >= 2 and cleaned_day_blocks < min(expected_days, 2):
        return "weak_day_structure"
    if expected_days == 1 and cleaned_day_blocks == 0 and len(TIME_OF_DAY_PATTERN.findall(answer)) < 2:
        return "weak_single_day_structure"
    if validate_chatml_dataset(
        [
            {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": answer},
                ]
            }
        ]
    ):
        return "invalid_chatml"
    return None


def _parse_days(record: dict[str, Any]) -> int:
    raw_days = _meta_value(record, "days", max_length=10)
    try:
        return int(raw_days)
    except ValueError:
        return 0


def _build_sample(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    destination = _meta_value(record, "destination", max_length=60)
    days = _meta_value(record, "days", max_length=10)
    audience = _meta_value(record, "audience", max_length=100)
    budget_level = _meta_value(record, "budget_level", max_length=40)
    travel_style = _meta_value(record, "travel_style", max_length=80)
    theme = _meta_value(record, "theme", max_length=100)
    extra_constraints = _meta_value(record, "extra_constraints", max_length=200)
    record_id = _safe_text(record.get("record_id"), max_length=120, mask_sensitive=False)

    user_query = _build_user_query(record)
    answer, clean_stats = _clean_assistant_content(record.get("assistant_content"))
    answer = _safe_text(answer, max_length=MAX_ASSISTANT_CHARS, mask_sensitive=False)

    reason = _classify_reason(
        record,
        user_query,
        answer,
        expected_days=_parse_days(record),
        cleaned_day_blocks=clean_stats["day_blocks"] if answer else 0,
    )
    if reason is not None:
        return None, reason

    sample = {
        "id": _make_id(destination, days, user_query, answer),
        "record_id": record_id or "",
        "task_type": "guide_generation",
        "scene": "guide_generation",
        "source": "guide_generation_raw_0424",
        "source_dataset": "guide_generation_raw_0424",
        "source_id": record_id or "",
        "city": destination,
        "destination": destination,
        "days": days,
        "audience": audience,
        "budget_level": budget_level,
        "travel_style": travel_style,
        "theme": theme,
        "extra_constraints": extra_constraints,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer},
        ],
    }
    return sample, None


def _length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p90": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "avg": round(sum(ordered) / len(ordered), 2),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
    }


def _day_block_count(answer: str) -> int:
    return len(DAY_MARKER_PATTERN.findall(answer))


def _sample_score(sample: dict[str, Any]) -> tuple[int, int, str]:
    answer = _sample_message(sample, "assistant")
    answer_length = len(answer)
    day_blocks = _day_block_count(answer)
    try:
        expected_days = int(str(sample.get("days", "")).strip())
    except ValueError:
        expected_days = 0

    score = 0
    if day_blocks >= min(max(expected_days, 1), 3):
        score += 4
    elif day_blocks >= 2:
        score += 2
    if 900 <= answer_length <= 2000:
        score += 3
    elif 700 <= answer_length <= 2300:
        score += 2
    else:
        score += 1
    if "住宿建议" in answer:
        score += 1
    if answer_length > 2200:
        score -= 1
    return score, -answer_length, sample.get("id", "")


def _select_diverse_samples(samples: list[dict[str, Any]], target_count: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not target_count or len(samples) <= target_count:
        return samples, {
            "candidate_count": len(samples),
            "selected_count": len(samples),
            "target_count": target_count,
            "selection_applied": False,
        }

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        destination = normalize_text(sample.get("destination")) or "未知目的地"
        grouped[destination].append(sample)

    ordered_destinations = sorted(grouped)
    for destination in ordered_destinations:
        grouped[destination].sort(key=_sample_score, reverse=True)

    destination_selected: Counter[str] = Counter()
    cursor_by_destination = {destination: 0 for destination in ordered_destinations}
    selected: list[dict[str, Any]] = []
    cap = 1
    while len(selected) < target_count:
        added_this_round = False
        for destination in ordered_destinations:
            items = grouped[destination]
            cursor = cursor_by_destination[destination]
            if cursor >= len(items):
                continue
            if destination_selected[destination] >= cap:
                continue
            selected.append(items[cursor])
            cursor_by_destination[destination] += 1
            destination_selected[destination] += 1
            added_this_round = True
            if len(selected) >= target_count:
                break
        if not added_this_round:
            break
        cap += 1

    return selected, {
        "candidate_count": len(samples),
        "selected_count": len(selected),
        "target_count": target_count,
        "selection_applied": True,
        "max_destination_count": max(destination_selected.values(), default=0),
    }


def process_guide_generation_strict(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    json_output_path: str = DEFAULT_JSON_OUTPUT_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    target_count: int | None = DEFAULT_TARGET_COUNT,
) -> list[dict[str, Any]]:
    configure_console_output()
    records = load_records(input_path)
    log_info(f"开始严格清洗 guide_generation 原始数据: {resolve_path(input_path)}")

    kept: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    reason_examples: defaultdict[str, deque[str]] = defaultdict(lambda: deque(maxlen=5))
    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    seen_answer_fingerprints: set[str] = set()

    for record in records:
        sample, reason = _build_sample(record)
        record_id = _safe_text(record.get("record_id"), max_length=120, mask_sensitive=False) or "unknown_record"
        if sample is None:
            reason_key = reason or "unknown_skip"
            reason_counts[reason_key] += 1
            reason_examples[reason_key].append(record_id)
            continue

        pair_fingerprint = hashlib.md5(
            f"{_sample_message(sample, 'user')}\n###\n{_sample_message(sample, 'assistant')}".encode("utf-8")
        ).hexdigest()
        answer_fingerprint = _answer_fingerprint(_sample_message(sample, "assistant"))

        if sample["id"] in seen_ids:
            reason_counts["duplicate_id"] += 1
            reason_examples["duplicate_id"].append(record_id)
            continue
        if pair_fingerprint in seen_pairs:
            reason_counts["duplicate_pair"] += 1
            reason_examples["duplicate_pair"].append(record_id)
            continue
        if answer_fingerprint in seen_answer_fingerprints:
            reason_counts["duplicate_answer"] += 1
            reason_examples["duplicate_answer"].append(record_id)
            continue

        seen_ids.add(sample["id"])
        seen_pairs.add(pair_fingerprint)
        seen_answer_fingerprints.add(answer_fingerprint)
        kept.append(sample)

    final_samples, selection_report = _select_diverse_samples(kept, target_count)

    write_jsonl(output_path, final_samples)
    write_json(json_output_path, final_samples)

    user_lengths = [len(_sample_message(sample, "user")) for sample in final_samples]
    assistant_lengths = [len(_sample_message(sample, "assistant")) for sample in final_samples]
    days_distribution = Counter(sample.get("days", "") for sample in final_samples)
    audience_distribution = Counter(sample.get("audience", "") for sample in final_samples)
    style_distribution = Counter(sample.get("travel_style", "") for sample in final_samples)

    report = {
        "dataset": "guide_generation_raw_0424",
        "input_path": str(resolve_path(input_path)),
        "output_path": str(resolve_path(output_path)),
        "json_output_path": str(resolve_path(json_output_path)),
        "report_path": str(resolve_path(report_path)),
        "batch_assessment": {
            "can_clean": len(kept) > 0,
            "current_raw_count": len(records),
            "recommended_raw_candidate_range": [STAGE1_GUIDE_CANDIDATE_MIN, STAGE1_GUIDE_CANDIDATE_MAX],
            "final_target_for_stage1": STAGE1_GUIDE_FINAL_TARGET,
            "meets_recommended_raw_pool": STAGE1_GUIDE_CANDIDATE_MIN <= len(records) <= STAGE1_GUIDE_CANDIDATE_MAX,
            "assessment": (
                "可作为严格清洗候选批次"
                if kept
                else "当前批次不建议进入清洗"
            ),
        },
        "stats": {
            "input_count": len(records),
            "strict_candidate_count": len(kept),
            "kept_count": len(final_samples),
            "dropped_count": len(records) - len(kept),
            "strict_keep_ratio": round(len(kept) / len(records), 4) if records else 0,
            "final_keep_ratio": round(len(final_samples) / len(records), 4) if records else 0,
            "user_length": _length_stats(user_lengths),
            "assistant_length": _length_stats(assistant_lengths),
            "days_distribution": dict(sorted(days_distribution.items(), key=lambda item: item[0])),
            "top_audience": dict(audience_distribution.most_common(12)),
            "top_travel_style": dict(style_distribution.most_common(12)),
        },
        "selection": selection_report,
        "drop_reason_counts": dict(reason_counts),
        "drop_reason_examples": {key: list(value) for key, value in reason_examples.items()},
    }
    write_json(report_path, report)

    if len(records) < STAGE1_GUIDE_CANDIDATE_MIN:
        log_warn(
            "guide_generation 原始候选量低于 32B stage1 建议值："
            f"{len(records)} < {STAGE1_GUIDE_CANDIDATE_MIN}。"
        )
    log_success(
        "guide_generation 严格清洗完成。"
        f"输入 {len(records)} 条，严格候选 {len(kept)} 条，最终输出 {len(final_samples)} 条，"
        f"剔除 {len(records) - len(kept)} 条。"
    )
    log_info(f"JSONL 输出: {resolve_path(output_path)}")
    log_info(f"JSON 输出: {resolve_path(json_output_path)}")
    log_info(f"报告输出: {resolve_path(report_path)}")
    return final_samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict cleaner for guide_generation_raw_0424.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="guide_generation 原始 JSONL 路径。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="严格清洗后的 JSONL 输出路径。")
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT_PATH, help="严格清洗后的 JSON 输出路径。")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="严格清洗报告输出路径。")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT, help="最终输出条数上限。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_guide_generation_strict(
        input_path=args.input,
        output_path=args.output,
        json_output_path=args.json_output,
        report_path=args.report,
        target_count=args.target_count,
    )


if __name__ == "__main__":
    main()
