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
from src.data_pipeline.handlers.handler_travel_qa import (
    DEFAULT_SYSTEM_PROMPT,
    build_travel_qa_sample,
)

DEFAULT_INPUT_PATH = "data/raw/travel_qa_raw_2026_04_22.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_travel_qa_2026_04_22_strict.jsonl"
DEFAULT_JSON_OUTPUT_PATH = "data/processed/sft_travel_qa.json"
DEFAULT_REPORT_PATH = "data/reports/travel_qa_2026_04_22_strict_round2_report.json"

STAGE1_TRAVEL_QA_TARGET = 3250
SOURCE_TASK_TARGETS = {
    "spot_qa": 1788,
    "city_qa": 812,
    "traffic_qa": 650,
}
CITY_CAP = 30
MAX_USER_CHARS = 180
MAX_ASSISTANT_CHARS = 300

TRAFFIC_ADVISORY = "具体线路、站点和运营情况建议出发前再确认。"
TRAFFIC_RECORD_OVERRIDES = {
    "qa_8579": "公交不能直接停到景区门口，但从湘潭站乘坐1路或46路，在“盘龙大观园”站下车即可，途中无需换乘，下车后步行一小段就到。具体线路、站点和运营情况建议出发前再确认。",
}

REALTIME_PRICE_PATTERN = re.compile(
    r"(?:票价|门票|购票|买票|售票|成人票|儿童票|学生票|优惠票|免费入园|无需门票|"
    r"价格|多少钱|人均|收费|起步价|包日价|"
    r"\d+(?:[.,]\d+)?\s*(?:元|韩元|日元|美元|马币|卢比|欧元|港币|泰铢)|"
    r"(?:RM|USD|THB|SGD|\$|€)\s*\d+)"
)
REALTIME_HOURS_PATTERN = re.compile(
    r"(?:营业时间|开放时间|全天开放|闭馆|开馆|开园|"
    r"\d{1,2}[:：]\d{2}|\d{1,2}\s*点(?:\d{1,2}\s*分)?|"
    r"班次|首班|末班|发车|车次|航班|每\s*\d+\s*分钟\s*一班|\d+\s*分钟\s*一班)"
)
BOOKING_OR_TOOL_PATTERN = re.compile(
    r"(?:库存|余票|有房|房态|预订|预约|下单|支付|订单|扫码|充值|下载APP|打开APP|"
    r"联系客服|立即购买|tool_calls?|function_call|```json|\"\s*tool\s*\"|\{\s*\"(?:intent|intentionName)\")",
    re.IGNORECASE,
)
MARKDOWN_PATTERN = re.compile(r"(?:\*\*|```|^#{1,6}\s*)", re.M)
MOJIBAKE_PATTERN = re.compile(r"\ufffd|(?:ç|è|é|å|æ|ä|ï|¼|½|œ|‰|¤|»){4,}")
GUIDE_LIKE_PATTERN = re.compile(
    r"(?:经典半日路线|半日路线|路线推荐：|行程推荐|一日游|两日游|分日|第1天|第一天)"
)
TRAFFIC_DURATION_PATTERN = re.compile(
    r"(?:全程|耗时|用时|车程|路程|步行)?\s*(?:约|大约|大概|约需|需)?\s*"
    r"\d+(?:[.-]\d+)?\s*(?:分钟|小时|公里|千米|米)(?:车程|路程|步行)?(?:左右|上下)?"
)
TRAFFIC_DURATION_QUERY_PATTERN = re.compile(r"(?:多久|多长时间|花的时间|省时间|耗时|大概要)")
TRAFFIC_ROUTE_MARKER_PATTERN = re.compile(
    r"(?:地铁|公交|巴士|专线|旅游专线|大巴|高铁|火车|BRT|快线|换乘|直达|下车|出站|步行|打车|自驾)"
)


def _sample_message(sample: dict[str, Any], role: str) -> str:
    for message in sample.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _fingerprint(sample: dict[str, Any]) -> str:
    return hashlib.md5(
        f"{_sample_message(sample, 'user')}\n###\n{_sample_message(sample, 'assistant')}".encode("utf-8")
    ).hexdigest()


def _answer_fingerprint(sample: dict[str, Any]) -> str:
    return hashlib.md5(normalize_text(_sample_message(sample, "assistant")).encode("utf-8")).hexdigest()


def _make_id(city: str, entity_name: str, user_query: str, answer: str) -> str:
    digest = hashlib.md5(f"{city}|{entity_name}|{user_query}|{answer}".encode("utf-8")).hexdigest()[:12]
    return f"travel_qa_{digest}"


def _strip_markdown(text: str) -> str:
    return normalize_text(MARKDOWN_PATTERN.sub("", text))


def _normalize_sentence_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", normalize_text(text))
    text = re.sub(r"\s*([。！？；;])\s*", r"\1", text)
    return text.strip()


def _polish_traffic_sentence(sentence: str) -> str:
    sentence = re.sub(r"(?:如|若)\s*prefer\s*公交", "如果想坐公交", sentence, flags=re.IGNORECASE)
    sentence = sentence.replace("；", "。").replace(";", "。")
    sentence = re.sub(r"[（(]\s*[）)]", "", sentence)
    sentence = re.sub(r"(?:^|[，,])\s*(?:需|约需|大约需|大概需)\s*(?=[，,。！？；;])", "", sentence)
    sentence = re.sub(r"(?:^|[，,])\s*(?:左右|上下)\s*(?=[，,。！？；;])", "", sentence)
    sentence = re.sub(r"上车后到达[，,]?\s*", "", sentence)
    sentence = re.sub(r"车程[，,]\s*", "", sentence)
    sentence = re.sub(r"(?<=[\u4e00-\u9fff])全程(?=(?:一车)?直达|无需换乘)", "。全程", sentence)
    sentence = re.sub(r"(?<=[\u4e00-\u9fff])自驾则", "。自驾则", sentence)
    sentence = re.sub(r"步行(?=(?:即可|可)?到达|可达|即到|即达)", "", sentence)
    sentence = re.sub(r"步行\s*(?=[，,。！？；;])", "", sentence)
    sentence = re.sub(r"[，,]\s*(?=[。！？；;])", "", sentence)
    sentence = re.sub(r"[，,]{2,}", "，", sentence)
    sentence = re.sub(r"[。！？]{2,}", "。", sentence)
    return sentence.strip("，,；; ")


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_sentence_spacing(text)
    parts = re.split(r"(?<=[。！？；;])", normalized)
    return [part.strip() for part in parts if part.strip()]


def _clean_traffic_answer(raw_answer: str) -> str:
    answer = _strip_markdown(raw_answer)
    answer = REALTIME_PRICE_PATTERN.sub("", answer)
    answer = REALTIME_HOURS_PATTERN.sub("", answer)
    answer = TRAFFIC_DURATION_PATTERN.sub("", answer)
    answer = re.sub(r"(?:如|若)\s*prefer\s*公交", "如果想坐公交", answer, flags=re.IGNORECASE)
    answer = re.sub(r"[（(]\s*(?:往|开往)[^）)]{1,30}[）)]", "", answer)
    answer = answer.replace("目前", "").replace("实时", "")
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"[，,；;：:]\s*[，,；;：:]+", "，", answer)

    kept: list[str] = []
    for sentence in _split_sentences(answer):
        sentence = _polish_traffic_sentence(sentence)
        if not sentence:
            continue
        if not TRAFFIC_ROUTE_MARKER_PATTERN.search(sentence):
            continue
        if REALTIME_PRICE_PATTERN.search(sentence) or REALTIME_HOURS_PATTERN.search(sentence):
            continue
        kept.append(sentence)
        if len("".join(kept)) >= 210 or len(kept) >= 3:
            break

    if not kept:
        return ""

    cleaned = _normalize_sentence_spacing("".join(kept))
    cleaned = _polish_traffic_sentence(cleaned)
    cleaned = re.sub(r"，{2,}", "，", cleaned).strip("，,；; ")
    if cleaned and not cleaned.endswith(("。", "！", "？")):
        cleaned += "。"
    if TRAFFIC_ADVISORY not in cleaned:
        cleaned += TRAFFIC_ADVISORY
    return cleaned


def _build_traffic_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    question = clean_text(record.get("user_query") or record.get("question"), max_length=800)
    raw_answer = clean_text(record.get("assistant_content") or record.get("answer"), max_length=2400)
    if not question or not raw_answer:
        return None
    if TRAFFIC_DURATION_QUERY_PATTERN.search(question):
        return None
    if REALTIME_PRICE_PATTERN.search(question) or BOOKING_OR_TOOL_PATTERN.search(question):
        return None

    record_id = clean_text(record.get("record_id"), max_length=100, mask_sensitive=False)
    answer = clean_text(_clean_traffic_answer(raw_answer), max_length=MAX_ASSISTANT_CHARS)
    if not answer:
        return None
    answer = TRAFFIC_RECORD_OVERRIDES.get(record_id, answer)

    city = clean_text(record.get("city"), max_length=100)
    entity_name = clean_text(record.get("entity_name"), max_length=200)
    system_prompt = clean_text(record.get("system_prompt") or DEFAULT_SYSTEM_PROMPT, max_length=1000, mask_sensitive=False)
    return {
        "id": _make_id(city, entity_name, question, answer),
        "record_id": record_id,
        "task_type": "travel_qa",
        "scene": "travel_qa",
        "source": clean_text(record.get("source") or "tripai_db", max_length=100, mask_sensitive=False),
        "source_dataset": "travel_qa_raw_2026_04_22",
        "source_id": clean_text(record.get("source_id"), max_length=100, mask_sensitive=False),
        "updated_at": clean_text(record.get("updated_at"), max_length=40, mask_sensitive=False),
        "source_task_type": "traffic_qa",
        "city": city,
        "entity_name": entity_name,
        "entity_type": clean_text(record.get("entity_type") or "traffic", max_length=50, mask_sensitive=False),
        "question_type": clean_text(record.get("question_type"), max_length=50, mask_sensitive=False),
        "tags": [clean_text(item, max_length=40, mask_sensitive=False) for item in record.get("tags", [])[:8]]
        if isinstance(record.get("tags"), list)
        else [],
        "is_time_sensitive": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def _classify_round2_reason(sample: dict[str, Any]) -> str | None:
    if sample.get("task_type") != "travel_qa":
        return "wrong_task_type"
    if sample.get("source_task_type") not in SOURCE_TASK_TARGETS:
        return "wrong_source_task_type"

    errors = validate_chatml_dataset([sample])
    if errors:
        return "invalid_chatml"

    question_type = normalize_text(sample.get("question_type"))
    user = _sample_message(sample, "user")
    answer = _sample_message(sample, "assistant")
    text = f"{user}\n{answer}"
    if not user or not answer:
        return "empty_content"
    if len(user) > MAX_USER_CHARS:
        return "overlong_user"
    if len(answer) > MAX_ASSISTANT_CHARS:
        return "overlong_answer"
    if question_type == "票务信息":
        return "ticket_question_type"
    if MOJIBAKE_PATTERN.search(text):
        return "mojibake"
    if MARKDOWN_PATTERN.search(text):
        return "markdown"
    if BOOKING_OR_TOOL_PATTERN.search(text):
        return "booking_or_tool"
    if REALTIME_PRICE_PATTERN.search(text):
        return "price_or_ticket"
    if REALTIME_HOURS_PATTERN.search(text):
        return "hours_or_schedule"
    if GUIDE_LIKE_PATTERN.search(text):
        return "guide_like"
    if sample.get("source_task_type") == "traffic_qa":
        if "一般可优先考虑" in answer:
            return "traffic_template"
        if not TRAFFIC_ROUTE_MARKER_PATTERN.search(answer):
            return "traffic_without_route_detail"
    return None


def _build_candidate(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if record.get("task_type") == "traffic_qa":
        sample = _build_traffic_sample(record)
    else:
        sample = build_travel_qa_sample(record)
    if sample is None:
        return None, "build_failed"
    reason = _classify_round2_reason(sample)
    if reason is not None:
        return None, reason
    return sample, None


def _dedupe(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    seen_pairs: set[str] = set()
    seen_answers: set[str] = set()
    deduped: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for sample in samples:
        pair_key = _fingerprint(sample)
        if pair_key in seen_pairs:
            reasons["duplicate_pair"] += 1
            continue
        answer_key = _answer_fingerprint(sample)
        if answer_key in seen_answers:
            reasons["duplicate_answer"] += 1
            continue
        seen_pairs.add(pair_key)
        seen_answers.add(answer_key)
        deduped.append(sample)
    return deduped, reasons


def _select_balanced(
    samples: list[dict[str, Any]],
    *,
    target_count: int,
    city_counts: Counter[str],
    selected_ids: set[str],
    city_cap: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for sample in samples:
        if sample["id"] in selected_ids:
            continue
        city = sample.get("city") or "__unknown__"
        buckets[city].append(sample)

    selected: list[dict[str, Any]] = []
    active = sorted(buckets, key=lambda item: (-len(buckets[item]), item))
    while active and len(selected) < target_count:
        next_active: list[str] = []
        for city in active:
            if len(selected) >= target_count:
                break
            if city_cap > 0 and city_counts[city] >= city_cap:
                continue
            bucket = buckets[city]
            if not bucket:
                continue
            sample = bucket.popleft()
            selected.append(sample)
            selected_ids.add(sample["id"])
            city_counts[city] += 1
            if bucket and (city_cap <= 0 or city_counts[city] < city_cap):
                next_active.append(city)
        active = next_active
    return selected


def _select_final(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample.get("source_task_type")].append(sample)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    city_counts: Counter[str] = Counter()
    for source_task_type in ("spot_qa", "city_qa", "traffic_qa"):
        selected.extend(
            _select_balanced(
                grouped.get(source_task_type, []),
                target_count=SOURCE_TASK_TARGETS[source_task_type],
                city_counts=city_counts,
                selected_ids=selected_ids,
                city_cap=CITY_CAP,
            )
        )

    if len(selected) < STAGE1_TRAVEL_QA_TARGET:
        leftovers = [sample for sample in samples if sample["id"] not in selected_ids]
        selected.extend(
            _select_balanced(
                leftovers,
                target_count=STAGE1_TRAVEL_QA_TARGET - len(selected),
                city_counts=city_counts,
                selected_ids=selected_ids,
                city_cap=CITY_CAP,
            )
        )

    return selected[:STAGE1_TRAVEL_QA_TARGET]


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min": 0, "avg": 0.0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
    ordered = sorted(lengths)

    def percentile(frac: float) -> int:
        index = max(0, min(len(ordered) - 1, round(len(ordered) * frac) - 1))
        return ordered[index]

    return {
        "min": min(lengths),
        "avg": round(sum(lengths) / len(lengths), 2),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": max(lengths),
    }


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    users = [_sample_message(sample, "user") for sample in samples]
    answers = [_sample_message(sample, "assistant") for sample in samples]
    cities = Counter(sample.get("city") or "__unknown__" for sample in samples)
    return {
        "count": len(samples),
        "task_type_counts": dict(Counter(sample.get("task_type") for sample in samples)),
        "source_task_type_counts": dict(Counter(sample.get("source_task_type") for sample in samples)),
        "question_type_counts": dict(Counter(sample.get("question_type") for sample in samples).most_common(20)),
        "city_count": len([city for city in cities if city != "__unknown__"]),
        "top_cities": dict(cities.most_common(20)),
        "city_cap_violations": {city: count for city, count in cities.items() if CITY_CAP > 0 and count > CITY_CAP},
        "user_length": _length_summary([len(text) for text in users]),
        "assistant_length": _length_summary([len(text) for text in answers]),
        "duplicate_pair_extra": len(samples) - len({(user, answer) for user, answer in zip(users, answers)}),
        "duplicate_answer_extra": len(samples) - len(set(answers)),
    }


def run_round2_cleaning(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    json_output_path: str = DEFAULT_JSON_OUTPUT_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_console_output()
    raw_records = load_records(input_path)
    log_info(f"开始 travel_qa 二轮清洗: {resolve_path(input_path)}")

    candidates: list[dict[str, Any]] = []
    filter_reasons: Counter[str] = Counter()
    removed_examples: list[dict[str, Any]] = []
    for record in raw_records:
        sample, reason = _build_candidate(record)
        if sample is None:
            filter_reasons[reason or "unknown"] += 1
            if len(removed_examples) < 80:
                removed_examples.append(
                    {
                        "record_id": record.get("record_id"),
                        "reason": reason,
                        "task_type": record.get("task_type"),
                        "question_type": record.get("question_type"),
                        "user": record.get("user_query"),
                        "assistant": normalize_text(record.get("assistant_content"))[:360],
                    }
                )
            continue
        candidates.append(sample)

    deduped, dedupe_reasons = _dedupe(candidates)
    filter_reasons.update(dedupe_reasons)
    selected = _select_final(deduped)

    chatml_errors = validate_chatml_dataset(selected)
    if chatml_errors:
        raise ValueError(f"输出数据 ChatML 校验失败，前 3 条错误: {chatml_errors[:3]}")
    if len(selected) < STAGE1_TRAVEL_QA_TARGET:
        log_warn(f"二轮清洗后仅 {len(selected)} 条，低于目标 {STAGE1_TRAVEL_QA_TARGET} 条。")

    output_file = write_jsonl(output_path, selected)
    json_output_file = write_json(json_output_path, selected)
    report = {
        "input_path": str(resolve_path(input_path)),
        "output_path": str(output_file),
        "json_output_path": str(json_output_file),
        "target_count": STAGE1_TRAVEL_QA_TARGET,
        "target_source_task_counts": SOURCE_TASK_TARGETS,
        "raw_count": len(raw_records),
        "candidate_count": len(candidates),
        "deduped_candidate_count": len(deduped),
        "selected_count": len(selected),
        "meets_target": len(selected) >= STAGE1_TRAVEL_QA_TARGET,
        "filter_reasons": dict(filter_reasons),
        "candidate_summary": _summarize(candidates),
        "output_summary": _summarize(selected),
        "removed_examples": removed_examples,
    }
    report_file = write_json(report_path, report)

    log_success(f"travel_qa 二轮清洗完成，输出 {len(selected)} 条。")
    log_info(f"输出 JSONL: {output_file}")
    log_info(f"同步 JSON: {json_output_file}")
    log_info(f"清洗报告: {report_file}")
    log_info(f"过滤原因: {dict(filter_reasons)}")
    return selected, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="travel_qa 32B stage1 二轮严格清洗。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="travel_qa 原始 JSONL。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="覆盖输出 JSONL。")
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT_PATH, help="同步输出 JSON 数组。")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="二轮清洗报告。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_round2_cleaning(args.input, args.output, args.json_output, args.report)


if __name__ == "__main__":
    main()
