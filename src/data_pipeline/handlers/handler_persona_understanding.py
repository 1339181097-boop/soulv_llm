from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    load_records,
    log_error,
    log_info,
    log_success,
    log_warn,
    resolve_path,
    write_json,
)
from src.data_pipeline.global_cleaner import clean_text, normalize_text
from src.data_pipeline.system_prompt_loader import load_system_prompt

DEFAULT_INPUT_PATH = "data/raw/persona_understanding_raw_2026_03_31.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_persona_understanding_strict.json"
DEFAULT_REPORT_PATH = "data/reports/persona_understanding_strict_report.json"
DEFAULT_TOTAL_SAMPLES = 500
DEFAULT_CITY_CAP = 8
DEFAULT_CITY_PERSONA_CAP = 1

_DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "你是专业的中文旅行需求分析助手。请根据用户的人群特征、预算、偏好和限制条件，"
    "提供因人而异的景点建议，并说明推荐逻辑。"
    "避免空泛套话和营销表达，不补充候选景点字段之外的新事实。"
)
DEFAULT_SYSTEM_PROMPT = load_system_prompt(
    "persona_understanding", _DEFAULT_SYSTEM_PROMPT_FALLBACK
)

CONTEXT_PREFIX = "参考信息："
USER_QUESTION_PREFIX = "用户问题："

RAW_TAG_SPLIT_PATTERN = re.compile(r"[、,，/；;|·\s]+")
PRICE_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")
GENERIC_QUERY_PREFIXES = (
    "有什么推荐",
    "有哪些推荐",
    "怎么选",
    "怎么安排",
    "去哪里",
    "值得去",
)
PLACEHOLDER_ANSWER_PREFIXES = (
    "根据您的需求，我们为您精选了几个适合的景点，希望您能喜欢。",
)
META_NOISE_TOKENS = (
    "无法凭空生成第3个",
    "严格遵循数据源",
    "重新审题",
    "仅能推荐已给的",
)

CANONICAL_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "亲子": ("亲子", "家庭", "家庭游"),
    "互动体验": ("互动", "互动体验", "沉浸", "沉浸式", "体验", "手作", "科普", "演艺"),
    "儿童友好": ("儿童友好", "儿童", "小朋友", "萌宠", "亲子"),
    "室内": ("室内", "展馆", "博物馆", "美术馆", "艺术馆", "科技馆", "海洋馆", "剧场"),
    "免费": ("免费", "免票", "免费开放", "免费入园", "免门票"),
    "低价": ("低价", "平价", "实惠", "经济"),
    "高消费": ("高消费", "昂贵", "奢华", "高端消费"),
    "古建": ("古建", "古建筑", "古镇", "古城", "古街", "古村", "寺", "祠", "宫", "塔", "书院", "园林"),
    "夜景": ("夜景", "夜游", "灯光", "灯会", "夜色"),
    "拍照": ("拍照", "打卡", "摄影", "出片", "网红"),
    "观景": ("观景", "全景", "俯瞰", "眺望", "登高", "景观", "天台"),
    "轻松": ("轻松", "休闲", "悠闲", "舒缓", "放松", "度假", "慢游", "慢逛", "平缓"),
    "少步行": ("少步行", "平地", "平坦", "无障碍", "短距离", "观光车", "电梯"),
    "休息设施": ("休息", "休憩", "座椅", "温泉", "度假"),
    "文化体验": ("文化", "历史", "非遗", "民俗", "展览", "文博", "遗址", "书院", "寺庙", "艺术"),
    "自然风景": ("自然", "山水", "草原", "森林", "湿地", "湖", "海", "沙滩", "峡谷", "瀑布", "溶洞", "洞穴"),
    "本地特色": ("本地特色", "地方特色", "海鲜市场", "夜市", "小吃", "美食", "古镇", "市集"),
    "交通便利": ("交通便利", "近市区", "市中心", "便捷", "核心区"),
    "浪漫": ("浪漫", "爱情", "情侣", "海誓山盟"),
    "私密": ("私密", "安静", "静谧", "小众", "避世"),
    "高品质体验": ("高品质", "品质", "精品", "高端", "度假酒店", "高尔夫", "游艇"),
    "小众探索": ("小众", "秘境", "冷门", "探索", "原生态"),
    "经典景点": ("经典", "地标", "世界遗产", "5A", "名胜", "标志性"),
    "商务便利": ("商务", "会展", "商旅"),
    "社交氛围": ("社交", "结伴", "派对", "市集"),
    "拥挤": ("拥挤", "人潮", "热门", "爆火", "排队"),
    "台阶多": ("台阶", "徒步", "栈道", "攀爬", "登山", "爬山"),
    "高强度活动": ("高空刺激", "高强度", "滑雪", "探险", "蹦极", "攀岩", "漂流"),
    "嘈杂": ("嘈杂", "喧闹", "轰趴"),
}

PREFERENCE_CANONICAL_MAP: dict[str, tuple[str, ...]] = {
    "性价比": ("免费", "低价"),
    "社交": ("社交氛围", "本地特色"),
    "特色体验": ("本地特色", "文化体验", "自然风景"),
    "预算控制": ("免费", "低价"),
    "拍照打卡": ("拍照",),
    "网红景点": ("拍照", "观景"),
    "浪漫": ("浪漫",),
    "拍照": ("拍照",),
    "私密": ("私密",),
    "高品质体验": ("高品质体验",),
    "互动体验": ("互动体验",),
    "儿童友好": ("儿童友好", "亲子"),
    "轻松节奏": ("轻松",),
    "轻松": ("轻松",),
    "室内或安全性高": ("室内",),
    "少步行": ("少步行",),
    "休息设施": ("休息设施",),
    "经典景点": ("经典景点", "文化体验"),
    "兼顾老人孩子": ("亲子", "轻松", "休息设施"),
    "景点衔接": ("交通便利",),
    "免费": ("免费",),
    "低价": ("低价",),
    "本地特色": ("本地特色",),
    "经济实惠": ("免费", "低价"),
    "古建": ("古建",),
    "夜景": ("夜景",),
    "观景": ("观景",),
    "小众探索": ("小众探索",),
    "安全与自由行程": ("轻松", "交通便利", "小众探索"),
}

AVOID_CANONICAL_MAP: dict[str, tuple[str, ...]] = {
    "高消费": ("高消费",),
    "昂贵": ("高消费",),
    "人潮拥挤": ("拥挤",),
    "人潮过密": ("拥挤",),
    "台阶多": ("台阶多",),
    "陡峭台阶": ("台阶多",),
    "长时间步行": ("台阶多",),
    "高强度活动": ("高强度活动",),
    "纯成人向": ("嘈杂",),
    "不适合老人儿童": ("高强度活动", "台阶多"),
    "不适合女性": ("嘈杂",),
    "儿童区": ("儿童友好", "亲子"),
    "嘈杂": ("嘈杂",),
    "无特色": (),
    "不出片": (),
}

PERSONA_DEFAULT_TAGS: dict[str, tuple[str, ...]] = {
    "学生党": ("免费", "低价", "本地特色", "文化体验"),
    "美食爱好者": ("本地特色", "文化体验", "轻松"),
    "情侣": ("浪漫", "拍照", "私密", "高品质体验"),
    "闺蜜游": ("拍照", "轻松", "文化体验", "本地特色"),
    "摄影爱好者": ("拍照", "夜景", "古建", "观景"),
    "独自旅行": ("轻松", "文化体验", "自然风景", "小众探索"),
    "亲子": ("亲子", "互动体验", "儿童友好", "室内", "轻松"),
    "三代同堂": ("轻松", "少步行", "休息设施", "亲子", "文化体验"),
    "预算型": ("免费", "低价", "本地特色", "文化体验"),
    "商务出差": ("交通便利", "轻松", "文化体验"),
    "老人": ("少步行", "轻松", "休息设施", "经典景点"),
}

POSITIVE_TAG_LABELS: dict[str, str] = {
    "亲子": "亲子友好",
    "互动体验": "互动体验",
    "儿童友好": "儿童友好",
    "室内": "室内或更稳妥",
    "免费": "免费",
    "低价": "低价",
    "古建": "古建氛围",
    "夜景": "夜景",
    "拍照": "拍照打卡",
    "观景": "观景视野",
    "轻松": "节奏轻松",
    "少步行": "步行压力更小",
    "休息设施": "更适合休闲放松",
    "文化体验": "文化体验",
    "自然风景": "自然风景",
    "本地特色": "本地特色",
    "交通便利": "衔接更顺",
    "浪漫": "浪漫氛围",
    "私密": "氛围更安静",
    "高品质体验": "品质感更强",
    "小众探索": "更有探索感",
    "经典景点": "经典辨识度",
    "商务便利": "商务便利",
    "社交氛围": "更有社交话题",
}

NEGATIVE_TAG_LABELS: dict[str, str] = {
    "高消费": "预算压力更高",
    "拥挤": "更容易拥挤",
    "台阶多": "步行或爬升压力更大",
    "高强度活动": "活动强度偏高",
    "嘈杂": "环境可能更嘈杂",
}


def _clean_list(raw_value: Any, *, max_items: int = 8, item_length: int = 40) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_value:
        text = clean_text(item, max_length=item_length, mask_sensitive=False)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _extract_price_numbers(text: str) -> list[float]:
    return [float(value) for value in PRICE_NUMBER_PATTERN.findall(text)]


def _price_canonical_tags(price_text: str) -> set[str]:
    normalized = normalize_text(price_text)
    if not normalized:
        return set()

    tags: set[str] = set()
    if any(token in normalized for token in ("免费", "免票", "免费开放", "免费入园", "免门票")):
        tags.add("免费")

    numbers = _extract_price_numbers(normalized)
    if numbers:
        low = min(numbers)
        if low <= 60:
            tags.add("低价")
        if low >= 200:
            tags.add("高消费")
    return tags


def _text_to_canonical_tags(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()

    tags: set[str] = set()
    for canonical, keywords in CANONICAL_TAG_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            tags.add(canonical)
    return tags


def _spot_canonical_tags(spot: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    raw_tags = _clean_list(spot.get("tags"), max_items=12, item_length=80)
    for raw_tag in raw_tags:
        for token in RAW_TAG_SPLIT_PATTERN.split(raw_tag):
            if token:
                tags.update(_text_to_canonical_tags(token))

    tags.update(_text_to_canonical_tags(clean_text(spot.get("brief"), max_length=1200)))
    tags.update(_price_canonical_tags(clean_text(spot.get("price"), max_length=120)))
    return tags


def _preference_to_canonical(preferences: list[str], persona_type: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for preference in preferences:
        mapped = PREFERENCE_CANONICAL_MAP.get(preference, ())
        for tag in mapped:
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)

    for fallback in PERSONA_DEFAULT_TAGS.get(persona_type, ()):
        if fallback not in seen:
            seen.add(fallback)
            tags.append(fallback)
    return tags


def _avoid_to_canonical(avoid_tags: list[str]) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for avoid in avoid_tags:
        for tag in AVOID_CANONICAL_MAP.get(avoid, ()):
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def _normalize_budget_level(value: str) -> str:
    normalized = normalize_text(value).lower()
    if normalized in {"low", "low_budget", "budget", "economy"}:
        return "low"
    if normalized in {"medium", "mid", "midrange"}:
        return "medium"
    if normalized in {"high", "high_budget", "premium"}:
        return "high"
    return normalized or "medium"


def _budget_matches(budget_level: str, spot_tags: set[str]) -> int:
    if budget_level == "low":
        score = 0
        if "免费" in spot_tags:
            score += 3
        if "低价" in spot_tags:
            score += 2
        if "高消费" in spot_tags:
            score -= 4
        return score
    if budget_level == "medium":
        score = 0
        if "免费" in spot_tags:
            score += 1
        if "低价" in spot_tags:
            score += 1
        if "高消费" in spot_tags:
            score -= 1
        return score
    if budget_level == "high":
        score = 0
        if "高品质体验" in spot_tags:
            score += 2
        if "高消费" in spot_tags:
            score += 1
        return score
    return 0


def _spot_score(
    spot_tags: set[str],
    preference_tags: list[str],
    avoid_tags: list[str],
    budget_level: str,
) -> tuple[int, list[str], list[str]]:
    matched = [tag for tag in preference_tags if tag in spot_tags]
    conflicts = [tag for tag in avoid_tags if tag in spot_tags]
    score = len(matched) * 3 - len(conflicts) * 3 + _budget_matches(budget_level, spot_tags)
    if "经典景点" in matched:
        score += 1
    if "轻松" in matched:
        score += 1
    return score, matched, conflicts


def _spot_name(spot: dict[str, Any]) -> str:
    return clean_text(spot.get("name"), max_length=80, mask_sensitive=False)


def _choose_template(sample_id: str) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:2], 16) % 3


def _join_labels(tags: list[str], label_map: dict[str, str], *, limit: int = 3) -> str:
    labels = [label_map[tag] for tag in tags if tag in label_map][:limit]
    if not labels:
        return "整体匹配度更高"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]}和{labels[1]}"
    return f"{labels[0]}、{labels[1]}和{labels[2]}"


def _build_selected_reason(spot_name: str, matched: list[str]) -> str:
    reason = _join_labels(matched, POSITIVE_TAG_LABELS, limit=3)
    return f"{spot_name}更符合{reason}"


def _build_rejected_reason(spot_name: str, conflicts: list[str]) -> str:
    reason = _join_labels(conflicts, NEGATIVE_TAG_LABELS, limit=2)
    return f"{spot_name}相对更容易出现{reason}"


def _render_assistant_content(
    *,
    record_id: str,
    persona_type: str,
    selected_spots: list[dict[str, Any]],
    rejected_spots: list[dict[str, Any]],
) -> str:
    selected_reasons = [
        _build_selected_reason(spot["name"], spot["matched"])
        for spot in selected_spots
    ]
    selected_names = [spot["name"] for spot in selected_spots]
    template_index = _choose_template(record_id)

    if template_index == 0:
        body = (
            f"如果按{persona_type}的需求来选，更适合优先考虑"
            f"{'、'.join(selected_names)}。"
            f"{'；'.join(selected_reasons)}。"
        )
    elif template_index == 1:
        body = (
            f"从这组候选景点里看，{persona_type}更适合把"
            f"{'、'.join(selected_names)}放在前面。"
            f"{'；'.join(selected_reasons)}。"
        )
    else:
        body = (
            f"结合画像偏好，优先推荐{'、'.join(selected_names)}。"
            f"{'；'.join(selected_reasons)}。"
        )

    if rejected_spots:
        reject_reasons = [
            _build_rejected_reason(spot["name"], spot["conflicts"])
            for spot in rejected_spots[:2]
            if spot["conflicts"]
        ]
        if reject_reasons:
            body += f"相比之下，{'；'.join(reject_reasons)}，所以优先级会靠后。"
    return body


def _render_reason_text(
    *,
    preference_tags: list[str],
    avoid_tags: list[str],
    selected_spots: list[dict[str, Any]],
    rejected_spots: list[dict[str, Any]],
) -> str:
    pref_text = _join_labels(preference_tags, POSITIVE_TAG_LABELS, limit=4)
    avoid_text = _join_labels(avoid_tags, NEGATIVE_TAG_LABELS, limit=3)
    selected_summary = "、".join(spot["name"] for spot in selected_spots)
    if rejected_spots:
        rejected_summary = "、".join(spot["name"] for spot in rejected_spots[:2])
        return (
            f"该画像核心看重{pref_text}，同时尽量避开{avoid_text}，"
            f"因此优先保留{selected_summary}，并将{rejected_summary}放到更后的位置。"
            "推荐仅依据候选景点的标签、价格和简介中可稳定支持的信息生成。"
        )
    return (
        f"该画像核心看重{pref_text}，同时尽量避开{avoid_text}，"
        f"因此优先保留{selected_summary}。"
        "推荐仅依据候选景点的标签、价格和简介中可稳定支持的信息生成。"
    )


def _build_contextual_user_query(query: str, city: str, persona_type: str) -> str:
    if not query:
        return ""
    normalized = normalize_text(query)
    if city and city in normalized:
        return normalized
    if any(normalized.startswith(prefix) for prefix in GENERIC_QUERY_PREFIXES):
        return (
            f"{CONTEXT_PREFIX}城市：{city}；画像：{persona_type}\n\n"
            f"{USER_QUESTION_PREFIX}{normalized}"
        )
    return normalized


def _raw_text_has_noise(record: dict[str, Any]) -> bool:
    combined = "\n".join(
        [
            clean_text(record.get("assistant_content"), max_length=2000),
            clean_text(record.get("reason_text"), max_length=1000),
        ]
    )
    if any(combined.startswith(prefix) for prefix in PLACEHOLDER_ANSWER_PREFIXES):
        return True
    return any(token in combined for token in META_NOISE_TOKENS)


def _fingerprint(sample: dict[str, Any]) -> str:
    user_text = sample["messages"][1]["content"]
    assistant_text = sample["messages"][2]["content"]
    return hashlib.md5(f"{user_text}\n###\n{assistant_text}".encode("utf-8")).hexdigest()


def _quality_score(sample: dict[str, Any]) -> tuple[int, int]:
    assistant = sample["messages"][2]["content"]
    matched_count = len(sample.get("matched_tags", []))
    rejected_count = len(sample.get("rejected_spots", []))
    answer_length = len(assistant)
    score = matched_count * 3 + rejected_count
    if 120 <= answer_length <= 320:
        score += 3
    elif 90 <= answer_length <= 420:
        score += 1
    score += min(2, len(sample.get("selected_spots", [])))
    return score, -answer_length


def build_persona_understanding_sample(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    task_type = clean_text(record.get("task_type"), max_length=80, mask_sensitive=False)
    if task_type != "persona_understanding":
        return None, "wrong_task_type"

    city = clean_text(record.get("city"), max_length=40, mask_sensitive=False)
    persona_type = clean_text(record.get("persona_type"), max_length=40, mask_sensitive=False)
    query = clean_text(record.get("user_query"), max_length=300)
    budget_level = _normalize_budget_level(
        clean_text(record.get("budget_level"), max_length=30, mask_sensitive=False)
    )
    people_count = record.get("people_count")
    audience = _clean_list(record.get("audience"), max_items=4, item_length=20)
    raw_preference_tags = _clean_list(record.get("preference_tags"), max_items=8, item_length=30)
    raw_avoid_tags = _clean_list(record.get("avoid_tags"), max_items=8, item_length=30)
    candidate_spots = record.get("candidate_spots")

    if (
        not city
        or not persona_type
        or not query
        or not isinstance(people_count, int)
        or not isinstance(candidate_spots, list)
    ):
        return None, "missing_required_fields"

    if len(candidate_spots) < 3:
        return None, "candidate_spots_lt3"
    if _raw_text_has_noise(record):
        return None, "raw_text_noise"

    canonical_preferences = _preference_to_canonical(raw_preference_tags, persona_type)
    canonical_avoid = _avoid_to_canonical(raw_avoid_tags)
    if not canonical_preferences:
        return None, "no_canonical_preferences"

    scored_spots: list[dict[str, Any]] = []
    for spot in candidate_spots[:5]:
        name = _spot_name(spot)
        if not name:
            continue
        spot_tags = _spot_canonical_tags(spot)
        score, matched, conflicts = _spot_score(
            spot_tags,
            canonical_preferences,
            canonical_avoid,
            budget_level,
        )
        scored_spots.append(
            {
                "name": name,
                "score": score,
                "matched": matched,
                "conflicts": conflicts,
                "canonical_tags": sorted(spot_tags),
                "price": clean_text(spot.get("price"), max_length=80, mask_sensitive=False),
            }
        )

    if len(scored_spots) < 3:
        return None, "candidate_spots_lt3_after_clean"

    scored_spots.sort(
        key=lambda item: (
            item["score"],
            len(item["matched"]),
            -len(item["conflicts"]),
            item["name"],
        ),
        reverse=True,
    )
    selected_spots = scored_spots[:3]
    if sum(len(spot["matched"]) for spot in selected_spots) < 2:
        return None, "insufficient_preference_alignment"

    rejected_spots = [
        spot
        for spot in scored_spots[3:]
        if spot["conflicts"] or spot["score"] < selected_spots[-1]["score"]
    ]

    record_id = clean_text(record.get("record_id"), max_length=80, mask_sensitive=False)
    assistant_content = _render_assistant_content(
        record_id=record_id or f"{city}_{persona_type}_{query}",
        persona_type=persona_type,
        selected_spots=selected_spots,
        rejected_spots=rejected_spots,
    )
    reason_text = _render_reason_text(
        preference_tags=canonical_preferences,
        avoid_tags=canonical_avoid,
        selected_spots=selected_spots,
        rejected_spots=rejected_spots,
    )

    assistant_message = f"{assistant_content}{reason_text}"
    if len(assistant_message) < 90:
        return None, "assistant_too_short"

    user_query = _build_contextual_user_query(query, city, persona_type)
    sample_id = "persona_understanding_" + hashlib.md5(
        f"{city}|{persona_type}|{user_query}|{assistant_message}".encode("utf-8")
    ).hexdigest()[:12]

    sample = {
        "id": sample_id,
        "task_type": "persona_understanding",
        "scene": "persona_understanding",
        "source": "tripai_persona_understanding_20260331",
        "source_id": record_id,
        "city": city,
        "persona_type": persona_type,
        "audience": audience,
        "budget_level": budget_level,
        "people_count": people_count,
        "raw_preference_tags": raw_preference_tags,
        "raw_avoid_tags": raw_avoid_tags,
        "canonical_preference_tags": canonical_preferences,
        "canonical_avoid_tags": canonical_avoid,
        "matched_tags": sorted(
            {tag for spot in selected_spots for tag in spot["matched"]}
        ),
        "selected_spots": [spot["name"] for spot in selected_spots],
        "rejected_spots": [spot["name"] for spot in rejected_spots[:2]],
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_message},
        ],
    }
    return sample, "ok"


def _calculate_persona_targets(
    samples: list[dict[str, Any]],
    total_samples: int,
) -> dict[str, int]:
    available = Counter(sample["persona_type"] for sample in samples)
    if total_samples <= 0:
        return dict(available)

    targets = {persona: 0 for persona in available}
    persona_order = sorted(available)
    remaining = total_samples
    while remaining > 0:
        progress = False
        for persona in persona_order:
            if remaining <= 0:
                break
            if targets[persona] >= available[persona]:
                continue
            targets[persona] += 1
            remaining -= 1
            progress = True
        if not progress:
            break
    return targets


def _select_balanced_subset(
    samples: list[dict[str, Any]],
    *,
    total_samples: int,
    city_cap: int,
    city_persona_cap: int,
) -> list[dict[str, Any]]:
    if total_samples <= 0 or len(samples) <= total_samples:
        return list(samples)

    targets = _calculate_persona_targets(samples, total_samples)
    persona_counts: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    city_persona_counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []

    ranked = sorted(samples, key=_quality_score, reverse=True)
    for sample in ranked:
        if len(selected) >= total_samples:
            break
        persona = sample["persona_type"]
        city = sample["city"]
        if persona_counts[persona] >= targets.get(persona, 0):
            continue
        if city_cap > 0 and city_counts[city] >= city_cap:
            continue
        city_persona_key = (city, persona)
        if city_persona_cap > 0 and city_persona_counts[city_persona_key] >= city_persona_cap:
            continue
        selected.append(sample)
        persona_counts[persona] += 1
        city_counts[city] += 1
        city_persona_counts[city_persona_key] += 1

    if len(selected) >= total_samples:
        return selected

    selected_ids = {sample["id"] for sample in selected}
    for sample in ranked:
        if len(selected) >= total_samples:
            break
        if sample["id"] in selected_ids:
            continue
        city = sample["city"]
        if city_cap > 0 and city_counts[city] >= city_cap:
            continue
        selected.append(sample)
        selected_ids.add(sample["id"])
        city_counts[city] += 1
    return selected


def process_persona_understanding_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    *,
    report_path: str = DEFAULT_REPORT_PATH,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    city_cap: int = DEFAULT_CITY_CAP,
    city_persona_cap: int = DEFAULT_CITY_PERSONA_CAP,
) -> list[dict[str, Any]]:
    configure_console_output()
    log_info(f"开始处理 persona_understanding 数据: {resolve_path(input_file_path)}")

    try:
        raw_records = load_records(input_file_path)
    except FileNotFoundError:
        log_warn(f"未找到 persona_understanding 原始数据: {resolve_path(input_file_path)}")
        return []
    except ValueError as exc:
        log_error(str(exc))
        return []

    processed: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    fingerprint_seen: set[str] = set()
    deduped = 0

    for record in raw_records:
        sample, reason = build_persona_understanding_sample(record)
        if sample is None:
            skip_reasons[reason] += 1
            continue
        fingerprint = _fingerprint(sample)
        if fingerprint in fingerprint_seen:
            deduped += 1
            continue
        fingerprint_seen.add(fingerprint)
        processed.append(sample)

    processed = sorted(processed, key=_quality_score, reverse=True)
    selected = _select_balanced_subset(
        processed,
        total_samples=total_samples,
        city_cap=city_cap,
        city_persona_cap=city_persona_cap,
    )

    output_path = write_json(output_json_path, selected)
    persona_counts = Counter(sample["persona_type"] for sample in selected)
    city_counts = Counter(sample["city"] for sample in selected)
    report = {
        "input_path": str(resolve_path(input_file_path)),
        "output_path": str(output_path),
        "raw_count": len(raw_records),
        "strict_kept_count": len(processed),
        "final_count": len(selected),
        "deduped_count": deduped,
        "skip_reasons": dict(skip_reasons),
        "persona_distribution": dict(persona_counts),
        "city_top_20": city_counts.most_common(20),
        "total_samples_target": total_samples,
        "city_cap": city_cap,
        "city_persona_cap": city_persona_cap,
    }
    report_output_path = write_json(report_path, report)

    log_success(
        "处理 persona_understanding 数据完成。"
        f"严格保留 {len(processed)} 条，"
        f"最终输出 {len(selected)} 条，"
        f"去重 {deduped} 条。"
    )
    log_info(f"persona 分布: {dict(persona_counts)}")
    log_info(f"城市 Top 10: {city_counts.most_common(10)}")
    if skip_reasons:
        log_info(f"过滤原因统计: {dict(skip_reasons)}")
    log_info(f"输出文件: {output_path}")
    log_info(f"报告文件: {report_output_path}")
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 persona_understanding 原始数据严格清洗为高质量 ChatML。"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="persona_understanding 原始数据路径，支持 JSON/JSONL。",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="strict ChatML JSON 输出路径。",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help="清洗报告输出路径。",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=DEFAULT_TOTAL_SAMPLES,
        help="目标样本数，默认 500；设为 0 保留严格清洗后的全量。",
    )
    parser.add_argument(
        "--city-cap",
        type=int,
        default=DEFAULT_CITY_CAP,
        help="单城市最多保留条数，设为 0 不限制。",
    )
    parser.add_argument(
        "--city-persona-cap",
        type=int,
        default=DEFAULT_CITY_PERSONA_CAP,
        help="同一城市+画像组合最多保留条数，设为 0 不限制。",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_persona_understanding_data(
        args.input,
        args.output,
        report_path=args.report,
        total_samples=args.total_samples,
        city_cap=args.city_cap,
        city_persona_cap=args.city_persona_cap,
    )


if __name__ == "__main__":
    main()
