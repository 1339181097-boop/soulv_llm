from __future__ import annotations

from src.data_pipeline.multi_turn_dialogue_round2_cleaner import clean_round2_sample, classify_round2_filter_reason


def _sample(*, later_user: str, constraint: str, assistant: str = "已根据新条件调整，节奏更轻松，也保留核心体验。") -> dict:
    return {
        "id": "sample",
        "record_id": "sample",
        "task_type": "multi_turn_dialogue",
        "source": "tripai_db",
        "source_id": "conv_sample",
        "updated_at": "2026-04-22",
        "constraint_changes": [
            "初始需求：参观西湖",
            constraint,
            "同行人员由2人变为2大1小，含一名6岁儿童",
        ],
        "messages": [
            {"role": "system", "content": "你是专业的中文旅行规划助手。"},
            {"role": "user", "content": "我想去杭州玩，主要想看看西湖"},
            {"role": "assistant", "content": "可以先围绕西湖安排半日到一日游。"},
            {"role": "user", "content": later_user},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "增加一位6岁孩子，减少连续步行"},
            {"role": "assistant", "content": "那就增加手划船和短距离停靠点，避免连续走完整条白堤。"},
        ],
    }


def test_round2_repairs_exact_generic_user_turn() -> None:
    sample = _sample(
        later_user="请调整行程",
        constraint="行程调整（增加或减少景点）：增加浙江省博物馆孤山馆区作为同日人文补充",
    )

    cleaned, reason, edits = clean_round2_sample(sample)

    assert reason is None
    assert cleaned is not None
    assert "repaired_user_constraint_turn" in edits
    assert cleaned["messages"][3]["content"] == "请调整行程：增加浙江省博物馆孤山馆区作为同日人文补充。"


def test_round2_drops_unrepairable_placeholder_constraint() -> None:
    sample = _sample(
        later_user="请调整行程",
        constraint="补充约束：预算调整（增加或减少）",
    )

    assert classify_round2_filter_reason(sample) == "unrepairable_pseudo_user_turn"


def test_round2_drops_wrapped_placeholder_user_turn() -> None:
    sample = _sample(
        later_user="住宿需求调整为：（变更住宿地点或标准）。",
        constraint="补充约束：（变更住宿地点或标准）",
    )

    assert classify_round2_filter_reason(sample) == "unrepairable_pseudo_user_turn"


def test_round2_repairs_slot_turn_with_inline_detail() -> None:
    sample = _sample(
        later_user="（变更交通方式为地铁+度假区免费穿梭巴士），再调整一下",
        constraint="补充约束：交通调整（变更交通方式为地铁+度假区免费穿梭巴士）",
    )

    cleaned, reason, edits = clean_round2_sample(sample)

    assert reason is None
    assert cleaned is not None
    assert "repaired_user_constraint_turn" in edits
    assert cleaned["messages"][3]["content"] == "交通方式调整为：地铁+度假区免费穿梭巴士。"


def test_round2_rebuilds_constraint_changes_from_user_turns() -> None:
    sample = _sample(
        later_user="预算想控制得更紧一些，住宿和餐饮都希望经济实惠",
        constraint="补充约束：预算调整（增加或减少）",
    )

    cleaned, reason, edits = clean_round2_sample(sample)

    assert reason is None
    assert cleaned is not None
    assert "rebuilt_constraint_changes" in edits
    assert cleaned["constraint_changes"][1] == "预算想控制得更紧一些，住宿和餐饮都希望经济实惠"


def test_round2_strips_generic_slot_when_detail_exists() -> None:
    sample = _sample(
        later_user="（更注重人文/自然/美食/购物等），再调整一下：我们特别想深入了解西湖诗词和白堤故事",
        constraint="偏好调整：更注重西湖诗词、白堤故事和人文讲解",
    )

    cleaned, reason, edits = clean_round2_sample(sample)

    assert reason is None
    assert cleaned is not None
    assert "repaired_user_constraint_turn" in edits
    assert cleaned["messages"][3]["content"] == "偏好调整：我们特别想深入了解西湖诗词和白堤故事。"


def test_round2_drops_ticket_or_hours_content() -> None:
    sample = _sample(
        later_user="改成两天一晚，节奏慢一点",
        constraint="时间调整：由半日游调整为两天一晚",
        assistant="第一天游西湖，第二天看博物馆。门票和开放时间建议以当天官方信息为准。",
    )

    assert classify_round2_filter_reason(sample) == "realtime_ticket_or_transaction"


def test_round2_drops_bad_alternation() -> None:
    sample = _sample(
        later_user="改成两天一晚，节奏慢一点",
        constraint="时间调整：由半日游调整为两天一晚",
    )
    sample["messages"].insert(3, {"role": "assistant", "content": "多余回答"})

    assert classify_round2_filter_reason(sample) == "bad_alternation"
