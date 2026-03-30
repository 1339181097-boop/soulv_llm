from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.handlers.handler_hotel_recommendation import (
    build_hotel_recommendation_sample,
    process_hotel_recommendation_data,
)


def test_build_hotel_recommendation_sample_adds_context_for_generic_query() -> None:
    record = {
        "record_id": "hotel_1",
        "task_type": "hotel_recommendation",
        "source": "tripai_hotel",
        "source_id": 1001,
        "city": "北京",
        "district": "大兴区",
        "user_query": "机场附近住哪里方便中转？",
        "assistant_content": (
            "如果主要目的是在大兴机场中转，建议优先选择靠近机场航站楼或礼贤镇一带的住宿。"
            "这类位置往返机场更省时间，也更适合短暂停留，能减少夜间抵达或赶早班机时的通勤压力。"
            "但这一区域远离市中心，生活配套和景点资源都比较有限，不太适合需要频繁进出城区或安排深度游的行程。"
        ),
        "hotel_tags": ["交通便利", "中转方便"],
        "audience": ["中转"],
        "budget_level": "medium",
        "updated_at": "2026-03-27",
        "hotel_name": "北京大兴机场某酒店",
        "location_desc": "靠近机场",
        "suitable_for": ["transit_stop"],
        "not_suitable_for": ["city_explore"],
        "reason_text": "靠近机场，适合短停。",
        "travel_style": "transit_stop",
        "query_intent": "区域选择",
        "question_mode": "open_recommendation",
    }

    sample = build_hotel_recommendation_sample(record)

    assert sample is not None
    assert "参考信息：" in sample["messages"][1]["content"]
    assert "城市：北京" in sample["messages"][1]["content"]
    assert "区域：大兴区" in sample["messages"][1]["content"]


def test_build_hotel_recommendation_sample_rejects_booking_noise() -> None:
    record = {
        "record_id": "hotel_2",
        "task_type": "hotel_recommendation",
        "source": "tripai_hotel",
        "source_id": 1002,
        "city": "三亚",
        "district": "海棠区",
        "user_query": "海棠湾附近亲子住哪里合适？",
        "assistant_content": "XX酒店今日价格899元，剩余2间，立即预订最划算。",
        "hotel_tags": ["亲子", "度假"],
        "audience": ["亲子"],
        "budget_level": "medium_high",
        "updated_at": "2026-03-27",
        "hotel_name": "XX酒店",
        "location_desc": "海棠湾",
        "suitable_for": ["family_resort"],
        "not_suitable_for": ["business_commute"],
        "reason_text": "亲子度假",
        "travel_style": "family_resort",
        "query_intent": "酒店类型选择",
        "question_mode": "open_recommendation",
    }

    assert build_hotel_recommendation_sample(record) is None


def test_process_hotel_recommendation_data_dedupes_same_query() -> None:
    temp_dir = Path(".tmp_hotel_recommendation_test")
    input_path = temp_dir / "hotel_recommendation_raw.jsonl"
    output_path = temp_dir / "sft_hotel_recommendation.json"

    keep_record = {
        "record_id": "hotel_keep",
        "task_type": "hotel_recommendation",
        "source": "tripai_hotel",
        "source_id": 2001,
        "city": "北京",
        "district": "大兴区",
        "user_query": "大兴机场附近住哪里方便中转？",
        "assistant_content": (
            "如果主要目的是在大兴机场中转，建议优先选择靠近机场航站楼或礼贤镇一带的住宿。"
            "这类位置往返机场更省时间，也更适合短暂停留，能减少赶早班机或深夜抵达时的折腾。"
            "但这一区域远离市中心，后续若想兼顾市区游览或高频通勤，就需要预留更长交通时间。"
        ),
        "hotel_tags": ["交通便利", "中转方便"],
        "audience": ["中转"],
        "budget_level": "medium",
        "updated_at": "2026-03-27",
        "hotel_name": "酒店A",
        "location_desc": "机场附近",
        "suitable_for": ["transit_stop"],
        "not_suitable_for": ["city_explore"],
        "reason_text": "靠近机场，适合短停。",
        "travel_style": "transit_stop",
        "query_intent": "区域选择",
        "question_mode": "open_recommendation",
    }
    drop_record = {
        "record_id": "hotel_drop",
        "task_type": "hotel_recommendation",
        "source": "tripai_hotel",
        "source_id": 2002,
        "city": "北京",
        "district": "大兴区",
        "user_query": "大兴机场附近住哪里方便中转？",
        "assistant_content": (
            "如果主要是机场中转，也可以住在机场周边功能区。"
            "位置确实方便，适合短停，也更容易衔接次日出发。"
            "但整体远离市区，生活配套较弱，不适合把它当作北京深度游的落脚点。"
        ),
        "hotel_tags": ["交通便利", "中转方便"],
        "audience": ["中转"],
        "budget_level": "low",
        "updated_at": "2026-03-27",
        "hotel_name": "酒店B",
        "location_desc": "机场附近",
        "suitable_for": ["transit_stop"],
        "not_suitable_for": ["city_explore"],
        "reason_text": "靠近机场，适合短停。",
        "travel_style": "transit_stop",
        "query_intent": "区域选择",
        "question_mode": "open_recommendation",
    }

    temp_dir.mkdir(exist_ok=True)
    try:
        with input_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(keep_record, ensure_ascii=False) + "\n")
            file.write(json.dumps(drop_record, ensure_ascii=False) + "\n")

        dataset = process_hotel_recommendation_data(
            str(input_path),
            str(output_path),
            total_samples=0,
            city_cap=0,
            query_cap=1,
        )

        assert len(dataset) == 1
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(saved) == 1
        assert saved[0]["task_type"] == "hotel_recommendation"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
