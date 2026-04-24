from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.handlers.handler_persona_understanding import (
    build_persona_understanding_sample,
    process_persona_understanding_data,
)


def test_build_persona_understanding_sample_rewrites_from_candidate_facts() -> None:
    record = {
        "record_id": "persona_1",
        "task_type": "persona_understanding",
        "city": "北京",
        "persona_type": "情侣",
        "user_query": "情侣去北京玩，想找浪漫又适合拍照的地方，有什么推荐？",
        "audience": ["情侣"],
        "preference_tags": ["浪漫", "拍照", "私密", "高品质体验"],
        "avoid_tags": ["拥挤", "嘈杂", "儿童区"],
        "budget_level": "high",
        "people_count": 2,
        "candidate_spots": [
            {
                "name": "景点A",
                "price": "门票¥40起",
                "address": "北京市A路1号",
                "open_hours": "09:00-18:00",
                "tags": ["浪漫、拍照打卡、观景、安静"],
                "brief": "适合慢慢看风景，整体氛围安静。",
            },
            {
                "name": "景点B",
                "price": "门票¥220起",
                "address": "北京市B路2号",
                "open_hours": "09:00-21:00",
                "tags": ["高品质体验、夜景、观景"],
                "brief": "更偏品质型体验。",
            },
            {
                "name": "景点C",
                "price": "免费开放",
                "address": "北京市C路3号",
                "open_hours": "全天开放",
                "tags": ["拍照打卡、古建、文化体验"],
                "brief": "适合拍照和慢逛。",
            },
            {
                "name": "景点D",
                "price": "门票¥80起",
                "address": "北京市D路4号",
                "open_hours": "10:00-22:00",
                "tags": ["儿童友好、亲子、互动体验"],
                "brief": "更适合亲子。",
            },
        ],
        "assistant_content": "这条旧文案会被重写。",
        "reason_text": "旧逻辑也会被重写。",
    }

    sample, reason = build_persona_understanding_sample(record)

    assert reason == "ok"
    assert sample is not None
    assistant = sample["messages"][2]["content"]
    assert "景点A" in assistant
    assert "景点B" in assistant
    assert "景点C" in assistant
    assert "景点D" in assistant
    assert "推荐仅依据候选景点的标签、价格和简介中可稳定支持的信息生成" in assistant


def test_build_persona_understanding_sample_rejects_too_few_candidates() -> None:
    record = {
        "record_id": "persona_2",
        "task_type": "persona_understanding",
        "city": "林芝",
        "persona_type": "亲子",
        "user_query": "带孩子去林芝玩有什么推荐？",
        "audience": ["亲子"],
        "preference_tags": ["互动体验", "儿童友好"],
        "avoid_tags": ["高强度活动"],
        "budget_level": "medium",
        "people_count": 3,
        "candidate_spots": [
            {"name": "景点A", "price": "免费", "tags": ["亲子、自然风景"], "brief": ""},
            {"name": "景点B", "price": "门票¥80起", "tags": ["儿童友好、文化体验"], "brief": ""},
        ],
    }

    sample, reason = build_persona_understanding_sample(record)

    assert sample is None
    assert reason == "candidate_spots_lt3"


def test_process_persona_understanding_data_writes_chatml_json() -> None:
    temp_dir = Path(".tmp_persona_understanding_test")
    input_path = temp_dir / "persona_raw.jsonl"
    output_path = temp_dir / "sft_persona.json"
    report_path = temp_dir / "persona_report.json"

    base_record = {
        "task_type": "persona_understanding",
        "city": "杭州",
        "persona_type": "摄影爱好者",
        "user_query": "摄影爱好者去杭州玩，想找拍照和观景都不错的地方。",
        "audience": ["摄影爱好者"],
        "preference_tags": ["拍照", "夜景", "观景"],
        "avoid_tags": ["无特色"],
        "budget_level": "medium",
        "people_count": 1,
        "candidate_spots": [
            {"name": "景点A", "price": "免费开放", "tags": ["拍照打卡、观景、夜景"], "brief": "适合看城市夜景和拍照。"},
            {"name": "景点B", "price": "门票¥50起", "tags": ["古建、拍照打卡、文化体验"], "brief": "古建氛围明显，适合慢慢拍照。"},
            {"name": "景点C", "price": "门票¥30起", "tags": ["观景、自然风景"], "brief": "视野开阔，整体节奏轻松。"},
            {"name": "景点D", "price": "门票¥120起", "tags": ["高空刺激、探险"], "brief": "活动强度偏高。"},
        ],
    }

    temp_dir.mkdir(exist_ok=True)
    try:
        with input_path.open("w", encoding="utf-8") as file:
            for index in range(2):
                record = dict(base_record)
                record["record_id"] = f"persona_{index}"
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

        dataset = process_persona_understanding_data(
            str(input_path),
            str(output_path),
            report_path=str(report_path),
            total_samples=0,
            city_cap=0,
            city_persona_cap=1,
        )

        assert len(dataset) == 1
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(saved) == 1
        assert saved[0]["task_type"] == "persona_understanding"
        assert json.loads(report_path.read_text(encoding="utf-8"))["final_count"] == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
