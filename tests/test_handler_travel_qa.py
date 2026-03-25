from __future__ import annotations

import json
from pathlib import Path
import shutil

from src.data_pipeline.handlers.handler_travel_qa import (
    build_travel_qa_sample,
    process_travel_qa_data,
)


def test_build_travel_qa_sample_adds_context_for_ambiguous_question() -> None:
    record = {
        "task_type": "spot_qa",
        "city": "\u6606\u660e",
        "entity_name": "\u4e91\u5357\u91ce\u751f\u52a8\u7269\u56ed",
        "entity_type": "spot",
        "question_type": "\u4eba\u7fa4\u9002\u914d",
        "is_time_sensitive": False,
        "user_query": "\u9002\u5408\u5e26\u5b69\u5b50\u53bb\u5417\uff1f",
        "assistant_content": "\u9002\u5408\uff0c\u56ed\u5185\u6709\u840c\u5ba0\u533a\u7b49\uff0c\u80fd\u8ba9\u5b69\u5b50\u8fd1\u8ddd\u79bb\u63a5\u89e6\u52a8\u7269\u3002",
    }

    sample = build_travel_qa_sample(record)

    assert sample is not None
    assert sample["task_type"] == "travel_qa"
    assert "\u53c2\u8003\u4fe1\u606f" in sample["messages"][1]["content"]
    assert "\u4e91\u5357\u91ce\u751f\u52a8\u7269\u56ed" in sample["messages"][1]["content"]


def test_build_travel_qa_sample_simplifies_traffic_answer() -> None:
    record = {
        "task_type": "traffic_qa",
        "city": "\u6606\u660e",
        "entity_name": "\u6606\u660e\u65c5\u6e38\u7d22\u9053",
        "entity_type": "traffic",
        "question_type": "\u4f4d\u7f6e\u4ea4\u901a",
        "is_time_sensitive": True,
        "user_query": "\u4ece\u6606\u660e\u5e02\u533a\u600e\u4e48\u53bb\u6606\u660e\u65c5\u6e38\u7d22\u9053\u6700\u65b9\u4fbf\uff1f",
        "assistant_content": (
            "\u4ece\u6606\u660e\u7ad9\u5230\u6606\u660e\u65c5\u6e38\u7d22\u9053\u516c\u5171\u4ea4\u901a\u7ea685\u5206\u949f\uff1b\u603b\u8ddd\u79bb\u7ea614.3\u516c\u91cc\uff1b"
            "\u53ef\u4e58\u5750\u5730\u94c12\u53f7\u7ebf\u300144\u8def\u7b49\u516c\u4ea4\u7ebf\u8def\u3002"
        ),
    }

    sample = build_travel_qa_sample(record)

    assert sample is not None
    assistant_content = sample["messages"][2]["content"]
    assert "85\u5206\u949f" not in assistant_content
    assert "14.3\u516c\u91cc" not in assistant_content
    assert "\u5730\u94c1\u6362\u4e58\u516c\u4ea4" in assistant_content
    assert "\u786e\u8ba4\u5f53\u65e5\u7ebf\u8def" in assistant_content


def test_build_travel_qa_sample_softens_time_sensitive_spot_answer() -> None:
    record = {
        "task_type": "spot_qa",
        "city": "\u91d1\u534e",
        "entity_name": "\u68a6\u5e7b\u8c37\u666f\u533a",
        "entity_type": "spot",
        "question_type": "\u6e38\u73a9\u5185\u5bb9",
        "is_time_sensitive": True,
        "user_query": "\u68a6\u5e7b\u8c37\u666f\u533a\u665a\u4e0a\u6709\u4ec0\u4e48\u6d3b\u52a8\uff1f",
        "assistant_content": "\u665a\u4e0a\u6709\u300a\u66b4\u96e8\u5c71\u6d2a\u300b\u8868\u6f14\uff0c\u6bcf\u59297\u70b9\u5de6\u53f3\u5f00\u59cb\uff0c\u6f14\u51fa\u65f6\u957f\u534a\u5c0f\u65f6\u5230\u4e00\u5c0f\u65f6\u4e0d\u7b49\u3002",
    }

    sample = build_travel_qa_sample(record)

    assert sample is not None
    assistant_content = sample["messages"][2]["content"]
    assert "7\u70b9" not in assistant_content
    assert "\u534a\u5c0f\u65f6" not in assistant_content
    assert "\u300a\u66b4\u96e8\u5c71\u6d2a\u300b\u8868\u6f14" in assistant_content
    assert "\u5f53\u65e5\u516c\u544a" in assistant_content
    assert "\u5b98\u65b9\u4fe1\u606f\u4e3a\u51c6" in assistant_content


def test_process_travel_qa_data_deduplicates_records() -> None:
    temp_dir = Path(".tmp_travel_qa_test")
    input_path = temp_dir / "travel_qa_raw.jsonl"
    output_path = temp_dir / "sft_travel_qa.json"
    record = {
        "record_id": "qa_1",
        "task_type": "city_qa",
        "city": "\u82cf\u5dde",
        "entity_name": "\u82cf\u5dde",
        "entity_type": "city",
        "question_type": "\u7279\u8272\u4eae\u70b9",
        "is_time_sensitive": False,
        "user_query": "\u82cf\u5dde\u6709\u4ec0\u4e48\u72ec\u7279\u7684\u9b45\u529b\uff1f",
        "assistant_content": "\u82cf\u5dde\u6709\u53e4\u5178\u56ed\u6797\u7684\u7cbe\u5de7\u96c5\u81f4\uff0c\u4e5f\u6709\u6c34\u5df7\u4ea4\u7ec7\u7684\u53e4\u9547\u98ce\u60c5\u3002",
    }

    temp_dir.mkdir(exist_ok=True)
    try:
        with input_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        dataset = process_travel_qa_data(str(input_path), str(output_path))

        assert len(dataset) == 1
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(saved) == 1
        assert saved[0]["task_type"] == "travel_qa"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
