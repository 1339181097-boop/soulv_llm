from __future__ import annotations

import json
import random
from pathlib import Path

from src.data_pipeline.data_mixer import (
    DatasetBucket,
    StageInput,
    StageRecipe,
    _load_dataset_file,
    _max_consecutive_task,
    _resolve_target_counts,
    _sample_records,
    build_stage_dataset_from_records,
)

TMP_ROOT = Path("data/test_tmp_data_mixer")


class CharTokenCounter:
    def count_text_tokens(self, text: str) -> int:
        return len(text)

    def count_chat_tokens(self, messages: list[dict]) -> int:
        total = 0
        for message in messages:
            total += len(str(message.get("role", "")))
            total += len(str(message.get("content", "")))
        return total


def _sample(task_type: str, index: int, *, assistant_length: int = 20, extra: dict | None = None) -> dict:
    payload = {
        "id": f"{task_type}_{index}",
        "record_id": f"source_{task_type}_{index}",
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": "你是旅行助手。"},
            {"role": "user", "content": f"{task_type} question {index}"},
            {"role": "assistant", "content": "答" * assistant_length},
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def _records(task_type: str, count: int, *, assistant_length: int = 20) -> list[dict]:
    return [_sample(task_type, index, assistant_length=assistant_length) for index in range(count)]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _case_dir(name: str) -> Path:
    path = TMP_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_resolve_target_counts_matches_requested_total() -> None:
    buckets = [
        DatasetBucket(filename="a.jsonl", weight=0.5, records=[]),
        DatasetBucket(filename="b.jsonl", weight=0.3, records=[]),
        DatasetBucket(filename="c.jsonl", weight=0.2, records=[]),
    ]

    counts = _resolve_target_counts(buckets, 19)

    assert sum(counts.values()) == 19
    assert counts["a.jsonl"] == 9
    assert counts["b.jsonl"] == 6
    assert counts["c.jsonl"] == 4


def test_sample_records_oversamples_only_for_ad_hoc_mixing() -> None:
    records = [{"id": index} for index in range(3)]
    sampled, duplicates = _sample_records(records, 8, random.Random(7))

    assert len(sampled) == 8
    assert duplicates == 5
    assert sorted({item["id"] for item in sampled}) == [0, 1, 2]


def test_load_dataset_file_accepts_jsonl() -> None:
    path = _case_dir("load_jsonl") / "samples.jsonl"
    expected = _records("travel_qa", 2, assistant_length=8)
    _write_jsonl(path, expected)

    loaded = _load_dataset_file(path)

    assert loaded == expected


def test_stage_build_writes_gated_interleaved_dataset() -> None:
    tmp_path = _case_dir("stage_pass")
    inputs = (
        StageInput("guide.jsonl", "guide_generation", 2),
        StageInput("travel.jsonl", "travel_qa", 4),
        StageInput("hotel.jsonl", "hotel_recommendation", 3),
        StageInput("traffic.jsonl", "traffic_planning", 3),
        StageInput("persona.jsonl", "persona_understanding", 2),
        StageInput("multi.jsonl", "multi_turn_dialogue", 3, stratify_by_tokens=True),
    )
    recipe = StageRecipe(
        name="test_stage",
        output_path=str(tmp_path / "stage.json"),
        seed=42,
        inputs=inputs,
        max_consecutive_task=3,
        tokenizer_path="unused",
    )
    records_by_filename = {
        "guide.jsonl": _records("guide_generation", 2, assistant_length=30),
        "travel.jsonl": _records("travel_qa", 4, assistant_length=10),
        "hotel.jsonl": _records("hotel_recommendation", 3, assistant_length=20),
        "traffic.jsonl": _records("traffic_planning", 3, assistant_length=20),
        "persona.jsonl": _records("persona_understanding", 2, assistant_length=20),
        "multi.jsonl": _records("multi_turn_dialogue", 6, assistant_length=15),
    }
    output_path = tmp_path / "stage.json"
    report_path = tmp_path / "report.json"
    output_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)

    mixed, report = build_stage_dataset_from_records(
        recipe=recipe,
        records_by_filename=records_by_filename,
        output_json_path=output_path,
        report_path=report_path,
        tokenizer=CharTokenCounter(),
    )

    assert output_path.exists()
    assert report_path.exists()
    assert report["status"] == "passed"
    assert report["output_written"] is True
    assert len(mixed) == 17
    assert report["quality"]["task_type_counts"] == {
        "guide_generation": 2,
        "travel_qa": 4,
        "hotel_recommendation": 3,
        "traffic_planning": 3,
        "persona_understanding": 2,
        "multi_turn_dialogue": 3,
    }
    assert _max_consecutive_task(mixed)["count"] <= 3
    assert all(file_report["oversample_count"] == 0 for file_report in report["selection"]["files"].values())
    assert report["selection"]["files"]["multi.jsonl"]["selection_mode"] == "assistant_token_budget"


def test_stage_build_fails_token_gate_without_writing_output() -> None:
    tmp_path = _case_dir("token_fail")
    inputs = (
        StageInput("guide.jsonl", "guide_generation", 2),
        StageInput("travel.jsonl", "travel_qa", 4),
        StageInput("hotel.jsonl", "hotel_recommendation", 3),
        StageInput("traffic.jsonl", "traffic_planning", 3),
        StageInput("persona.jsonl", "persona_understanding", 2),
        StageInput("multi.jsonl", "multi_turn_dialogue", 3),
    )
    recipe = StageRecipe(
        name="bad_tokens",
        output_path=str(tmp_path / "bad.json"),
        seed=42,
        inputs=inputs,
        tokenizer_path="unused",
    )
    records_by_filename = {
        "guide.jsonl": _records("guide_generation", 2, assistant_length=500),
        "travel.jsonl": _records("travel_qa", 4, assistant_length=2),
        "hotel.jsonl": _records("hotel_recommendation", 3, assistant_length=5),
        "traffic.jsonl": _records("traffic_planning", 3, assistant_length=5),
        "persona.jsonl": _records("persona_understanding", 2, assistant_length=5),
        "multi.jsonl": _records("multi_turn_dialogue", 3, assistant_length=5),
    }
    output_path = tmp_path / "bad.json"
    output_path.unlink(missing_ok=True)

    _, report = build_stage_dataset_from_records(
        recipe=recipe,
        records_by_filename=records_by_filename,
        output_json_path=output_path,
        tokenizer=CharTokenCounter(),
    )

    assert not output_path.exists()
    assert report["status"] == "failed"
    assert report["output_written"] is False
    assert report["token_gates"]["passed"] is False
    assert report["token_gates"]["checks"]["guide_assistant_share"]["passed"] is False


def test_stage_build_fails_quality_gate_for_tool_trace() -> None:
    tmp_path = _case_dir("quality_fail")
    inputs = (
        StageInput("guide.jsonl", "guide_generation", 1),
        StageInput("travel.jsonl", "travel_qa", 1),
        StageInput("hotel.jsonl", "hotel_recommendation", 1),
        StageInput("traffic.jsonl", "traffic_planning", 1),
        StageInput("persona.jsonl", "persona_understanding", 1),
        StageInput("multi.jsonl", "multi_turn_dialogue", 1),
    )
    recipe = StageRecipe(
        name="bad_quality",
        output_path=str(tmp_path / "bad_quality.json"),
        seed=42,
        inputs=inputs,
        tokenizer_path="unused",
    )
    bad_guide = _sample(
        "guide_generation",
        0,
        assistant_length=20,
        extra={"tool_calls": [{"name": "should_not_be_here"}]},
    )
    records_by_filename = {
        "guide.jsonl": [bad_guide],
        "travel.jsonl": _records("travel_qa", 1, assistant_length=20),
        "hotel.jsonl": _records("hotel_recommendation", 1, assistant_length=20),
        "traffic.jsonl": _records("traffic_planning", 1, assistant_length=20),
        "persona.jsonl": _records("persona_understanding", 1, assistant_length=20),
        "multi.jsonl": _records("multi_turn_dialogue", 1, assistant_length=20),
    }
    output_path = tmp_path / "bad_quality.json"
    output_path.unlink(missing_ok=True)

    _, report = build_stage_dataset_from_records(
        recipe=recipe,
        records_by_filename=records_by_filename,
        output_json_path=output_path,
        tokenizer=CharTokenCounter(),
    )

    assert not output_path.exists()
    assert report["status"] == "failed"
    assert report["quality"]["passed"] is False
    assert report["quality"]["strict_error_count"] == 1
