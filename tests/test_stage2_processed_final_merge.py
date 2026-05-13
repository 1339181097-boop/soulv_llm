from __future__ import annotations

import json

from src.data_pipeline.finalize_stage2_processed import (
    Stage2ProcessedInput,
    build_final_dataset_from_records,
    finalize_stage2_processed,
)
from src.tool_use.datasets import validate_sharegpt_tool_dataset


def _function_call(name: str, arguments: dict) -> dict:
    return {
        "from": "function_call",
        "value": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False),
    }


def _observation(payload: dict | None = None) -> dict:
    return {
        "from": "observation",
        "value": json.dumps(payload or {"status": "success", "data": {"ok": True}}, ensure_ascii=False),
    }


def _base_record(
    sample_id: str,
    task_type: str,
    *,
    conversations: list[dict],
    scene: str = "amap_plan_route",
    subtype: str = "unit",
    expected_behavior: str = "should_call_tool",
) -> dict:
    return {
        "id": sample_id,
        "task_type": task_type,
        "scene": scene,
        "subtype": subtype,
        "expected_behavior": expected_behavior,
        "expected_tool_chain": [],
        "expected_arguments_subset": {},
        "quality_tags": [],
        "conversations": conversations,
    }


def _single_tool_record(sample_id: str = "single_001") -> dict:
    return _base_record(
        sample_id,
        "single_tool_call",
        conversations=[
            {"from": "system", "value": "system"},
            {"from": "human", "value": "route"},
            _function_call("amap_plan_route", {"origin": "A", "destination": "B"}),
            _observation(),
            {"from": "gpt", "value": "已为你整理好路线。"},
        ],
    )


def _no_tool_record(sample_id: str = "no_tool_001") -> dict:
    return _base_record(
        sample_id,
        "no_tool_needed",
        scene="travel_qa",
        expected_behavior="should_answer_directly",
        conversations=[
            {"from": "system", "value": "system"},
            {"from": "human", "value": "advice"},
            {"from": "gpt", "value": "建议放慢节奏，优先安排休息。"},
        ],
    )


def _inputs() -> tuple[Stage2ProcessedInput, ...]:
    return (
        Stage2ProcessedInput("single.jsonl", "single_tool_call", 2),
        Stage2ProcessedInput("no_tool.jsonl", "no_tool_needed", 2),
    )


def test_finalizer_injects_tools_and_outputs_sharegpt_shape() -> None:
    dataset, report = build_final_dataset_from_records(
        {
            "single_tool_call": [_single_tool_record()],
            "no_tool_needed": [_no_tool_record()],
        },
        inputs=_inputs(),
    )

    assert report["status"] == "passed"
    assert len(dataset) == 2
    assert validate_sharegpt_tool_dataset(dataset) == []
    assert all(set(item) == {"id", "task_type", "scene", "expected_behavior", "tools", "conversations"} for item in dataset)
    assert all(isinstance(json.loads(item["tools"]), list) for item in dataset)


def test_finalizer_keeps_shortfall_without_oversampling() -> None:
    dataset, report = build_final_dataset_from_records(
        {
            "single_tool_call": [_single_tool_record("single_only")],
            "no_tool_needed": [],
        },
        inputs=_inputs(),
    )

    assert len(dataset) == 1
    assert [item["id"] for item in dataset] == ["single_only"]
    assert report["target_total"] == 4
    assert report["shortfall_total"] == 3
    assert report["shortfall_by_task"] == {"single_tool_call": 1, "no_tool_needed": 2}


def test_finalizer_rejects_duplicate_ids() -> None:
    _, report = build_final_dataset_from_records(
        {
            "single_tool_call": [_single_tool_record("dup")],
            "no_tool_needed": [_no_tool_record("dup")],
        },
        inputs=_inputs(),
    )

    assert report["status"] == "failed"
    assert report["quality"]["duplicate_id_count"] == 1
    assert any("duplicates id" in error for error in report["validation"]["sample_errors"])


def test_finalizer_rejects_disallowed_two_step_chain() -> None:
    bad_chain = _base_record(
        "bad_chain",
        "single_tool_call",
        conversations=[
            {"from": "system", "value": "system"},
            {"from": "human", "value": "bad"},
            _function_call("amap_search_poi", {"keyword": "hotel"}),
            _observation(),
            _function_call("amap_plan_route", {"origin": "A", "destination": "B"}),
            _observation(),
            {"from": "gpt", "value": "done"},
        ],
    )

    _, report = build_final_dataset_from_records(
        {
            "single_tool_call": [bad_chain],
            "no_tool_needed": [_no_tool_record()],
        },
        inputs=_inputs(),
    )

    assert report["status"] == "failed"
    assert any("disallowed tool chain" in error for error in report["validation"]["sample_errors"])


def test_finalizer_rejects_no_tool_trace() -> None:
    bad_no_tool = _base_record(
        "bad_no_tool",
        "no_tool_needed",
        scene="travel_qa",
        expected_behavior="should_answer_directly",
        conversations=[
            {"from": "system", "value": "system"},
            {"from": "human", "value": "advice"},
            _function_call("amap_geocode", {"address": "A"}),
            _observation(),
            {"from": "gpt", "value": "answer"},
        ],
    )

    _, report = build_final_dataset_from_records(
        {
            "single_tool_call": [_single_tool_record()],
            "no_tool_needed": [bad_no_tool],
        },
        inputs=_inputs(),
    )

    assert report["status"] == "failed"
    assert any("no_tool_needed contains tool trace" in error for error in report["validation"]["sample_errors"])


def test_finalizer_allows_altitude_meter_context() -> None:
    altitude_no_tool = _base_record(
        "altitude_no_tool",
        "no_tool_needed",
        scene="travel_qa",
        expected_behavior="should_answer_directly",
        conversations=[
            {"from": "system", "value": "system"},
            {"from": "human", "value": "family travel"},
            {"from": "gpt", "value": "当地海拔较高，平均3000米以上，建议先适应。"},
        ],
    )

    _, report = build_final_dataset_from_records(
        {
            "single_tool_call": [_single_tool_record()],
            "no_tool_needed": [altitude_no_tool],
        },
        inputs=_inputs(),
    )

    assert report["status"] == "passed"


def test_finalizer_writes_valid_files_from_jsonl_inputs(tmp_path) -> None:
    processed_dir = tmp_path / "processed_stage2"
    processed_dir.mkdir()
    for filename, record in (
        ("single.jsonl", _single_tool_record()),
        ("no_tool.jsonl", _no_tool_record()),
    ):
        (processed_dir / filename).write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    output_path = tmp_path / "final" / "stage2_amap_tool_use_sft.json"
    report_path = tmp_path / "final" / "stage2_amap_tool_use_report.json"
    dataset, report = finalize_stage2_processed(
        processed_dir=processed_dir,
        output_path=output_path,
        report_path=report_path,
        inputs=_inputs(),
    )

    assert report["status"] == "passed"
    assert report["output_written"] is True
    assert output_path.exists()
    assert report_path.exists()
    assert validate_sharegpt_tool_dataset(dataset) == []
