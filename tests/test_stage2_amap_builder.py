from __future__ import annotations

from src.data_pipeline.build_stage2_amap_tool_use import (
    _build_poi_snapshot,
    _build_route_snapshot,
    _compute_target_counts,
    _is_safe_no_tool_record,
    _poi_answer_from_snapshot,
    _route_answer_from_snapshot,
    _semantic_validation_errors,
    _travel_no_tool,
    build_dataset,
)


def test_compute_target_counts_matches_expected_stage2_mix() -> None:
    counts = _compute_target_counts(1600)

    assert sum(counts.values()) == 1600
    assert counts["single_tool_call"] == 560
    assert counts["slot_filling_tool_call"] == 320
    assert counts["clarify_then_call"] == 240
    assert counts["tool_result_grounded_answer"] == 160
    assert counts["no_tool_needed"] == 160
    assert counts["tool_failure_fallback"] == 160


def test_travel_no_tool_builder_preserves_direct_answer() -> None:
    record = {
        "id": "travel_001",
        "city": "杭州",
        "entity_name": "西湖",
        "messages": [
            {"role": "system", "content": "你是 TripAI 旅行助手。"},
            {"role": "user", "content": "西湖适合带老人慢慢逛吗？"},
            {"role": "assistant", "content": "适合，节奏可以放慢一些，沿湖走走会比较舒服。"},
        ],
    }

    sample = _travel_no_tool(record, None)

    assert sample["task_type"] == "no_tool_needed"
    assert sample["expected_behavior"] == "should_answer_directly"
    assert sample["messages_with_answer"][-1]["content"] == "适合，节奏可以放慢一些，沿湖走走会比较舒服。"


def test_route_answer_is_grounded_to_summary_and_step() -> None:
    snapshot = _build_route_snapshot("北京南站", "颐和园", "北京", "transit")
    answer = _route_answer_from_snapshot(snapshot)

    assert snapshot["data"]["summary"] in answer
    assert snapshot["data"]["route_steps"][0]["instruction"] in answer


def test_poi_answer_is_grounded_to_count_and_top_name() -> None:
    snapshot = _build_poi_snapshot("restaurant", "杭州", anchor_label="西湖")
    answer = _poi_answer_from_snapshot(snapshot)

    assert str(snapshot["data"]["count"]) in answer
    assert snapshot["data"]["pois"][0]["name"] in answer


def test_safe_no_tool_filter_rejects_route_boundary_question() -> None:
    record = {
        "id": "travel_boundary_001",
        "question_type": "位置交通",
        "entity_type": "spot",
        "is_time_sensitive": False,
        "messages": [
            {"role": "system", "content": "你是 TripAI 旅行助手。"},
            {"role": "user", "content": "从南京市区怎么去明孝陵最方便？"},
            {"role": "assistant", "content": "更建议查路线工具。"},
        ],
    }

    assert not _is_safe_no_tool_record(record)


def test_full_build_has_diverse_modes_keywords_and_passes_semantic_checks() -> None:
    dataset, report = build_dataset(1600, 42)
    summary = report["semantic_summary"]

    assert _semantic_validation_errors(dataset) == []
    assert set(summary["route_mode_distribution"]) == {"transit", "driving", "walking", "bicycling"}
    assert len(summary["search_keyword_distribution"]) >= 4
    assert summary["envelope_status_distribution"]["empty"] > 0
    assert summary["envelope_status_distribution"]["error"] > 0
    assert summary["clarify_first_turn_unique_count"] >= 5
