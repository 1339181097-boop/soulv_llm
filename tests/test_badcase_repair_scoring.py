from __future__ import annotations

from src.data_pipeline.score_badcase_repairs_with_dashscope import (
    _extract_json_object,
    _filter_status,
    _normalize_judge_payload,
    _resolve_chat_completions_url,
)


def test_resolve_chat_completions_url_accepts_base_or_full_url() -> None:
    assert (
        _resolve_chat_completions_url("https://dashscope.aliyuncs.com/compatible-mode/v1")
        == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    assert (
        _resolve_chat_completions_url("https://example.test/v1/chat/completions")
        == "https://example.test/v1/chat/completions"
    )


def test_extract_json_object_handles_fenced_response() -> None:
    payload = _extract_json_object('```json\n{"overall_score": 4, "confidence": 0.82}\n```')

    assert payload == {"overall_score": 4, "confidence": 0.82}


def test_normalize_judge_payload_and_filter_status() -> None:
    normalized = _normalize_judge_payload(
        {
            "correctness": 5,
            "instruction_following": 4,
            "completeness": 4,
            "actionability": 4,
            "safety_and_honesty": 5,
            "information_control": 4,
            "language_quality": 4,
            "training_value": 4,
            "overall_score": "4",
            "confidence": 85,
            "verdict": "keep",
            "issue_tags": [],
            "judge_reason": "usable",
        }
    )

    status, reasons = _filter_status(normalized, min_score=4, min_confidence=0.7)

    assert normalized["confidence"] == 0.85
    assert normalized["scores"]["overall_score"] == 4
    assert status == "keep"
    assert reasons == []


def test_filter_status_sends_low_confidence_to_review() -> None:
    normalized = _normalize_judge_payload(
        {
            "correctness": 5,
            "instruction_following": 5,
            "completeness": 5,
            "actionability": 5,
            "safety_and_honesty": 5,
            "information_control": 5,
            "language_quality": 5,
            "training_value": 5,
            "overall_score": 5,
            "confidence": 0.4,
            "verdict": "keep",
        }
    )

    status, reasons = _filter_status(normalized, min_score=4, min_confidence=0.7)

    assert status == "review"
    assert "low_confidence" in reasons
