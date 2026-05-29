from __future__ import annotations

from src.data_pipeline.dedup_badcase_repairs import deduplicate_records


def _record(user: str, answer: str, *, repair_type: str = "paraphrase") -> dict:
    return {
        "source_badcase_id": "case_001",
        "repair_type": repair_type,
        "task_type": "travel_qa",
        "messages": [{"role": "user", "content": user}],
        "answer": answer,
    }


def test_deduplicate_records_rejects_exact_hash_duplicate() -> None:
    records = [
        _record("How should I plan a two-day Beijing trip?", "Keep the route light and verify live ticket details."),
        _record("How should I plan a two-day Beijing trip?", "Keep the route light and verify live ticket details."),
    ]

    kept, rejected, summary = deduplicate_records(records, skip_similarity=True)

    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0]["_dedup"]["reason"] == "exact_hash"
    assert summary["rejection_reasons"] == {"exact_hash": 1}


def test_deduplicate_records_rejects_near_duplicate_by_similarity() -> None:
    answer = (
        "Day one should focus on the nearby old town and riverfront. "
        "Day two should keep one museum plus one relaxed food area. "
        "Do not invent exact opening hours or ticket prices."
    )
    records = [
        _record("Please arrange a relaxed two-day Beijing route for a first-time visitor.", answer),
        _record("Please arrange a relaxed two-day Beijing route for first time visitors.", answer, repair_type="paraphrase_2"),
    ]

    kept, rejected, summary = deduplicate_records(records, similarity_threshold=0.9)

    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0]["_dedup"]["reason"] == "similarity"
    assert summary["rejection_reasons"] == {"similarity": 1}
