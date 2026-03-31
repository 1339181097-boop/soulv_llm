from __future__ import annotations

import random

from src.data_pipeline.data_mixer import (
    DatasetBucket,
    _build_bucket_audit,
    _choose_assistant_aware_target_counts,
    _resolve_target_counts,
    _sample_records,
)


def test_resolve_target_counts_matches_requested_total() -> None:
    buckets = [
        DatasetBucket(filename="a.json", weight=0.5, records=[]),
        DatasetBucket(filename="b.json", weight=0.3, records=[]),
        DatasetBucket(filename="c.json", weight=0.2, records=[]),
    ]

    counts = _resolve_target_counts(buckets, 19)

    assert sum(counts.values()) == 19
    assert counts["a.json"] == 9
    assert counts["b.json"] == 6
    assert counts["c.json"] == 4


def test_sample_records_oversamples_in_full_cycles() -> None:
    records = [{"id": index} for index in range(3)]
    sampled, duplicates = _sample_records(records, 8, random.Random(7))

    assert len(sampled) == 8
    assert duplicates == 5
    ids = [item["id"] for item in sampled]
    assert sorted(set(ids)) == [0, 1, 2]


def test_choose_assistant_aware_target_counts_keeps_docs_baseline_when_assistant_share_is_controlled() -> None:
    guide_records = [
        {
            "task_type": "guide_generation",
            "messages": [
                {"role": "user", "content": "plan trip"},
                {"role": "assistant", "content": "a" * 300},
            ],
        }
        for _ in range(4)
    ]
    qa_records = [
        {
            "task_type": "travel_qa",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "b" * 220},
            ],
        }
        for _ in range(4)
    ]
    multi_records = [
        {
            "task_type": "multi_turn_dialogue",
            "messages": [
                {"role": "user", "content": "follow up"},
                {"role": "assistant", "content": "c" * 120},
            ],
        }
        for _ in range(4)
    ]
    buckets = [
        DatasetBucket("sft_guide_generation.json", 1.0, guide_records),
        DatasetBucket("sft_travel_qa.json", 1.0, qa_records),
        DatasetBucket("sft_multi_turn_dialogue.json", 1.0, multi_records),
    ]
    baseline_counts = {
        "sft_guide_generation.json": 1,
        "sft_travel_qa.json": 4,
        "sft_multi_turn_dialogue.json": 1,
    }

    audits = _build_bucket_audit(buckets, baseline_counts)
    final_counts, strategy = _choose_assistant_aware_target_counts(audits, baseline_counts)

    assert final_counts == baseline_counts
    assert strategy["mode"] == "docs_baseline"


def test_choose_assistant_aware_target_counts_switches_when_assistant_share_is_too_high() -> None:
    baseline_counts = {
        "sft_guide_generation.json": 800,
        "sft_travel_qa.json": 1000,
        "sft_hotel_recommendation.json": 750,
        "sft_traffic_planning.json": 500,
        "sft_persona_understanding_strict.json": 500,
        "sft_multi_turn_dialogue.json": 500,
    }
    audits = {
        "sft_guide_generation.json": {"projected_assistant_share": 0.62},
        "sft_multi_turn_dialogue.json": {"projected_assistant_share": 0.07},
    }

    final_counts, strategy = _choose_assistant_aware_target_counts(audits, baseline_counts)

    assert final_counts["sft_guide_generation.json"] == 250
    assert final_counts["sft_travel_qa.json"] == 1250
    assert final_counts["sft_traffic_planning.json"] == 794
    assert final_counts["sft_multi_turn_dialogue.json"] == 350
    assert strategy["mode"] == "assistant_aware_correction"

