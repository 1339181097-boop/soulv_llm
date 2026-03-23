from __future__ import annotations

import random

from src.data_pipeline.data_mixer import DatasetBucket, _resolve_target_counts, _sample_records


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
