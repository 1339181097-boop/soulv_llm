from __future__ import annotations

from src.data_pipeline.dataset_audit import audit_dataset_file, summarize_dataset


def test_summarize_dataset_counts_lengths_and_sources() -> None:
    dataset = [
        {
            "source": "pseudo_real_dialogue",
            "messages": [
                {"role": "user", "content": "帮我做个成都三日游攻略"},
                {"role": "assistant", "content": "下面给你一份成都三日游的轻松安排。"},
            ],
        },
        {
            "source": "synthetic_rule",
            "messages": [
                {"role": "user", "content": "北京亲子游住哪里方便"},
                {"role": "assistant", "content": "如果以亲子出行为主，建议优先考虑交通便利、周边餐饮完善的区域。"},
            ],
        },
    ]

    summary = summarize_dataset(dataset, dataset_name="stage1_general_sft.json")

    assert summary["sample_count"] == 2
    assert summary["source_counts"] == {
        "pseudo_real_dialogue": 1,
        "synthetic_rule": 1,
    }
    assert summary["source_origin_counts"] == {
        "real_like": 1,
        "synthetic": 1,
    }
    assert summary["user_length"]["count"] == 2
    assert summary["assistant_length"]["count"] == 2


def test_audit_dataset_file_reports_missing_path(tmp_path) -> None:
    report = audit_dataset_file(tmp_path / "missing.json")

    assert report["exists"] is False
    assert report["valid_chatml"] is False
    assert report["errors"]
