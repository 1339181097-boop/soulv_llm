from __future__ import annotations

import json
from pathlib import Path

from src.eval.scripts.collect_badcases import _merge_records, collect_badcases


def _write_json(path: Path, payload) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_collect_badcases_merges_rule_and_judge_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "reports" / "run_a"
    raw_path = run_dir / "raw_outputs_travel_qa.json"
    _write_json(
        run_dir / "summary.json",
        {
            "run_name": "run_a",
            "model_name": "model-a",
            "request_model": "served-a",
            "tasks": {"travel_qa": {"output_file": str(raw_path), "sample_count": 3}},
        },
    )
    _write_json(
        raw_path,
        [
            {
                "id": "case_ok",
                "task_type": "travel_qa",
                "messages": [{"role": "user", "content": "西湖适合老人吗？"}],
                "prediction": "适合，建议慢慢逛。",
                "status": "ok",
                "error": "",
            },
            {
                "id": "case_rule",
                "task_type": "travel_qa",
                "messages": [{"role": "user", "content": "门票多少钱？"}],
                "prediction": "门票100元。",
                "status": "ok",
                "error": "",
                "must_not_do": ["Do not invent live prices"],
            },
            {
                "id": "case_judge",
                "task_type": "travel_qa",
                "messages": [{"role": "user", "content": "杭州怎么玩？"}],
                "prediction": "随便玩都可以。",
                "status": "ok",
                "error": "",
            },
        ],
    )
    _write_json(
        run_dir / "rule_results_travel_qa.json",
        [
            {"id": "case_ok", "rule_pass": True},
            {
                "id": "case_rule",
                "rule_pass": False,
                "rule_score": 0,
                "max_severity": "blocker",
                "triggered_rule_ids": ["exact_price"],
                "rule_hits": [{"rule_id": "exact_price", "message": "invented price"}],
            },
        ],
    )
    _write_json(
        run_dir / "judge_results_travel_qa.json",
        [
            {"id": "case_ok", "pass_or_fail": True, "verdict": "pass"},
            {
                "id": "case_judge",
                "pass_or_fail": False,
                "verdict": "hold",
                "overall_score": 2,
                "issue_tags": ["generic_answer"],
                "judge_reason": "回答过于空泛。",
                "missed_must_include": ["具体建议"],
                "violated_must_not_do": [],
            },
        ],
    )

    records = collect_badcases(run_dir)

    assert [record["sample_id"] for record in records] == ["case_judge", "case_rule"]
    assert records[0]["gold_answer"] == ""
    assert records[0]["gold_answer_status"] == "pending"
    assert records[0]["failure_sources"] == ["judge"]
    assert records[0]["error_analysis"]["summary"] == "回答过于空泛。"
    assert records[1]["failure_sources"] == ["rule"]
    assert records[1]["error_analysis"]["rule"]["triggered_rule_ids"] == ["exact_price"]


def test_merge_records_preserves_existing_gold_answer() -> None:
    existing = [
        {
            "badcase_id": "stage1::run_a::case_001",
            "gold_answer": "已有修正答案",
            "gold_answer_status": "completed",
            "created_at_utc": "old",
        }
    ]
    incoming = [
        {
            "badcase_id": "stage1::run_a::case_001",
            "gold_answer": "",
            "gold_answer_status": "pending",
            "updated_at_utc": "new",
        }
    ]

    merged = _merge_records(existing, incoming)

    assert len(merged) == 1
    assert merged[0]["gold_answer"] == "已有修正答案"
    assert merged[0]["gold_answer_status"] == "completed"
    assert merged[0]["created_at_utc"] == "old"
    assert merged[0]["updated_at_utc"] == "new"
