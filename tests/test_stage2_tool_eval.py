from __future__ import annotations

from src.tool_eval.scripts.build_stage2_amap_golden import build_golden_cases, build_thinking_canary_cases
from src.tool_eval.scripts.score_tool_eval import _rate, _summarize_record
from src.tool_use.protocol import ALLOWED_TWO_STEP_CHAINS


def test_stage2_golden_builder_outputs_50_cases_with_allowed_chains() -> None:
    cases = build_golden_cases()

    assert len(cases) == 50
    assert len(build_thinking_canary_cases()) == 8
    assert {case["task_type"] for case in cases} == {
        "single_tool_call",
        "clarify_then_call",
        "no_tool_needed",
        "tool_failure_fallback",
        "tool_result_grounded_answer",
    }

    for case in cases:
        chain = case.get("expected_tool_chain", [])
        assert len(chain) <= 2
        if len(chain) == 2:
            assert tuple(chain) in ALLOWED_TWO_STEP_CHAINS


def test_score_treats_clarification_without_tool_call_as_correct() -> None:
    summary = _summarize_record(
        {
            "id": "clarify_case",
            "task_type": "clarify_then_call",
            "expected_behavior": "should_clarify",
            "expected_tool_chain": [],
            "messages": [{"role": "user", "content": "帮我规划去机场的路线。"}],
            "result": {
                "executed_calls": [],
                "messages": [
                    {"role": "user", "content": "帮我规划去机场的路线。"},
                    {"role": "assistant", "content": "可以，我需要先确认你的出发地和城市。"},
                ],
                "final_answer": "可以，我需要先确认你的出发地和城市。",
            },
        }
    )

    assert summary["clarify_correct"]
    assert summary["tool_selection_correct"]
    assert summary["overall_pass"]


def test_score_rates_no_tool_on_relevant_subset_only() -> None:
    per_case = [
        {"expected_behavior": "should_answer_directly", "no_tool_correct": False},
        {"expected_behavior": "should_call_tool", "no_tool_correct": True},
        {"expected_behavior": "should_call_tool", "no_tool_correct": True},
    ]

    assert _rate(per_case, "no_tool_correct", expected_behavior="should_answer_directly") == 0.0
