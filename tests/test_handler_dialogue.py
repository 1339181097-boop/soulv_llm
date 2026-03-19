from __future__ import annotations

import json

from src.data_pipeline.handlers.handler_dialogue import build_dialogue_sample, process_dialogue_data


def test_build_dialogue_sample_cleans_brand_noise_and_literal_newlines() -> None:
    record = {
        "session_id": "sid_test",
        "dialogue": [
            {"role": "user", "content": "北京五天四晚攻略"},
            {
                "role": "assistant",
                "content": (
                    "嘿，旅行者，欢迎你来到北京。接下来，就让我这个旅游界的“小奇”带你一起出发。\\n\\n"
                    "⭐ 常用App\\ntripAI小奇旅行\\n点击“商城”抢票\\n\\n"
                    "**第一天**：故宫、景山。建议在tripAI上提前预订门票。\\n"
                    "**第二天**：长城、鸟巢和水立方。建议早点出发避开人流。\\n"
                    "**第三天**：颐和园、圆明园，晚上可以去五道口用餐。\\n"
                    "**贴士**：热门景点提前预约，步行很多，鞋子一定要舒服。\\n\\n"
                    "希望这份攻略能帮助您畅游北京！祝您旅途愉快！有任何具体问题，欢迎随时咨询。"
                ),
            },
        ],
    }

    sample = build_dialogue_sample(record)

    assert sample is not None
    assistant_content = sample["messages"][2]["content"]
    assert "\\n" not in assistant_content
    assert "tripAI" not in assistant_content
    assert "常用App" not in assistant_content
    assert "点击“商城”" not in assistant_content
    assert "希望这份攻略能帮助您" not in assistant_content
    assert "第一天" in assistant_content
    assert "第二天" in assistant_content
    assert "平台" in assistant_content


def test_process_dialogue_data_deduplicates_samples(tmp_path) -> None:
    input_path = tmp_path / "dialogue.jsonl"
    output_path = tmp_path / "sft_dialogue.json"

    record = {
        "session_id": "sid_dup",
        "dialogue": [
            {"role": "user", "content": "北京三天两晚"},
            {
                "role": "assistant",
                "content": (
                    "好的，为您规划一份北京三天两晚攻略。\\n\\n"
                    "**第一天：** 故宫与景山。\\n\\n"
                    "**第二天：** 长城与奥林匹克公园。\\n\\n"
                    "**第三天：** 颐和园与圆明园。\\n\\n"
                    "**贴士：** 提前预约热门景点，穿舒适的鞋子。"
                ),
            },
        ],
    }

    with input_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    dataset = process_dialogue_data(str(input_path), str(output_path))

    assert len(dataset) == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved) == 1
