from __future__ import annotations

import json

from src.data_pipeline.handlers.handler_dialogue import build_dialogue_sample, process_dialogue_data


def test_build_dialogue_sample_keeps_short_consult_and_cleans_brand_noise() -> None:
    record = {
        "session_id": "sid_test",
        "dialogue": [
            {"role": "user", "content": "614站是到北海公园哪个门？"},
            {
                "role": "assistant",
                "content": (
                    "嘿，我是tripAI，你的专属智能助手。\\n\\n"
                    "614路在“北海”站下车，对应北海公园南门。下车后沿景山前街步行约100米即可到入口。\\n"
                    "不要和北海北地铁站混淆，那个更靠近北门。\\n"
                    "贴有tripAI二维码可扫码查实时公交。"
                ),
            },
        ],
    }

    sample = build_dialogue_sample(record)

    assert sample is not None
    assistant_content = sample["messages"][2]["content"]
    assert "tripAI" not in assistant_content
    assert "二维码" not in assistant_content
    assert "扫码" not in assistant_content
    assert "南门" in assistant_content
    assert "北门" in assistant_content


def test_build_dialogue_sample_rejects_itinerary_like_long_guide() -> None:
    record = {
        "session_id": "sid_itinerary",
        "dialogue": [
            {"role": "user", "content": "上海四天攻略"},
            {
                "role": "assistant",
                "content": (
                    "哈喽，我是tripAI，你的专属智能助手。\\n\\n"
                    "出行准备：提前预约门票。\\n\\n"
                    "第一天：外滩、南京路、豫园。\\n"
                    "第二天：迪士尼一整天。\\n"
                    "第三天：武康路、静安寺、城隍庙。\\n"
                    "第四天：朱家角古镇。\\n\\n"
                    "交通指南：可以在tripAI小奇旅行App查看优惠航线，点击“商城”预订。"
                ),
            },
        ],
    }

    assert build_dialogue_sample(record) is None


def test_build_dialogue_sample_rejects_structured_day_plan_even_without_promo() -> None:
    record = {
        "session_id": "sid_day_plan",
        "dialogue": [
            {"role": "user", "content": "附近有什么浪漫的景点"},
            {
                "role": "assistant",
                "content": (
                    "下面给你一份轻松浪漫的一日安排。\\n\\n"
                    "上午：先去滨海栈道散步看海。\\n"
                    "中午：找一家景观餐厅吃午饭。\\n"
                    "下午：去美术馆和咖啡馆坐坐。\\n"
                    "晚上：到摩天轮附近看夜景，再找酒吧小酌。\\n"
                    "这条路线节奏轻松，也比较适合拍照。"
                ),
            },
        ],
    }

    assert build_dialogue_sample(record) is None


def test_process_dialogue_data_deduplicates_and_filters_invalid_samples(tmp_path) -> None:
    input_path = tmp_path / "dialogue.jsonl"
    output_path = tmp_path / "sft_dialogue.json"

    valid_record = {
        "session_id": "sid_valid",
        "dialogue": [
            {"role": "user", "content": "苏稽古镇收费吗"},
            {
                "role": "assistant",
                "content": (
                    "嘿，我是tripAI，你的专属智能助手～\\n\\n"
                    "苏稽古镇本身免费开放，不需要门票。\\n"
                    "如果你打算去里面的个别展馆，再单独确认是否收费会更稳妥。"
                ),
            },
        ],
    }
    itinerary_record = {
        "session_id": "sid_drop",
        "dialogue": [
            {"role": "user", "content": "北京三天两晚攻略"},
            {
                "role": "assistant",
                "content": (
                    "第一天：故宫与景山。\\n\\n"
                    "第二天：长城与奥林匹克公园。\\n\\n"
                    "第三天：颐和园与圆明园。\\n\\n"
                    "交通指南：点击“商城”抢票。"
                ),
            },
        ],
    }

    with input_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(valid_record, ensure_ascii=False) + "\n")
        file.write(json.dumps(valid_record, ensure_ascii=False) + "\n")
        file.write(json.dumps(itinerary_record, ensure_ascii=False) + "\n")

    dataset = process_dialogue_data(str(input_path), str(output_path))

    assert len(dataset) == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert "免费开放" in saved[0]["messages"][2]["content"]


def test_build_dialogue_sample_rejects_style_heavy_platform_voice() -> None:
    record = {
        "session_id": "sid_style",
        "dialogue": [
            {"role": "user", "content": "帮我推荐一个手机"},
            {
                "role": "assistant",
                "content": (
                    "这边给你整理一份热门旗舰机推荐，重点看拍照、续航和256G大存储版本。\n\n"
                    "### 推荐清单\n"
                    "1. 机型A\n"
                    "2. 机型B\n"
                    "3. 机型C\n\n"
                    "有需要我再继续展开。"
                ),
            },
        ],
    }

    assert build_dialogue_sample(record) is None
def test_build_dialogue_sample_strips_intro_line_and_platform_listing() -> None:
    record = {
        "session_id": "sid_listing",
        "dialogue": [
            {"role": "user", "content": "深圳罗湖口岸附近有什么吃的？"},
            {
                "role": "assistant",
                "content": (
                    "哈喽，我是tripAI，你的专属智能助手\n"
                    "**住宿**：tripAI\n"
                    "罗湖口岸附近想吃得方便一点，可以优先去向西村和国贸商圈\n"
                    "向西村更适合宵夜和粤式小店，国贸一带连锁选择更多。"
                ),
            },
        ],
    }

    sample = build_dialogue_sample(record)

    assert sample is not None
    assistant_content = sample["messages"][2]["content"]
    assert "tripAI" not in assistant_content
    assert "住宿" not in assistant_content
    assert "向西村" in assistant_content
    assert "国贸" in assistant_content
