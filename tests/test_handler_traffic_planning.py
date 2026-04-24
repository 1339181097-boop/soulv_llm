from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.handlers.handler_traffic_planning import (
    build_traffic_planning_sample,
    filter_traffic_planning_samples,
    process_traffic_planning_data,
)


def test_build_traffic_planning_sample_cleans_precise_dynamic_info() -> None:
    record = {
        "record_id": "traffic_1",
        "task_type": "traffic_planning",
        "city": "北京",
        "origin": "颐和园",
        "destination": "天安门城楼",
        "user_query": "现在我人在颐和园，打算去天安门城楼，想问问有没有不绕远路的交通路线可以走呀？",
        "assistant_content": (
            "### 地铁出行建议：从颐和园附近的地铁4号线西苑站乘坐4号线（安河桥北方向），"
            "到西直门站换乘2号线（外环方向），坐至前门站下车，从前门站步行前往天安门城楼。"
            "### 出租车出行建议：直接使用导航软件输入目的地“天安门城楼”，按照导航指引驾车前往，"
            "全程约26.5公里，预计75分钟左右，可直接到达目的地附近。"
        ),
        "transport_modes": ["地铁", "出租车"],
        "transfer_tips": "地铁需要注意首末班车时间，避开高峰期",
        "avoid_text": "避免高峰期出行，可能会遇到交通拥堵",
        "scenario": "scenic_connection",
        "updated_at": "2026-03-26",
    }

    sample = build_traffic_planning_sample(record)

    assert sample is not None
    assistant_content = sample["messages"][2]["content"]
    assert "26.5公里" not in assistant_content
    assert "75分钟" not in assistant_content
    assert "地铁4号线" in assistant_content
    assert "首末班" in assistant_content
    assert "高峰期" in assistant_content


def test_build_traffic_planning_sample_adds_context_for_short_query() -> None:
    record = {
        "record_id": "traffic_2",
        "task_type": "traffic_planning",
        "city": "杭州",
        "origin": "杭州东站",
        "destination": "良渚文化村",
        "user_query": "哪个更方便？",
        "assistant_content": "建议优先坐地铁，再接驳公交，整体更稳一些。",
        "transport_modes": ["地铁", "公交车"],
        "transfer_tips": "地铁需要注意首末班车时间，避开高峰期",
        "avoid_text": "无特殊避免事项",
        "scenario": "train_to_city",
        "updated_at": "2026-03-26",
    }

    sample = build_traffic_planning_sample(record)

    assert sample is not None
    user_content = sample["messages"][1]["content"]
    assert "参考信息" in user_content
    assert "杭州东站" in user_content
    assert "良渚文化村" in user_content


def test_build_traffic_planning_sample_rejects_promo_noise() -> None:
    record = {
        "record_id": "traffic_3",
        "task_type": "traffic_planning",
        "city": "深圳",
        "origin": "深圳北站",
        "destination": "世界之窗",
        "user_query": "从深圳北站去世界之窗怎么走更方便？",
        "assistant_content": "打开tripAI商城可以直接比价订票，还能扫码看优惠路线。",
        "transport_modes": ["地铁"],
        "transfer_tips": "地铁需要注意首末班车时间，避开高峰期",
        "avoid_text": "无特殊避免事项",
        "scenario": "train_to_city",
        "updated_at": "2026-03-26",
    }

    assert build_traffic_planning_sample(record) is None


def test_filter_traffic_planning_samples_removes_short_and_non_executable_answers() -> None:
    keep_sample = build_traffic_planning_sample(
        {
            "record_id": "keep_1",
            "task_type": "traffic_planning",
            "city": "\u5317\u4eac",
            "origin": "\u5317\u4eac\u7ad9",
            "destination": "\u5965\u6797\u5339\u514b\u516c\u56ed",
            "user_query": "\u4ece\u5317\u4eac\u7ad9\u5230\u5965\u6797\u5339\u514b\u516c\u56ed\u600e\u4e48\u8d70\u66f4\u65b9\u4fbf\uff1f",
            "assistant_content": "\u4ece\u5317\u4eac\u7ad9\u4e58\u5750\u5730\u94c12\u53f7\u7ebf\uff0c\u5230\u5d07\u6587\u95e8\u7ad9\u6362\u4e58\u5730\u94c15\u53f7\u7ebf\uff0c\u5750\u5230\u5927\u5c6f\u8def\u4e1c\u7ad9\u4e0b\u8f66\uff0c\u4ece\u51fa\u7ad9\u53e3\u51fa\u7ad9\u540e\u6b65\u884c\u5373\u53ef\u5230\u8fbe\u3002\u7ad9\u5185\u6307\u5f15\u6bd4\u8f83\u6e05\u6670\uff0c\u6574\u4f53\u6362\u4e58\u4e5f\u4e0d\u7b97\u590d\u6742\u3002",
            "transport_modes": ["\u5730\u94c1"],
            "transfer_tips": "\u5730\u94c1\u9700\u8981\u6ce8\u610f\u9996\u672b\u73ed\u8f66\u65f6\u95f4\uff0c\u907f\u5f00\u9ad8\u5cf0\u671f",
            "avoid_text": "\u65e0\u7279\u6b8a\u907f\u514d\u4e8b\u9879",
            "scenario": "train_to_city",
            "updated_at": "2026-03-26",
        }
    )
    drop_sample = build_traffic_planning_sample(
        {
            "record_id": "drop_1",
            "task_type": "traffic_planning",
            "city": "\u676d\u5dde",
            "origin": "\u96f7\u5cf0\u5854",
            "destination": "\u897f\u6e56\u6e38\u8239\u7801\u5934",
            "user_query": "\u4ece\u96f7\u5cf0\u5854\u51fa\u53d1\uff0c\u6253\u7b97\u53bb\u897f\u6e56\u5750\u6e38\u8239\uff0c\u9009\u62e9\u600e\u6837\u7684\u4ea4\u901a\u65b9\u5f0f\u80fd\u6700\u5feb\u5230\u8fbe\u4e14\u6700\u7701\u4e8b\u5462\uff1f",
            "assistant_content": "\u63a8\u8350\u9009\u62e9\u51fa\u79df\u8f66\u51fa\u884c\u3002",
            "transport_modes": ["\u51fa\u79df\u8f66"],
            "transfer_tips": "\u9ad8\u5cf0\u671f\u53ef\u80fd\u4f1a\u96be\u6253\u8f66\uff0c\u5efa\u8bae\u63d0\u524d\u53eb\u8f66",
            "avoid_text": "\u907f\u514d\u643a\u5e26\u5927\u91cf\u884c\u674e\u4e58\u5750\u516c\u5171\u4ea4\u901a\u5de5\u5177",
            "scenario": "scenic_connection",
            "updated_at": "2026-03-26",
        }
    )

    assert keep_sample is not None
    assert drop_sample is not None

    filtered, reasons = filter_traffic_planning_samples([keep_sample, drop_sample])

    assert len(filtered) == 1
    assert filtered[0]["record_id"] == "keep_1"
    assert sum(reasons.values()) == 1


def test_filter_traffic_planning_samples_removes_placeholder_routes() -> None:
    placeholder_sample = build_traffic_planning_sample(
        {
            "record_id": "placeholder_1",
            "task_type": "traffic_planning",
            "city": "长沙",
            "origin": "靖港古镇",
            "destination": "太平街",
            "user_query": "从靖港古镇出发前往太平街，需要换乘吗？要是需要换乘的话，怎样换乘能更便捷呢？",
            "assistant_content": (
                "方案二：公交换乘。先从靖港古镇附近找到公交站，乘坐前往市区方向的公交，"
                "到达市区后，再转乘前往太平街附近的公交，具体需查询公交线路。"
            ),
            "transport_modes": ["公交车", "地铁"],
            "transfer_tips": "地铁需要注意首末班车时间，避开高峰期",
            "avoid_text": "避免高峰期出行，可能会遇到交通拥堵",
            "scenario": "city_hub_connection",
            "updated_at": "2026-04-22",
        }
    )

    assert placeholder_sample is not None

    filtered, reasons = filter_traffic_planning_samples([placeholder_sample])

    assert filtered == []
    assert reasons["vague_route"] == 1


def test_filter_traffic_planning_samples_removes_generic_public_routes_without_details() -> None:
    generic_sample = {
        "id": "traffic_planning_generic_public_route",
        "record_id": "generic_public_1",
        "task_type": "traffic_planning",
        "source": "tripai_traffic_planning_raw_2026_04_22",
        "source_id": "generic_public_source",
        "city": "昆明",
        "origin": "西山附近小区",
        "destination": "东川红土地景区",
        "messages": [
            {"role": "system", "content": "你是专业的中文旅行交通规划助手。"},
            {"role": "user", "content": "从西山附近小区出发去东川红土地景区，公交和地铁哪种方式更便捷？"},
            {
                "role": "assistant",
                "content": (
                    "如果不赶时间，可以优先考虑公交接驳，整体更省预算。 "
                    "公交地铁方式规划；第一步：前往地铁站；第二步：搭乘地铁前往能换乘的站点。 "
                    "出发前建议再确认线路、站点、首末班或发车安排，并预留换乘时间。"
                ),
            },
        ],
    }

    filtered, reasons = filter_traffic_planning_samples([generic_sample])

    assert filtered == []
    assert sum(reasons.values()) == 1


def test_process_traffic_planning_data_can_write_strict_output() -> None:
    temp_dir = Path(".tmp_traffic_planning_test")
    input_path = temp_dir / "traffic_planning_raw.jsonl"
    output_path = temp_dir / "sft_traffic_planning.json"
    strict_output_path = temp_dir / "sft_traffic_planning_strict.json"
    keep_record = {
        "record_id": "traffic_keep",
        "task_type": "traffic_planning",
        "city": "北京",
        "origin": "北京南站",
        "destination": "国贸",
        "user_query": "从北京南站出发，打算前往国贸，想知道哪种方式更方便？",
        "assistant_content": "从北京南站乘坐地铁4号线到西单站，再换乘1号线到国贸站下车即可。如果更看重少换乘，也可以直接打车。",
        "transport_modes": ["地铁", "网约车"],
        "transfer_tips": "地铁需要注意首末班车时间，避开高峰期",
        "avoid_text": "避免高峰期出行，可能会遇到交通拥堵",
        "scenario": "train_to_city",
        "updated_at": "2026-03-26",
    }
    drop_record = {
        "record_id": "traffic_drop",
        "task_type": "traffic_planning",
        "city": "上海",
        "origin": "南京路步行街",
        "destination": "东方明珠",
        "user_query": "从南京路步行街去东方明珠，哪种方式更节省时间？",
        "assistant_content": "推荐选择驾车方式，驾车距离约2.083公里，预计需要12.2分钟，相比公共交通预计的24.4分钟能更节省时间。",
        "transport_modes": ["公交车", "自驾"],
        "transfer_tips": "地铁需要注意首末班车时间，避开高峰期 高峰期可能会难打车，建议提前叫车",
        "avoid_text": "避免在恶劣天气下选择共享单车",
        "scenario": "scenic_connection",
        "updated_at": "2026-03-25",
    }

    temp_dir.mkdir(exist_ok=True)
    try:
        with input_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(keep_record, ensure_ascii=False) + "\n")
            file.write(json.dumps(drop_record, ensure_ascii=False) + "\n")

        dataset = process_traffic_planning_data(
            str(input_path),
            str(output_path),
            strict_output_json_path=str(strict_output_path),
        )

        assert len(dataset) == 2
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        strict_saved = json.loads(strict_output_path.read_text(encoding="utf-8"))
        assert len(saved) == 2
        assert len(strict_saved) == 1
        assert strict_saved[0]["record_id"] == "traffic_keep"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_process_traffic_planning_data_tracks_input_source_label() -> None:
    temp_dir = Path(".tmp_traffic_planning_source_test")
    input_path = temp_dir / "traffic_planning_batch_apr22.jsonl"
    output_path = temp_dir / "sft_traffic_planning.json"
    record = {
        "record_id": "traffic_source",
        "task_type": "traffic_planning",
        "source": "tripai_db",
        "source_id": "src_1",
        "city": "北京",
        "origin": "北京南站",
        "destination": "国贸",
        "user_query": "从北京南站出发，打算前往国贸，想知道哪种方式更方便？",
        "assistant_content": "从北京南站乘坐地铁4号线到西单站，再换乘1号线到国贸站下车即可。如果更看重少换乘，也可以直接打车。",
        "transport_modes": ["地铁", "网约车"],
        "transfer_tips": "地铁需要注意首末班车时间，避开高峰期",
        "avoid_text": "避免高峰期出行，可能会遇到交通拥堵",
        "scenario": "train_to_city",
        "updated_at": "2026-04-22",
    }

    temp_dir.mkdir(exist_ok=True)
    try:
        with input_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        dataset = process_traffic_planning_data(str(input_path), str(output_path))

        assert len(dataset) == 1
        assert dataset[0]["source"] == "tripai_traffic_planning_batch_apr22"
        assert dataset[0]["raw_source"] == "tripai_db"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
