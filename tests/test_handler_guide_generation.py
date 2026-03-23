from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from src.data_pipeline.handlers.handler_guide_generation import (
    build_guide_generation_sample,
    prepare_guide_generation_raw,
    process_guide_generation_data,
)


def test_build_guide_generation_sample_removes_preface_platform_and_stale_detail_lines() -> None:
    rng = random.Random(0)
    record = {
        "destination": "东京",
        "days": "3",
        "itinerary_content": (
            "哈喽，我是tripAI，你的专属智能助手。\n\n"
            "出行准备\n1. 证件\n2. 行李\n\n"
            "常用App\n**住宿**：tripAI\n点击“商城”预订\n\n"
            "### 第一天：浅草寺与上野\n"
            "**上午**：先去浅草寺，顺着仲见世街慢慢走到本堂。\n"
            "- **景点介绍**：这条线路适合第一天轻松逋街，不需要走很远就能感受东京的旧城氛围。\n"
            "- **门票**：免费开放。\n"
            "- **开放时间**：09:00-17:00。\n"
            "- **小贴士**：早一点出门会更好拍照，也能避开人潮。\n"
            "**中午**：在附近吃天丿或荞麦面\n"
            "- **推荐美食**：选一家排队适中的小店即可，吃完可以顺路去上野公园。\n"
            "- **1. 浅草老铺餐厅**：位置在雷门附近，人均120元。\n"
            "**下午**：上野公园与阿美横丁\n"
            "- **建议**：如果体力还可以，可以加一个博物馆或美术馆。\n\n"
            "### 第二天：银座与筑地\n"
            "**上午**：先在银座主街散步，再去小巷子里找咖啡馆休息。\n"
            "- **建议**：不用硬冲大商场，以街区观察和慢逛为主就够了。\n"
            "**下午**：去筑地场外市场吃点海鲜，傍晚再回滨水区域散步。\n"
            "- **小贴士**：如果不想排长队，尽量避开正餐高峰。\n\n"
            "### 第三天：新宿与涩谷\n"
            "**上午**：新宿御苑\n"
            "**下午**：涩谷十字路口与周边街区。\n\n"
            "希望这份攻略能帮到你。"
        ),
    }

    sample = build_guide_generation_sample(record, rng)

    assert sample is not None
    assistant_content = sample["messages"][2]["content"]
    assert "出行准备" not in assistant_content
    assert "常用App" not in assistant_content
    assert "商城" not in assistant_content
    assert "tripAI" not in assistant_content
    assert "第一天" in assistant_content
    assert "第二天" in assistant_content
    assert "门票" not in assistant_content
    assert "开放时间" not in assistant_content
    assert "09:00-17:00" not in assistant_content
    assert "人均120元" not in assistant_content


def test_build_guide_generation_sample_rejects_overlong_content() -> None:
    rng = random.Random(0)
    repeated = "这个景点适合慢慢逛、拍照和休息，也适合留出富裕时间体验周边街区。"
    long_body = " ".join([repeated] * 120)
    record = {
        "destination": "巴黎",
        "days": "3",
        "itinerary_content": (
            "### 第一天：城区\n" + long_body + "\n\n"
            "### 第二天：博物馆\n" + long_body + "\n\n"
            "### 第三天：塞河\n" + long_body
        ),
    }

    assert build_guide_generation_sample(record, rng) is None


def test_build_guide_generation_sample_rejects_time_dense_schedule() -> None:
    rng = random.Random(0)
    record = {
        "destination": "阿尔山",
        "days": "2",
        "itinerary_content": (
            "### 第一天：森林公园\n"
            "- 08:00 从市区出发。\n"
            "- 09:00 到游客中心取票。\n"
            "- 10:00 前往天池。\n"
            "- 11:30 景区内午餐。\n"
            "- 13:00 去三潭峡。\n"
            "- 15:00 返回酒店。\n\n"
            "### 第二天：火山地貌\n"
            "- 08:30 出门。\n"
            "- 10:00 抵达景区。\n"
            "- 12:00 午饭。\n"
            "- 14:00 返程。"
        ),
    }

    assert build_guide_generation_sample(record, rng) is None


def test_process_guide_generation_data_filters_risky_sample() -> None:
    good_record = {
        "destination": "香港",
        "days": "3",
        "itinerary_content": (
            "### 第一天：中环\n"
            "**上午**：太平山与山顶步道\n"
            "- **景点介绍**：上午先乘索道上山，在凌霄阁附近看维港天际线，然后沿步道慢慢散步。\n"
            "- **小贴士**：早上上山视野更好，也比较容易避开人流。\n"
            "**下午**：中环街区、石板街与嘉咸街\n"
            "- **建议**：以步行为主，中间找家茶餐厅休息就很合适，傍晚可以再去滨水方向看夜景。\n\n"
            "### 第二天：尖沙咀\n"
            "**上午**：星光大道与海滨长廊\n"
            "- **景点介绍**：这一段适合放慢速度看海、拍照和休息，不需要安排太多紧张行程。\n"
            "**下午**：天星小轮与弥敦道散步\n"
            "- **建议**：働晚再坐轮渡，之后就近安排晚饭，整体节奏会比较舒服。\n\n"
            "### 第三天：西九文化区\n"
            "**上午**：先去海滨附近散步，再进文化场馆转转\n"
            "**下午**：在附近找家咖啡馆或书店坐一会，作为这次短途的收尾。"
        ),
    }
    risky_record = {
        "destination": "巴黎",
        "days": "3",
        "itinerary_content": (
            "### 第一天：城区\n"
            "**上午**：先去中心街区散步\n"
            "- **建议**：这条路线对新手很友好，而且可以一路串起几个地标。\n"
            "**下午**：去塞纳河沿岸散步\n"
            "- **小贴士**：这条路线闭眼冲就可以，同时还能买到官方最低价行程组合。\n\n"
            "### 第二天：博物馆\n"
            "**上午**：先去附近广场看建筑\n"
            "**下午**：继续去博物馆和周边街区散步\n"
            "- **建议**：时间富裕的话可以在附近再多停留一会。\n\n"
            "### 第三天：街区慢逛\n"
            "**上午**：在小街巷间走走停停\n"
            "**下午**：找家咖啡馆休息，做为整个行程的收尾。"
        ),
    }

    temp_dir = Path(".tmp_guide_generation_test")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        input_path = temp_dir / "guide_generation_raw.jsonl"
        output_path = temp_dir / "sft_guide_generation.json"
        with input_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(good_record, ensure_ascii=False) + "\n")
            file.write(json.dumps(risky_record, ensure_ascii=False) + "\n")

        dataset = process_guide_generation_data(str(input_path), str(output_path), seed=0)

        assert len(dataset) == 1
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(saved) == 1
        assert "太平山" in saved[0]["messages"][2]["content"]
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)



def test_prepare_guide_generation_raw_reaches_targets(tmp_path) -> None:
    input_path = tmp_path / "guide_generation_raw.jsonl"
    output_path = tmp_path / "guide_generation_raw_expanded.jsonl"
    seed_records = [
        {
            "destination": "??",
            "days": "3",
            "itinerary_content": (
                "??????\n\n"
                "### ?????????\n????????????????????????\n\n"
                "### ?????????\n???????????????????????\n\n"
                "### ?????????\n??????????????????????"
            ),
        },
        {
            "destination": "??",
            "days": "3",
            "itinerary_content": (
                "??????\n\n"
                "### ??????\n???????????????????\n\n"
                "### ???????????\n????????????????\n\n"
                "### ???????????\n???????????????????"
            ),
        },
    ]
    with input_path.open("w", encoding="utf-8") as file:
        for record in seed_records:
            file.write(__import__("json").dumps(record, ensure_ascii=False) + "\n")

    raw_records = prepare_guide_generation_raw(
        str(input_path),
        str(output_path),
        target_raw_count=4,
        target_cleanable_count=3,
    )

    assert len(raw_records) >= 4
    assert output_path.exists()
    saved_lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(saved_lines) == len(raw_records)
    assert any(record.get("variant_type") == "contiguous_day_window" for record in raw_records)
