from __future__ import annotations

import json
from pathlib import Path
import shutil

from src.data_pipeline.handlers.handler_multiturn import build_multiturn_sample, process_multiturn_data


def test_build_multiturn_sample_keeps_valid_raw_dialogue_and_adds_system() -> None:
    record = {
        "record_id": "multi_900001",
        "task_type": "multi_turn_dialogue",
        "source": "tripai_db",
        "conversation_id": "conv_900001",
        "city": "杭州",
        "constraint_changes": [
            "初始需求：杭州旅游",
            "补充约束：加入亲子需求",
            "补充约束：改成两天一晚",
        ],
        "messages": [
            {"role": "user", "content": "我想去杭州玩，主要看看西湖。"},
            {"role": "assistant", "content": "建议先安排一日西湖慢游，上午从断桥和白堤开始，下午转到花港观鱼和苏堤，整体节奏轻松，适合第一次来杭州的人慢慢建立方位感。"},
            {"role": "user", "content": "这次还会带一个6岁孩子，希望加点适合亲子的内容。"},
            {"role": "assistant", "content": "可以把花港观鱼停留时间拉长，再加入杭州动物园或少年儿童公园，减少纯步行和密集讲解，让孩子有更多观察动物和自由活动的时间。"},
            {"role": "user", "content": "再改成两天一晚，想住西湖边。"},
            {"role": "assistant", "content": "那就拆成两天：第一天游西湖南线，第二天走北线并住在湖滨步行可达的安静酒店，晚上还能沿湖慢慢散步，整体会比单日往返更从容。"},
        ],
    }

    sample = build_multiturn_sample(record)

    assert sample is not None
    assert sample["task_type"] == "multi_turn_dialogue"
    assert sample["id"] == "multi_900001"
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["content"] == "我想去杭州玩，主要看看西湖。"
    assert len(sample["messages"]) == 7


def test_build_multiturn_sample_rejects_editorial_placeholder_tail() -> None:
    record = {
        "record_id": "multi_900002",
        "task_type": "multi_turn_dialogue",
        "source": "tripai_db",
        "conversation_id": "conv_900002",
        "city": "临沂",
        "constraint_changes": [
            "初始需求：地下大峡谷",
            "补充约束：改高铁出行",
            "补充约束：加入老人",
        ],
        "messages": [
            {"role": "user", "content": "我想去临沂玩，主要想看看地下大峡谷。"},
            {"role": "assistant", "content": "建议安排半天游览，重点看溶洞和地下暗河，洞内主通道完整、节奏不赶，适合第一次去的游客先抓住最有代表性的地貌和游览体验。"},
            {"role": "user", "content": "交通调整：改成高铁+接驳。"},
            {
                "role": "assistant",
                "content": "已按高铁出行优化：先到临沂北站，再接驳去景区。其他内容不变……（后续同上，从‘形成于约0.65亿至2.3亿年前’起完全保留）",
            },
            {"role": "user", "content": "再加一位65岁老人。"},
            {"role": "assistant", "content": "老人可优先走主通道，减少台阶和长距离步行，同时把休息点、饮水点和无障碍设施提前纳入动线安排，整体更稳妥。"},
        ],
    }

    assert build_multiturn_sample(record) is None


def test_build_multiturn_sample_trims_soft_editorial_tail() -> None:
    record = {
        "record_id": "multi_900003",
        "task_type": "multi_turn_dialogue",
        "source": "tripai_db",
        "conversation_id": "conv_900003",
        "city": "福州",
        "constraint_changes": [
            "初始需求：福州旅游",
            "补充约束：加入夜游",
            "补充约束：控制预算",
        ],
        "messages": [
            {"role": "user", "content": "我想去福州玩两天。"},
            {"role": "assistant", "content": "建议白天逛三坊七巷，第二天去鼓山，整体很适合两天安排，节奏比较稳定，适合第一次来福州先抓核心景点，也方便把福州的城市步行和山景体验自然串起来。"},
            {"role": "user", "content": "想加闽江夜游。"},
            {"role": "assistant", "content": "可以把第一天傍晚安排成闽江夜游，返回后再去老药洲街吃小吃，这样能把白天的人文步行和夜间江景自然衔接起来，也能让第一天的体验层次更丰富。其余安排保持不变。"},
            {"role": "user", "content": "预算想控制在人均1200以内。"},
            {"role": "assistant", "content": "把住宿改成经济型双床房，鼓山缆车改为步行上山，总体可压在预算内，同时核心景点和体验顺序都还能保住，对两天行程的完整性影响也比较小。"},
        ],
    }

    sample = build_multiturn_sample(record)

    assert sample is not None
    assistant_text = sample["messages"][4]["content"]
    assert "其余安排保持不变" not in assistant_text
    assert "闽江夜游" in assistant_text


def test_build_multiturn_sample_rejects_constraint_mismatch() -> None:
    record = {
        "record_id": "multi_900004",
        "task_type": "multi_turn_dialogue",
        "source": "tripai_db",
        "conversation_id": "conv_900004",
        "city": "合肥",
        "constraint_changes": [
            "初始需求：三河古镇",
            "补充约束：住一晚",
            "补充约束：预算提高",
            "补充约束：偏人文",
        ],
        "messages": [
            {"role": "user", "content": "我想去合肥玩，主要想看看三河古镇。"},
            {"role": "assistant", "content": "三河古镇适合半天到一天游览，慢慢逛街巷、旧居和桥头水巷，能比较完整地感受古镇的生活气息和建筑层次。"},
            {"role": "user", "content": "我想住一晚。"},
            {"role": "assistant", "content": "那就可以安排两天一夜，晚上住在古镇里，清晨和傍晚都能看到人少、氛围更安静的一面。"},
            {"role": "user", "content": "预算也能提高一些。"},
            {"role": "assistant", "content": "可以升级河鲜晚餐和更有质感的徽派民宿，把行程从走马观花调整为慢节奏沉浸式体验。"},
        ],
    }

    assert build_multiturn_sample(record) is None


def test_build_multiturn_sample_rejects_excessive_rewrite() -> None:
    repeated = (
        "长江索道是山城标志性体验，建议傍晚乘坐，从索道上俯瞰两岸灯火和江面流光。"
        "随后步行至龙门浩老街，再去南山一棵树观景台，形成完整的夜景动线。"
    )
    record = {
        "record_id": "multi_900005",
        "task_type": "multi_turn_dialogue",
        "source": "tripai_db",
        "conversation_id": "conv_900005",
        "city": "重庆",
        "constraint_changes": [
            "初始需求：长江索道",
            "补充约束：加入龙门浩和南山",
            "补充约束：带孩子出行",
            "补充约束：住两晚",
        ],
        "messages": [
            {"role": "user", "content": "我想去重庆玩，主要想看看长江索道。"},
            {"role": "assistant", "content": repeated},
            {"role": "user", "content": "请再加上龙门浩和南山一棵树。"},
            {"role": "assistant", "content": repeated + "这样索道、老街和观景台就串起来了，视角和动线会更完整。"},
            {"role": "user", "content": "再带一个6岁孩子。"},
            {"role": "assistant", "content": repeated + "考虑孩子体力，南山段改为接驳车上山，避免长距离爬坡。"},
            {"role": "user", "content": "我们准备住两晚。"},
            {"role": "assistant", "content": repeated + "第二天清晨可以再坐一次索道看晨光，这样昼夜光线变化都能体验到。"},
        ],
    }

    assert build_multiturn_sample(record) is None


def test_process_multiturn_data_writes_expected_dataset() -> None:
    temp_dir = Path(".tmp_multiturn_test")
    input_path = temp_dir / "dialogue.jsonl"
    output_path = temp_dir / "sft_multi_turn_dialogue.json"

    valid_record = {
        "record_id": "multi_900006",
        "task_type": "multi_turn_dialogue",
        "source": "tripai_db",
        "conversation_id": "conv_900006",
        "city": "南京",
        "constraint_changes": [
            "初始需求：牛首山",
            "补充约束：加住景区附近民宿",
            "补充约束：加入轻自然体验",
        ],
        "messages": [
            {"role": "user", "content": "我想去南京玩，主要想看看牛首山。"},
            {"role": "assistant", "content": "建议安排牛首山一日游，上午看佛顶宫和佛顶塔，下午在山间步道慢行，整体兼顾建筑观览和自然呼吸感，也能让第一次来的游客把核心区域走得比较完整。"},
            {"role": "user", "content": "住宿想住景区附近安静些的民宿。"},
            {"role": "assistant", "content": "可以住在牛首山东门步行十分钟范围内的安静民宿，方便第二天早起，也能减少往返折腾，对想把清晨山景和主要建筑都看全的人会更友好。"},
            {"role": "user", "content": "还想加一点轻自然体验，比如林间步道和观鸟。"},
            {"role": "assistant", "content": "那就把第二天上午留给林间步道和观鸟点，整体节奏更舒缓，也更适合把牛首山看得更完整，同时不会和前一天的建筑参观互相挤占时间。"},
        ],
    }
    invalid_record = {
        "record_id": "multi_900007",
        "task_type": "multi_turn_dialogue",
        "source": "tripai_db",
        "conversation_id": "conv_900007",
        "city": "临沂",
        "constraint_changes": [
            "初始需求：地下大峡谷",
            "补充约束：改高铁出行",
            "补充约束：加入老人",
        ],
        "messages": [
            {"role": "user", "content": "我想去地下大峡谷。"},
            {"role": "assistant", "content": "建议安排半天游览，先看溶洞主通道，再根据体力决定是否延伸到更多观景段。"},
            {"role": "user", "content": "改成高铁+接驳。"},
            {"role": "assistant", "content": "其他内容不变……（后续同上，从某句起完全保留）"},
            {"role": "user", "content": "还带一位老人。"},
            {"role": "assistant", "content": "可以照顾老人需求，把休息点和主通道优先纳入安排。"},
        ],
    }

    temp_dir.mkdir(exist_ok=True)
    try:
        with input_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(valid_record, ensure_ascii=False) + "\n")
            file.write(json.dumps(invalid_record, ensure_ascii=False) + "\n")

        dataset = process_multiturn_data(str(input_path), str(output_path))

        assert len(dataset) == 1
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(saved) == 1
        assert saved[0]["task_type"] == "multi_turn_dialogue"
        assert saved[0]["messages"][0]["role"] == "system"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


