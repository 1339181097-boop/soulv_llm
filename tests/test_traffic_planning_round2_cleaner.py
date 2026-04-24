from __future__ import annotations

from src.data_pipeline.traffic_planning_round2_cleaner import classify_round2_filter_reason


def _sample(user: str, assistant: str) -> dict:
    return {
        "id": "sample",
        "record_id": "sample",
        "task_type": "traffic_planning",
        "messages": [
            {"role": "system", "content": "你是专业的中文旅行交通规划助手。"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_round2_keeps_concrete_public_route() -> None:
    sample = _sample(
        "从北京南站出发去国贸，地铁怎么走更方便？",
        "建议坐地铁。可以从北京南站乘坐地铁4号线到西单站，站内换乘地铁1号线到国贸站下车；如果行李较多，也可以直接打车，少换乘但费用更高。",
    )

    assert classify_round2_filter_reason(sample) is None


def test_round2_keeps_special_audience_direct_car_reason() -> None:
    sample = _sample(
        "我是一位行动不便的残疾人，从正定国际机场出发去隆兴寺，什么方式最方便？",
        "如果更看重少换乘和门到门体验，直接打车通常会更省心。出租车或网约车可以从机场直接送到隆兴寺附近，减少上下车和长距离步行，对行动不便的人更友好。",
    )

    assert classify_round2_filter_reason(sample) is None


def test_round2_drops_online_dependent_public_route() -> None:
    sample = _sample(
        "从抱犊寨去正定古城，有没有直接能坐的公共交通？",
        "可以先到抱犊寨附近公交站点，乘坐前往市区方向的公交到能转乘去正定古城的站点，然后在转乘点换乘去正定古城的公交，具体需查询公交线路。",
    )

    assert classify_round2_filter_reason(sample) == "online_dependent_route"


def test_round2_drops_category_only_route() -> None:
    sample = _sample(
        "从巢湖出发去安徽博物院，大家一般会考虑哪些交通方式？",
        "跨城段更建议先用高铁衔接，到站后再转市内交通。公共交通方式一：长途客车转地铁；公共交通方式二：高铁转地铁。",
    )

    assert classify_round2_filter_reason(sample) == "template_without_specifics"


def test_round2_drops_query_nearby_transfer_template() -> None:
    sample = _sample(
        "从中山北站去岐江公园，公交路线怎么安排？",
        "可以先坐018路到富华总站，再查询附近能换乘到前往岐江公园方向的公交线路；如果不想折腾，也可以打车。",
    )

    assert classify_round2_filter_reason(sample) == "online_dependent_route"


def test_round2_drops_hard_vague_transfer_station() -> None:
    sample = _sample(
        "从趵突泉去九如山，公共交通怎么走？",
        "建议先乘坐K51路到燕山立交桥站，再换乘到能转乘去九如山公交的站点，之后坐前往景区方向的公交。",
    )

    assert classify_round2_filter_reason(sample) == "hard_vague_route"


def test_round2_drops_final_online_dependency() -> None:
    sample = _sample(
        "从南塘街去五马街，公交怎么安排？",
        "先到南塘街附近公交站，乘坐途经该站的公交线路，具体可通过地图查询实时线路。",
    )

    assert classify_round2_filter_reason(sample) == "final_online_dependency"


def test_round2_drops_unexpanded_public_option() -> None:
    sample = _sample(
        "带老人孩子从东关街去大明寺，怎么走舒服？",
        "自驾比较直接方便，能送到景区附近；方案二：公交出行。 如果更看重稳定性，也可以改走地铁或公交接驳。",
    )

    assert classify_round2_filter_reason(sample) == "final_unexpanded_public_option"


def test_round2_drops_conditional_public_option() -> None:
    sample = _sample(
        "带老人孩子从机场去景区，有哪些舒适方式？",
        "方案一：打车/网约车；方案二：公交换乘（若有合适线路）。",
    )

    assert classify_round2_filter_reason(sample) == "final_unexpanded_public_option"


def test_round2_drops_truncated_duration() -> None:
    sample = _sample(
        "从雷峰塔去六和塔，公交怎么走？",
        "可在雷峰塔站乘坐公交315路到九溪站，再步行前往六和塔，大概需要30 -。",
    )

    assert classify_round2_filter_reason(sample) == "final_truncated_duration"


def test_round2_drops_placeholder_like_route() -> None:
    sample = _sample(
        "从南宁西站去石门森林公园，公交怎么走？",
        "先乘坐能往市区方向去的公交到某个人流量较大的公交站点，然后转乘前往石门森林公园方向的公交。",
    )

    assert classify_round2_filter_reason(sample) == "final_placeholder_like"


def test_round2_drops_broken_full_trip_fragment() -> None:
    sample = _sample(
        "从大明寺去瘦西湖，公交怎么走？",
        "可从大明寺公交站乘坐37路公交车到瘦西湖站下车，全程，公交路线较为直接。",
    )

    assert classify_round2_filter_reason(sample) == "final_broken_fragment"
