from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_utils import configure_console_output, ensure_parent_dir, log_info, log_success
from pipeline.handler_intent import DEFAULT_SYSTEM_PROMPT

DEFAULT_OUTPUT_PATH = "data/raw/intent.jsonl"
DEFAULT_TOTAL_SAMPLES = 800
DEFAULT_SEED = 42

INTENT_ORDER = [
    "FUNCTION_FLIGHTS_SEARCH_STRATEGY",
    "FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY",
    "FUNCTION_FLIGHTS_PASSENGER_STRATEGY",
    "FUNCTION_HOTELS_STRATEGY",
    "TRAVEL_STRATEGY",
    "TRAVEL_LOCATION_STRATEGY",
    "FUNCTION_TICKETS_STRATEGY",
    "FUNCTION_CAR_RENTAL_STRATEGY",
    "FUNCTION_VISA_STRATEGY",
    "DEFAULT_STRATEGY",
]

CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "成都",
    "重庆",
    "西安",
    "三亚",
    "昆明",
    "厦门",
    "南京",
    "苏州",
    "长沙",
    "武汉",
    "天津",
    "青岛",
    "大理",
    "丽江",
    "哈尔滨",
]

MONTHS = ["下周", "下个月", "国庆", "五一", "暑假", "春节", "本周末", "明天", "后天", "月底"]

AIRPORTS = [
    "首都机场",
    "大兴机场",
    "浦东机场",
    "虹桥机场",
    "白云机场",
    "宝安机场",
    "天府机场",
    "双流机场",
    "凤凰机场",
    "咸阳机场",
]

SPOTS = [
    "故宫",
    "迪士尼",
    "环球影城",
    "外滩",
    "锦绣中华",
    "长隆野生动物园",
    "兵马俑",
    "西湖",
    "玉龙雪山",
    "东方明珠",
    "天坛",
    "鼓浪屿",
]

COUNTRIES = [
    "日本",
    "美国",
    "英国",
    "法国",
    "意大利",
    "新加坡",
    "泰国",
    "韩国",
    "澳大利亚",
    "加拿大",
]

HOTEL_AREAS = [
    "春熙路",
    "解放碑",
    "外滩",
    "西湖",
    "天河路",
    "三里屯",
    "五四广场",
    "南锣鼓巷",
    "鼓楼",
    "海棠湾",
]

CAR_TYPES = ["SUV", "商务车", "7座车", "经济型轿车", "新能源车"]
PASSENGER_FIELDS = ["护照号码", "乘机人信息", "身份证号", "英文姓名", "常用旅客"]
PAYMENT_WORDS = ["现在", "立刻", "今天", "马上", "这会儿"]
CUSTOMER_ROLES = ["我", "我们", "一家三口", "两个人", "我和朋友"]
DEFAULT_TOPICS = ["活动", "优惠", "会员", "客服", "平台", "功能", "服务范围", "售后"]


def _samples_per_intent(total_samples: int) -> dict[str, int]:
    base = total_samples // len(INTENT_ORDER)
    remainder = total_samples % len(INTENT_ORDER)
    counts: dict[str, int] = {}
    for index, name in enumerate(INTENT_ORDER):
        counts[name] = base + (1 if index < remainder else 0)
    return counts


def _flight_search_query(rng: random.Random) -> str:
    origin, destination = rng.sample(CITIES, 2)
    when = rng.choice(MONTHS)
    templates = [
        f"{when}{origin}飞{destination}的机票帮我查一下",
        f"我想看{origin}到{destination}的航班",
        f"{when}从{origin}去{destination}还有票吗",
        f"帮我搜一下{origin}飞{destination}最便宜的机票",
        f"{origin}到{destination}的飞机票现在多少钱",
        f"查下{when}{origin}往返{destination}航班",
        f"我准备去{destination}，先看看从{origin}出发的航班",
        f"{origin}飞{destination}明早最早一班航班是什么",
    ]
    return rng.choice(templates)


def _flight_confirm_query(rng: random.Random) -> str:
    destination = rng.choice(CITIES)
    when = rng.choice(MONTHS)
    actor = rng.choice(CUSTOMER_ROLES)
    pay_word = rng.choice(PAYMENT_WORDS)
    templates = [
        f"{actor}{when}去{destination}这班机票现在下单",
        f"帮我确认下单这张去{destination}的机票",
        f"这个去{destination}的票可以直接支付吗",
        f"去{destination}这班机票我{pay_word}就订",
        f"{when}飞{destination}这个航班确认预订并付款",
        f"把刚才那个去{destination}的航班下单吧",
        f"这个去{destination}的航班没问题的话直接帮我提交",
        f"我决定买去{destination}的这张机票，下一步怎么支付",
    ]
    return rng.choice(templates)


def _passenger_query(rng: random.Random) -> str:
    field = rng.choice(PASSENGER_FIELDS)
    destination = rng.choice(CITIES)
    actor = rng.choice(CUSTOMER_ROLES)
    templates = [
        f"怎么添加{field}",
        f"{actor}想修改乘机人的{field}",
        f"去{destination}的订单里常用旅客在哪里维护",
        f"乘机人信息填错了，{field}怎么改",
        f"可以新增一个乘机人给{destination}这趟行程用吗",
        f"护照过期了，{field}怎么更新",
        f"帮我看看去{destination}这单的乘机人资料在哪管理",
        f"我想删除一个乘机人并重填{field}",
    ]
    return rng.choice(templates)


def _hotel_query(rng: random.Random) -> str:
    city = rng.choice(CITIES)
    area = rng.choice(HOTEL_AREAS)
    when = rng.choice(MONTHS)
    templates = [
        f"{when}去{city}想订酒店",
        f"{city}{area}附近有什么住宿推荐",
        f"帮我找一下{city}住得方便的酒店",
        f"想看{city}民宿，有没有性价比高的",
        f"{when}去{city}，帮我看看亲子酒店",
        f"{city}市中心酒店怎么选",
        f"{city}{area}附近住哪里比较方便",
        f"推荐一下{city}的民宿或者酒店",
    ]
    return rng.choice(templates)


def _travel_query(rng: random.Random) -> str:
    city = rng.choice(CITIES)
    days = rng.choice(["1天", "2天", "3天", "4天", "5天"])
    templates = [
        f"{city}有什么好玩的",
        f"{city}{days}旅游攻略",
        f"去{city}玩怎么安排比较好",
        f"{city}值得推荐的景点有哪些",
        f"{days}时间在{city}怎么玩",
        f"第一次去{city}，给个攻略",
        f"{city}自由行有什么推荐",
        f"想去{city}旅游，先看看攻略",
    ]
    return rng.choice(templates)


def _travel_location_query(rng: random.Random) -> str:
    city = rng.choice(CITIES)
    spot = rng.choice(SPOTS)
    days = rng.choice(["半天", "1天", "2天"])
    templates = [
        f"{city}{spot}{days}攻略",
        f"{city}{spot}附近玩一天怎么安排",
        f"{spot}值不值得去，有什么推荐",
        f"{city}{spot}怎么玩比较合适",
        f"我在{city}，想去{spot}看看",
        f"{spot}周边还有什么景点可以一起玩",
        f"{city}{spot}游玩路线怎么排",
        f"{spot}附近好玩好吃的有哪些",
    ]
    return rng.choice(templates)


def _ticket_query(rng: random.Random) -> str:
    spot = rng.choice(SPOTS)
    templates = [
        f"{spot}门票多少钱",
        f"帮我买{spot}门票",
        f"{spot}景区票现在还有吗",
        f"{spot}需要提前预约门票吗",
        f"{spot}乐园票怎么买",
        f"查一下{spot}成人票价格",
        f"{spot}今天的门票能订吗",
        f"我想买两张{spot}门票",
    ]
    return rng.choice(templates)


def _car_rental_query(rng: random.Random) -> str:
    city = rng.choice(CITIES)
    car_type = rng.choice(CAR_TYPES)
    templates = [
        f"{city}租{car_type}多少钱",
        f"帮我看看{city}机场租车",
        f"{city}自驾租车怎么订",
        f"{city}有没有日租车服务",
        f"去{city}旅游想租车",
        f"{city}租车价格大概多少",
        f"推荐一下{city}靠谱的租车方案",
        f"{city}想租一辆{car_type}出游",
    ]
    return rng.choice(templates)


def _visa_query(rng: random.Random) -> str:
    country = rng.choice(COUNTRIES)
    templates = [
        f"{country}签证怎么办",
        f"{country}旅游签证需要什么材料",
        f"{country}签证办理流程是什么",
        f"我想办{country}旅游签证",
        f"{country}签证多久能下来",
        f"{country}签证现在好办吗",
        f"申请{country}签证要准备哪些资料",
        f"{country}自由行签证怎么弄",
    ]
    return rng.choice(templates)


def _default_query(rng: random.Random) -> str:
    topic = rng.choice(DEFAULT_TOPICS)
    city = rng.choice(CITIES)
    when = rng.choice(MONTHS)
    actor = rng.choice(CUSTOMER_ROLES)
    templates = [
        "你是谁",
        "你能帮我做什么",
        "怎么联系人工客服",
        "你们平台靠谱吗",
        f"最近有什么{topic}吗",
        f"我有点没想好去哪玩，先给我介绍下你能提供什么服务",
        f"你和别的旅游助手有什么区别",
        "你们是做什么的",
        f"你能不能先告诉我{city}值不值得去，不用推荐具体产品",
        f"tripAI 主要能帮我解决哪些旅游问题",
        f"我现在还没决定去哪，先随便聊聊可以吗",
        f"你们平台有什么会员或者{topic}",
        f"{actor}{when}想出去走走，你先介绍一下自己",
        f"我还没确定目的地，先问下你都支持哪些旅游服务",
        f"{city}先不聊具体产品，我想知道你能帮我做哪些事情",
        f"如果我只是想咨询{topic}，你能处理吗",
        f"你们和普通旅游平台相比有什么不一样",
        f"我还不准备下单，只想先了解一下平台能力",
        f"{when}我可能想出行，但现在先问问你能做什么",
        f"先别推荐具体内容，告诉我你主要负责哪些业务",
    ]
    return rng.choice(templates)


QUERY_BUILDERS = {
    "FUNCTION_FLIGHTS_SEARCH_STRATEGY": _flight_search_query,
    "FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY": _flight_confirm_query,
    "FUNCTION_FLIGHTS_PASSENGER_STRATEGY": _passenger_query,
    "FUNCTION_HOTELS_STRATEGY": _hotel_query,
    "TRAVEL_STRATEGY": _travel_query,
    "TRAVEL_LOCATION_STRATEGY": _travel_location_query,
    "FUNCTION_TICKETS_STRATEGY": _ticket_query,
    "FUNCTION_CAR_RENTAL_STRATEGY": _car_rental_query,
    "FUNCTION_VISA_STRATEGY": _visa_query,
    "DEFAULT_STRATEGY": _default_query,
}


def _build_record(intention_name: str, rng: random.Random, seen_queries: set[str]) -> dict[str, str]:
    query_builder = QUERY_BUILDERS[intention_name]
    for _ in range(500):
        user_query = query_builder(rng)
        if user_query not in seen_queries:
            seen_queries.add(user_query)
            return {
                "user_query": user_query,
                "intentionName": intention_name,
                "source": "synthetic_seed",
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
            }
    raise RuntimeError(f"无法为意图 {intention_name} 生成足够多的不重复 query。")


def generate_intent_seed_dataset(total_samples: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    counts = _samples_per_intent(total_samples)
    seen_queries: set[str] = set()
    records: list[dict[str, str]] = []

    for intention_name in INTENT_ORDER:
        for _ in range(counts[intention_name]):
            records.append(_build_record(intention_name, rng, seen_queries))

    rng.shuffle(records)
    return records


def write_jsonl(records: list[dict[str, str]], output_path: str) -> Path:
    path = ensure_parent_dir(output_path)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 tripAI intent 种子数据。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 JSONL 路径。")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL_SAMPLES, help="总样本数。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_console_output()
    log_info(f"开始生成 intent 种子数据，总量 {args.total} 条。")
    records = generate_intent_seed_dataset(args.total, args.seed)
    output_path = write_jsonl(records, args.output)
    log_success(f"intent 原始数据已生成，共 {len(records)} 条。")
    log_info(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
