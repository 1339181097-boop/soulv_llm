from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_utils import (
    configure_console_output,
    iter_jsonl,
    log_error,
    log_info,
    log_success,
    resolve_path,
    write_json,
)
from pipeline.global_cleaner import clean_text, normalize_text

DEFAULT_INPUT_PATH = "data/raw/travel_guide.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/sft_itinerary.json"
DEFAULT_SEED = 42
MIN_CONTENT_LENGTH = 100

SYSTEM_PROMPTS = (
    "你是搜旅智慧科技的专属 AI 旅行管家“小奇”。请使用结构清晰、信息完整、富有亲和力的中文，为旅行者生成可执行的攻略。",
    "你是 tripAI 的王牌旅行顾问“小奇”。回复时要保持人设稳定，擅长使用分段、标题和适量 Emoji，输出完整行程规划。",
    "你是旅游规划专家“小奇”。请针对用户目的地和出行天数，给出清晰的 Day1、Day2 式行程安排，并补充交通、住宿和贴士。",
)

USER_QUERY_TEMPLATES = (
    "帮我规划一份 {destination} {days} 天的旅游攻略，尽量详细一点。",
    "我打算去 {destination} 玩 {days} 天，请你给我一个完整行程安排。",
    "tripAI，帮我做一份 {destination} {days} 天自由行攻略，要有每天的重点。",
    "准备和家人去 {destination} 旅行 {days} 天，给我一份省心又清晰的路线。",
    "想去 {destination} 玩 {days} 天，麻烦你按天帮我排一下吃住行。",
)


def synthesize_user_query(destination: str, days: str, rng: random.Random) -> str:
    template = rng.choice(USER_QUERY_TEMPLATES)
    return template.format(destination=destination, days=days)


def build_itinerary_sample(record: dict[str, Any], rng: random.Random) -> dict[str, list[dict[str, str]]] | None:
    destination = normalize_text(record.get("destination") or "未知目的地")
    days = normalize_text(record.get("days") or "3")
    content = clean_text(record.get("itinerary_content"), max_length=20000)

    if len(content) < MIN_CONTENT_LENGTH:
        return None

    return {
        "messages": [
            {"role": "system", "content": rng.choice(SYSTEM_PROMPTS)},
            {"role": "user", "content": synthesize_user_query(destination, days, rng)},
            {"role": "assistant", "content": content},
        ]
    }


def process_itinerary_data(
    input_file_path: str = DEFAULT_INPUT_PATH,
    output_json_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, list[dict[str, str]]]]:
    configure_console_output()
    rng = random.Random(seed)
    processed_data: list[dict[str, list[dict[str, str]]]] = []
    skipped_short = 0
    total_records = 0

    log_info(f"开始处理攻略数据: {resolve_path(input_file_path)}")

    try:
        for _, record in iter_jsonl(input_file_path):
            total_records += 1
            sample = build_itinerary_sample(record, rng)
            if sample is None:
                skipped_short += 1
                continue
            processed_data.append(sample)
    except FileNotFoundError:
        log_error(f"未找到输入文件: {resolve_path(input_file_path)}")
        return []

    output_path = write_json(output_json_path, processed_data)
    log_success(
        "攻略数据处理完成。"
        f"读取 {total_records} 条，输出 {len(processed_data)} 条，"
        f"跳过 {skipped_short} 条过短或空内容。"
    )
    log_info(f"输出文件: {output_path}")
    return processed_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将攻略原始 JSONL 转换为 ChatML 训练数据。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="原始攻略 JSONL 路径。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 ChatML JSON 路径。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，保证结果可复现。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_itinerary_data(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
