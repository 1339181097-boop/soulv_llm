from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.data_utils import (
    configure_console_output,
    log_info,
    log_success,
    log_warn,
    read_json,
    resolve_path,
    validate_chatml_dataset,
    write_json,
)

DEFAULT_INPUT_PATH = "data/processed/sft_traffic_planning_strict.json"
DEFAULT_OUTPUT_PATH = "data/processed/sft_traffic_planning_strict_round2.json"
DEFAULT_REPORT_PATH = "data/reports/traffic_planning_round2_report.json"
STAGE1_TRAFFIC_TARGET = 2000

REALTIME_OR_TRANSACTION_PATTERN = re.compile(
    r"实时票价|实时余票|余票|库存|下单|支付|预订|订票|购票|售票|二维码|扫码|票务"
)
PLACEHOLDER_PATTERN = re.compile(
    r"假设|某路公交|某一路公交|某条公交|某个(?:中转|换乘|站点|公交)|某中间站点|某换乘|某某路口|"
    r"XX|X路|Y路|公交X|公交Y|若干站|某大型换乘公交站"
)
ONLINE_DEPENDENT_PATTERN = re.compile(
    r"需查询|需要查询|查询具体|具体需|具体以|以实际|根据实际|实际公交|实际站点|具体公交|具体线路|"
    r"需提前查询|提前查询|需要提前查询|看看是否有|若有此类|可能有接驳|查询附近能|查找附近能"
)
VAGUE_ROUTE_PATTERN = re.compile(
    r"前往市区方向|到达市区后|市区换乘点|市区方向的公交|附近找到公交站|找到公交站|"
    r"先到[^。；]*(?:公交站|公交站点)|能转乘[^。；]*公交|合适的换乘点|合适公交线路|"
    r"相应公交|对应公交|乘坐合适的公交|乘坐前往[^。；]*方向的公交|公交换乘参考|路线示例|示例方案"
)
HARD_VAGUE_PATTERN = re.compile(
    r"\[具体[^\]]+\]|前往市区方向的公交|查找能到达附近主干道|"
    r"到达市区后[^。；]*(?:找附近|看看|再找|选择)[^。；]*(?:公交|线路|站点)|"
    r"能转乘去[^。；]*(?:的站点|公交|方向)|"
    r"正确的公交路线|需要先到正确的公交路线|前往目标景点方向的公交|前往目的地方向的公交"
)
NAVIGATION_ONLY_PATTERN = re.compile(r"导航软件|打开导航|按导航|按照导航|导航会|导航路线|跟着导航")
FINAL_ONLINE_DEPENDENCY_PATTERN = re.compile(
    r"查询附近|查询实时线路|查询附近公交|查询到能|具体可查询|需要查找公交站点|查找公交站点"
)
FINAL_PLACEHOLDER_LIKE_PATTERN = re.compile(r"\[具体|XX|X路|Y路|某个|若干站")
FINAL_UNEXPANDED_PUBLIC_OPTION_PATTERN = re.compile(
    r"(?:方案|交通方式)[一二三四五六七八九十\d]*[：:]公交(?:出行|换乘)?(?:（若有合适线路）|。)(?:\s*如果|$)|"
    r"公交(?:换乘)?（若有合适线路）|若有合适线路|如果有合适线路"
)
FINAL_TRUNCATED_DURATION_PATTERN = re.compile(
    r"大概需要\d+\s*-|需要\d+\s*-|大概步行\d+\s*-|大概骑行\d+\s*-|大概需要到|大概需要，|大概需要；|大概，|大概需要$"
)
FINAL_BROKEN_FRAGMENT_PATTERN = re.compile(
    r"路程左右|全程，|但相对自驾可能，|但，大概|优点是。|缺点是。|司机会行驶，"
)
FINAL_NAVIGATION_DEPENDENCY_PATTERN = re.compile(r"导航提示|直接导航|可根据导航实时调整路线")

PUBLIC_QUERY_KEYWORDS = (
    "公共交通",
    "公交",
    "地铁",
    "换乘",
    "路线",
    "怎么走",
    "怎么坐",
    "怎么规划",
    "具体",
    "直达",
)
SPECIAL_AUDIENCE_KEYWORDS = (
    "老人",
    "小孩",
    "孩子",
    "宝宝",
    "带娃",
    "行动不便",
    "残疾",
    "轮椅",
    "长辈",
    "爷爷",
    "奶奶",
    "行李",
)
DIRECT_CAR_KEYWORDS = ("打车", "网约车", "出租车", "包车", "自驾", "叫车")
DIRECT_CAR_REASON_KEYWORDS = ("门到门", "少换乘", "无需换乘", "不用换乘", "直接", "省心", "舒适", "方便上下车")

ROUTE_LINE_PATTERNS = (
    re.compile(r"(?:地铁|轨道交通|轻轨)\d+号线"),
    re.compile(r"(?:公交|巴士|摆渡车|大巴|机场巴士|机场大巴)?[A-Z]?\d+[A-Z]?(?:路|线)"),
    re.compile(r"[A-Z]口"),
)
STATION_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9·]{2,24}站")
GENERIC_STATION_NAMES = {
    "地铁站",
    "公交站",
    "站点",
    "附近站",
    "附近站点",
    "中转站",
    "换乘站",
    "公交站点",
    "乘车点",
}
GENERIC_STATION_PREFIXES = (
    "前往",
    "附近",
    "就近",
    "合适",
    "景区附近",
    "目的地附近",
    "能换乘的",
    "换乘的",
    "能到达的",
    "相关",
    "对应",
)
COMMON_TEMPLATE_PREFIXES = (
    "如果不赶时间，可以优先考虑公交接驳",
    "更建议优先走地铁或城际轨道",
    "如果更看重少换乘和门到门体验",
    "跨城段更建议先用高铁衔接",
    "从机场进城时，先走机场大巴或轨道交通",
)


def _message_content(sample: dict[str, Any], role: str) -> str:
    for message in sample.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _route_specific_hits(answer: str) -> int:
    hits = sum(bool(pattern.search(answer)) for pattern in ROUTE_LINE_PATTERNS)
    station_hits = {
        match.group(0)
        for match in STATION_PATTERN.finditer(answer)
        if match.group(0) not in GENERIC_STATION_NAMES
        and not any(match.group(0).startswith(prefix) for prefix in GENERIC_STATION_PREFIXES)
    }
    return hits + min(len(station_hits), 3)


def _is_public_route_query(query: str) -> bool:
    return any(keyword in query for keyword in PUBLIC_QUERY_KEYWORDS)


def _is_special_audience_query(query: str) -> bool:
    return any(keyword in query for keyword in SPECIAL_AUDIENCE_KEYWORDS)


def _has_direct_car_reason(answer: str) -> bool:
    return any(keyword in answer for keyword in DIRECT_CAR_KEYWORDS) and any(
        keyword in answer for keyword in DIRECT_CAR_REASON_KEYWORDS
    )


def _is_template_without_specifics(answer: str) -> bool:
    return answer.startswith(COMMON_TEMPLATE_PREFIXES) and _route_specific_hits(answer) < 2


def classify_round2_filter_reason(sample: dict[str, Any]) -> str | None:
    if sample.get("task_type") != "traffic_planning":
        return "wrong_task_type"

    query = _message_content(sample, "user")
    answer = _message_content(sample, "assistant")
    if not query or not answer:
        return "empty_chatml_content"

    route_hits = _route_specific_hits(answer)
    is_special_direct = _is_special_audience_query(query) and _has_direct_car_reason(answer)

    if REALTIME_OR_TRANSACTION_PATTERN.search(answer):
        return "realtime_or_transaction"
    if PLACEHOLDER_PATTERN.search(answer):
        return "placeholder_route"
    if ONLINE_DEPENDENT_PATTERN.search(answer):
        return "online_dependent_route"
    if HARD_VAGUE_PATTERN.search(answer):
        return "hard_vague_route"
    if NAVIGATION_ONLY_PATTERN.search(answer):
        return "navigation_only_route"
    if FINAL_ONLINE_DEPENDENCY_PATTERN.search(answer):
        return "final_online_dependency"
    if FINAL_PLACEHOLDER_LIKE_PATTERN.search(answer):
        return "final_placeholder_like"
    if FINAL_UNEXPANDED_PUBLIC_OPTION_PATTERN.search(answer):
        return "final_unexpanded_public_option"
    if FINAL_TRUNCATED_DURATION_PATTERN.search(answer):
        return "final_truncated_duration"
    if FINAL_BROKEN_FRAGMENT_PATTERN.search(answer):
        return "final_broken_fragment"
    if FINAL_NAVIGATION_DEPENDENCY_PATTERN.search(answer):
        return "final_navigation_dependency"
    if VAGUE_ROUTE_PATTERN.search(answer) and route_hits < 2:
        return "vague_route"
    if _is_template_without_specifics(answer) and not is_special_direct:
        return "template_without_specifics"
    if "公交换乘" in answer and route_hits < 2 and not is_special_direct:
        return "generic_public_transfer"
    if _is_public_route_query(query) and route_hits < 2 and not is_special_direct:
        return "insufficient_public_route_detail"
    if route_hits == 0 and not is_special_direct:
        return "insufficient_route_detail"
    return None


def filter_round2_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    filtered: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        reason = classify_round2_filter_reason(sample)
        if reason is None:
            filtered.append(sample)
            continue

        reasons[reason] += 1
        if len(examples) < 40:
            examples.append(
                {
                    "index": index,
                    "record_id": sample.get("record_id"),
                    "reason": reason,
                    "user": _message_content(sample, "user"),
                    "assistant": _message_content(sample, "assistant"),
                }
            )

    return filtered, reasons, examples


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(samples),
        "scene_counts": dict(Counter(sample.get("scene") for sample in samples)),
        "source_counts": dict(Counter(sample.get("source") for sample in samples)),
        "specific_route_hits": dict(
            sorted(Counter(_route_specific_hits(_message_content(sample, "assistant")) for sample in samples).items())
        ),
    }


def run_round2_cleaning(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_console_output()
    resolved_input = resolve_path(input_path)
    log_info(f"开始 traffic_planning 二轮清洗: {resolved_input}")

    samples = read_json(input_path)
    if not isinstance(samples, list):
        raise ValueError(f"{resolved_input} must be a JSON array.")

    chatml_errors = validate_chatml_dataset(samples)
    if chatml_errors:
        raise ValueError(f"输入数据 ChatML 校验失败，前 3 条错误: {chatml_errors[:3]}")

    filtered, reasons, removed_examples = filter_round2_samples(samples)
    output_file = write_json(output_path, filtered)

    report = {
        "input_path": str(resolved_input),
        "output_path": str(output_file),
        "target_count": STAGE1_TRAFFIC_TARGET,
        "meets_target": len(filtered) >= STAGE1_TRAFFIC_TARGET,
        "input_summary": _summarize(samples),
        "output_summary": _summarize(filtered),
        "removed_count": len(samples) - len(filtered),
        "removed_reasons": dict(reasons),
        "removed_examples": removed_examples,
    }
    report_file = write_json(report_path, report)

    log_success(f"二轮清洗完成，保留 {len(filtered)} / {len(samples)} 条。")
    if len(filtered) < STAGE1_TRAFFIC_TARGET:
        log_warn(f"保留量低于 traffic_planning 目标 {STAGE1_TRAFFIC_TARGET} 条，需要补数。")
    log_info(f"二轮产物: {output_file}")
    log_info(f"二轮报告: {report_file}")
    return filtered, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="traffic_planning 32B stage1 二轮严格清洗。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="上一轮 strict ChatML JSON。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="二轮 strict ChatML JSON 输出。")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="二轮清洗报告输出。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_round2_cleaning(args.input, args.output, args.report)


if __name__ == "__main__":
    main()
