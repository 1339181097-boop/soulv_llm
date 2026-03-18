# tripAI Intent Schema

本文件用于在工程内固定 tripAI 当前 intent 规范。

唯一来源：

- `TripAI 意图识别 System Prompt.docx`

原则：

- 以公司现行规范为准
- 以当前后端联动规范为准
- 不做命名修正
- 不做语义扩展

## Canonical Intent Names

```text
FUNCTION_FLIGHTS_SEARCH_STRATEGY
FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY
FUNCTION_FLIGHTS_PASSENGER_STRATEGY
FUNCTION_HOTELS_STRATEGY
TRAVEL_STRATEGY
TRAVEL_LOCATION_STRATEGY
FUNCTION_TICKETS_STRATEGY
FUNCTION_CAR_RENTAL_STRATEGY
FUNCTION_VISA_STRATEGY
DEFAULT_STRATEGY
```

## Priority Rules

1. 明确包含预订关键词，优先处理机票/酒店需求
2. 包含具体城市/地点，优先展示攻略或地点信息
3. 模糊请求，默认返回城市旅行攻略

## Output Contract

模型只能输出单一意图，格式如下：

```json
{"intentionName":"FUNCTION_FLIGHTS_SEARCH_STRATEGY"}
```

禁止：

- 多意图
- 数组输出
- 解释性文本
- 寒暄和额外补充
