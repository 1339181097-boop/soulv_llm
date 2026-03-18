# tripAI Intent 数据需求文档（数据员版）

## 1. 说明

本次 intent 数据标注，全部以公司当前在用的
`TripAI 意图识别 System Prompt`
为准。

请注意：

- 不要修改意图名称
- 不要使用自定义命名
- 不要把看起来像笔误的意图名改掉
- 后端当前就是按这套名称联调

## 2. 只能使用以下意图名称

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

## 3. 各意图对应说明

### 机票搜索

- `FUNCTION_FLIGHTS_SEARCH_STRATEGY`
- 典型关键词：
  - 查机票
  - 航班
  - 飞往
  - 从…到…

### 机票预订确认

- `FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY`
- 典型关键词：
  - 确认预订
  - 下单
  - 支付机票

### 乘客信息管理

- `FUNCTION_FLIGHTS_PASSENGER_STRATEGY`
- 典型关键词：
  - 乘机人
  - 护照
  - 乘客信息

### 酒店需求

- `FUNCTION_HOTELS_STRATEGY`
- 典型关键词：
  - 酒店
  - 住宿
  - 民宿

### 城市旅行攻略

- `TRAVEL_STRATEGY`
- 典型关键词：
  - 攻略
  - 推荐
  - 好玩

### 地点详情查询

- `TRAVEL_LOCATION_STRATEGY`
- 判定特征：
  - 城市
  - 具体地点
  - 攻略 / 好玩 / 推荐 / 游玩 / 景点

### 门票服务

- `FUNCTION_TICKETS_STRATEGY`
- 典型关键词：
  - 门票
  - 景区票
  - 乐园票
  - 展览票

### 租车服务

- `FUNCTION_CAR_RENTAL_STRATEGY`
- 典型关键词：
  - 租车
  - 自驾租车
  - 日租车
  - 租车价格

### 签证服务

- `FUNCTION_VISA_STRATEGY`
- 典型关键词：
  - 签证
  - 旅游签证
  - 签证办理
  - 签证材料

### 其他问题

- `DEFAULT_STRATEGY`
- 当以上规则都不明确命中时使用

## 4. 标注优先级

请按以下优先级判断：

1. 明确包含预订关键词，优先处理机票/酒店需求
2. 包含具体城市/地点，优先展示攻略或地点信息
3. 模糊请求，默认返回城市旅行攻略

补充要求：

- 一条数据只能标一个意图
- 禁止多意图

## 5. 交付格式

请交付 `jsonl` 文件：

- 编码：`UTF-8`
- 一行一条
- 文件名建议：`intent.jsonl`

## 6. 最小字段

每条数据至少有两个字段：

```json
{
  "user_query": "帮我查一下明天上海飞北京的机票",
  "intentionName": "FUNCTION_FLIGHTS_SEARCH_STRATEGY"
}
```

字段要求：

- `user_query`
  - 用户原始问题
  - 尽量保留真实说法
- `intentionName`
  - 必须严格来自上面的固定枚举

## 7. 推荐补充字段

如方便，可补充：

```json
{
  "user_query": "下周去成都，帮我看看酒店",
  "intentionName": "FUNCTION_HOTELS_STRATEGY",
  "source": "客服日志",
  "note": "用户未明确入住日期"
}
```

## 8. 内容要求

### 尽量保留真实问法

例如：

- “上海飞北京机票”
- “帮我看看下周去杭州住哪方便”
- “国庆去三亚，想找亲子酒店”
- “我想去日本玩，签证怎么办”
- “给我查一下明天最早一班去深圳的航班”

### 一条数据只标一个主意图

例如：

```json
{
  "user_query": "下周去上海，先帮我看看机票和酒店",
  "intentionName": "FUNCTION_FLIGHTS_SEARCH_STRATEGY",
  "note": "复合诉求，当前按主意图机票处理"
}
```

### 当前不需要模型回答

不需要额外提供 `assistant_response`。

## 9. 第一批数据量

建议第一批先给 `300 ~ 500` 条。

## 10. 示例

```json
{"user_query":"我想查北京飞上海的航班","intentionName":"FUNCTION_FLIGHTS_SEARCH_STRATEGY"}
{"user_query":"我要订这个航班","intentionName":"FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY"}
{"user_query":"如何添加乘机人","intentionName":"FUNCTION_FLIGHTS_PASSENGER_STRATEGY"}
{"user_query":"想订三亚的酒店","intentionName":"FUNCTION_HOTELS_STRATEGY"}
{"user_query":"巴黎三日游攻略","intentionName":"TRAVEL_STRATEGY"}
{"user_query":"上海外滩附近玩一天","intentionName":"TRAVEL_LOCATION_STRATEGY"}
{"user_query":"迪士尼门票多少钱","intentionName":"FUNCTION_TICKETS_STRATEGY"}
{"user_query":"北京机场租车","intentionName":"FUNCTION_CAR_RENTAL_STRATEGY"}
{"user_query":"美国签证办理流程","intentionName":"FUNCTION_VISA_STRATEGY"}
{"user_query":"你是谁","intentionName":"DEFAULT_STRATEGY"}
```

## 11. 请尽量避免

- `user_query` 为空
- `intentionName` 为空
- 完全重复
- 同一句 query 被标成不同意图
- 明显无关内容
- 乱码
- 一大段聊天记录直接塞成一条

## 12. 一句话要求

请按公司现行意图规范，先提供一批 `300 ~ 500` 条、字段简单、标签统一、保留真实用户问法的 `intent.jsonl` 数据。
