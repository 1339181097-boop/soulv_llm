# TripAI Stage2 高德工具调用 SFT 数据需求说明 v2

版本：v2.0  
适用模型：Qwen/Qwen3-32B，在 Stage1 merged base 基础上继续做 Stage2 LoRA  
适用目标：TripAI 高德 AMap 工具协议对齐、澄清边界、两步工具链、失败回退、最终回答风格  
训练格式：LLaMA-Factory `sharegpt + tools`，不包含 `<think>` 或 `reasoning_content`

## 1. 背景与结论

Stage2 v1 已经完成训练和评测。结果显示模型能学习工具协议，但当前数据存在明显质量和配比问题，导致模型偏向“能调就调”，对缺参场景不够谨慎，最终回答也出现机械复述工具字段的问题。

Stage1 merged baseline 在 50 条 golden eval 上的主要结果：

| 指标 | Stage1 merged baseline |
|---|---:|
| tool_selection_accuracy | 0.76 |
| argument_accuracy | 0.82 |
| clarify_accuracy | 0.375 |
| no_tool_accuracy | 1.00 |
| fallback_accuracy | 0.3333 |
| execution_success_rate | 0.86 |
| final_answer_grounded_rate | 0.96 |
| overall_pass_rate | 0.50 |

Stage2 v1 训练后，300/392 step 的效果基本一致。以较好的 300 step 为例：

| 指标 | Stage2 v1 step-300 | 变化 |
|---|---:|---:|
| tool_selection_accuracy | 0.76 | 持平 |
| argument_accuracy | 0.98 | 明显提升 |
| clarify_accuracy | 0.25 | 明显下降 |
| no_tool_accuracy | 1.00 | 保持 |
| fallback_accuracy | 1.00 | 明显提升 |
| execution_success_rate | 0.94 | 提升 |
| final_answer_grounded_rate | 0.72 | 明显下降 |
| overall_pass_rate | 0.54 | 轻微提升 |

核心结论：

1. Stage2 训练方向有效，模型已经学会更准确地生成参数、处理 fallback、提高执行成功率。
2. Stage2 v1 数据配比不合理，过多强化了单步调用、slot filling 和 fallback，导致模型倾向过调用。
3. Stage2 v1 数据质量不足，最终回答过度复述工具原始字段，例如“2537秒”“18753米”，不符合真实旅行助手表达。
4. Stage2 v2 必须重做数据，不建议继续用 v1 数据单纯重训。

## 2. 当前 v1 数据暴露的问题

### 2.1 澄清边界不足

模型在以下场景应该先问用户，而不是直接调用工具：

- 缺起点：例如“帮我规划去机场的公共交通路线”，没有当前位置。
- 缺终点：例如“从北京南站怎么走”，没有目的地。
- 缺城市：例如“火车站到人民公园”，全国多个城市都有同名地点。
- 缺出行方式：例如“西湖到灵隐寺怎么去”，用户没有说明公交、驾车、步行、骑行偏好。
- POI 缺城市：例如“万达广场附近找酒店”，全国有大量万达广场。
- POI 缺 anchor：例如“附近找餐厅”，没有当前位置。
- POI 缺 keyword/category：例如“西湖附近有什么方便的地方”，搜索目标不明确。
- geocode 缺城市：例如“人民公园在哪里”，全国多个人民公园。

Stage2 v1 训练后在这些样本上仍然经常直接调用工具，说明数据中“先澄清”的监督不够强。

### 2.2 两步工具链没有学会

当前工具协议允许最多两步，且只允许：

- `amap_geocode -> amap_plan_route`
- `amap_geocode -> amap_search_poi`

但 Stage2 v1 模型在需要两步链路的样本中，往往只调用最后一个工具，例如只调用 `amap_search_poi` 或 `amap_plan_route`，没有先显式调用 `amap_geocode`。

这会导致 tool selection 评分不通过，也说明训练数据没有把“先定位 anchor，再围绕坐标搜索/规划”的链路训清楚。

### 2.3 最终回答不够人性化

当前回答中出现了大量工具原始字段复述：

错误表达示例：

- “预计耗时 2537 秒”
- “距离 18753 米”
- “步行约 94 米”
- “全程约 2363 米，约 360 秒”

这些表达不适合面向用户。旅行助手应做单位转换和自然语言改写：

- `2537 秒` 应表达为 “约 42 分钟”
- `18753 米` 应表达为 “约 18.8 公里”
- `94 米` 应表达为 “步行约 100 米” 或 “步行一小段”
- `360 秒` 应表达为 “约 6 分钟”

最终回答应像正常旅行助手，而不是接口字段打印器。

### 2.4 grounded answer 下降

Stage1 baseline 的 `final_answer_grounded_rate` 为 0.96，但 Stage2 v1 降到 0.70-0.72。主要原因包括：

- 回答过于模板化，只保留秒数、米数，没有稳定包含路线、站点、目的地等关键事实。
- 对 POI 和路线结果的摘要不够自然，忽略了用户真正关心的“怎么选、怎么走、是否方便”。
- 工具失败时 fallback 学得很好，但正常成功时回答质量下降。

### 2.5 fallback 比例和强度过高

Stage2 v1 把 fallback 从 0.3333 拉到 1.0，说明该能力已经充分学会。v2 不应继续大比例堆 fallback，否则会挤占澄清和两步链路的训练空间。

## 3. Stage2 v2 数据目标

Stage2 v2 不再教通用 function calling，而是做 TripAI 专属工具行为对齐。目标是：

1. 明确什么时候调用工具，什么时候不调用工具。
2. 参数缺失时先澄清，不能猜。
3. 需要 anchor 坐标时，显式走 `geocode -> search_poi/plan_route` 两步链。
4. 工具失败、空结果或不确定时，可靠 fallback，不编造。
5. 成功拿到工具结果后，用自然、人类可读、旅行助手风格的中文回答。
6. 保持 Stage1 的自然语言能力，不让模型变成机械工具结果复述器。

## 4. 工具协议冻结要求

只允许以下三个工具：

### 4.1 `amap_geocode`

用途：把景点、酒店、商圈、地址解析成结构化位置和经纬度。

参数要求：

- `address`：必填，地点名/地址/景点名/商圈名。
- `city`：可选，但遇到重名地点或用户已给城市时必须填写。

### 4.2 `amap_search_poi`

用途：搜索酒店、餐厅、地铁站、停车场、商场、景点等 POI。

参数要求：

- `keyword`：必填，搜索关键词，例如“酒店”“餐厅”“地铁站”“停车场”“景点”。
- `city`：可选，但有城市信息时必须填写。
- `around_location`：可选，可以是地点名，也可以是 `经度,纬度`。两步链路中必须使用 geocode 返回的坐标。
- `radius_m`：可选，建议根据场景使用 500、1000、1500、3000。

### 4.3 `amap_plan_route`

用途：规划起点到终点路线。

参数要求：

- `origin`：必填，起点，可以是地点名、地址或坐标。
- `destination`：必填，终点，可以是地点名、地址或坐标。
- `mode`：可选但建议明确，枚举值为 `transit`、`driving`、`walking`、`bicycling`。
- `city`：公共交通规划强烈建议填写城市。

### 4.4 工具链限制

最多两步工具调用。

允许：

- `amap_geocode -> amap_plan_route`
- `amap_geocode -> amap_search_poi`
- 单独调用 `amap_geocode`
- 单独调用 `amap_search_poi`
- 单独调用 `amap_plan_route`

禁止：

- 任意三步或更多工具链。
- `search_poi -> plan_route`
- `plan_route -> search_poi`
- 并行多个工具调用。
- 虚构不存在的工具。
- 在没有足够参数时强行调用工具。

## 5. Stage2 v2 推荐配比

总量建议仍为 3200 条，先保证质量，不盲目扩到 10K。

| 桶 | 数量 | 占比 | 目的 |
|---|---:|---:|---|
| clarify_then_call | 1200 | 37.5% | 修复缺参过调用，是 v2 最大重点 |
| two_step_chain | 800 | 25.0% | 显式训练 `geocode -> route/search` |
| tool_result_grounded_answer | 400 | 12.5% | 修复最终回答风格和单位转换 |
| single_tool_call | 240 | 7.5% | 保留直接调用能力，防止退化 |
| slot_filling_tool_call | 240 | 7.5% | 保留参数填充能力，避免 argument accuracy 回落 |
| tool_failure_fallback | 160 | 5.0% | 保留失败回退能力，避免过拟合 fallback |
| no_tool_needed | 160 | 5.0% | 保留无需工具直答能力 |
| 合计 | 3200 | 100% |  |

## 6. 各桶详细要求

### 6.1 `clarify_then_call`：1200 条

目标：让模型学会“缺关键参数时必须先问，不允许猜”。

子类配比：

| 子类 | 数量 | 要求 |
|---|---:|---|
| route_missing_origin | 120 | 缺起点，必须问“你从哪里出发/在哪个城市” |
| route_missing_destination | 120 | 缺终点，必须问“你要去哪里” |
| route_missing_city_or_ambiguous_city | 120 | 起终点重名或城市缺失，必须问城市 |
| route_missing_mode | 120 | 用户未说明公交/驾车/步行/骑行，必须问偏好 |
| poi_missing_city | 160 | 全国同名 anchor，例如万达广场、人民公园，必须问城市 |
| poi_missing_anchor | 160 | 只有“附近”，没有位置，必须问当前位置/参考点 |
| poi_missing_keyword | 160 | 只有“方便的地方”，目标不明确，必须问想找什么 |
| geocode_missing_city | 160 | 地名重名，必须问城市 |
| geocode_ambiguous_same_name | 80 | 明确提示“全国有多个同名地点”，必须问城市或附近地标 |
| 合计 | 1200 |  |

轨迹要求：

1. 第一轮用户请求缺参。
2. 第一轮 assistant 只能追问，不能包含任何 tool call。
3. 用户补充缺失信息。
4. assistant 再调用对应工具。
5. observation 后给出自然语言最终回答。

错误样例：

用户：“从北京南站怎么走？”  
错误：直接调用 `amap_plan_route`，并猜目的地为“北京站”。

正确样例：

用户：“从北京南站怎么走？”  
assistant：“可以帮你规划，先确认一下你要去哪里？另外你希望坐地铁公交、打车还是步行？”  
用户：“去颐和园，坐地铁公交。”  
assistant 调用 `amap_plan_route`。

### 6.2 `two_step_chain`：800 条

目标：让模型显式学习两步工具链，不再只调用最后一步。

子类配比：

| 子类 | 数量 | 工具链 |
|---|---:|---|
| geocode_then_search_hotel | 120 | `amap_geocode -> amap_search_poi` |
| geocode_then_search_restaurant | 100 | `amap_geocode -> amap_search_poi` |
| geocode_then_search_subway | 80 | `amap_geocode -> amap_search_poi` |
| geocode_then_search_parking | 80 | `amap_geocode -> amap_search_poi` |
| geocode_then_search_mall_or_spot | 120 | `amap_geocode -> amap_search_poi` |
| geocode_then_route_transit | 140 | `amap_geocode -> amap_plan_route` |
| geocode_then_route_walking | 80 | `amap_geocode -> amap_plan_route` |
| geocode_then_route_driving_or_bicycling | 80 | `amap_geocode -> amap_plan_route` |
| 合计 | 800 |  |

强制要求：

1. 第一工具必须是 `amap_geocode`。
2. 第二工具必须使用第一步返回的坐标作为 `around_location`、`origin` 或 `destination`。
3. 不能把 geocode 隐含在 `amap_search_poi` 或 `amap_plan_route` 内部。
4. 最终回答必须同时体现 anchor 和搜索/路线结果。

正确轨迹示例：

用户：“鼓浪屿码头附近找步行方便的酒店。”

assistant 调用：

```json
{"name": "amap_geocode", "arguments": {"address": "鼓浪屿码头", "city": "厦门"}}
```

observation 返回坐标。

assistant 调用：

```json
{"name": "amap_search_poi", "arguments": {"keyword": "酒店", "city": "厦门", "around_location": "118.073520,24.446243", "radius_m": 1000}}
```

最终回答：列出 2-4 个酒店，说明距离和选择建议。

### 6.3 `tool_result_grounded_answer`：400 条

目标：修复最终回答风格，让模型学会把工具结果转成用户能读懂的旅行建议。

子类配比：

| 子类 | 数量 |
|---|---:|
| route_grounded_answer | 180 |
| poi_grounded_answer | 160 |
| geocode_grounded_answer | 60 |
| 合计 | 400 |

重点要求：

1. 不得裸露工具原始秒数和大米数。
2. 不得机械复述所有 segments。
3. 优先总结关键路径、线路、站点、时长、距离、选择建议。
4. 工具结果中的关键事实必须保留，不能编造。

错误表达：

- “预计耗时 2537 秒，距离 18753 米。”
- “步行约 94 米到火车东站。”
- “全程约 6690 米，约 12 分钟。”

正确表达：

- “全程大约 42 分钟，距离约 18.8 公里。”
- “从杭州东站出来后，步行一小段到火车东站地铁站。”
- “地铁段大约 12 分钟，出站后步行约 500 米到水上公园。”

### 6.4 `single_tool_call`：240 条

目标：保留模型本来已经较强的直接工具调用能力。

子类配比：

| 子类 | 数量 |
|---|---:|
| direct_route | 80 |
| direct_geocode | 80 |
| direct_poi | 80 |
| 合计 | 240 |

要求：

- 用户请求必须参数充足。
- 只调用一个工具。
- 参数必须准确。
- 最终回答要自然，不得裸露不友好的原始字段。

### 6.5 `slot_filling_tool_call`：240 条

目标：保留 slot filling 和 argument accuracy。

子类配比：

| 子类 | 数量 |
|---|---:|
| route_slot_filling | 120 |
| poi_slot_filling | 120 |
| 合计 | 240 |

要求：

- 用户请求里有足够信息，但表达较口语化。
- assistant 要正确抽取 origin、destination、mode、city、keyword、around_location、radius_m。
- 不得凭空补用户未给且不可推断的信息。

### 6.6 `tool_failure_fallback`：160 条

目标：保持 fallback 能力，但不让 fallback 挤占训练重心。

子类配比：

| 子类 | 数量 |
|---|---:|
| route_error_or_empty | 60 |
| poi_error_or_empty | 60 |
| geocode_error_or_empty | 40 |
| 合计 | 160 |

要求：

- 工具返回 `error` 或 `empty`。
- assistant 必须说明当前无法确认或无法获取可靠结果。
- 不得编造路线、坐标、店名、距离。
- 可以建议用户补充城市、详细地址、换时间重试，或使用地图 app 实时确认。

合格表达：

“刚才路线工具没有返回可靠结果，我先不硬给你编。你可以补充更具体的起终点，或者稍后再试，我再继续帮你查。”

不合格表达：

“虽然工具失败了，但你可以从 A 站坐 2 号线到 B 站。”  
该回答属于工具失败后编造。

### 6.7 `no_tool_needed`：160 条

目标：防止模型变成凡事调用工具。

要求：

- 旅游建议、行程节奏、注意事项、优缺点分析等不需要实时位置或 POI 检索的问题。
- 绝不能出现 function_call 或 observation。
- 回答应保持 Stage1 的自然语言能力。

示例：

- “西湖适合带老人慢慢逛吗？”
- “冬天去哈尔滨需要准备什么？”
- “住在春熙路附近有什么优缺点？”

## 7. 最终回答风格规范

### 7.1 单位转换规则

时长：

| 原始秒数 | 展示方式 |
|---:|---|
| < 60 | “不到 1 分钟”或“几十秒” |
| 60-3599 | 四舍五入为分钟，例如 “约 42 分钟” |
| >= 3600 | “约 1 小时 20 分钟” |

距离：

| 原始米数 | 展示方式 |
|---:|---|
| < 100 | “几十米”或“约 90 米” |
| 100-999 | “约 300 米” |
| >= 1000 | 转为公里，保留 1 位小数，例如 “约 18.8 公里” |

特殊规则：

- 公共交通总路程不一定必须报距离，优先报线路和耗时。
- 步行接驳很短时，可写“步行一小段”。
- 驾车路线可以写“约 40 分钟、约 11 公里”。
- 不得出现“2537 秒”“18753 米”这类裸原始字段。

### 7.2 路线回答规范

公共交通回答应包含：

- 起点和终点。
- 主要线路，例如地铁 10 号线、公交 278 路。
- 关键上下车站或换乘点。
- 总耗时，使用分钟/小时分钟。
- 必要时提醒出发前确认实时运营。

驾车回答应包含：

- 预计时长和距离，单位人类可读。
- 不需要逐条复述道路转向。
- 可提醒避开拥堵、出发前确认实时路况。

步行/骑行回答应包含：

- 大致时长和距离。
- 简洁方向或沿途参考点。
- 若工具失败，说明无法获取可靠路线，不编造。

### 7.3 POI 回答规范

POI 推荐回答应包含：

- 2-5 个结果，优先列距离近、名称清晰的。
- 每个结果包含名称、距离、必要时包含地址。
- 给出简短选择建议，例如“赶时间优先选最近”“带老人优先选步行少、入口清晰的”。
- 不要堆满所有原始字段。

### 7.4 geocode 回答规范

geocode 回答应包含：

- 地点名。
- 城市/区县。
- 坐标可选，用户问“位置/经纬度”时再给。
- 对重名地点，如果用户未给城市，应先澄清，不应直接给某个城市结果。

## 8. 数据格式要求

### 8.1 源数据格式

建议数据部门交付 source JSONL 或 JSON 数组，每条包含：

```json
{
  "id": "stage2_v2_xxx_000001",
  "task_type": "clarify_then_call",
  "scene": "amap_plan_route",
  "expected_behavior": "should_clarify",
  "tools": [],
  "messages": [],
  "messages_with_answer": [],
  "expected_tool_chain": ["amap_plan_route"],
  "expected_arguments_subset": {},
  "quality_tags": ["missing_destination", "human_readable_units"]
}
```

字段说明：

- `id`：全局唯一。
- `task_type`：必须是本需求定义的桶名。
- `scene`：工具场景，例如 `amap_plan_route`、`amap_search_poi`、`amap_geocode`、`two_step_chain`。
- `expected_behavior`：`should_call_tool`、`should_clarify`、`should_answer_directly`、`should_fallback`。
- `tools`：完整工具 schema。
- `messages`：评测输入消息。
- `messages_with_answer`：完整训练轨迹，含 tool call、observation、最终回答。
- `expected_tool_chain`：期望工具链，用于评测和抽检。
- `expected_arguments_subset`：需要严格命中的参数子集。
- `quality_tags`：便于抽检和问题定位。

### 8.2 LLaMA-Factory 导出格式

最终训练文件仍使用 `sharegpt + tools`：

```json
{
  "id": "stage2_v2_xxx_000001",
  "task_type": "clarify_then_call",
  "scene": "amap_plan_route",
  "expected_behavior": "should_clarify",
  "tools": "[...]",
  "conversations": [
    {"from": "system", "value": "..."},
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "为了帮你规划路线，我需要先确认..."},
    {"from": "human", "value": "..."},
    {"from": "function_call", "value": "{\"name\":\"amap_plan_route\",\"arguments\":{...}}"},
    {"from": "observation", "value": "{\"status\":\"success\",\"data\":{...}}"},
    {"from": "gpt", "value": "..."}
  ]
}
```

注意：

- `tools` 必须是 JSON 字符串。
- `function_call` 的 `value` 必须是合法 JSON。
- `observation` 的 `value` 必须是合法 JSON。
- 不允许写 `<think>`、`reasoning_content`、推理过程。

## 9. 数据质量验收标准

### 9.1 自动校验

必须 100% 通过：

- JSON parse。
- source validator。
- sharegpt validator。
- 工具名白名单校验。
- 工具链长度校验，最多两步。
- 工具链合法性校验，只允许 `geocode -> plan_route` 和 `geocode -> search_poi`。
- no-tool 样本不得包含 function_call/observation。
- clarify 第一轮 assistant 不得包含 tool call。
- fallback 样本不得在 error/empty 后编造具体路线、坐标或 POI。

### 9.2 人工抽检

每个桶至少抽检 5%，且每个子类至少抽检 20 条。

通过标准：

- 严重错误为 0。
- 每 100 条轻微错误不超过 2 条。
- 严重错误包括：工具链错、缺参仍调工具、工具失败后编造、JSON 无效、单位不转换、最终回答与 observation 矛盾。
- 轻微错误包括：表达略啰嗦、距离四舍五入不够优雅、推荐排序不够好但事实正确。

### 9.3 去重要求

- 训练集内部重复率不超过 2%。
- 与 golden/eval 集不得有完全重复问题。
- 同一城市、同一地点、同一问题模板不得连续大量重复。
- 同名地点必须覆盖多个城市，不得全部落在热门城市。

### 9.4 覆盖要求

城市覆盖：

- 一线和新一线城市必须覆盖。
- 二线和旅游城市必须覆盖。
- 同名地点样本必须覆盖不同省市。

场景覆盖：

- 路线：公交/地铁、驾车、步行、骑行。
- POI：酒店、餐厅、地铁站、停车场、商场、景点。
- geocode：景点、商圈、酒店、码头、广场、公园。
- 失败：API error、empty result、地点模糊、城市缺失。

## 10. 独立评测集要求

除了 3200 条训练集，建议额外准备 100 条 Stage2 v2 golden eval，不能从训练集复制。

推荐配比：

| 类别 | 数量 |
|---|---:|
| clarify_then_call | 30 |
| two_step_chain | 25 |
| single_tool_call | 15 |
| tool_result_grounded_answer | 10 |
| tool_failure_fallback | 10 |
| no_tool_needed | 10 |
| 合计 | 100 |

评测指标目标：

| 指标 | 最低门槛 | 理想目标 |
|---|---:|---:|
| tool_selection_accuracy | >= 0.90 | >= 0.93 |
| argument_accuracy | >= 0.90 | >= 0.95 |
| clarify_accuracy | >= 0.85 | >= 0.90 |
| no_tool_accuracy | >= 0.90 | >= 0.95 |
| fallback_accuracy | >= 0.85 | >= 0.90 |
| execution_success_rate | >= 0.90 | >= 0.95 |
| final_answer_grounded_rate | >= 0.90 | >= 0.95 |
| overall_pass_rate | >= 0.75 | >= 0.80 |

## 11. 交付物清单

数据部门需要交付：

1. `stage2_amap_tool_use_v2_source.json` 或 `.jsonl`
2. `stage2_amap_tool_use_v2_sft.json`
3. `stage2_amap_tool_use_v2_report.json`
4. `stage2_amap_tool_use_v2_golden.json`
5. 人工抽检表，包含抽检条数、错误类型、修复状态。

report 至少包含：

- 总样本数。
- 各桶数量。
- 各子类数量。
- 城市分布。
- 工具链分布。
- 平均消息轮数。
- 最大样本长度。
- 自动 validator 结果。
- 人工抽检结果。

## 12. 最终判断

Stage2 v2 的核心不是增加数据量，而是提升数据质量和重新分配训练信号。

当前最需要强化：

1. 先澄清，不猜参数。
2. 显式两步链路，不省略 geocode。
3. 成功工具结果后的自然语言表达。
4. 单位转换和用户体验。

当前需要降低：

1. fallback 占比。
2. 单步工具调用占比。
3. 机械模板化回答。
4. 秒数、米数等原始字段复述。

只有当 v2 数据满足以上质量要求后，才建议重新进行 Stage2 LoRA 训练。
