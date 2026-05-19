# Stage2 文本能力 Badcase 标注与扩展指南

## 1. 背景

本轮评测对象：

```text
qwen3_32b_stage2_amap_tool_use_merged_v2
```

评测集为 TripAI 六类自然语言能力评测集，共 `360` 条：

```text
guide_generation
travel_qa
hotel_recommendation
traffic_planning
persona_understanding
multi_turn_dialogue
```

本轮自动评测后，已抽取 `90` 条 badcase。主要问题集中在：

```text
traffic_planning        36 条
travel_qa               19 条
persona_understanding   16 条
multi_turn_dialogue     13 条
guide_generation         6 条
hotel_recommendation     0 条
```

本轮数据工作的目标不是重做全量数据，而是围绕 badcase 做定向修复：

1. 给每条 badcase 补充高质量正确答案。
2. 基于 badcase 扩展同类修复样本。
3. 为下一轮 SFT/修复训练提供干净、可追溯的数据。

重要原则：

```text
原始 badcase 用于回归测试，不建议直接进入训练集。
训练修复应优先使用 badcase 的同类变体和扩展样本，避免污染评测集。
```

## 2. 交付给数据部门的文件

建议交付以下文件：

```text
data/handoff/qwen3_32b_stage2_text_full_badcases_for_labeling.jsonl
docs/badcase_labeling_and_expansion_guide.md
data/eval/stage1/qwen3_32b_stage2_text_full_badcases_summary.json
```

其中最重要的是：

```text
data/handoff/qwen3_32b_stage2_text_full_badcases_for_labeling.jsonl
```

这是给数据部门标注的主文件。

如果该文件还没有生成，可以在服务器或本地仓库执行：

```bash
mkdir -p data/handoff

jq -c '{
  badcase_id,
  sample_id,
  task_type,
  scene,
  difficulty,
  tags,
  messages,
  reference_answer,
  must_include,
  must_not_do,
  model_prediction,
  failure_sources,
  issue_tags: .error_analysis.judge.issue_tags,
  judge_reason: .error_analysis.judge.judge_reason,
  scores: .error_analysis.judge.scores,
  gold_answer: "",
  gold_answer_notes: "",
  gold_answer_status: "pending"
}' \
data/eval/stage1/qwen3_32b_stage2_text_full_badcases.jsonl \
> data/handoff/qwen3_32b_stage2_text_full_badcases_for_labeling.jsonl
```

检查条数：

```bash
wc -l data/handoff/qwen3_32b_stage2_text_full_badcases_for_labeling.jsonl
```

预期为：

```text
90
```

## 3. 数据部门需要返回的文件

数据部门应返回两个 JSONL 文件。

### 3.1 标注后的 badcase 文件

建议文件名：

```text
qwen3_32b_stage2_text_full_badcases_labeled.jsonl
```

要求：

1. 保留原始字段，不要删除 `badcase_id`、`sample_id`、`task_type`、`messages`、`model_prediction`。
2. 补充 `gold_answer`。
3. 将 `gold_answer_status` 从 `pending` 改成 `completed`。
4. 如有必要，在 `gold_answer_notes` 里说明这条修复什么问题。

### 3.2 Badcase 扩展修复样本文件

建议文件名：

```text
qwen3_32b_stage2_text_badcase_repair_expanded.jsonl
```

这个文件用于后续训练修复。

要求：

1. 不直接复制原始 badcase 题面。
2. 基于 badcase 生成同类变体。
3. 每条扩展样本必须能追溯到原始 `badcase_id`。
4. 每条扩展样本必须有高质量答案。

推荐字段：

```json
{
  "source_badcase_id": "stage1::qwen3_32b_stage2_text_full::traffic_planning::eval_traffic_planning_000002",
  "repair_type": "paraphrase",
  "task_type": "traffic_planning",
  "messages": [
    {
      "role": "system",
      "content": "你是 TripAI 的中文旅行助手，回答要自然、可靠、可执行。"
    },
    {
      "role": "user",
      "content": "扩展后的用户问题"
    }
  ],
  "answer": "高质量修复答案",
  "fix_tags": [
    "no_invented_schedule",
    "uncertainty_acknowledgement",
    "actionable_advice"
  ],
  "notes": "说明这条样本修复什么问题"
}
```

## 4. `gold_answer` 标注要求

每条 badcase 都需要补一个高质量 `gold_answer`。

`gold_answer` 的目标是给出正确示范，不是解释模型为什么错。

基本要求：

1. 直接回答用户问题。
2. 语言自然，像真实旅行顾问。
3. 信息可执行，不空泛。
4. 不输出 JSON、工具调用、函数名、内部推理过程、`<think>`。
5. 不编造实时票价、车次、库存、营业时间、精确耗时。
6. 如果用户信息不足，要澄清或给稳妥的判断方式。
7. 如果涉及强实时信息，要说明以地图、官方平台或实时查询结果为准。

标注完成后字段应类似：

```json
{
  "gold_answer": "正确示范回答...",
  "gold_answer_notes": "修复点：避免编造具体班次，补充路线选择逻辑。",
  "gold_answer_status": "completed"
}
```

## 5. Badcase 扩展原则

每条 badcase 不只补答案，还应扩展同类修复样本。

推荐每条 badcase 扩展：

```text
1 条同义改写
1 条城市/场景替换
1 条约束增强版
1 条边界/缺信息版
可选 1 条多轮变体
```

扩展样本不要机械换城市名，要改变真实约束和表达方式。

错误示例：

```text
把“上海到苏州”改成“北京到天津”，其他全不变。
```

更好的扩展示例：

```text
原问题：从上海去苏州玩一天，高铁往返合适吗？

扩展 1：我从杭州去南京当天往返，坐高铁玩一天现实吗？
扩展 2：广州去深圳一日游，公共交通怎么安排更稳妥？
扩展 3：带老人从上海去苏州当天往返，怎么安排不累？
扩展 4：我想当天往返去周边城市玩，但还没确定出发地，应该先确认哪些信息？
扩展 5：上次你给了苏州一日游安排，现在我想减少换乘，不要整篇重写，只告诉我怎么调整。
```

## 6. 扩展数量建议

本轮建议优先修复高风险类别。

推荐扩展量：

```text
traffic_planning        每条 5-8 条
travel_qa               每条 3-5 条
persona_understanding   每条 3-5 条
multi_turn_dialogue     每条 3-4 条
guide_generation         每条 2-3 条
hotel_recommendation     本轮无需重点扩展
```

建议总量控制在：

```text
400-700 条
```

先做定向小步修复，不建议一次扩到几千条。

## 7. 各任务类型修复重点

### 7.1 traffic_planning

这是本轮最高优先级。

常见问题：

```text
factual_invention
invented_schedule
unrealistic_route
invented_timing
invented_route_detail
unverified_transfer_instructions
```

禁止：

1. 编造具体车次。
2. 编造发车时间。
3. 编造票价。
4. 编造精确耗时。
5. 编造换乘站名。
6. 把不确定路线说成确定事实。

正确答案应包含：

1. 先给结论：这个走法是否合适。
2. 给方案取舍：高铁、地铁、公交、打车、步行各自适合什么情况。
3. 说明风险：高峰、换乘、行李、老人小孩、天气、末班车。
4. 给核验建议：出发前用地图或官方平台确认实时班次和路线。

推荐回答结构：

```text
结论：可以/不太建议/要看条件。
推荐方案：优先 A，备选 B。
为什么：从时间稳定性、换乘复杂度、体力消耗解释。
注意事项：哪些信息需要出发前确认。
```

### 7.2 travel_qa

常见问题：

```text
答非所问
no_clear_recommendation
incomplete_recommendation
lacks_conciseness
```

修复重点：

1. 先回答结论。
2. 不要把短问答写成长攻略。
3. 用户问比较题时，要明确推荐优先级。
4. 用户问是否值得去时，要给适合/不适合人群。
5. 不确定的信息要谨慎，不要硬编。

推荐回答结构：

```text
结论：更推荐 A / 不建议 B / 要看情况。
理由：2-4 点即可。
适合人群：谁适合，谁不适合。
补充建议：必要时给下一步。
```

### 7.3 persona_understanding

常见问题：

```text
generic_recommending
persona_ignored
no_deprioritization
no_ranking_logic
no_recommendation_logic
```

修复重点：

1. 必须体现“因人而异”。
2. 不要只列目的地，要解释为什么适合这个人。
3. 要说明不优先推荐什么，以及原因。
4. 要处理预算、体力、同行人群、偏好之间的冲突。
5. 要给优先级。

错误风格：

```text
你可以去 A、B、C，都不错。
```

正确风格：

```text
如果你不喜欢人多打卡，A 比 B 更适合；B 虽然经典但人流更集中，可以降级为备选。
```

### 7.4 multi_turn_dialogue

常见问题：

```text
rewrote_entire_plan
ignored_context_continuity
no_change_explanation
no_context_preservation
```

修复重点：

1. 继承上文，不要每轮从头开始。
2. 用户要求局部修改时，只改相关部分。
3. 用户推翻旧条件时，要明确哪些旧建议失效。
4. 用户新增限制时，要保留未被推翻的条件。
5. 不要整篇重写，除非用户明确要求。

推荐回答结构：

```text
收到，这次只调整 X。
保留：A、B、C。
修改：把原来的 Y 换成 Z。
原因：更符合你新增的限制。
```

### 7.5 guide_generation

常见问题：

```text
minor_overload_risk
structural_overload
time_sensitive_fact
```

修复重点：

1. 控制行程强度。
2. 不要堆景点。
3. 每天节奏要合理。
4. 不编具体门票、营业时间。
5. 对老人、亲子、雨天、预算等约束要有明显体现。

### 7.6 hotel_recommendation

本轮自动评测表现较好，不是重点扩展对象。

如果扩展，重点仍然是：

1. 不编实时房价和房态。
2. 推荐区域或住宿类型时要解释适合原因。
3. 对预算、交通、老人小孩、安静程度做取舍。

## 8. 质量验收标准

标注和扩展完成后，数据部门应自查以下问题。

硬性失败：

```text
输出 <think> 或推理过程
输出工具调用、函数名、JSON trace
编造实时票价、车次、库存、营业时间
答非所问
多轮对话忽略最新约束
```

质量不足：

```text
只泛泛建议，没有判断
只列景点或区域，没有解释原因
没有优先级
没有说明不推荐什么
没有承认不确定性
交通规划给了无法验证的细节
```

建议每 50 条抽查 5 条，重点看：

1. 是否符合任务类型。
2. 是否修复原 badcase 的失败点。
3. 是否有新的编造或过度承诺。
4. 是否自然、可读、可执行。

## 9. 最终交付清单

数据部门最终应交付：

```text
qwen3_32b_stage2_text_full_badcases_labeled.jsonl
qwen3_32b_stage2_text_badcase_repair_expanded.jsonl
```

其中：

```text
badcases_labeled.jsonl
```

用于人工回归、错误分析和评测对照。

```text
badcase_repair_expanded.jsonl
```

用于下一轮训练修复。

交付前请确认：

```bash
wc -l qwen3_32b_stage2_text_full_badcases_labeled.jsonl
wc -l qwen3_32b_stage2_text_badcase_repair_expanded.jsonl
```

标注版预期：

```text
90 行
```

扩展版建议：

```text
400-700 行
```

## 10. 给数据部门的一句话

请不要只“改答案”，而要把每条 badcase 背后的错误模式抽象出来，做同类变体扩展。原始 badcase 留作回归测试，扩展变体用于训练修复。
