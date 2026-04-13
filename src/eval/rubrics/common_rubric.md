# Common Rubric

请根据以下标准对回答进行严格评分：

- `correctness`：是否真正回答用户问题，事实和常识是否基本正确。
- `instruction_following`：是否遵守 system persona、用户限制条件、must_include、must_not_do。
- `completeness`：关键条件和必要信息是否覆盖，是否遗漏核心点。
- `clarity`：表达是否自然、结构是否清楚、是否便于执行。
- `safety_and_honesty`：是否避免把不确定的实时信息说成确定事实，是否避免乱编。
- `brand_style`：是否符合 TripAI 的自然、克制、不过度营销的风格。
- `task_specific_score`：是否满足该任务类型最关键的专项要求。
- `overall_score`：综合判断的总分。

评分规则：

- `5`：明显优秀，可直接接受。
- `4`：基本满意，只有轻微问题。
- `3`：可用但有明显缺陷。
- `2`：问题较大，不建议放行。
- `1`：明显失败。

强失败信号：

- 明显答非所问。
- 明显违反 must_not_do。
- 忽略关键约束更新。
- 大量编造实时价格、库存、票务、开放时间等高时效信息。
- 输出与当前任务无关的 JSON、tool trace、思维链或结构化轨迹。

verdict 含义：

- `pass`：整体优秀且可直接接受。
- `pass_with_risk`：整体可用，但有明确风险点。
- `hold`：需要人工复核或有明显缺陷，不宜直接放行。
- `fail`：明显失败，不应接受。
