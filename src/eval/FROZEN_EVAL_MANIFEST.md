# Stage1 Frozen Eval Manifest

## 1. 结论

当前 `src/eval/` 下的 6 类评测集已经整理为可冻结、可复用的 `stage1` 主评测集。

冻结结论：**可以冻结**。

建议冻结名：`stage1_main_eval_v1_frozen`

适用范围：

- `8B Domain SFT`
- `stage1 baseline` 训练后评测
- 后续轮次横向对比
- bad case 修复后的回归对照

当前这套评测集不用于：

- tool calling / function calling 阶段
- JSON 路由输出评测
- 实时票务、库存、价格类能力评测

---

## 2. 冻结文件清单

当前主评测集共 6 个文件，每类 `60` 条，总计 `360` 条。

- `eval_guide_generation.json`
- `eval_travel_qa.json`
- `eval_hotel_recommendation.json`
- `eval_traffic_planning.json`
- `eval_persona_understanding.json`
- `eval_multi_turn_dialogue.json`

目录位置：`src/eval/`

---

## 3. 冻结前核查结果

本轮已完成如下核查：

1. 结构一致性
- 六类文件均使用统一字段：
  - `id`
  - `split`
  - `task_type`
  - `scene`
  - `difficulty`
  - `tags`
  - `messages`
  - `reference_answer`
  - `must_include`
  - `must_not_do`
  - `notes`
  - `seed_source`
  - `seed_sample_id`

2. 训练隔离
- 与对应 `processed` 训练集做了精确字符串核查
- 结果：六类均为
  - `exact_user_overlap = 0`
  - `exact_reference_overlap = 0`

3. 样本规模
- 每类 `60` 条
- 总计 `360` 条

4. `reference_answer` 去重
- 六类均已达到逐条可区分
- 六类均为 `unique_reference_answers = 60`

5. 多轮深度分层
- `eval_multi_turn_dialogue.json` 已覆盖：
  - `4` 条消息深度：`20` 条
  - `6` 条消息深度：`20` 条
  - `8` 条消息深度：`20` 条

6. 可冻结性
- 当前已无结构性阻塞问题
- 可作为长期主 `eval` 使用

---

## 4. 当前设计约定

### 4.1 语言约定

- 用户题面与对话上下文以中文为主
- 一部分 `reference_answer`、`notes`、`must_include`、`must_not_do` 使用英文说明

这样做的原因：

1. 当前环境下英文 rubric 更稳定，避免编码问题反复污染冻结集
2. 题面和实际待评测输入仍然保持中文，不影响模型评测
3. rubric 字段主要供评测脚本、人工复核和 AI Judge 使用

### 4.2 `reference_answer` 的角色

`reference_answer` 在本项目中是“评分参考方向”，不是唯一标准答案。

它的作用是：

- 帮助人工评测快速理解这题在测什么
- 帮助 AI Judge 把注意力放在正确维度上
- 保持后续多轮版本比较口径稳定

它不要求：

- 与模型答案逐字匹配
- 提供唯一正确表达
- 充当训练标签回灌模型

### 4.3 `seed_source` / `seed_sample_id`

这两个字段用于离线追溯：

- 说明该评测题从哪一类 `processed` 数据启发而来
- 用于后续审计和排查
- 不参与模型输入

---

## 5. 冻结后使用规则

冻结后建议遵守以下规则：

1. 不改题面
- 不改 `messages`
- 不改最后一轮用户要求
- 不改多轮上下文结构

2. 不改评测意图
- 不改 `task_type`
- 不改 `scene`
- 不改这题原本要测的能力点

3. 允许的小修改
- 修正明显错字
- 修正元信息拼写
- 补充说明文档
- 增加外部评测脚本或报告，不改题本身

4. 如果必须改题
- 不能原地静默覆盖
- 应复制出新版本，例如：
  - `stage1_main_eval_v2_candidate`
- 待审核后再决定是否替换主冻结集

---

## 6. 建议的评测执行方式

建议按以下三层执行：

1. 规则检查
- 空回答
- 非法结构化输出
- 明显违反 `must_not_do`

2. AI Judge
- 基于 `reference_answer`
- 结合 `must_include` / `must_not_do`
- 输出分项评分和总评

3. 人工复核
- 复核低分样本
- 复核失败样本
- 复核多轮难样本
- 复核版本差异最大的样本

---

## 7. 当前冻结集的覆盖说明

### `guide_generation`

- 重点覆盖按天规划
- 覆盖首访、情侣、老人、已有安排补全、粗计划优化等题型

### `travel_qa`

- 覆盖城市落脚点、区域比较、景点判断、交通类问答
- 覆盖 `easy / medium / hard`

### `hotel_recommendation`

- 覆盖预算、人群、区域、便利性、住宿类型判断
- 重点考察“推荐对象 - 推荐理由”关系

### `traffic_planning`

- 覆盖不同交通衔接场景
- 重点考察稳定性、换乘负担、步行强度、可执行性

### `persona_understanding`

- 覆盖不同画像、预算和偏好组合
- 重点考察“因人而异”的推荐逻辑

### `multi_turn_dialogue`

- 覆盖新增约束、推翻旧条件、缩短天数、延长天数、删除景点、局部修改
- 已做深度分层，适合长期复用

---

## 8. 后续建议

下一步建议：

1. 继续沿用 [run_eval.py](/d:/soulv_llm/src/eval/scripts/run_eval.py) 作为冻结集的统一推理入口
2. 训练完成后优先在阿里云上用 `vLLM` 部署模型，并暴露 OpenAI 兼容接口
3. 固定一份标准评测报告模板
4. 把每次 checkpoint 的评测结果落到 `reports/`
5. 若后续新增第 7 类或进入 tool-use 阶段，不直接修改本冻结集，而是新增新阶段评测集

---

## 9. 一句话说明

当前 `src/eval/` 下这 6 个文件，已经可以作为 `stage1` 后续训练轮次长期复用的主冻结评测集。

