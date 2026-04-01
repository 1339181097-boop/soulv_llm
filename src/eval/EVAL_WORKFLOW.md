# Stage1 Eval Workflow

## 1. 目的

本文件定义 `stage1_main_eval_v1_frozen` 的复用流程。

目标不是说明如何建设评测集，
而是说明在评测集已经冻结后，后续每一轮模型训练、对比、回归和版本评审应如何稳定复用这套评测。

适用范围：

- `8B Domain SFT`
- `stage1 baseline`
- `stage1` 后续多轮数据迭代
- 小规模修复实验
- 正式版本横向比较

不适用范围：

- tool calling / function calling 阶段
- JSON 路由输出阶段
- 新任务类型大幅扩展后的评测

---

## 2. 固定输入

每次复用评测时，固定输入为：

1. 冻结评测集
- `eval_guide_generation.json`
- `eval_travel_qa.json`
- `eval_hotel_recommendation.json`
- `eval_traffic_planning.json`
- `eval_persona_understanding.json`
- `eval_multi_turn_dialogue.json`

2. 待评测模型
- 某个 checkpoint
- 某个 LoRA adapter
- 某个 merge 后模型

3. 固定评测配置
- 同一套 system prompt 使用原则
- 同一套推理参数
- 同一套打分标准

评测时应尽量避免在不同模型之间切换推理参数，
否则结果会掺入额外变量。

---

## 3. 标准流程

后续每次版本评测统一按以下步骤执行：

1. 选择模型版本
2. 用冻结评测集逐题推理
3. 保存原始回答
4. 跑规则检查
5. 跑 AI Judge
6. 人工复核重点样本
7. 生成报告
8. 做版本结论
9. 归档结果

---

## 4. Step 1：选择模型版本

每次评测前先明确本次要评的模型版本。

建议至少记录：

- 模型名
- checkpoint 路径
- 是否是 base 重训
- 是否是 adapter 续训
- 训练集版本
- 训练轮次
- 备注

示例：

- `qwen3_8b_stage1_baseline`
- `qwen3_8b_stage1_round2_accumulated`
- `qwen3_8b_stage1_bugfix_adapter`

---

## 5. Step 2：逐题推理

对冻结评测集中的 6 个文件逐题推理。

要求：

1. 每条样本按原始 `messages` 输入模型
2. 不改题面
3. 不补充额外提示词
4. 不人工修饰模型输出
5. 多轮样本必须完整输入上下文

建议输出字段：

- `id`
- `task_type`
- `model_name`
- `messages`
- `prediction`
- `inference_config`
- `timestamp`

建议按文件分别保存，也可以额外汇总成总文件。

---

## 6. Step 3：保存原始回答

模型原始输出必须保留，不能只保留分数。

原因：

1. 方便人工复核
2. 方便 bad case 回放
3. 方便后续对比两个版本的真实差异
4. 方便排查 AI Judge 的误判

建议保存位置：

```text
src/eval/reports/
└─ <model_name>/
   ├─ raw_outputs_guide_generation.json
   ├─ raw_outputs_travel_qa.json
   ├─ raw_outputs_hotel_recommendation.json
   ├─ raw_outputs_traffic_planning.json
   ├─ raw_outputs_persona_understanding.json
   ├─ raw_outputs_multi_turn_dialogue.json
   └─ summary.json
```

---

## 7. Step 4：规则检查

规则检查用于发现硬错误。

建议优先检查：

1. 空回答
2. 回答过短
3. 输出 JSON / tool trace
4. 明显违反 `must_not_do`
5. 多轮样本忽略最后一轮用户约束
6. 明显编造实时票价、库存、班次、开放时间

规则检查的定位：

- 不是完整评分
- 是“硬错误过滤器”
- 可以作为直接失败信号

建议输出：

- 是否通过规则检查
- 触发了哪些规则
- 严重程度

---

## 8. Step 5：AI Judge

AI Judge 用于做大盘评分和版本比较。

Judge 输入建议包括：

- 原始题目 `messages`
- 模型输出 `prediction`
- `reference_answer`
- `must_include`
- `must_not_do`
- 对应任务类型 rubric

Judge 输出建议包括：

- `correctness`
- `instruction_following`
- `completeness`
- `clarity`
- `safety_and_honesty`
- `task_specific_score`
- `overall_score`
- `pass_or_fail`
- `judge_reason`

Judge 的定位：

- 用于全量批跑
- 用于横向比较模型版本
- 用于筛出争议样本和低分样本

Judge 不是最终唯一裁决。

---

## 9. Step 6：人工复核

人工复核不需要全量逐条精批，
但必须保留。

建议至少复核：

1. 所有失败样本
2. 所有规则检查未通过样本
3. 所有多轮高难样本中的低分样本
4. 与上一版本差异最大的样本
5. 每类随机抽样若干条

人工复核重点：

- Judge 是否误判
- 模型是否确实回答到点上
- 是否存在表面得分不错、实际业务价值很低的回答
- 是否存在多轮继承错误

---

## 10. Step 7：生成评测报告

每次正式评测都应生成固定格式报告。

报告最少应包含：

1. 模型版本
2. 训练集版本
3. 评测集版本
4. 评测时间
5. 总样本数
6. 六类样本数
7. 总体通过率
8. 六类平均分
9. 六类通过率
10. 规则检查失败数
11. 低分样本示例
12. bad case 示例
13. 与上一版本对比结论

报告不建议只输出一个总分。

---

## 11. Step 8：做版本结论

每次评测后都应形成明确结论，至少回答以下问题：

1. 这版是否优于上一版
2. 哪几类变好
3. 哪几类变差
4. 是否存在明显回归
5. 是否适合进入下一轮
6. 是否适合做正式 baseline 或正式版本

建议结论分级：

- `pass`
- `pass_with_risk`
- `hold`
- `fail`

说明：

- `pass`：可作为当前阶段候选正式版
- `pass_with_risk`：整体可用，但有明显风险点
- `hold`：需要补充验证或修复后再判断
- `fail`：不建议继续推进

---

## 12. Step 9：归档结果

每次评测完成后，应把结果归档。

建议归档内容：

- 原始模型输出
- 规则检查结果
- AI Judge 结果
- 人工复核记录
- 最终 summary
- 对比结论

建议命名：

```text
src/eval/reports/<model_name>/
```

例如：

```text
src/eval/reports/qwen3_8b_stage1_baseline/
src/eval/reports/qwen3_8b_stage1_round2_accumulated/
src/eval/reports/qwen3_8b_stage1_bugfix_adapter/
```

---

## 13. 推荐的复用节奏

### 13.1 第一轮 baseline

在第一轮正式训练完成后：

1. 跑全套冻结 eval
2. 生成人工可读报告
3. 确认 `stage1 baseline`

### 13.2 第二轮及以后正式版

每次正式重训后：

1. 跑同一套冻结 eval
2. 与上一正式版对比
3. 看是否整体提升

### 13.3 小规模修复实验

对于 adapter 续训或 bad case 修复：

1. 仍然跑全套 eval
2. 重点看相关类别和历史 bad case
3. 避免“修了一类，退化三类”

---

## 14. 什么时候不应继续直接复用这套 eval

出现以下情况时，应考虑新增新阶段 eval，而不是直接修改本冻结集：

1. 任务边界变化很大
2. 进入 tool-use / function-calling 阶段
3. 新增全新核心能力类型
4. 线上产品目标已明显超出 `8B Domain SFT`

原则：

- 老冻结集继续保留，用于历史可比性
- 新阶段另建新 eval
- 不直接覆盖旧冻结集

---

## 15. 一句话工作流

以后每次模型迭代统一按：

`训练 -> 冻结 eval 推理 -> 规则检查 -> AI Judge -> 人工复核 -> 报告 -> 版本结论 -> 归档`

这套流程应保持稳定，避免每一轮都临时改口径。

---

## 16. 推荐的云端部署方式

对当前项目，推荐的评测链路是：

1. 在阿里云 GPU 机器上完成训练
2. 将待评测 checkpoint 或 merge 后模型部署为 `vLLM` 服务
3. 暴露 OpenAI 兼容接口，例如 `/v1/chat/completions`
4. 在本地开发机或任意可访问该服务的机器上运行 `src/eval/scripts/run_eval.py`
5. 把原始输出与报告保存在仓库 `src/eval/reports/` 下

这样做的优点：

- 本地不需要部署模型
- 训练环境和评测环境解耦
- 后续 baseline、round2、round3 都能复用同一套 eval 脚本
- 与 `vLLM` 的主流部署方式兼容

推荐的 `vLLM` 服务形态示例：

```bash
vllm serve /path/to/merged-or-base-plus-lora-model \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192
```

评测命令示例：

```bash
python src/eval/scripts/run_eval.py \
  --base-url http://<your-server>:8000/v1 \
  --api-key EMPTY \
  --model <served-model-name> \
  --model-name qwen3_8b_stage1_baseline \
  --run-name stage1_baseline_eval \
  --max-new-tokens 512 \
  --temperature 0
```

如果后续接入网关、鉴权层或内部推理平台，只要它仍然兼容 OpenAI chat completions 输入输出格式，就不需要改冻结 eval 本身，只需要继续复用这套 runner。
