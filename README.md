# TripAI 模型训练工程上下文

## 一、仓库当前状态

当前仓库已经完成了 `Qwen3-8B-Instruct` 在旅游领域自然语言能力上的第一阶段验证，并正式进入 **工具调用 / function calling 对齐阶段**。

需要明确的项目定位是：

- `8B` 的角色是 **MVP 全链路验证模型**
- `32B` 才是后续正式产品的主力模型
- 当前仓库的价值，不是把 `8B` 打磨成最终上线模型，而是先把：
  - 数据构造
  - 本地训练
  - 云端部署
  - 冻结评测
  - 规则打分
  - LLM Judge
  - 工具调用训练链路
  全部跑通

因此，当前仓库的主任务已经从：

- `8B Domain SFT`

切换为：

- `8B Tool Use / Function Calling MVP`

一句话理解：

> 8B 负责验证“这套方法能不能跑通”，32B 负责承接“正式产品能力”。

---

## 二、已经完成的阶段

### 2.1 Stage1：旅游领域自然语言 SFT

这部分已经完成的事情包括：

- 基于 `Qwen3-8B-Instruct` 做旅游领域 SFT
- 训练六类自然语言任务：
  - `guide_generation`
  - `travel_qa`
  - `hotel_recommendation`
  - `traffic_planning`
  - `persona_understanding`
  - `multi_turn_dialogue`
- 完成远程推理评测链路
- 完成规则层打分脚本
- 完成基于 `qwen-plus` 的 LLM-as-a-Judge

这一阶段的结论是：

- `8B` 已经足够作为 **自然语言链路验证基线**
- 没必要再继续重投入打磨 `8B` 的纯自然语言效果
- 后续这套自然语言 eval 主要承担 **回归检查** 的角色

### 2.2 Stage1 产物的用途

Stage1 现在的用途不是继续作为主战场，而是：

- 作为后续 tool-use 阶段的回归基线
- 验证工具训练没有把旅游自然语言能力训坏
- 证明 `8B -> 训练 -> 部署 -> 评测 -> 打分 -> Judge` 这条链路成立

---

## 三、当前阶段目标：进入 Tool Use / Function Calling

当前阶段不再把“纯自然语言旅游助手能力”作为唯一目标，而是开始训练模型具备 **工具调用编排能力**。

这里的目标不是训练一个完整 Agent 平台，而是先让模型具备以下核心能力：

1. 识别什么时候应该调用工具
2. 识别什么时候不应该调用工具
3. 正确选择工具
4. 正确抽取工具参数
5. 在参数不充分时先发起澄清
6. 在工具失败时做合理回退
7. 在拿到工具结果后生成对用户有用的最终回答

当前阶段的正确目标是：

- 把 `Qwen3-8B-Instruct` 从“旅游领域自然语言助手”
- 推进到“能调用 TripAI 工具链的 MVP 编排模型”

---

## 四、关于 Qwen3-8B-Instruct 的能力判断

`Qwen3-8B-Instruct` **本身就具备基础 function calling / tool use 能力**，不是完全从零开始。

这意味着当前阶段要做的，不是“从 0 教模型学会调用工具”，而是：

- 把它对齐到 TripAI 的工具定义
- 把它对齐到 TripAI 的参数 schema
- 把它对齐到 TripAI 的调用风格
- 把它对齐到 TripAI 的 no-tool / fallback / clarification 规则

因此，当前阶段的训练本质上是：

- **通用工具能力 -> 业务工具协议对齐**

而不是：

- **完全从零发明工具调用能力**

---

## 五、8B 与 32B 的分工

### 5.1 8B 的职责

`8B` 当前主要负责：

- 验证工具调用数据格式是否可训练
- 验证工具调用 eval 是否可落地
- 验证部署链路是否支持工具调用推理
- 验证训练配置、LoRA、推理模板、工具协议是否闭环
- 低成本快速迭代

### 5.2 32B 的职责

`32B` 后续主要负责：

- 承接正式产品上线能力
- 承接更强的推理、规划、复杂多工具链路
- 承接更高质量的用户体验与任务完成度

### 5.3 当前仓库的工程判断

当前仓库的工程重点应当是：

- 用 `8B` 把方法跑通
- 再把同样的方法迁移到 `32B`

而不是：

- 在 `8B` 上无限抠最终质量

---

## 六、当前阶段边界

### 当前阶段做什么

- 做 tool use / function calling 数据构造
- 做 TripAI 业务工具协议对齐
- 做 `Qwen3-8B-Instruct` 的工具调用 SFT
- 做工具调用评测
- 保留 stage1 自然语言回归评测
- 验证工具调用训练是否能稳定部署与推理

### 当前阶段不强求什么

- 不要求 `8B` 成为最终生产模型
- 不要求复杂通用 Agent 自主规划
- 不要求一开始就做大规模多工具长链推理
- 不要求一开始就做 DPO / RLHF
- 不要求一开始就替代全部旧业务系统

当前阶段优先级应该是：

1. 工具协议统一
2. 数据格式统一
3. 调用行为稳定
4. 可评测
5. 可部署
6. 可迁移到 32B

---

## 七、推荐的 Tool Use 数据目标

### 7.1 第一轮推荐规模

如果当前只是做 `8B MVP` 验证，推荐的 tool-use 数据规模为：

- `1500 ~ 3000` 条高质量样本

这个阶段不建议一开始就做超大规模数据。

重点是：

- schema 清晰
- 行为稳定
- eval 可复用
- 失败样本可定位

### 7.2 推荐数据桶

当前阶段建议按以下 6 类任务桶构造 tool-use SFT：

1. `single_tool_call`
   - 单工具直接调用
2. `slot_filling_tool_call`
   - 参数抽取 / 参数映射 / schema 对齐
3. `clarify_then_call`
   - 参数缺失时先澄清再调用
4. `no_tool_needed`
   - 不应调用工具，直接自然语言回答
5. `tool_result_grounded_answer`
   - 使用工具结果组织最终回复
6. `tool_failure_fallback`
   - 工具失败、空结果、超时、信息不足时的回退

### 7.3 推荐配比

- `single_tool_call`: `35%`
- `slot_filling_tool_call`: `20%`
- `clarify_then_call`: `15%`
- `no_tool_needed`: `10%`
- `tool_result_grounded_answer`: `10%`
- `tool_failure_fallback`: `10%`

32B stage2 当前默认采用测试结果驱动的 `3200` 条 v2 配比：

- `single_tool_call`: `480`
- `slot_filling_tool_call`: `640`
- `clarify_then_call`: `800`
- `tool_result_grounded_answer`: `480`
- `no_tool_needed`: `320`
- `tool_failure_fallback`: `480`

---

## 八、当前阶段最关键的设计：统一工具协议

开始训练前，必须先冻结一套统一协议。

当前仓库不应混用：

- 旧 `intentionName` JSON
- 自定义自由格式 JSON
- OpenAI tools schema
- 不同工具模板的混搭版本

建议统一采用：

- **OpenAI-compatible tools / function schema**
- 在训练数据中显式保留：
  - `tools`
  - `assistant tool call`
  - `tool result`
  - `assistant final answer`

建议每个工具至少包含：

- `name`
- `description`
- `parameters`（JSON Schema）

工具返回也要统一：

- 成功返回格式
- 空结果格式
- 错误返回格式

只有工具协议冻结后，训练数据才不会越训越乱。

---

## 九、推荐的 Tool Use 样本格式

推荐使用 ChatML 风格样本，并显式保留工具定义和工具结果。

参考结构：

```json
{
  "id": "tool_000001",
  "task_type": "single_tool_call",
  "scene": "hotel_search",
  "difficulty": "medium",
  "messages": [
    {"role": "system", "content": "你是 TripAI 旅行助手。"},
    {"role": "user", "content": "帮我找明天晚上在西湖附近能住的酒店。"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_hotel",
        "description": "根据城市、区域、入住日期等条件搜索酒店",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "area": {"type": "string"},
            "check_in_date": {"type": "string"}
          },
          "required": ["city", "area"]
        }
      }
    }
  ],
  "expected_behavior": "should_call_tool",
  "messages_with_answer": [
    {"role": "system", "content": "你是 TripAI 旅行助手。"},
    {"role": "user", "content": "帮我找明天晚上在西湖附近能住的酒店。"},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "search_hotel",
            "arguments": "{\"city\":\"杭州\",\"area\":\"西湖附近\",\"check_in_date\":\"明天\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": "{\"results\":[...]}"
    },
    {
      "role": "assistant",
      "content": "根据当前搜索结果，我更推荐你优先考虑..."
    }
  ]
}
```

说明：

- `messages` 用于描述原始输入
- `tools` 用于显式定义工具集合
- `messages_with_answer` 用于训练完整交互轨迹
- `expected_behavior` 用于区分：
  - `should_call_tool`
  - `should_clarify`
  - `should_answer_directly`
  - `should_fallback`

---

## 十、当前训练建议

### 10.1 起始模型

当前阶段建议的工具训练起点是：

- 推理基线验证：原生 `Qwen/Qwen3-32B`
- 正式 tool-use SFT：从当前最好的旅游领域 SFT checkpoint 继续训练

也就是说：

- 先用原生模型测基础 tool use 能力
- 再从 TripAI 旅游版 checkpoint 继续训，保留已有旅游理解能力

### 10.2 训练方式

建议继续复用当前 LoRA / QLoRA 链路：

- `finetuning_type: lora`
- `template: qwen`
- `cutoff_len: 4096` 起步
- 如果工具结果较长，再扩到 `8192`

当前阶段不建议一开始就上：

- DPO
- RLHF
- 复杂多阶段对齐

先把 SFT 跑通最重要。

### 10.3 建议新增配置

建议后续在 `configs/` 下新增：

- `llamafactory_stage2_tool_use_sft.yaml`
- `llamafactory_dataset_info_stage2_tool_use.json`

并保持与 stage1 配置物理隔离。

---

## 十一、评测策略也要切换

### 11.1 现有自然语言 eval 的角色

`src/eval/` 下现有这套评测，已经主要用于：

- stage1 自然语言能力回归
- 检查工具训练后是否把基础旅游助手能力训坏

它不再是 tool-use 阶段的主评测。

### 11.2 Tool Use 阶段需要新增主评测

接下来需要单独建设 tool-use eval。

推荐至少覆盖这些指标：

1. 工具选择是否正确
2. 参数抽取是否正确
3. 什么时候应该澄清
4. 什么时候不应该调用工具
5. 工具失败时是否回退合理
6. 工具结果是否被正确整合进最终回复
7. 多工具链路是否稳定

### 11.3 当前推荐的双评测机制

后续每轮 tool-use 训练都建议同时跑两套评测：

1. **tool-use 主评测**
   - 判断工具能力是否提升
2. **stage1 自然语言回归评测**
   - 判断旅游基础能力是否退化

---

## 十二、当前仓库目录理解

### 已经可用的部分

- `configs/`
  - 当前已有 stage1 SFT 配置
- `src/eval/scripts/run_eval.py`
  - 远程推理评测
- `src/eval/scripts/score_rules.py`
  - 规则层打分
- `src/eval/scripts/judge_with_llm.py`
  - LLM as a Judge
- `src/eval/`
  - 当前 stage1 自然语言冻结评测集

### 接下来建议新增的部分

- `data/tool_use/`
  - tool-use SFT 数据
- `src/tool_eval/` 或 `src/eval_tool_use/`
  - tool-use 专项评测
- `configs/llamafactory_stage2_tool_use_sft.yaml`
  - tool-use 训练配置

---

## 十三、建议的执行顺序

当前阶段最稳妥的推进顺序是：

1. 冻结当前 stage1 自然语言基线
2. 冻结 TripAI 工具 schema
3. 用原生 `Qwen/Qwen3-32B` 做 0-shot tool baseline
4. 构造 `stage2_tool_use_sft` 数据集
5. 从当前旅游 SFT checkpoint 继续做 tool-use LoRA
6. 建立 tool-use eval
7. 同时跑：
   - tool-use eval
   - stage1 自然语言回归 eval
8. 稳定后再迁移到 `32B`

---

## 十四、一句话原则

当前仓库已经不再以“继续打磨 8B 旅游自然语言能力”为主要目标，  
而是以 `8B` 为低成本验证基线，先把 **TripAI 的工具调用训练、部署、评测和回归闭环** 跑通，再迁移到 `32B` 正式模型。

---

## 十五、Stage2 高德 Tool-Use MVP 落地清单

当前仓库已经补齐了一条独立的 `stage2_amap` 轨道，执行时按下面顺序推进：

1. **先跑原生 Qwen/Qwen3-32B 基线摸底**
   - 数据集：`src/tool_eval/datasets/native_tool_baseline.json`
   - 脚本：`src/tool_eval/scripts/run_native_tool_baseline.py`
   - 分析：`src/tool_eval/scripts/analyze_native_tool_baseline.py`
   - 目标：确认原生模型对 `tools / tool_calls / arguments` 的天然偏好，再对齐 TripAI 协议

2. **冻结 TripAI 高德工具协议**
   - 协议定义：`src/tool_use/protocol.py`
   - 固定工具：
     - `amap_geocode`
     - `amap_search_poi`
     - `amap_plan_route`
   - 固定边界：
     - 优先单工具
     - 最多两步链路
     - 缺参先澄清
     - 无需工具时直接回答
     - 工具失败时 fallback

3. **构造 stage2 训练数据**
   - 构造脚本：`src/data_pipeline/build_stage2_amap_tool_use.py`
   - 默认输出：
     - `data/tool_use/stage2_amap_tool_use_source.json`
     - `data/final/stage2_amap_tool_use_sft.json`
     - `data/final/stage2_amap_tool_use_report.json`
   - 也可直接运行：`bash scripts/05_build_stage2_amap_data.sh`

4. **校验 source 和导出格式**
   - 校验脚本：`src/data_pipeline/validate_tool_use_dataset.py`
   - 导出脚本：`src/data_pipeline/export_stage2_amap_tool_use.py`
   - 说明：
     - source 继续保留 `{messages, tools, expected_behavior, messages_with_answer}`
     - 导出后对齐 LLaMA-Factory 可训练的 `sharegpt + tools` 结构

5. **准备 stage2 起始模型**
   - merge 配置：`configs/llamafactory_stage1_merge_for_stage2.yaml`
   - merge 脚本：`scripts/04_merge_stage1_for_stage2.sh`
   - 目标：先把 stage1 旅游 LoRA merge 成 `stage1_merged_base`，再新开 stage2 LoRA

6. **跑 stage2 smoke / 正式 SFT**
   - 训练配置：`configs/llamafactory_stage2_amap_tool_use_sft.yaml`
   - 数据注册：`configs/llamafactory_dataset_info_stage2_amap_tool.json`
   - 训练入口：`bash scripts/02_run_sft.sh stage2_amap`

7. **接真实高德执行链路**
   - 高德客户端：`src/tool_use/amap_client.py`
   - 工具编排器：`src/tool_use/orchestrator.py`
   - 运行时环境变量：
     - `AMAP_API_KEY`

8. **跑 stage2 tool-use 专项评测**
   - 评测说明：`src/tool_eval/README.md`
   - golden 数据：`src/tool_eval/datasets/stage2_amap_golden.json`
   - 执行脚本：
     - `src/tool_eval/scripts/run_tool_eval.py`
     - `src/tool_eval/scripts/score_tool_eval.py`

9. **继续跑 stage1 自然语言回归**
   - 保持 `src/eval/` 只负责自然语言回归，不与 tool-use 主评测混用

更完整的 step-by-step 执行说明见：

- `docs/stage2_amap_tool_use_workflow.md`
