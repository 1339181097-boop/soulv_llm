# SFT 数据格式约定

本文件描述当前 8B Domain SFT 阶段的数据契约。  
当前阶段只训练旅游领域自然语言能力，不训练意图识别 JSON，不训练工具调用。

## 1. 顶层结构

本项目统一使用 ChatML 风格的 JSON 数组作为 SFT 训练输入。

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "系统提示词"
      },
      {
        "role": "user",
        "content": "用户问题"
      },
      {
        "role": "assistant",
        "content": "模型标准答案"
      }
    ]
  }
]
```

## 2. 当前阶段约束

- 顶层必须是 JSON 数组。
- 每条样本必须包含 `messages` 字段。
- `messages` 必须是数组，且至少包含一条 `user` 和一条 `assistant` 消息。
- 每条消息必须包含 `role` 和 `content`。
- `content` 必须是非空字符串。
- 当前阶段允许多轮 `user` / `assistant`，也允许保留一条 `system`。
- 当前阶段训练样本默认只使用 `system`、`user`、`assistant` 三种角色。

## 3. 当前阶段禁止项

以下内容不属于当前 Domain SFT 数据：

- 纯 `{"intentionName": ...}` 分类输出
- `tool_calls` / `function_call` / `function_calls`
- `tool` / `observation` 轨迹消息
- 任何模拟工具执行结果的样本

当前阶段 `assistant.content` 必须是自然语言回复。

## 4. 离线元信息

如有需要，每条样本可以在顶层保留离线元信息，例如：

- `id`
- `task_type`
- `scene`
- `source`
- `brand_style`
- `difficulty`

这些字段只用于离线分析、采样、审计，不应成为模型要学习输出的内容。

## 5. 当前处理模块

当前仓库已经按任务类型拆分处理器，分别负责不同 SFT 桶的数据清洗与格式化：

- `handler_guide_generation.py`
  - 负责 `guide_generation` 数据
  - 输出结构化、可直接训练的攻略生成样本
- `handler_travel_qa.py`
  - 负责 `travel_qa` 数据
  - 覆盖景点 / 城市 / 交通相关问答
- `handler_hotel_recommendation.py`
  - 负责 `hotel_recommendation` 数据
  - 覆盖住宿推荐与需求理解
- `handler_traffic_planning.py`
  - 负责 `traffic_planning` 数据
  - 覆盖交通规划 / 路线建议 / 出行方式选择
- `handler_multiturn.py`
  - 负责 `multi_turn_dialogue` 数据
  - 覆盖多轮上下文承接与约束更新

当前阶段目标是落实 6 类 SFT 任务：

- `guide_generation`
- `travel_qa`
- `hotel_recommendation`
- `traffic_planning`
- `persona_understanding`
- `multi_turn_dialogue`

其中 `persona_understanding` 目前还没有进入 `data/processed/` 的正式产物，因此当前 `data/final/stage1_general_sft.json` 仍可能以五类候选版方式先行混合。

## 6. 路径约定

- 原始数据：`data/raw/`
- 单模块产物：`data/processed/`
- 当前阶段主混合产物：`data/final/stage1_general_sft.json`
