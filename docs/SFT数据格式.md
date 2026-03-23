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

## 5. 当前可用模块

当前阶段可作为主路径的数据模块：

- `handler_guide_generation.py`
  - 输出攻略生成样本
  - `assistant.content` 为自然语言攻略正文或局部规划
- `handler_dialogue.py`
  - 输出多轮需求补充、改口、约束更新样本
- `handler_roleplay_safety.py`
  - 输出品牌身份和安全约束样本

## 6. 路径约定

- 原始数据：`data/raw/`
- 单模块产物：`data/processed/`
- 当前阶段主混合产物：`data/final/stage1_general_sft.json`
