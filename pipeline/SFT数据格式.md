# SFT 数据格式约定

本项目统一使用 ChatML 风格的 JSON 数组作为 SFT 训练输入。

## 顶层结构

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

## 约束

- 顶层必须是 JSON 数组。
- 每条样本必须包含 `messages` 字段。
- `messages` 必须是数组，且至少包含一条 `user` 和一条 `assistant` 消息。
- 每条消息必须包含：
  - `role`
  - `content`
- `content` 必须是非空字符串。
- 多轮对话允许出现多个 `user` 和 `assistant`，也允许额外出现 `system`、`tool`、`observation`。

## 各模块建议

- `handler_itinerary.py`
  - 输出长攻略型样本
  - `assistant.content` 保留完整攻略正文
- `handler_intent.py` / `handler_rag.py`
  - `assistant.content` 优先输出纯 JSON 字符串
- `handler_multiturn.py`
  - 直接保留上下文多轮消息
- `handler_roleplay_safety.py`
  - 用于“小奇”人设与安全拒答
- `handler_basic_qa.py`
  - 用于景点百科类问答

## 路径约定

- 原始数据：`data/raw/`
- 单模块产物：`data/processed/`
- 最终混合产物：`data/final/`
