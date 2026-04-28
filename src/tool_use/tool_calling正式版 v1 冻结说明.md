# Stage2 AMap 工具协议冻结说明

冻结评审日期：`2026-04-13`

冻结结论：**可以冻结，可以开始正式生成 stage2_amap tool-use 数据。**

建议冻结名：`stage2_amap_tool_protocol_v1_frozen`

## 1. 冻结范围

本次冻结覆盖以下内容：

- `src/tool_use/protocol.py`
- `src/tool_use/datasets.py`
- `src/tool_use/orchestrator.py`
- `src/tool_use/amap_client.py`
- `src/data_pipeline/build_stage2_amap_tool_use.py`
- `src/data_pipeline/export_stage2_amap_tool_use.py`
- `src/data_pipeline/validate_tool_use_dataset.py`
- `configs/llamafactory_dataset_info_stage2_amap_tool.json`

如果以上文件和本说明冲突，以代码实现为准，其中 `src/tool_use/protocol.py` 是协议单一事实源。

## 2. 冻结后的协议主张

当前 stage2_amap 轨道只冻结 3 个工具：

- `amap_geocode`
- `amap_search_poi`
- `amap_plan_route`

当前只允许两类两步工具链：

- `amap_geocode -> amap_plan_route`
- `amap_geocode -> amap_search_poi`

其他规则：

- 单轮可以不调工具，直接自然回答
- 最多 2 轮工具调用
- 参数缺失时先澄清，再调工具
- 工具失败或结果为空时必须 fallback，不能编造

## 3. 工具 Schema

### 3.1 `amap_geocode`

用途：把地点名称或地址解析为位置结果。

参数：

- `address`: `string`，必填
- `city`: `string`，可选

### 3.2 `amap_search_poi`

用途：搜索酒店、景点、站点等 POI。

参数：

- `keyword`: `string`，必填
- `city`: `string`，可选
- `around_location`: `string`，可选
- `radius_m`: `integer`，可选

### 3.3 `amap_plan_route`

用途：规划路线。

参数：

- `origin`: `string`，必填
- `destination`: `string`，必填
- `mode`: `string`，可选，枚举值固定为 `transit`、`driving`、`walking`、`bicycling`
- `city`: `string`，可选

## 4. 行为标签

协议层冻结 4 个行为标签：

- `should_call_tool`
- `should_clarify`
- `should_answer_directly`
- `should_fallback`

当前 builder 中 6 类样本与行为标签的映射为：

- `single_tool_call` -> `should_call_tool`
- `slot_filling_tool_call` -> `should_call_tool`
- `clarify_then_call` -> `should_clarify`
- `tool_result_grounded_answer` -> `should_call_tool`
- `no_tool_needed` -> `should_answer_directly`
- `tool_failure_fallback` -> `should_fallback`

## 5. Source Dataset 消息协议

source 数据集固定保留 4 类 role：

- `system`
- `user`
- `assistant`
- `tool`

结构要求：

- `messages` 保存原始输入上下文
- `messages_with_answer` 保存完整监督轨迹
- `assistant` 消息必须至少有 `content` 或 `tool_calls`
- `assistant.tool_calls[*]` 必须满足：
- `id` 为非空字符串
- `type` 固定为 `function`
- `function.name` 必须是已冻结工具名
- `function.arguments` 必须是可解析为对象的 JSON 字符串
- `tool` 消息必须满足：
- `tool_call_id` 为非空字符串
- `content` 为可解析 JSON 的字符串

## 6. Tool Result Envelope

工具返回格式冻结为 3 类：

成功：

```json
{"status":"success","data":{...}}
```

空结果：

```json
{"status":"empty","reason":"no_result"}
```

错误：

```json
{"status":"error","reason":"...","retryable":false}
```

后续如果新增字段、改字段名、改 `status` 语义，都应视为协议变更，而不是普通实现细节调整。

## 7. ShareGPT 导出协议

导出到 LLaMA-Factory 时，当前冻结映射为：

- `system` -> `system`
- `user` -> `human`
- `assistant` 自然语言 -> `gpt`
- `assistant.tool_calls` -> `function_call`
- `tool` -> `observation`

额外约束：

- 导出后的 `tools` 字段是 JSON 字符串
- `function_call.value` 是 JSON 字符串，内部必须含 `name` 和 `arguments`
- `observation.value` 是 JSON 字符串

对应注册配置已固定在 `configs/llamafactory_dataset_info_stage2_amap_tool.json`。

## 8. 正式造数默认配比

`build_stage2_amap_tool_use.py` 当前 32B 默认总样本数是 `3200`，默认配比已经在代码中固化为：

- `single_tool_call`: `20%`，即 `640`
- `slot_filling_tool_call`: `18%`，即 `576`
- `clarify_then_call`: `18%`，即 `576`
- `tool_result_grounded_answer`: `22%`，即 `704`
- `no_tool_needed`: `12%`，即 `384`
- `tool_failure_fallback`: `10%`，即 `320`

说明：

- 如果仓库里其他文档出现旧比例描述，以这里和 builder 实现为准
- 调整配比虽然不一定改变 tool schema，但会改变 stage2 训练分布，应作为新数据版本处理

## 9. 可冻结性核查结果

本次核查已确认：

- `build_amap_tool_schemas()` 被 builder、dataset validator、native baseline、orchestrator 复用
- `ALLOWED_TWO_STEP_CHAINS` 同时被数据校验和执行器复用
- tool result envelope 由 `protocol.py` helper 和 `amap_client.py` 共同实现
- `pytest tests/test_tool_use_dataset.py tests/test_tool_use_orchestrator.py tests/test_stage2_amap_builder.py -q` 通过，结果为 `7 passed`
- `python src/data_pipeline/build_stage2_amap_tool_use.py --total-samples 3200 ...` 已成功生成完整训练数据
- source 与 sharegpt 两种格式的校验均已通过

因此，当前协议已经不是“草案状态”，而是“可训练、可导出、可执行、可评测”的收敛状态。

## 10. 冻结后哪些改动算破坏性变更

以下变更都不应直接覆盖本冻结版本，而应新开版本：

- 新增、删除、重命名工具
- 修改工具参数名、必填项、枚举值
- 修改允许的两步链
- 修改 source dataset role 体系
- 修改 `tool_calls` 或 `tool` 消息结构
- 修改 success / empty / error 的 envelope 结构
- 修改 ShareGPT 导出标签映射
- 修改 stage2 默认任务配比并希望保留历史可比性

## 11. 冻结后允许的非破坏性变更

以下变更可以继续做，但不应宣称为“协议升级”：

- 文案润色
- tool description 的非语义性措辞优化
- builder 内部采样实现优化
- 日志、报错提示、CLI 参数说明增强
- 新增不改变协议语义的测试

## 12. 一句话使用规则

从 `2026-04-13` 起，stage2_amap 的造数、训练、tool eval、部署联调，统一按 `stage2_amap_tool_protocol_v1_frozen` 执行；如果要改协议，先升版本，再改数据和评测，不直接覆盖本冻结版。
