# Qwen3-32B 两张 L20 推进指南

> 状态说明（2026-04-20）：
> 当前仓库已经补齐 32B 训练与部署配置，但 32B 基座权重、训练数据、merged 产物和远端 dataset 注册并不默认存在。
> 下文中的 32B 路径都应理解为“目标约定路径”，不是“服务器现状”。

这份指南对应当前仓库里已经跑通的 `8B stage1/stage2` 工程链路，目标是把正式主线迁移到 `Qwen/Qwen3-32B`，并固定为“两张 L20 48G + 4bit QLoRA + 双卡分布式”的执行口径。

## 1. 目标边界

- 正式基座：`Qwen/Qwen3-32B`
- `stage1`：只做六类自然语言 SFT，不混入 tool-use 轨迹
- `stage2`：继续复用当前冻结的高德工具协议，并使用 `3200` 条 32B tool-use 数据；`1600` 条口径只保留为 smoke/ablation 子集
- 第一轮目标：先把 `32B` 的训练、merge、部署、评测闭环打通，再决定是否扩容 stage2 数据和 golden eval

## 2. 关键配置

新增的 32B 配置已经放在 `configs/`：

- `llamafactory_dataset_info_stage1_general_sft.json`
- `llamafactory_stage1_32b_smoke_sft.yaml`
- `llamafactory_stage1_32b_formal_sft.yaml`
- `llamafactory_stage1_32b_merge_for_stage2.yaml`
- `llamafactory_stage2_32b_amap_tool_use_smoke_sft.yaml`
- `llamafactory_stage2_32b_amap_tool_use_formal_sft.yaml`
- `llamafactory_stage2_32b_merge_for_deploy.yaml`

32B 训练默认策略已经固定为：

- `4bit QLoRA`
- `LoRA(all, r=64, alpha=128, dropout=0.05)`
- `bf16`
- `gradient_checkpointing=true`
- `deepspeed=/root/soulv_llm/configs/deepspeed_zero2.json`
- 数据目录统一读取 `/root/llama-factory/data`

## 3. 环境校准

正式开始前，先在阿里云训练机确认：

1. 两张 `L20 48G` 都能被 `nvidia-smi` 看到
2. `LLaMA-Factory` 已安装，且支持 `FORCE_TORCHRUN=1`
3. `vLLM` 已安装
4. 准备训练前，需要先把 `Qwen/Qwen3-32B` 基座下载到 `/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B` 或等价软链接路径
5. 准备训练前，需要先把训练数据放到 `/root/llama-factory/data` 或等价软链接路径
6. 准备训练前，如果你使用 LLaMA-Factory 的 dataset 名称训练方式，再自行完成远端 `dataset_info.json` 注册

## 4. Stage1 执行顺序

### 4.1 Smoke

先跑小样确认显存、吞吐、checkpoint、eval loss 都正常：

```bash
bash scripts/02_run_sft.sh stage1_32b smoke
```

默认行为：

- 走双卡分布式
- `max_samples=800`
- `max_steps=150`
- `cutoff_len=2048`

如果 smoke OOM，降级顺序固定为：

1. 先降 `cutoff_len`
2. 再降 stage2 的 `cutoff_len`
3. 最后才考虑把 `lora_rank` 从 `64` 降到 `32`

不要通过继续缩小数据量来伪通过正式配置。

### 4.2 Formal

Smoke 通过后再跑正式版：

```bash
bash scripts/02_run_sft.sh stage1_32b formal
```

正式版特点：

- 不带 `max_samples`
- 默认读取完整 `stage1_general_sft.json`
- 默认输出到 `/root/soulv_assets/runs/checkpoints/qwen3_32b_stage1_general_sft`

### 4.3 Merge

```bash
bash scripts/04_merge_stage1_for_stage2.sh stage1_32b
```

输出目录：

```text
/root/soulv_assets/runs/merged/qwen3_32b_stage1_merged_base
```

## 5. Stage1 数据策略

32B 的 `stage1` 正式目标不是沿用 8B 的 `3894` 条，而是把清洗后可用池拉到 `8K-12K`，默认按 `10K` 组织。

推荐最终可用量：

- `guide_generation: 650`
- `travel_qa: 3250`
- `hotel_recommendation: 1900`
- `traffic_planning: 2000`
- `persona_understanding: 1300`
- `multi_turn_dialogue: 900`

补数原则：

- 继续做 token-aware mix，不按样本数盲混
- `guide_generation` 和 `multi_turn_dialogue` 不能靠长文模板堆量
- 优先补强 `traffic_planning`、`persona_understanding`、`multi_turn_dialogue`

## 6. Stage1 评测门槛

继续复用当前冻结的 `360` 条 stage1 eval，不重建主评测集。

建议门槛：

- `overall rule pass >= 0.92`
- `overall judge pass >= 0.50`
- `traffic_planning`、`persona_understanding`、`multi_turn_dialogue` 三类必须明显高于 8B 当前基线

## 7. Stage2 执行顺序

### 7.1 数据

第一轮使用当前冻结协议重新构建的 32B stage2 数据：

- `data/final/stage2_amap_tool_use_sft.json`
- `data/final/stage2_amap_tool_use_report.json`
- `configs/llamafactory_dataset_info_stage2_amap_tool.json`

当前轮次不改：

- `src/tool_use/protocol.py`
- 工具 schema
- 两步链路限制

### 7.2 Smoke

```bash
bash scripts/02_run_sft.sh stage2_amap_32b smoke
```

### 7.3 Formal

```bash
bash scripts/02_run_sft.sh stage2_amap_32b formal
```

### 7.4 Merge

```bash
bash scripts/06_merge_stage2_for_deploy.sh stage2_amap_32b
```

输出目录：

```text
/root/soulv_assets/runs/merged/qwen3_32b_stage2_amap_tool_use_merged
```

## 8. Stage2 评测门槛

继续沿用当前 tool eval 指标：

- `tool selection >= 0.90`
- `argument accuracy >= 0.85`
- `no-tool accuracy >= 0.90`
- `fallback accuracy >= 0.85`
- `execution success >= 0.90`

同时要求：

- stage1 回归总体通过率相对 stage1-only 模型下降不超过 `5pp`

当前 `src/tool_eval/datasets/stage2_amap_golden.json` 已扩到 `50` 条，覆盖直调、澄清、no-tool、fallback、两步链路、路线模式和 POI 搜索；`stage2_amap_thinking_canary.json` 只用于 thinking-mode 兼容性观察。

## 9. 部署

`scripts/03_run_vllm_api.sh` 现在支持 32B 轨道，但不再建议依赖“默认 32B 文件已经在服务器上”。

- `MODEL_VARIANT=32b`
- `TOKENIZER_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B`
- `SERVED_MODEL_NAME=qwen3_32b_stage2_amap_tool_use`
- `TENSOR_PARALLEL_SIZE=2`

直接启动：

```bash
MODEL_VARIANT=32b \
MODEL_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B \
TOKENIZER_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B \
SERVED_MODEL_NAME=qwen3_32b_base \
HOST=127.0.0.1 \
PORT=8000 \
bash scripts/03_run_vllm_api.sh
```

如果要回到旧 8B 轨道：

```bash
MODEL_VARIANT=8b \
MODEL_PATH=/root/soulv_assets/runs/merged/qwen3_8b_stage2_amap_tool_use_merged \
TOKENIZER_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-8B \
bash scripts/03_run_vllm_api.sh
```

## 10. 验收清单

数据侧必须过：

- 六个 bucket 结构校验
- `data_mixer` 报告
- stage1 混合后的 task count 校验
- `guide_generation` / `multi_turn_dialogue` 长文本占比校验
- `persona_understanding` 的 city/persona cap 校验

训练 smoke 必须过：

- 双卡启动成功
- 无 OOM
- 能保存 checkpoint
- 能产出 `eval_loss`
- merge 后模型能被 vLLM 正常加载

最终上线前必须过：

- stage1 回归门槛
- stage2 tool eval 门槛
- `/v1/chat/completions` 正常
- 首轮 `tool_calls` 正常
- `/api/tool-orchestrate` 端到端工具执行正常
