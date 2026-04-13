# Stage2 高德 Tool-Use Workflow

## Step 0. 对齐原生 Qwen3-8B

先跑 `src/tool_eval/datasets/native_tool_baseline.json`，确认原生模型在 `tools / tool_calls / arguments` 上的天然偏好。

目标不是重新发明协议，而是把 TripAI 对齐到原生 `Qwen3-8B-Instruct` 已经稳定的 OpenAI-compatible tool-calling 风格。

## Step 1. 冻结协议

固定 3 个工具：

- `amap_geocode`
- `amap_search_poi`
- `amap_plan_route`

固定行为边界：

- 优先单工具
- 最多两步链路
- 缺参先澄清
- 无需工具时直接回答
- 工具失败时 fallback，不硬编

## Step 2. 生成训练数据

```bash
python src/data_pipeline/build_stage2_amap_tool_use.py
```

默认会同时产出：

- `data/tool_use/stage2_amap_tool_use_source.json`
- `data/final/stage2_amap_tool_use_sft.json`
- `data/final/stage2_amap_tool_use_report.json`

## Step 3. 校验和导出

```bash
python src/data_pipeline/validate_tool_use_dataset.py \
  --file data/tool_use/stage2_amap_tool_use_source.json \
  --format source

python src/data_pipeline/validate_tool_use_dataset.py \
  --file data/final/stage2_amap_tool_use_sft.json \
  --format sharegpt
```

如果只想单独导出一次：

```bash
python src/data_pipeline/export_stage2_amap_tool_use.py
```

## Step 4. Merge stage1 起始模型

```bash
bash scripts/04_merge_stage1_for_stage2.sh
```

默认读取 [`llamafactory_stage1_merge_for_stage2.yaml`](/d:/soulv_llm/configs/llamafactory_stage1_merge_for_stage2.yaml)。

## Step 5. 跑 stage2 smoke / 正式训练

```bash
bash scripts/02_run_sft.sh stage2_amap
```

默认配置是 [`llamafactory_stage2_amap_tool_use_sft.yaml`](/d:/soulv_llm/configs/llamafactory_stage2_amap_tool_use_sft.yaml)。

## Step 6. 接真实高德执行

运行 stage2 tool eval 之前，部署环境里需要设置：

```bash
export AMAP_API_KEY=<your-key>
```

`src/tool_use/orchestrator.py` 会负责：

- 传入 `tools`
- 解析 `tool_calls`
- 调高德 API
- 回填 `tool` 消息
- 再向模型请求最终回答

## Step 7. 跑专项评测

```bash
python src/tool_eval/scripts/run_tool_eval.py \
  --base-url http://<server>:8000/v1 \
  --api-key EMPTY \
  --model <served-model-name>

python src/tool_eval/scripts/score_tool_eval.py
```

## Step 8. 跑 stage1 回归

继续使用现有 `src/eval/` 链路做自然语言能力回归，不要把 stage2 tool eval 混进去。
