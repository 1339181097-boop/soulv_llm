# vLLM + 前端部署落地方案

这份方案是按当前仓库的真实结构整理出来的，不是空泛模板。

## 1. 先明确仓库里已经有什么

- `configs/llamafactory_stage1_merge_for_stage2.yaml`
  - stage1 LoRA merge 后的基座输出到 `/root/soulv_assets/runs/merged/stage1_merged_base`
- `configs/llamafactory_stage2_amap_tool_use_sft.yaml`
  - stage2 tool-use LoRA 默认输出到 `/root/soulv_assets/runs/checkpoints/qwen3_8b_stage2_amap_tool_use`
- `src/tool_use/orchestrator.py`
  - 真实调用 OpenAI-compatible `/v1/chat/completions`
  - 默认把 `chat_template_kwargs.enable_thinking=false` 传给服务
- `src/tool_eval/scripts/run_native_tool_baseline.py`
  - 可以测“模型首轮是否会主动产出 `tool_calls`”
- `src/tool_eval/scripts/run_tool_eval.py`
  - 会执行真实高德工具链路
- `src/tool_eval/reports/native_tool_baseline_summary.json`
  - 当前仓库已经留下了实证：6 个 baseline case 里有 4 个返回了 `tool_calls`，4 个 arguments 都是合法 JSON

所以当前项目已经具备：

1. OpenAI-compatible 推理接口调用代码
2. 工具调用协议
3. 原生工具调用基线
4. 端到端 tool eval

差的主要是：

1. stage2 merge 的正式脚本
2. 真正可执行的 vLLM 启动脚本
3. 能给别人直接访问的网页和网关

## 2. 推荐的最终拓扑

推荐你按下面的方式部署：

1. `vLLM` 只监听本机 `127.0.0.1:8000`
2. 轻量网关监听公网 `0.0.0.0:7860`
3. 网关同时负责：
   - 托管网页
   - 代理 `/v1/*` 到本机 vLLM
   - 暴露 `/api/tool-orchestrate` 给网页做端到端工具执行测试

这样做的好处：

1. 别人只需要访问一个网页链接，例如 `http://<server-ip>:7860`
2. 浏览器不需要跨域访问 `8000`
3. 你不必把 vLLM 的真实 `API Key` 写进前端源码
4. 如果只开放 `7860`，安全面更小

## 3. stage2 merge

仓库现在新增了：

- `configs/llamafactory_stage2_merge_for_deploy.yaml`
- `scripts/06_merge_stage2_for_deploy.sh`

默认假设：

- stage1 merged base: `/root/soulv_assets/runs/merged/stage1_merged_base`
- stage2 LoRA: `/root/soulv_assets/runs/checkpoints/qwen3_8b_stage2_amap_tool_use`
- 最终 merged 输出: `/root/soulv_assets/runs/merged/qwen3_8b_stage2_amap_tool_use_merged`

如果你还没有做最终 stage2 merge，直接执行：

```bash
bash scripts/06_merge_stage2_for_deploy.sh
```

如果你已经 merge 完成，只要把下面 vLLM 启动脚本里的 `MODEL_PATH` 指向你的最终 merged 模型目录即可。

## 4. vLLM 启动命令

仓库现在把 `scripts/03_run_vllm_api.sh` 改成了可直接执行的版本，重点是：

1. 显式带上了 `--tokenizer`
2. 带上了 `--served-model-name`
3. 默认加了 `--generation-config vllm`
4. 默认加了 Qwen3 reasoning parser
5. 默认开启 auto tool choice，并把 parser 暴露成环境变量

推荐启动方式：

```bash
HOST=127.0.0.1 \
PORT=8000 \
MODEL_PATH=/root/soulv_assets/runs/merged/qwen3_8b_stage2_amap_tool_use_merged \
TOKENIZER_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-8B \
SERVED_MODEL_NAME=qwen3_8b_stage2_amap_tool_use \
bash scripts/03_run_vllm_api.sh
```

如果你需要给外部客户端直接开放 vLLM，也可以把 `HOST=0.0.0.0`，但对“网页给别人访问”这个目标来说，不推荐先暴露 `8000`。

### 4.1 为什么这些参数要补上

- `--tokenizer`
  - vLLM 官方 CLI 明确支持单独指定 tokenizer；如果不写，默认回退到 `--model`
  - 对你这种“merge 后模型目录”和“原始 tokenizer 目录”分离的场景，显式写最稳
- `--served-model-name`
  - 仓库里的评测脚本都要求你明确传 `--model <served-model-name>`
- `--generation-config vllm`
  - vLLM 官方说明：默认会读取模型仓库里的 `generation_config.json`，这会覆盖部分采样默认值
  - 你的网页和脚本都想自己控制采样参数，所以这里建议固定成 `vllm`
- `--reasoning-parser qwen3`
  - 仓库里的 `run_eval.py` 和 `orchestrator.py` 都会在请求里发送 `chat_template_kwargs.enable_thinking=false`
  - vLLM 的 Qwen3 reasoning parser 文档也明确提到了 `enable_thinking=False`
- `--enable-auto-tool-choice --tool-call-parser hermes`
  - vLLM 官方明确说明：`tool_choice="auto"` 必须同时打开 auto-tool-choice 和 parser
  - 官方公开点名的 Qwen 支持是 `Qwen2.5 / QwQ-32B -> hermes parser`
  - 你的模型是 `Qwen3-8B`，官方没有直接写到这一型号，所以这里是“结合仓库实证后的推荐默认值”，不是 100% 官方点名结论
  - 之所以先给 `hermes`，是因为你仓库里已经保存了成功返回 `tool_calls` 的基线输出，这说明你这条 Qwen 系列链路已经跑通过 parser-style tool calling

### 4.2 如果 `auto` 模式不稳定怎么办

vLLM 官方文档明确区分了四种模式：

- `named`
- `required`
- `auto`
- `none`

其中：

- `required` / `named`
  - 走结构化约束解码
  - 参数 JSON 最稳
- `auto`
  - 不做 schema 级约束
  - 完全依赖模型原始输出 + parser 提取

所以如果你只是想手动验证“模型有没有工具调用能力”，建议先这样测：

1. 先在网页里用 `tool_choice=required`
2. 再改成 `tool_choice=auto`

## 5. 启动公网网页和同源网关

仓库现在新增了：

- `src/deploy/frontend_server.py`
- `src/deploy/web/index.html`
- `scripts/07_run_frontend_gateway.sh`

启动方式：

```bash
HOST=0.0.0.0 \
PORT=7860 \
UPSTREAM_VLLM_BASE_URL=http://127.0.0.1:8000 \
DEFAULT_MODEL_NAME=qwen3_8b_stage2_amap_tool_use \
bash scripts/07_run_frontend_gateway.sh
```

如果你的 vLLM 配了 `--api-key`，可以把同一个 key 配到网关环境变量里：

```bash
export UPSTREAM_VLLM_API_KEY=<your-vllm-key>
```

这样浏览器端就不需要知道这个 key。

## 6. 别人的电脑怎么访问

最小步骤如下：

1. 服务器安全组 / 防火墙放行 `7860`
2. 启动 vLLM
3. 启动前端网关
4. 让别人打开：

```text
http://<server-ip>:7860
```

如果你走的是这套网关方案：

- 网页访问端口：`7860`
- OpenAI-compatible API 也可以通过：

```text
http://<server-ip>:7860/v1
```

外部就不需要再访问 `8000`

## 7. 网页里能做什么

新的网页已经支持三类场景：

1. 普通多轮对话
2. 模型首轮工具调用检查
3. 端到端 AMap 工具执行测试

### 7.1 普通对话

用于验证：

- 模型是否正常响应
- 网页是否能被别人访问
- streaming 是否正常

### 7.2 模型首轮工具调用检查

用于验证：

- `tools` 能不能正常传进去
- 模型是否产出 `tool_calls`
- arguments JSON 是否合法

这块直接打：

```text
/v1/chat/completions
```

### 7.3 端到端工具执行测试

用于验证：

- 模型先出 `tool_calls`
- 服务端执行真实高德工具
- 工具结果回填后模型是否能生成最终回答

这块直接打：

```text
/api/tool-orchestrate
```

也就是网页已经能直接复用仓库里的：

- `src/tool_use/orchestrator.py`
- `src/tool_use/amap_client.py`

## 8. 接口层手动测试命令

### 8.1 测模型首轮 tool_calls

推荐先用 `required`：

```bash
curl http://<server-ip>:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_8b_stage2_amap_tool_use",
    "messages": [
      {"role": "system", "content": "你是 TripAI 旅行助手。用户需要实时路线、位置和周边信息时优先使用工具。"},
      {"role": "user", "content": "我从北京南站出发，想去颐和园，优先公共交通，帮我看看怎么走。"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "amap_plan_route",
          "description": "规划路线",
          "parameters": {
            "type": "object",
            "properties": {
              "origin": {"type": "string"},
              "destination": {"type": "string"},
              "mode": {"type": "string", "enum": ["transit", "driving", "walking", "bicycling"]},
              "city": {"type": "string"}
            },
            "required": ["origin", "destination"]
          }
        }
      }
    ],
    "tool_choice": "required",
    "parallel_tool_calls": false,
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512
  }'
```

如果这一步稳，再把 `tool_choice` 改成：

```json
"tool_choice": "auto"
```

### 8.2 测端到端工具执行

```bash
curl http://<server-ip>:7860/api/tool-orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_8b_stage2_amap_tool_use",
    "messages": [
      {"role": "system", "content": "你是 TripAI 旅行助手。用户需要实时路线、位置和周边信息时优先使用工具。"},
      {"role": "user", "content": "帮我找一下杭州西湖附近评分不错的酒店。"}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "top_p": 1,
    "disable_thinking": true
  }'
```

注意：

- 这一步需要服务端已经设置 `AMAP_API_KEY`
- 没有 key 时，结果里会返回 `missing_amap_api_key`

## 9. 继续复用仓库里的评测脚本

### 9.1 native baseline

```bash
python src/tool_eval/scripts/run_native_tool_baseline.py \
  --base-url http://<server-ip>:7860/v1 \
  --api-key EMPTY \
  --model qwen3_8b_stage2_amap_tool_use
```

### 9.2 stage2 tool eval

```bash
export AMAP_API_KEY=<your-key>

python src/tool_eval/scripts/run_tool_eval.py \
  --base-url http://<server-ip>:7860/v1 \
  --api-key EMPTY \
  --model qwen3_8b_stage2_amap_tool_use

python src/tool_eval/scripts/score_tool_eval.py
```

### 9.3 stage1 回归

```bash
python src/eval/scripts/run_eval.py \
  --base-url http://<server-ip>:7860/v1 \
  --api-key EMPTY \
  --model qwen3_8b_stage2_amap_tool_use \
  --model-name qwen3_8b_stage2_amap_tool_use \
  --disable-thinking
```

## 10. 额外修正点

你桌面的旧前端有一个小问题：

- 它把 `top_k` / `repetition_penalty` 放到了 `extra_body` 里

这在 OpenAI 官方 Python client 里没问题，但你这个网页是直接发原始 HTTP JSON 给 vLLM。

vLLM 官方文档明确写了：

- 如果你是直接发 HTTP，可以把这些额外参数直接合并到 JSON payload 顶层

所以仓库里的新网页已经改成了顶层字段写法。

## 11. 官方文档对应点

- vLLM `serve` CLI:
  - `--tokenizer`
  - `--served-model-name`
  - `--allowed-origins`
  - `--api-key`
  - `--enable-auto-tool-choice`
  - `--tool-call-parser`
  - https://docs.vllm.ai/en/stable/cli/serve/
- vLLM OpenAI-compatible server:
  - 支持 `/v1/chat/completions`
  - 直接 HTTP 请求时可把扩展参数合并到 JSON 顶层
  - 默认会读 `generation_config.json`，可用 `--generation-config vllm` 关闭
  - https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- vLLM Tool Calling:
  - `required` / `named` / `auto` / `none`
  - `auto` 模式必须配 parser
  - Qwen 官方列出的 parser 是 `hermes`，但当前公开支持条目写的是 `Qwen2.5 / QwQ-32B`
  - https://docs.vllm.ai/en/stable/features/tool_calling/
- vLLM Qwen3 reasoning parser:
  - 明确提到 `enable_thinking=False`
  - https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/
