# Stage2 高德 Tool-Use 训练指南

## 1. 这份指南解决什么问题

这份指南面向当前仓库里的 `stage2_amap` 训练链路，目标是把：

- `Qwen3-8B-Instruct`
- stage1 旅游领域基线能力
- stage2 高德工具调用数据
- LLaMA-Factory 训练
- vLLM/OpenAI-compatible 部署
- 高德真实执行评测

串成一条可以反复执行的闭环。

这份指南默认你现在做的是：

- 用 `8B` 跑通工具调用训练方法
- 先验证工程闭环
- 不追求一步做到最终线上效果

如果后面你切到 `32B`，可以沿用同样的流程，只把模型路径、训练资源和配置放大。

---

## 2. 先理解当前仓库的训练思路

当前仓库不是“从零教会模型工具调用”，而是：

1. 先用 stage1 训练出一个旅游领域自然语言基线模型
2. 再把这个模型对齐到 TripAI 的高德工具协议
3. 再做 stage2 工具调用 SFT
4. 再部署成 OpenAI-compatible 接口
5. 再跑专项 tool eval 和 stage1 回归

所以 stage2 训练的核心不是“让模型会不会调用工具”，而是：

- 会不会按 TripAI 规定的 schema 调
- 会不会在该调的时候调
- 会不会在不该调的时候不调
- 会不会缺参数先澄清
- 会不会在工具失败时 fallback

---

## 3. 训练前你要知道的关键文件

先把这些文件认熟，后面几乎每一步都会用到。

### 3.1 工具协议

- `src/tool_use/protocol.py`

这个文件定义了：

- 系统提示词
- 允许的工具列表
- 每个工具的 JSON Schema
- 合法的两步工具链

当前只冻结了 3 个工具：

- `amap_geocode`
- `amap_search_poi`
- `amap_plan_route`

当前只允许两种两步链路：

- `amap_geocode -> amap_plan_route`
- `amap_geocode -> amap_search_poi`

如果你改了这里的工具名、参数名、两步链路，必须重建数据并重跑评测。

### 3.2 数据构建

- `src/data_pipeline/build_stage2_amap_tool_use.py`

这个脚本会：

- 读取 `data/processed/` 下的 stage1 处理后数据
- 生成 stage2 的 source dataset
- 导出 stage2 的 sharegpt dataset
- 生成构建报告

### 3.3 数据校验与导出

- `src/data_pipeline/validate_tool_use_dataset.py`
- `src/data_pipeline/export_stage2_amap_tool_use.py`

### 3.4 训练配置

- `configs/llamafactory_stage1_merge_for_stage2.yaml`
- `configs/llamafactory_stage2_amap_tool_use_sft.yaml`
- `configs/llamafactory_dataset_info_stage2_amap_tool.json`

### 3.5 训练入口

- `scripts/04_merge_stage1_for_stage2.sh`
- `scripts/02_run_sft.sh`
- `scripts/05_build_stage2_amap_data.sh`

### 3.6 部署与评测

- `src/tool_use/orchestrator.py`
- `src/tool_use/amap_client.py`
- `scripts/03_run_vllm_api.sh`
- `src/tool_eval/scripts/run_native_tool_baseline.py`
- `src/tool_eval/scripts/run_tool_eval.py`
- `src/tool_eval/scripts/score_tool_eval.py`
- `src/eval/scripts/run_eval.py`

---

## 4. 环境怎么分层

建议你把整个流程拆成 3 个环境来理解。

### 4.1 本地开发机

本地主要做这些事：

- 看代码
- 构建和校验数据
- 跑单测
- 产出要上传的训练文件

你现在这个 Windows 仓库环境更适合做这部分。

### 4.2 远端训练机

远端训练机主要做这些事：

- 装 LLaMA-Factory
- 放基座模型
- 放 stage1 LoRA
- merge stage1 基座
- 跑 stage2 QLoRA/SFT

这部分通常是 Linux 机器。

### 4.3 远端服务机

服务机主要做这些事：

- 启动 vLLM 或别的 OpenAI-compatible 服务
- 设置 `AMAP_API_KEY`
- 提供 `/v1/chat/completions`
- 给 tool eval 和后续联调用

有时训练机和服务机会是同一台机器，也可以分开。

---

## 5. 第零步：明确你的目标是否符合当前仓库边界

当前仓库的 `stage2_amap` 轨道，适合做：

- 单工具直接调用
- 参数抽取
- 参数缺失先澄清
- no-tool 判断
- 工具结果归纳成最终回答
- 工具失败 fallback

当前不适合一上来就做：

- 多工具长链规划
- 任意 Agent 自主拆解
- RL/DPO 主导的工具优化
- 把所有线上业务链路都塞进来

如果你现在的目标是“先把 stage2 高德工具调用训通”，那这条链路是匹配的。

---

## 6. 第一步：确认本地依赖和目录结构

### 6.1 安装本地 Python 依赖

仓库里已经有 `requirements.txt`，先装它：

```powershell
pip install -r requirements.txt
```

它主要覆盖：

- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `peft`
- `trl`
- `bitsandbytes`
- `pandas`
- `pyarrow`
- `pytest`

注意：

- 这里只能覆盖“本地数据处理 + 基础依赖”
- 不代表你已经装好了 `LLaMA-Factory`
- 也不代表你已经装好了 `vLLM`

### 6.2 确认这几个目录存在

本地至少要有：

- `data/raw/`
- `data/processed/`
- `data/final/`
- `data/tool_use/`
- `configs/`
- `scripts/`
- `src/tool_use/`
- `src/tool_eval/`

### 6.3 确认 stage2 依赖的 processed 数据已经存在

`build_stage2_amap_tool_use.py` 会直接读取：

- `data/processed/sft_traffic_planning.json`
- `data/processed/sft_hotel_recommendation.json`
- `data/processed/sft_travel_qa.json`

如果这三份文件不在，你就不能直接构建 stage2 数据。

---

## 7. 第二步：如果 processed 数据不全，先补 stage1 数据处理

如果你是新环境，或者换了一份原始数据，需要先重新处理 stage1 数据。

仓库里已经有一个总入口：

```bash
bash scripts/01_run_pipeline.sh
```

这个脚本会依次跑：

- `guide_generation`
- `travel_qa`
- `hotel_recommendation`
- `traffic_planning`
- `multi_turn_dialogue`
- `data_mixer`

但要注意：

- stage2 真正依赖的只有 `traffic_planning`、`hotel_recommendation`、`travel_qa`
- 所以你不是必须先把所有 stage1 类目都重跑一遍
- 如果只是为了构建 stage2 数据，可以只跑这三个 handler

如果你只是沿用当前仓库现成数据，这一步可以跳过。

---

## 8. 第三步：冻结工具协议

这一步特别重要。

开始生成数据前，你要先确认：

- 工具名不再改
- 参数 schema 不再改
- 工具返回格式尽量稳定
- 两步链路限制不再改

当前仓库在 `src/tool_use/protocol.py` 里已经冻结了最小协议。

### 8.1 当前协议包含什么

1. 系统提示词 `TRIPAI_TOOL_USE_SYSTEM_PROMPT`
2. 工具 schema
3. 合法行为标签
4. 合法工具名集合
5. 允许的两步链路

### 8.2 为什么一定要先冻结

因为 stage2 数据不是纯自然语言数据，而是“工具协议对齐数据”。

如果你边训练边改协议，会直接导致：

- 旧样本和新样本 schema 混杂
- 评测集和训练集口径不一致
- 模型学到多个冲突格式
- 出现“能调工具但格式飘”的问题

### 8.3 什么时候可以改协议

只有在下面几种情况才建议改：

- 你确认工具参数设计本身不合理
- 你确认 tool eval 暴露出系统性 schema 问题
- 你准备重建训练集和 golden eval

如果只是“小优化描述文案”，尽量不要动协议主结构。

---

## 9. 第四步：本地生成 stage2 训练数据

这是你实际开始训练前最重要的本地步骤。

### 9.1 直接使用默认构建命令

```powershell
python src/data_pipeline/build_stage2_amap_tool_use.py
```

默认会产出 3 个文件：

- `data/tool_use/stage2_amap_tool_use_source.json`
- `data/final/stage2_amap_tool_use_sft.json`
- `data/final/stage2_amap_tool_use_report.json`

### 9.2 这三个文件分别是什么

`stage2_amap_tool_use_source.json`

- 更接近原始训练逻辑的“源数据”
- 显式保留 `tools`
- 显式保留 `messages`
- 显式保留 `messages_with_answer`
- 适合检查样本结构是否合理

`stage2_amap_tool_use_sft.json`

- 导出的 `sharegpt + tools` 训练格式
- 是 LLaMA-Factory 真正要吃的训练数据

`stage2_amap_tool_use_report.json`

- 样本总量
- 每个桶的目标数
- 每个桶的候选池大小
- 每个桶的采样结果

### 9.3 默认样本量是多少

默认是 `1600` 条。

这 1600 条不是平均分，而是按比例分桶的：

- `single_tool_call`
- `slot_filling_tool_call`
- `clarify_then_call`
- `tool_result_grounded_answer`
- `no_tool_needed`
- `tool_failure_fallback`

### 9.4 想先 smoke 一下怎么办

如果你不想一上来就生成正式数据，可以先生成一个小样本版本：

```powershell
python src/data_pipeline/build_stage2_amap_tool_use.py --total-samples 32 --source-output data/tool_use/stage2_amap_tool_use_source_smoke_tmp.json --export-output data/final/stage2_amap_tool_use_sft_smoke_tmp.json --report-output data/final/stage2_amap_tool_use_report_smoke_tmp.json
```

先 smoke 的好处是：

- 可以快速确认脚本没挂
- 可以先看样本长什么样
- 可以先检查输出字段和角色标签

### 9.5 什么时候生成正式版

下面几件事都确认后，再生成正式版：

- 协议不改了
- 构建逻辑不改了
- smoke 样本看起来没问题
- 训练配置路径已经想清楚

---

## 10. 第五步：校验 stage2 数据

这一步不要省。

而且一定要串行执行，不要并行。

### 10.1 先校验 source dataset

```powershell
python src/data_pipeline/validate_tool_use_dataset.py --file data/tool_use/stage2_amap_tool_use_source.json --format source
```

### 10.2 再校验 sharegpt dataset

```powershell
python src/data_pipeline/validate_tool_use_dataset.py --file data/final/stage2_amap_tool_use_sft.json --format sharegpt
```

### 10.3 为什么要分两次校验

因为这两份数据负责不同的风险点。

`source` 校验主要防：

- tool name 不合法
- arguments 不是合法 JSON string
- role 序列不合法
- 工具链超出最大轮数
- 使用了不允许的两步链路

`sharegpt` 校验主要防：

- 导出后的 `tools` 字段不合法
- `function_call` / `observation` 不是合法 JSON
- LLaMA-Factory 读取字段名不对

### 10.4 如果你只想重导出一次

```powershell
python src/data_pipeline/export_stage2_amap_tool_use.py
```

这个场景通常适合：

- source 数据已经确定没问题
- 你只调整了导出格式
- 你不想整份数据重新 build

---

## 11. 第六步：本地抽查数据质量

这一步虽然不是“必须命令”，但很建议你做。

你至少要随机抽看几类样本：

- `single_tool_call`
- `clarify_then_call`
- `no_tool_needed`
- `tool_failure_fallback`

重点看下面这些问题。

### 11.1 工具名有没有漂

应该只出现：

- `amap_geocode`
- `amap_search_poi`
- `amap_plan_route`

### 11.2 arguments 是不是 JSON string

不是 Python dict，不是伪 JSON，不是自然语言。

### 11.3 澄清样本是不是先问再调

`clarify_then_call` 必须先出一条 assistant 自然语言澄清，再等用户补充，再出 tool call。

### 11.4 no-tool 样本是不是没有 tool call

`no_tool_needed` 不应该偷偷混进工具调用。

### 11.5 failure 样本是不是明确 fallback

失败样本不能编结果，回答里要承认本次工具失败，并给稳妥建议。

---

## 12. 第七步：跑本地最小测试

训练前建议先把 stage2 相关的关键测试跑一遍。

```powershell
pytest tests/test_tool_use_dataset.py tests/test_tool_use_orchestrator.py tests/test_stage2_amap_builder.py -q
```

这几组测试分别在防什么：

`test_tool_use_dataset.py`

- source 数据合法
- sharegpt 导出合法
- 不合法工具链会被拒绝

`test_tool_use_orchestrator.py`

- 工具调用能执行
- 第二个不合法工具链会被拦截

`test_stage2_amap_builder.py`

- 桶配比没漂
- no-tool 样本保留自然回答

如果这几组测试不过，不建议直接开训。

---

## 13. 第八步：准备远端训练机

到这一步开始从“本地准备”切到“远端训练”。

### 13.1 远端训练机上需要什么

你至少需要：

- Linux 环境
- Python 环境
- `LLaMA-Factory`
- `Qwen3-8B-Instruct` 基座模型
- stage1 LoRA 权重
- 足够磁盘空间
- 足够显存

### 13.2 为什么 stage2 训练更推荐远端 Linux

因为仓库里的训练脚本默认就是：

- `bash` 脚本
- Linux 路径
- `/root/...` 风格的输出目录

也就是说，当前配置天然更贴近 Linux 训练机。

### 13.3 远端机器上至少要准备的目录

按当前配置，默认会用到这些目录：

- `/root/llama-factory/data`
- `/root/soulv_llm/models/base_models/Qwen3-8B-Instruct`
- `/root/soulv_assets/runs/checkpoints/qwen3_8b_stage1_general_sft`
- `/root/soulv_assets/runs/merged/stage1_merged_base`
- `/root/soulv_assets/runs/checkpoints/qwen3_8b_stage2_amap_tool_use`

如果你不想用这套目录，可以改 YAML，但一定要前后一致。

---

## 14. 第九步：检查并修改训练配置

这是最容易卡住的一步。

### 14.1 先看 stage2 训练配置

文件：

- `configs/llamafactory_stage2_amap_tool_use_sft.yaml`

你重点检查下面几个字段。

`model_name_or_path`

- 当前默认是 merge 后的 stage1 基座
- 如果路径不存在，训练一定起不来

`dataset_dir`

- 当前默认是 `/root/llama-factory/data`
- 必须和你远端数据上传位置一致

`dataset`

- 当前是 `soulv_stage2_amap_tool_use_sft`
- 必须和 dataset registration 名字一致

`output_dir`

- 这是 stage2 LoRA 的输出目录
- 磁盘空间不够会直接爆

`bf16`

- 默认是 `true`
- 如果你的卡不支持，就改成 `fp16: true`

`quantization_bit`

- 默认是 `4`
- 说明当前是 4bit QLoRA 路线

### 14.2 再看 merge 配置

文件：

- `configs/llamafactory_stage1_merge_for_stage2.yaml`

重点检查：

- `model_name_or_path`
- `adapter_name_or_path`
- `export_dir`

如果 stage1 LoRA 权重位置不对，merge 这一步就会失败。

### 14.3 再看 dataset registration

文件：

- `configs/llamafactory_dataset_info_stage2_amap_tool.json`

这里定义了：

- 数据集名
- 文件名
- `sharegpt` 格式
- `messages` 和 `tools` 对应列
- 各类角色 tag

如果远端 LLaMA-Factory 用的是它自己的 `dataset_info.json`，你还需要把这段注册到远端版本里。

---

## 15. 第十步：把本地产物上传到远端

你至少要上传两类东西：

### 15.1 上传训练数据

最核心的是：

- `data/final/stage2_amap_tool_use_sft.json`

建议上传到：

- `/root/llama-factory/data/stage2_amap_tool_use_sft.json`

示例：

```bash
scp data/final/stage2_amap_tool_use_sft.json <your_ssh_host>:/root/llama-factory/data/stage2_amap_tool_use_sft.json
```

### 15.2 上传配置文件

建议至少上传：

- `configs/llamafactory_stage2_amap_tool_use_sft.yaml`
- `configs/llamafactory_stage1_merge_for_stage2.yaml`
- `configs/llamafactory_dataset_info_stage2_amap_tool.json`

如果你会直接在远端改，也可以不上传全量，但最好保留一份一致版本。

---

## 16. 第十一步：在远端注册 dataset

这是很多人第一次会漏掉的步骤。

即使你本地已经有：

- `configs/llamafactory_dataset_info_stage2_amap_tool.json`

也不代表远端的 `LLaMA-Factory` 自动知道它。

### 16.1 你需要做什么

你要把下面这个数据集注册到远端的 dataset 注册文件中：

- `soulv_stage2_amap_tool_use_sft`

对应信息是：

- `file_name`: `stage2_amap_tool_use_sft.json`
- `formatting`: `sharegpt`
- `messages`: `conversations`
- `tools`: `tools`

### 16.2 为什么这一步重要

因为训练配置里写的是：

- `dataset: soulv_stage2_amap_tool_use_sft`

如果远端没注册这个名字，`llamafactory-cli train` 会直接报找不到 dataset。

---

## 17. 第十二步：先 merge stage1 起始模型

stage2 当前不是直接拿原始 `Qwen3-8B-Instruct` 去训，而是：

1. 先把 stage1 LoRA merge 回基座
2. 得到一个 stage1 旅游领域基线模型
3. 再在它上面开 stage2 LoRA

### 17.1 merge 命令

```bash
bash scripts/04_merge_stage1_for_stage2.sh
```

这个脚本本质上会调用：

```bash
llamafactory-cli export configs/llamafactory_stage1_merge_for_stage2.yaml
```

### 17.2 merge 完成后你应该得到什么

你应该能在 `export_dir` 下看到 merge 后的模型目录。

当前默认是：

- `/root/soulv_assets/runs/merged/stage1_merged_base`

### 17.3 如果 merge 失败，优先排查什么

优先看：

- 基座模型路径是否存在
- stage1 LoRA 路径是否存在
- 磁盘空间是否足够
- `transformers` / `peft` 版本是否兼容

---

## 18. 第十三步：开始 stage2 正式训练

### 18.1 推荐命令

```bash
bash scripts/02_run_sft.sh stage2_amap
```

脚本会自动选用：

- `configs/llamafactory_stage2_amap_tool_use_sft.yaml`

### 18.2 这一步实际干了什么

本质上是：

```bash
llamafactory-cli train configs/llamafactory_stage2_amap_tool_use_sft.yaml
```

### 18.3 开训前最后检查一次

建议你在真正回车前，再核一遍：

- merge 后模型目录在不在
- dataset 注册好了没有
- 训练 JSON 已经传上去了没有
- `output_dir` 磁盘空间够不够
- `bf16` 是否和机器兼容
- 训练机能不能看到基座模型路径

### 18.4 训练过程重点看什么

重点看：

- 有没有在 loading dataset 阶段就报错
- 有没有在 tokenizer/template 阶段报错
- 有没有 OOM
- `eval_loss` 是否在正常下降
- checkpoint 是否正常保存

---

## 19. 第十四步：训练完成后怎么确认产物

训练成功后，你至少应该拿到：

- stage2 LoRA 输出目录
- 若干 checkpoint
- loss 曲线或日志

当前默认输出目录是：

- `/root/soulv_assets/runs/checkpoints/qwen3_8b_stage2_amap_tool_use`

你后续部署时，一般有两种做法：

- 直接拿“基座 + stage2 LoRA”提供服务
- 先 merge stage2，再拿 merge 后模型提供服务

这取决于你的服务框架是否直接支持 LoRA 挂载。

---

## 20. 第十五步：部署成 OpenAI-compatible 接口

stage2 tool eval 不是直接对权重文件打分，而是对一个服务接口打分。

### 20.1 服务要满足什么要求

至少要能提供：

- OpenAI-compatible `/v1/chat/completions`
- 支持传入 `tools`
- 支持输出 `tool_calls`

### 20.2 仓库里给了什么

仓库里有一个启动提示脚本：

- `scripts/03_run_vllm_api.sh`

里面给的是示意命令：

```bash
vllm serve $MODEL_PATH --host $HOST --port $PORT --max-model-len 8192
```

### 20.3 这一步你需要自己补什么

你需要根据你的部署方式补全：

- 具体模型路径
- 是否加载 LoRA
- 端口
- 显存配置
- 并发配置

---

## 21. 第十六步：设置高德 API Key

如果你只是在本地 build 数据，不需要高德 API。

如果你要跑真实工具执行评测，就必须设置：

```bash
export AMAP_API_KEY=<your-key>
```

### 21.1 为什么必须设置

因为：

- `src/tool_use/amap_client.py`

会直接访问高德 Web API。

没有 `AMAP_API_KEY` 的话，工具执行会返回：

- `missing_amap_api_key`

### 21.2 哪些步骤会用到 Key

会用到：

- `run_tool_eval.py`
- 真实联调
- 任何通过 `ToolCallingOrchestrator` 调真实高德的场景

不会用到：

- 本地构建训练数据
- 本地静态校验
- 只看格式的单元测试

---

## 22. 第十七步：先跑 native baseline

这个步骤很有价值，不要跳。

### 22.1 为什么先跑原生 baseline

因为你要先知道：

- 原生 `Qwen3-8B-Instruct` 对 OpenAI tools 的天然习惯是什么
- 它会不会天然输出 tool_calls
- arguments 格式稳不稳

这样你后面做 stage2，不是在瞎对齐，而是在顺着模型原本的风格对齐。

### 22.2 命令

```bash
python src/tool_eval/scripts/run_native_tool_baseline.py --base-url http://<server>:8000/v1 --api-key EMPTY --model <served-model-name>
```

然后再汇总：

```bash
python src/tool_eval/scripts/analyze_native_tool_baseline.py
```

### 22.3 你要关注哪些结果

重点看：

- 有多少 case 会主动出 tool_call
- arguments 能不能稳定解析成 JSON
- no-tool case 会不会直接自然回答

如果 baseline 就已经很顺，你的 stage2 重点应该是协议对齐，不是从零发明格式。

---

## 23. 第十八步：跑 stage2 tool eval

训练完并部署后，开始跑专项评测。

### 23.1 命令

```bash
python src/tool_eval/scripts/run_tool_eval.py --base-url http://<server>:8000/v1 --api-key EMPTY --model <served-model-name>
```

### 23.2 这一步在干什么

它会：

1. 读取 golden dataset
2. 调 OpenAI-compatible 模型接口
3. 把 `tools` 传进去
4. 通过 `ToolCallingOrchestrator` 执行真实高德工具
5. 把工具结果回填给模型
6. 让模型生成最终回答
7. 保存完整输出

### 23.3 结果文件在哪

默认会写到：

- `src/tool_eval/reports/stage2_amap_tool_eval_outputs.json`

---

## 24. 第十九步：给 stage2 tool eval 打分

### 24.1 命令

```bash
python src/tool_eval/scripts/score_tool_eval.py
```

### 24.2 它会看哪些指标

主要包括：

- tool selection accuracy
- argument accuracy
- clarify accuracy
- no-tool accuracy
- fallback accuracy
- execution success rate
- final answer grounded rate
- overall pass rate

### 24.3 当前 release gate 是什么

当前仓库里定义的是：

- tool selection accuracy `>= 0.85`
- argument accuracy `>= 0.80`
- execution success rate `>= 0.90`

如果达不到这些门槛，说明还不能算一个稳定的 stage2 版本。

---

## 25. 第二十步：不要忘了跑 stage1 回归

这是很多人做 tool use 时最容易漏的。

stage2 不是只看工具调用对不对，还要看有没有把原本的旅游自然语言能力训坏。

所以 stage2 评测后，你还应该继续跑：

- `src/eval/` 里的 stage1 自然语言回归

核心原则是：

- `src/tool_eval/` 负责专项工具调用评测
- `src/eval/` 负责 stage1 自然语言回归
- 两条线不要混

如果 stage2 分数变好了，但 stage1 回归掉太多，这版仍然不算稳定。

---

## 26. 推荐的完整执行顺序

如果你想从头到尾稳稳地跑一遍，推荐按这个顺序：

1. 安装本地依赖
2. 确认 `data/processed/` 三份 stage2 依赖数据存在
3. 冻结 `src/tool_use/protocol.py`
4. 本地先 smoke 生成小样本
5. 校验 smoke 的 source 和 sharegpt
6. 抽查若干样本质量
7. 生成正式版 stage2 数据
8. 再校验正式版 source 和 sharegpt
9. 跑本地 stage2 关键测试
10. 上传数据与配置到远端
11. 在远端注册 dataset
12. merge stage1 起始模型
13. 跑 `stage2_amap` 正式训练
14. 部署成 OpenAI-compatible 接口
15. 设置 `AMAP_API_KEY`
16. 跑 native baseline
17. 跑 stage2 tool eval
18. 跑 score
19. 跑 stage1 回归
20. 汇总结论，决定是否继续迭代数据或配置

---

## 27. 常见问题与排查方法

### 27.1 build 数据时报找不到 `data/processed/...`

原因通常是：

- stage1 handler 没跑
- 文件名不一致
- 当前环境不是完整仓库数据环境

先确认：

- `data/processed/sft_traffic_planning.json`
- `data/processed/sft_hotel_recommendation.json`
- `data/processed/sft_travel_qa.json`

### 27.2 validate 时报 `unknown tool`

原因通常是：

- 你改过 `protocol.py`
- 数据里遗留旧工具名

要么回退协议，要么整批重建数据。

### 27.3 validate 时报 `arguments must be valid JSON`

说明 tool call 的 `arguments` 不是合法 JSON string。

常见原因：

- 手动改样本时写成了 Python dict
- 少了引号
- 多了注释

### 27.4 训练时报 dataset 找不到

先排查：

- 远端有没有注册 `soulv_stage2_amap_tool_use_sft`
- 数据文件是不是放在 `dataset_dir` 对应的位置
- 文件名是不是 `stage2_amap_tool_use_sft.json`

### 27.5 训练时报路径不存在

优先看 3 个地方：

- `model_name_or_path`
- `dataset_dir`
- `output_dir`

尤其当前默认 YAML 里是 Linux 的 `/root/...` 路径，本地 Windows 路径不能直接照抄。

### 27.6 训练时报 `bf16` 不支持

把配置从：

- `bf16: true`

改成：

- `fp16: true`

不要同时乱开多个精度字段。

### 27.7 tool eval 全部失败且工具返回 `missing_amap_api_key`

说明服务机或执行环境没设置：

- `AMAP_API_KEY`

### 27.8 tool eval 能调工具，但最终回答很怪

优先看这几个层面：

1. 训练数据里 `tool_result_grounded_answer` 是否太少
2. 系统提示词是否把“基于工具结果作答”说清楚
3. tool result 返回结构是否过于原始，模型难以消费
4. 模型服务模板是否正确支持 tools

### 27.9 stage2 指标提升了，但 stage1 回归掉很多

说明可能出现了：

- 工具调用样本配比过高
- no-tool 样本不足
- 训练轮数过多
- 学到了过度调用工具的倾向

这时优先做的是：

- 调整数据配比
- 加强 no-tool 与 direct-answer 样本
- 降低训练强度

不要第一反应就是继续堆训练轮数。

---

## 28. 每次正式开训前的检查清单

每次你准备真正跑 stage2 正式训练前，建议照着这张表过一遍。

- 工具协议已经冻结
- `data/processed/` 三份依赖文件存在
- stage2 数据已经重新 build
- source 校验通过
- sharegpt 校验通过
- 本地关键测试通过
- 远端 `dataset_dir` 与上传目录一致
- dataset registration 已完成
- stage1 merge 配置路径正确
- stage2 train 配置路径正确
- 训练机支持当前精度设置
- 磁盘空间足够
- 训练完成后的部署路径想清楚
- `AMAP_API_KEY` 已准备
- 评测命令与输出路径已准备

---

## 29. 推荐的第一次执行策略

如果这是你第一次完整跑 `stage2_amap`，我建议不要直接冲正式训练。

更稳的方式是：

1. 本地先生成 `32` 或 `64` 条 smoke 数据
2. 校验通过后抽查样本
3. 远端先确认 dataset 注册逻辑没问题
4. 先走一遍 merge
5. 先拿小规模配置或短轮次试跑
6. 确认训练日志正常
7. 再开正式版 stage2

第一次先证明“链路通”，第二次再追求“结果好”。

---

## 30. 你后面最值得优先来问我的点

如果你后面卡住，最值得直接来问我的通常是这几类：

- 我该不该改 `protocol.py`
- 这条报错是数据问题还是训练环境问题
- 这个 YAML 路径该怎么改
- 远端 dataset registration 该怎么配
- 训练 loss 正不正常
- tool eval 分数为什么低
- stage1 回归掉了该怎么调配比

你只要把：

- 你卡住的命令
- 完整报错
- 你改过的配置文件
- 你当时的目标

贴给我，我就可以继续顺着这份指南帮你往下定位。
