# 搜旅大模型 (tripAI) - Qwen3-8B MVP 阶段全链路工程上下文 (Project Context)

## 🎯 一、 项目全局定位与核心目标
本项目旨在打造搜旅智慧科技的专属 AI 旅行管家“小奇”。当前处于 **MVP（最小可行性产品）验证阶段**，核心目标是跑通大模型后训练（Post-Training）的全自动化 Pipeline。
- **基座模型**：Qwen3-8B-Instruct
- **算力环境**：单卡 NVIDIA A10 (24GB 显存)
- **业务验证双核心**：
  1. **复杂格式生成**：根据知识库排版旅游攻略（严格分8大模块、大量使用Emoji、维持“小奇”人设）。
  2. **精准意图与工具调用 (Function Calling)**：面对查机票、订酒店等强业务诉求，静默输出纯 JSON 字符串（如 `{"intentionName":"FUNCTION_FLIGHTS_SEARCH_STRATEGY"}`），绝不带任何解释性废话。

---

## 🛤️ 二、 全生命周期 (End-to-End) 任务拆解

本项目将严格划分为 **5 个工程阶段**，请 AI 助手在协助编写代码时，时刻明确当前处于哪个阶段。

### Phase 1: 数据管道建设 (Data Pipeline) —— 【当前进行中】
构建约 3000 条高质量、无噪音的黄金 SFT 数据集，统一转换为 **ChatML** 格式。
- **[已完成]** `src/data_pipeline/handlers/handler_itinerary.py`：生成攻略长文数据（739 条）。
- **[已完成]** `src/data_pipeline/handlers/handler_intent.py`：解析真实日志或合成意图 JSON 产出数据（约 800 条）。
- **[已完成]** `src/data_pipeline/handlers/handler_roleplay_safety.py`：合成自我认知（“我是小奇”）与安全拒答数据（约 400 条）。
- **[已完成]** `src/data_pipeline/handlers/handler_multiturn.py`：处理多轮客服对话数据（约 1000 条）。
- **[已完成]** `src/data_pipeline/data_mixer.py`：终极融合器，按配比随机抽取、Shuffle，输出最终的 `soulv_mixed_sft.json`。

#### 当前已落地数据资产（2026-03）
- `sft_itinerary.json`：739 条
- `sft_intent.json`：800 条
- `sft_roleplay_safety.json`：400 条

#### 当前阶段 SFT（LoRA）推荐配比
在 **多轮对话数据尚未补齐** 的前提下，建议先用这三类数据做一版可跑通的 LoRA 微调，推荐混合配比如下：

- **攻略长文（itinerary）**：55%
- **意图识别 / 纯 JSON（intent）**：30%
- **角色设定 / 安全拒答（roleplay_safety）**：15%

这样设计的原因：
1. **攻略长文必须占主导**：当前产品里用户最容易感知的能力，仍然是“小奇”能否稳定输出结构清晰、版式统一的长攻略。
2. **意图识别必须足够强，但不能压过攻略风格**：intent 数据虽然重要，但如果占比过高，模型会更容易被拉向“短答、结构化输出优先”，影响攻略生成手感。
3. **角色与安全更适合作为稳态约束**：这部分要有，但不需要压太高，避免把训练重心从“旅行能力”拉走。

#### 当前阶段不建议的做法
- 不建议按现有样本数量直接混合（739:800:400），因为这样会让 intent 权重偏高。
- 不建议在多轮数据缺失时，过早把训练重点转向客服式对话能力。
- 不建议为了凑量，把 roleplay/safety 过度放大到 20% 以上。

#### 等多轮数据补齐后的下一版目标配比
等 `src/data_pipeline/handlers/handler_multiturn.py` 对应数据补齐后，建议把训练配比切换为更接近真实产品形态的版本：

- 攻略长文（itinerary）：40%
- 多轮对话 / 意图澄清（multiturn）：25%
- 意图识别 / 纯 JSON（intent）：20%
- 角色设定 / 安全拒答（roleplay_safety）：10%
- 基础景点百科 QA（basic_qa，可选）：5%

### Phase 2: LLaMA-Factory 训练引擎挂载 (Environment & Framework)
将我们清洗出的数据无缝接入业界最强微调框架 LLaMA-Factory。
- **[待执行]** 编写/修改 `dataset_info.json`：向 LLaMA-Factory 注册我们的混合数据集。
- **[待执行]** 调整 `configs/train_sft_qlora.yaml`：配置 Qwen3-8B 的训练超参数（Batch Size, Learning Rate, Epochs 等），确保在 A10 24G 显存下不 OOM。

### Phase 3: 模型指令微调 (SFT - LoRA)
在单卡 A10 上执行高效参数微调。
- **技术路线**：LoRA (Low-Rank Adaptation) 或 QLoRA。
- **优化目标**：让模型掌握排版肌肉记忆，并极度服从 JSON 输出指令，克服灾难性遗忘。
- **[待执行]** 编写自动化训练启动 Shell 脚本 `run_sft.sh`。

### Phase 4: 模型合并与推理部署 (Inference & Deployment)
微调结束后，对模型进行权重验证和本地化部署测试。
- **[待执行]** `merge_lora.sh`：将训练好的 LoRA 权重与 Qwen3-8B 基础权重合并。
- **[待执行]** `inference_test.py`：利用 vLLM 或 HuggingFace Transformers 编写简易 CLI 对话脚本，进行第一手直观测试。

### Phase 5: 业务自动化评估 (Evaluation & Metrics)
不依赖人工肉眼看，而是通过代码自动化验证模型是否达标。
- **[待执行]** `eval_json_format.py`：针对意图识别测试集，用正则和 `json.loads` 测试 JSON 输出的成功率（目标 99%+）。
- **[待执行]** `eval_persona.py`：测试模型是否在回复中自称“小奇”并正确使用 8 大模块。

---

## 🛠️ 三、 代码与工程规范 (AI 助手必读)
1. **防爆防御**：所有数据处理脚本必须包含异常捕获（`try-except JSONDecodeError`），决不能因为一条脏数据导致整个 Pipeline 崩溃。
2. **日志打印**：使用丰富的终端输出（如 `print("✅ 成功加载数据...")`），带有进度指示，方便人类工程师在控制台监控。
3. **相对路径**：所有脚本执行路径默认以项目根目录为基准（如 `data/processed/...`）。
4. **ChatML 铁律**：模型的多轮对话必须严格采用 `{"messages": [{"role": "system", "content": "..."}, ...]}` 格式。

---

## 💬 四、 给 AI 助手的初始指令
阅读完以上上下文后，请回复：
**“✅ 搜旅 8B MVP 全链路架构（Phase 1 到 Phase 5）已载入核心记忆！请主人下达具体的编码指令，我们立刻开工！”**

---

## 📁 五、 当前项目目录结构（重构后）

当前仓库已经按“文档 / 配置 / 源码 / 脚本 / 测试 / 模型 / 日志”拆分，便于后续在云端继续扩训练、评测和部署链路。

```text
SOULV_LLM/
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
├── docs/
├── models/
│   ├── base_models/
│   └── checkpoints/
├── configs/
├── src/
│   ├── data_pipeline/
│   │   ├── handlers/
│   │   ├── data_utils.py
│   │   ├── global_cleaner.py
│   │   └── data_mixer.py
│   ├── train/
│   ├── eval/
│   └── deploy/
├── scripts/
├── tests/
├── logs/
├── main.py
├── pyproject.toml
├── .gitignore
└── README.md
```

说明：
- `docs/`：集中存放规范文档与数据说明
- `src/data_pipeline/handlers/`：集中存放各类数据处理入口
- `configs/`：训练、对齐、分布式配置文件
- `scripts/`：云端一键执行脚本入口
- `models/` 与 `logs/`：为云端训练和部署预留目录
- 旧 `pipeline/` 兼容层已清理，后续统一使用 `src/data_pipeline/` 下的新路径

