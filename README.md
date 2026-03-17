# 搜旅大模型 (tripAI) - Qwen-7B MVP 阶段全链路工程上下文 (Project Context)

## 🎯 一、 项目全局定位与核心目标
本项目旨在打造搜旅智慧科技的专属 AI 旅行管家“小奇”。当前处于 **MVP（最小可行性产品）验证阶段**，核心目标是跑通大模型后训练（Post-Training）的全自动化 Pipeline。
- **基座模型**：Qwen3-8B-Instruct (或同级别 Qwen2.5 系列)
- **算力环境**：单卡 NVIDIA A10 (24GB 显存)
- **业务验证双核心**：
  1. **复杂格式生成**：根据知识库排版旅游攻略（严格分8大模块、大量使用Emoji、维持“小奇”人设）。
  2. **精准意图与工具调用 (Function Calling)**：面对查机票、订酒店等强业务诉求，静默输出纯 JSON 字符串（如 `{"intentionName":"FUNCTION_FLIGHTS_SEARCH_STRATEGY"}`），绝不带任何解释性废话。

---

## 🛤️ 二、 全生命周期 (End-to-End) 任务拆解

本项目将严格划分为 **5 个工程阶段**，请 AI 助手在协助编写代码时，时刻明确当前处于哪个阶段。

### Phase 1: 数据管道建设 (Data Pipeline) —— 【当前进行中】
构建约 3000 条高质量、无噪音的黄金 SFT 数据集，统一转换为 **ChatML** 格式。
- **[已完成]** `handler_itinerary.py`：生成攻略长文数据（739 条）。
- **[已完成]** `handler_intent.py`：解析真实日志或合成意图 JSON 产出数据（约 800 条）。
- **[待执行]** `handler_roleplay.py`：合成自我认知（“我是小奇”）与安全拒答数据（约 400 条）。
- **[待执行]** `handler_multiturn.py`：处理多轮客服对话数据（约 1000 条）。
- **[待执行]** `data_mixer.py`：终极融合器，按配比随机抽取、Shuffle，输出最终的 `soulv_mixed_sft.json`。

### Phase 2: LLaMA-Factory 训练引擎挂载 (Environment & Framework)
将我们清洗出的数据无缝接入业界最强微调框架 LLaMA-Factory。
- **[待执行]** 编写/修改 `dataset_info.json`：向 LLaMA-Factory 注册我们的混合数据集。
- **[待执行]** 生成 `train_qwen7b_lora.yaml`：配置训练超参数（Batch Size, Learning Rate, Epochs 等），确保在 A10 24G 显存下不 OOM。

### Phase 3: 模型指令微调 (SFT - LoRA)
在单卡 A10 上执行高效参数微调。
- **技术路线**：LoRA (Low-Rank Adaptation) 或 QLoRA。
- **优化目标**：让模型掌握排版肌肉记忆，并极度服从 JSON 输出指令，克服灾难性遗忘。
- **[待执行]** 编写自动化训练启动 Shell 脚本 `run_sft.sh`。

### Phase 4: 模型合并与推理部署 (Inference & Deployment)
微调结束后，对模型进行权重验证和本地化部署测试。
- **[待执行]** `merge_lora.sh`：将训练好的 LoRA 权重与 Qwen-8B 基础权重合并。
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
**“✅ 搜旅 7B MVP 全链路架构（Phase 1 到 Phase 5）已载入核心记忆！请主人下达具体的编码指令，我们立刻开工！”**
