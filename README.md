# 搜旅大模型（tripAI）- Qwen3-8B 当前工程上下文

## 一、项目当前定位

当前仓库的主目标是训练一个旅游领域基础模型，用于：

1. 攻略生成
2. 旅游问答
3. 需求理解与偏好表达
4. 多轮补充、改口、约束更新
5. TripAI 品牌风格与安全边界

当前活跃阶段是 **8B Domain SFT**。  
这个阶段只训练自然语言能力，不训练意图识别 JSON，不训练工具调用，不替代完整业务链路。

## 二、当前阶段边界

### 当前阶段做什么

- 训练旅游领域自然语言能力
- 建设可训练、可评测、可部署的基础数据与训练链路
- 跑通 LoRA / QLoRA 的本地训练和推理闭环

### 当前阶段不做什么

- 不输出 `{"intentionName": ...}`
- 不做 tool calling / function calling
- 不做机票、酒店、门票、签证、租车、乘机人等业务执行
- 不把强交易型请求混进当前阶段主训练集

## 三、当前主路径数据资产

当前 Domain SFT 主路径以自然语言样本为主，核心模块包括：

- `src/data_pipeline/handlers/handler_guide_generation.py`
  - 生成攻略 / 行程规划样本
- `src/data_pipeline/handlers/handler_dialogue.py`
  - 生成多轮补充、改口、约束更新样本
- `src/data_pipeline/handlers/handler_roleplay_safety.py`
  - 生成品牌身份与安全样本

当前主混合数据集输出为：

- `data/final/stage1_general_sft.json`

当前阶段推荐混合配比：

- 攻略生成（itinerary）：50%
- 多轮对话（dialogue）：30%
- 角色设定 / 安全拒答（roleplay_safety）：20%

## 四、训练与脚本入口

### 当前 Domain SFT 数据管道

`scripts/01_run_pipeline.sh`

- 处理 itinerary / dialogue / roleplay_safety
- 生成 `data/final/stage1_general_sft.json`

### 当前训练入口

`scripts/02_run_sft.sh stage1`

- 使用 `configs/llamafactory_stage1_sft.yaml`
- 训练当前阶段 Domain SFT 模型

## 五、评测重点

当前阶段重点评估：

- 攻略质量
- 问答准确性
- 需求理解能力
- 多轮上下文承接
- 品牌风格稳定性

当前阶段不再以以下指标为目标：

- 单意图分类准确率
- 纯 JSON 输出成功率
- 工具调用成功率

## 六、下一阶段

完成当前 Domain SFT 后，再单独进入 Tool-use / Function Calling 阶段。  
下一阶段会替代旧的 `intentionName` 路由思路，但当前仓库还没有锁定最终工具协议。

## 七、目录结构

```text
SOULV_LLM/
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
├── docs/
├── models/
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
└── README.md
```
