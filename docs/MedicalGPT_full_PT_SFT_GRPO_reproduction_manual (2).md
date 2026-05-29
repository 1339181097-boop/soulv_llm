# MedicalGPT 完整复现项目方案：H20 版医学 PT + SFT + GRPO

## 1. 项目定位

项目名建议：**MedGRPO-PT: 基于 MedicalGPT 的中文医疗大模型 PT + SFT + GRPO 复现项目**

这个项目的核心原则是：**尽量复用 MedicalGPT 仓库已经提供的数据格式、脚本、工具和推荐数据集；只有在项目亮点需要的地方，才新增少量数据治理脚本。**

最终训练链路：

```text
医学继续预训练 PT
  -> 医疗问诊 SFT
  -> 医学结构化 GRPO
  -> 标准评测 + 自建病例评测
```

项目要回答三个问题：

1. **PT 是否让模型更适应医学文本分布？**
   - 复用 `shibing624/medical` 的 `pretrain` 数据。
   - 主要观察医学文本 eval loss / PPL 是否下降。

2. **SFT 是否把医学文本能力转成问诊问答能力？**
   - 复用 `shibing624/medical` 的 `finetune` 数据。
   - 复用 `shibing624/huatuo_medical_qa_sharegpt` 的 ShareGPT 格式对话数据。
   - 复用 MedicalGPT 的 `training/supervised_finetuning.py` 和 `scripts/run_sft.sh`。

3. **GRPO 是否进一步提升结构化诊断、证据引用和医疗安全？**
   - 复用 MedicalGPT 的 `training/grpo_training.py`、`scripts/run_grpo.sh`、`data/grpo/sample.jsonl` 格式。
   - 在默认 GRPO 奖励函数基础上改成医学规则奖励。

你后续包装 HealthAI-2025 的地方，不是说“数据是我原创的”，而是说：

```text
我复用了公开医疗语料，但没有粗暴全量拼接训练；
我借鉴 HealthAI-2025 的数据治理思路，
把公开医疗 QA 做成结构化病例、医学目标分布筛选数据和可验证 GRPO 数据，
再用 PT/SFT/GRPO 三阶段训练和评测消融证明每一步是否有效。
```

## 2. 仓库扫描结论：哪些直接复用

### 2.1 MedicalGPT 训练脚本

MedicalGPT 仓库已经覆盖完整训练链路，我们不重写训练框架。

| 阶段 | 直接复用 Python 入口 | 直接复用 shell 模板 | 本项目怎么用 |
|---|---|---|---|
| PT | `training/pretraining.py` | `scripts/run_pt.sh` | 复制一份改成 H20 + 医学 PT 数据 |
| SFT | `training/supervised_finetuning.py` | `scripts/run_sft.sh` | 复制两份，分别跑 Random-SFT 和 Retrieval-SFT |
| GRPO | `training/grpo_training.py` | `scripts/run_grpo.sh` | 复制一份，替换成医学 GRPO 数据和奖励函数 |
| RM/PPO/DPO/ORPO | 仓库已支持 | 仓库已支持 | 本项目先不作为主线，后续可扩展 |

### 2.2 MedicalGPT 工具脚本

这些工具优先复用：

| 工具 | 用途 | 本项目怎么用 |
|---|---|---|
| `tools/convert_dataset.py` | `alpaca/qa/json` 转 ShareGPT JSONL 或 JSONL | 转换 `shibing624/medical` 的 SFT 数据；转换 PT JSON 为 JSONL |
| `tools/validate_jsonl.py` | 校验 ShareGPT JSONL 是否包含 `conversations/from/value` | 每次生成 SFT 数据后都跑 |
| `tools/merge_peft_adapter.py` | 合并 LoRA adapter 到 base model | PT 后合并一次，SFT 后合并一次 |
| `tools/merge_tokenizers.py` 等 | 词表/后处理工具 | 第一版不需要 |

### 2.3 MedicalGPT 仓库内置样例数据

仓库内 `data/` 目录的样例数据只用于 smoke test，不作为主实验规模数据。

| 目录 | 用途 |
|---|---|
| `data/pretrain` | PT 格式 smoke |
| `data/sft` | ShareGPT SFT 格式 smoke |
| `data/reward` | RM/DPO/PPO 数据格式参考 |
| `data/grpo/sample.jsonl` | GRPO 格式参考，字段是 `question/answer` |

### 2.4 MedicalGPT README 推荐数据集

主实验复用 README 推荐的医疗数据集：

| 数据集 | 用途 | 本项目地位 |
|---|---|---|
| `shibing624/medical` | 240 万中文医疗数据，包括 PT、SFT、reward | 主数据集 |
| `shibing624/huatuo_medical_qa_sharegpt` | 22 万中文医疗对话，ShareGPT 格式 | SFT 辅助数据，格式最省心 |
| `shibing624/sharegpt_gpt4` | 通用多轮 ShareGPT 数据 | 可选，少量混入防止模型太窄 |

## 3. H20 硬件策略

你当前服务器优先用 **H20**。本文档默认按 **单张 H20** 写命令。

H20 策略：

```text
默认：CUDA_VISIBLE_DEVICES=0
默认：bf16
默认：LoRA
默认：smoke 用 Qwen3-4B-Base，正式实验用 Qwen3-8B-Base
默认：不用 DeepSpeed，除非后面上 14B/32B
```

推荐模型：

| 阶段 | 模型 | 说明 |
|---|---|---|
| smoke | `Qwen/Qwen3-4B-Base` | 只用于验证模型下载、数据格式和训练脚本 |
| 正式 PT | `Qwen/Qwen3-8B-Base` | Base 模型更适合继续预训练 |
| 正式 SFT | PT 后 merged model | 从医学 PT 结果继续做指令微调 |
| 正式 GRPO | SFT 后 merged model | 从指令模型继续做结构化对齐 |

注意：

```text
这里的 4B/8B 指 Base 版本：
Qwen/Qwen3-4B-Base
Qwen/Qwen3-8B-Base
```

如果你已经下载过 `Qwen/Qwen3-4B`，它可以用于推理或 SFT smoke；但本项目有 PT 阶段，所以主文档统一使用 `Qwen/Qwen3-4B-Base` 和 `Qwen/Qwen3-8B-Base`。

如果 H20 资源稳定，正式主线直接做 8B。再往上加分可以考虑 `Qwen/Qwen3-14B-Base`，但不要在数据和脚本没跑通前上 14B。

如果后面没有 H20，换两张 5090：

```text
把 CUDA_VISIBLE_DEVICES=0 改为 CUDA_VISIBLE_DEVICES=0,1
把 python training/xxx.py 改为 torchrun --nproc_per_node 2 training/xxx.py
适当降低 per_device_train_batch_size
```

## 4. 数据集设计

### 4.1 主数据集：`shibing624/medical`

下载后重点使用这些文件：

| 文件 | 规模 | 原始字段 | 项目用途 |
|---|---:|---|---|
| `pretrain/train_encyclopedia.json` | 约 36 万 | `text` | PT 主语料 |
| `pretrain/medical_book_zh.json` | 8475 | `text` | PT 高质量医学教材语料 |
| `pretrain/valid_encyclopedia.json` | 500 | `text` | PT 验证 |
| `pretrain/test_encyclopedia.json` | 500 | `text` | PT 测试 / PPL |
| `finetune/train_zh_0.json` | 约 195 万 | `instruction/input/output` | SFT 主候选池 |
| `finetune/valid_zh_0.json` | 500 | `instruction/input/output` | SFT 验证 |
| `finetune/test_zh_0.json` | 500 | `instruction/input/output` | 自建病例评测 holdout |
| `reward/train.json` | 约 3800 | `question/response_chosen/response_rejected` | 安全样例、GRPO safety 参考 |

### 4.2 辅助 SFT 数据：`shibing624/huatuo_medical_qa_sharegpt`

这个数据已经是 MedicalGPT 支持的 ShareGPT 格式，适合快速跑通 SFT。

| 文件 | 用途 |
|---|---|
| `HuatuoGPT_sft_data_v1_sharegpt.jsonl` | SFT smoke 和辅助混入 |
| `HuatuoGPT2_sft_instruct_GPT4_sharegpt.jsonl` | 提升格式、表达和指令跟随 |

### 4.3 数据版本规划

第一版先准备这些数据：

```text
data/medgrpo/pt/train/train.jsonl
data/medgrpo/pt/valid/valid.jsonl
data/medgrpo/pt/test/test.jsonl

data/medgrpo/sft/medical_all/train_zh_0_sharegpt.jsonl
data/medgrpo/sft/valid/valid.jsonl
data/medgrpo/sft/smoke_1k/train.jsonl
data/medgrpo/sft/random_30k/train.jsonl

data/medgrpo/structured/medical_cases_200k.jsonl
data/medgrpo/anchors/medical_anchor_300.jsonl
data/medgrpo/retrieval/retrieval_sft_30k/train.jsonl

data/medgrpo/grpo/grpo_medical_6k.jsonl
```

注意：前两阶段先只需要 PT 和 SFT 数据；Retrieval 和 GRPO 可以在基础链路跑通后再做。

## 5. HealthAI-2025 方法如何包装进来

HealthAI-2025 的思路可以概括为：

```text
医疗 QA
  -> 病例结构化
  -> 向量化目标分布和候选样本
  -> 相似度筛选
  -> 构造推理/诊断格式数据
  -> SFT
```

本项目的对应实现：

| HealthAI-2025 思路 | 本项目实现 |
|---|---|
| 原始医疗 QA | `shibing624/medical/finetune/train_zh_0.json` |
| 病例结构化 | 生成 `feature_content` 字段 |
| 目标分布向量 | 医学 anchor，不用 CEval 原题 |
| 相似度筛选 | `BAAI/bge-large-zh-v1.5` 或同类中文 embedding |
| R1/推理蒸馏 | Retrieval-SFT 中 20%-30% 改写成结构化诊断 JSON |
| 评估闭环 | CEval + PPL + 自建病例评测 |

要点：

```text
数据可以复用公开数据；
方法亮点体现在“如何筛、如何构造、如何评测”，而不是声称数据原创。
```

## 6. 项目目录规划

基于 MedicalGPT 原仓库，不重建训练框架。

```text
MedicalGPT/
  data/
    medgrpo/
      raw/
      pt/
      sft/
      structured/
      anchors/
      retrieval/
      grpo/
      eval/
  scripts/
    run_pt_medgrpo_h20.sh
    run_sft_smoke_medgrpo_h20.sh
    run_sft_random_medgrpo_h20.sh
    run_sft_retrieval_medgrpo_h20.sh
    run_grpo_medgrpo_h20.sh
  tools/
    convert_dataset.py                 # 原仓库复用
    validate_jsonl.py                   # 原仓库复用
    merge_peft_adapter.py               # 原仓库复用
    medgrpo_select_jsonl.py             # 新增：抽样/切分
    medgrpo_extract_feature.py          # 新增：HealthAI 风格结构化
    medgrpo_retrieve_sft.py             # 新增：向量筛选
    medgrpo_build_grpo.py               # 新增：GRPO 数据
  outputs/
    pt_smoke_qwen3_4b/
    sft_smoke_qwen3_4b/
    pt_med_qwen3_8b/
    pt_med_qwen3_8b_merged/
    sft_random_qwen3_8b/
    sft_retrieval_qwen3_8b/
    sft_retrieval_qwen3_8b_merged/
    grpo_med_qwen3_8b/
  results/
```

新增脚本控制在 4 个以内，且只做数据加工，不重写 MedicalGPT 的训练逻辑。

## 7. 复现手册

## Step 0: 环境准备

### 0.1 克隆 MedicalGPT

```bash
git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT
```

### 0.2 创建环境

```bash
conda create -n medgpt python=3.10 -y
conda activate medgpt
```

安装依赖：

```bash
nvidia-smi
pip install -r requirements.txt --upgrade
pip install -U huggingface_hub datasets "transformers>=4.51.0" accelerate tokenizers
```

H20 通常支持 bf16。第一版如果 `flash-attn` 报错，就先关掉，不要卡环境。

### 0.3 下载 Qwen3 模型到本地固定目录

为了训练命令稳定，模型统一放到 MedicalGPT 仓库内的 `models/` 目录：

```bash
mkdir -p models
```

下载 4B smoke 模型：

```bash
huggingface-cli download Qwen/Qwen3-4B-Base \
  --local-dir ./models/Qwen3-4B-Base
```

下载 8B 正式模型：

```bash
huggingface-cli download Qwen/Qwen3-8B-Base \
  --local-dir ./models/Qwen3-8B-Base
```

如果服务器访问 Hugging Face 慢，可以临时使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

下载后检查：

```bash
ls ./models/Qwen3-4B-Base
ls ./models/Qwen3-8B-Base
```

至少应该看到：

```text
config.json
tokenizer.json
tokenizer_config.json
*.safetensors
```

如果模型是分片保存，还会看到 `model.safetensors.index.json` 和多个 `model-*.safetensors` 文件。

## Step 1: 下载数据

```bash
mkdir -p data/medgrpo/raw
```

下载 `shibing624/medical`：

```bash
huggingface-cli download shibing624/medical \
  --repo-type dataset \
  --local-dir data/medgrpo/raw/medical
```

下载 Huatuo ShareGPT：

```bash
huggingface-cli download shibing624/huatuo_medical_qa_sharegpt \
  --repo-type dataset \
  --local-dir data/medgrpo/raw/huatuo_sharegpt
```

检查：

```bash
find data/medgrpo/raw/medical -maxdepth 3 -type f | sort
find data/medgrpo/raw/huatuo_sharegpt -maxdepth 2 -type f | sort
```

## Step 2: 准备 PT 数据

### 2.1 原仓库工具能做什么

MedicalGPT 的 `tools/convert_dataset.py` 支持 `json2jsonl`。如果原始 PT 文件是 JSON 数组，可以直接转 JSONL：

```bash
mkdir -p data/medgrpo/pt/train data/medgrpo/pt/valid data/medgrpo/pt/test

python tools/convert_dataset.py \
  --in_file data/medgrpo/raw/medical/pretrain/train_encyclopedia.json \
  --out_file data/medgrpo/pt/train/train_encyclopedia.jsonl \
  --data_type json2jsonl

python tools/convert_dataset.py \
  --in_file data/medgrpo/raw/medical/pretrain/medical_book_zh.json \
  --out_file data/medgrpo/pt/train/medical_book_zh.jsonl \
  --data_type json2jsonl

python tools/convert_dataset.py \
  --in_file data/medgrpo/raw/medical/pretrain/valid_encyclopedia.json \
  --out_file data/medgrpo/pt/valid/valid_encyclopedia.jsonl \
  --data_type json2jsonl

python tools/convert_dataset.py \
  --in_file data/medgrpo/raw/medical/pretrain/test_encyclopedia.json \
  --out_file data/medgrpo/pt/test/test_encyclopedia.jsonl \
  --data_type json2jsonl
```

### 2.2 第一版不急着清洗

为了最大复用，第一版 PT 数据可以先只做格式转换，不写复杂清洗脚本。训练时用：

```text
--max_train_samples 100000
--max_eval_samples 500
```

这样不用物理切分 10 万条，也能先跑一版有效 PT。

后续如果要更像项目，再补 `medgrpo_select_jsonl.py` 做：

- 去重
- 长度过滤
- 随机抽样
- 数据统计

### 2.3 PT 数据格式检查

```bash
head -n 1 data/medgrpo/pt/train/train_encyclopedia.jsonl
head -n 1 data/medgrpo/pt/train/medical_book_zh.jsonl
wc -l data/medgrpo/pt/train/*.jsonl
wc -l data/medgrpo/pt/valid/*.jsonl
```

每行应该至少包含：

```json
{"text": "..."}
```

## Step 3: 用 H20 跑医学 PT

先复制原仓库脚本：

```bash
cp scripts/run_pt.sh scripts/run_pt_medgrpo_h20.sh
```

### 3.1 4B smoke PT

4B smoke 只验证环境、数据、脚本和 LoRA 保存是否正常，不作为最终结果。

```bash
CUDA_VISIBLE_DEVICES=0 python training/pretraining.py \
  --model_name_or_path ./models/Qwen3-4B-Base \
  --train_file_dir ./data/medgrpo/pt/train \
  --validation_file_dir ./data/medgrpo/pt/valid \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --do_train \
  --do_eval \
  --use_peft True \
  --seed 42 \
  --max_train_samples 1000 \
  --max_eval_samples 100 \
  --num_train_epochs 0.1 \
  --learning_rate 1e-4 \
  --warmup_steps 5 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 50 \
  --eval_strategy steps \
  --save_steps 100 \
  --save_strategy steps \
  --save_total_limit 2 \
  --gradient_accumulation_steps 4 \
  --preprocessing_num_workers 4 \
  --block_size 1024 \
  --packing True \
  --output_dir outputs/pt_smoke_qwen3_4b \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache
```

### 3.2 8B 正式 PT

4B smoke 通过后，再跑 8B 正式 PT：

```bash
CUDA_VISIBLE_DEVICES=0 python training/pretraining.py \
  --model_name_or_path ./models/Qwen3-8B-Base \
  --train_file_dir ./data/medgrpo/pt/train \
  --validation_file_dir ./data/medgrpo/pt/valid \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --do_train \
  --do_eval \
  --use_peft True \
  --seed 42 \
  --max_train_samples 100000 \
  --max_eval_samples 500 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 200 \
  --eval_strategy steps \
  --save_steps 1000 \
  --save_strategy steps \
  --save_total_limit 3 \
  --gradient_accumulation_steps 4 \
  --preprocessing_num_workers 8 \
  --block_size 1024 \
  --packing True \
  --output_dir outputs/pt_med_qwen3_8b \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache
```

运行：

```bash
bash scripts/run_pt_medgrpo_h20.sh
```

如果显存富余，可以逐步提高：

```text
per_device_train_batch_size: 4 -> 8
block_size: 1024 -> 2048
max_train_samples: 100000 -> 全量
```

如果 8B 正式 PT 出现 OOM，优先按这个顺序降配置：

```text
per_device_train_batch_size: 4 -> 2
block_size: 1024 保持不变
gradient_accumulation_steps: 4 -> 8
```

## Step 4: 合并 PT LoRA

复用 MedicalGPT 的合并工具：

```bash
python tools/merge_peft_adapter.py \
  --base_model_name_or_path ./models/Qwen3-8B-Base \
  --peft_model_path outputs/pt_med_qwen3_8b \
  --output_dir outputs/pt_med_qwen3_8b_merged
```

如果脚本参数和当前版本略有差异，以：

```bash
python tools/merge_peft_adapter.py --help
```

输出为准。

## Step 5: 准备 SFT 数据

### 5.1 转换 `shibing624/medical` 全量 SFT 数据

`train_zh_0.json` 是 Alpaca 风格：

```json
{"instruction": "...", "input": "...", "output": "..."}
```

复用 MedicalGPT 官方转换工具：

```bash
mkdir -p data/medgrpo/sft/medical_all data/medgrpo/sft/valid

python tools/convert_dataset.py \
  --in_file data/medgrpo/raw/medical/finetune/train_zh_0.json \
  --out_file data/medgrpo/sft/medical_all/train_zh_0_sharegpt.jsonl \
  --data_type alpaca \
  --file_type json

python tools/convert_dataset.py \
  --in_file data/medgrpo/raw/medical/finetune/valid_zh_0.json \
  --out_file data/medgrpo/sft/valid/valid.jsonl \
  --data_type alpaca \
  --file_type json
```

校验：

```bash
python tools/validate_jsonl.py \
  --file_path data/medgrpo/sft/medical_all/train_zh_0_sharegpt.jsonl

python tools/validate_jsonl.py \
  --file_path data/medgrpo/sft/valid/valid.jsonl
```

### 5.2 准备 SFT smoke 数据

优先复用 Huatuo ShareGPT 原生格式。第一版可以直接训练时通过 `--max_train_samples 1000` 控制规模，不必先切文件。

如果你想物理切一个 smoke 文件：

```bash
mkdir -p data/medgrpo/sft/smoke_1k
head -n 1000 data/medgrpo/raw/huatuo_sharegpt/HuatuoGPT_sft_data_v1_sharegpt.jsonl \
  > data/medgrpo/sft/smoke_1k/train.jsonl
head -n 100 data/medgrpo/raw/huatuo_sharegpt/HuatuoGPT_sft_data_v1_sharegpt.jsonl \
  > data/medgrpo/sft/smoke_1k/valid.jsonl

python tools/validate_jsonl.py \
  --file_path data/medgrpo/sft/smoke_1k/train.jsonl
```

### 5.3 准备 Random-SFT

第一版最大化复用，可以直接让训练脚本读全量转换文件，并用：

```text
--max_train_samples 30000
```

如果你要严格做可复现 Random-SFT，后续新增一个极小脚本 `tools/medgrpo_select_jsonl.py`：

```text
输入：train_zh_0_sharegpt.jsonl
输出：random_30k/train.jsonl
逻辑：固定 seed、打乱、取 30000、写出
```

目录：

```bash
mkdir -p data/medgrpo/sft/random_30k
```

第一版先不急，先用 `max_train_samples` 跑通。

## Step 6: H20 跑 SFT

先复制官方脚本：

```bash
cp scripts/run_sft.sh scripts/run_sft_medgrpo_h20.sh
```

### 6.1 SFT smoke

```bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path ./models/Qwen3-4B-Base \
  --train_file_dir ./data/medgrpo/sft/smoke_1k \
  --validation_file_dir ./data/medgrpo/sft/smoke_1k \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 1000 \
  --max_eval_samples 100 \
  --model_max_length 1024 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_steps 5 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 50 \
  --eval_strategy steps \
  --save_steps 200 \
  --save_strategy steps \
  --gradient_accumulation_steps 4 \
  --preprocessing_num_workers 4 \
  --output_dir outputs/sft_smoke_qwen3_4b \
  --target_modules all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --tool_format default \
  --cache_dir ./cache \
  --flash_attn False
```

### 6.2 Random-SFT 第一版

```bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path outputs/pt_med_qwen3_8b_merged \
  --train_file_dir ./data/medgrpo/sft/medical_all \
  --validation_file_dir ./data/medgrpo/sft/valid \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 30000 \
  --max_eval_samples 500 \
  --model_max_length 1024 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 200 \
  --eval_strategy steps \
  --save_steps 500 \
  --save_strategy steps \
  --save_total_limit 3 \
  --gradient_accumulation_steps 4 \
  --preprocessing_num_workers 4 \
  --output_dir outputs/sft_random_qwen3_8b \
  --target_modules all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --tool_format default \
  --cache_dir ./cache \
  --flash_attn False
```

如果 H20 显存富余：

```text
per_device_train_batch_size 4 -> 8
model_max_length 1024 -> 2048
```

## Step 7: HealthAI 风格 Retrieval-SFT

这一步是项目亮点，不是 MedicalGPT 原生工具能完全完成的，所以允许新增少量脚本。

### 7.1 结构化病例

输入：

```text
data/medgrpo/raw/medical/finetune/train_zh_0.json
```

输出：

```text
data/medgrpo/structured/medical_cases_200k.jsonl
```

字段：

```json
{
  "id": "medical_000001",
  "question": "患者问题",
  "origin_answer": "原始回答",
  "feature_content": "性别: 未知\n年龄: 未知\n主诉: ...\n现病史: ...\n既往史: ...\n体格检查: ...",
  "quality_tags": ["has_symptom"]
}
```

第一版可以规则抽取；第二版可以用 Qwen3 Instruct 系列模型或 API 模型批量结构化。这里的结构化抽取属于数据加工，不参与 PT 底座选择，所以可以用指令模型。

### 7.2 医学 anchor

输出：

```text
data/medgrpo/anchors/medical_anchor_300.jsonl
```

来源：

- `pretrain/train_encyclopedia.json` 中的疾病/症状文本。
- 常见科室。
- 常见任务：疾病判断、检查建议、用药安全、急症识别。

### 7.3 向量筛选

embedding 模型：

```text
BAAI/bge-large-zh-v1.5
```

输出：

```text
data/medgrpo/retrieval/retrieval_ranked_200k.jsonl
data/medgrpo/retrieval/retrieval_sft_30k/train.jsonl
```

`retrieval_sft_30k/train.jsonl` 最终仍然必须是 MedicalGPT SFT 支持的 ShareGPT 格式：

```json
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
```

做完继续用原仓库校验：

```bash
python tools/validate_jsonl.py \
  --file_path data/medgrpo/retrieval/retrieval_sft_30k/train.jsonl
```

## Step 8: Retrieval-SFT 训练

复制 `scripts/run_sft.sh`，只改数据和输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path outputs/pt_med_qwen3_8b_merged \
  --train_file_dir ./data/medgrpo/retrieval/retrieval_sft_30k \
  --validation_file_dir ./data/medgrpo/sft/valid \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --do_train \
  --do_eval \
  --use_peft True \
  --model_max_length 1024 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 200 \
  --eval_strategy steps \
  --save_steps 500 \
  --save_strategy steps \
  --save_total_limit 3 \
  --gradient_accumulation_steps 4 \
  --preprocessing_num_workers 4 \
  --output_dir outputs/sft_retrieval_qwen3_8b \
  --target_modules all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --tool_format default \
  --cache_dir ./cache \
  --flash_attn False
```

合并：

```bash
python tools/merge_peft_adapter.py \
  --base_model_name_or_path outputs/pt_med_qwen3_8b_merged \
  --peft_model_path outputs/sft_retrieval_qwen3_8b \
  --output_dir outputs/sft_retrieval_qwen3_8b_merged
```

## Step 9: 构造 GRPO 数据

MedicalGPT 原生 GRPO 数据格式是：

```json
{"question": "...", "answer": "..."}
```

本项目保持这个格式，不发明新训练格式。

构造三类数据：

| 数据 | 数量 | 来源 |
|---|---:|---|
| 医学短答 | 1k | `train_zh_0.json` 中答案明确的 QA |
| 结构化病例 | 4k | Retrieval-SFT top 样本 |
| 医疗安全 | 1k | `reward/train.json` 和高风险问诊样本 |

合并输出：

```text
data/medgrpo/grpo/grpo_medical_6k.jsonl
```

例子：

```json
{"question":"患者出现多饮、多尿、体重下降，最需要考虑哪类疾病？请只输出疾病名称。","answer":"糖尿病"}
```

结构化病例也保持 `question/answer`，只是 `answer` 存 reward 需要的标准信息：

```json
{"question":"性别: 女\n年龄: 36\n主诉: 鼻塞、打喷嚏、鼻痒\n请输出 JSON，字段包括 reasoning_content、reason、diseases、suggestion。","answer":"{\"gold_diseases\":[\"过敏性鼻炎\"],\"must_evidence\":[\"鼻塞\",\"打喷嚏\",\"鼻痒\"],\"forbidden\":[\"抗生素剂量\",\"未提供的影像检查\"]}"}
```

## Step 10: 修改 GRPO 奖励函数

复用 `training/grpo_training.py`，只改奖励函数。

默认 GRPO 示例偏 `accuracy_reward` 和 `format_reward`。医疗版本改成：

| reward | 作用 |
|---|---|
| `medical_format_reward` | JSON 可解析、字段完整 |
| `diagnosis_match_reward` | `diseases` 命中 gold disease |
| `evidence_coverage_reward` | `reason` 覆盖 must_evidence |
| `no_hallucination_reward` | 不出现 forbidden 内容 |
| `medical_safety_reward` | 高风险场景提示就医，不给危险剂量 |

注意：

```text
训练框架不重写；
数据格式不重写；
只替换 reward function。
```

## Step 11: H20 跑 GRPO

复制官方脚本：

```bash
cp scripts/run_grpo.sh scripts/run_grpo_medgrpo_h20.sh
```

H20 单卡版本：

```bash
CUDA_VISIBLE_DEVICES=0 python training/grpo_training.py \
  --model_name_or_path outputs/sft_retrieval_qwen3_8b_merged \
  --train_file_dir data/medgrpo/grpo \
  --train_samples -1 \
  --max_steps 1000 \
  --save_steps 100 \
  --save_strategy steps \
  --save_total_limit 5 \
  --output_dir outputs/grpo_med_qwen3_8b \
  --dtype bfloat16 \
  --bf16 True \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing False \
  --beta 0.001 \
  --learning_rate 5.0e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --use_vllm False \
  --logging_steps 10 \
  --use_peft True \
  --qlora False \
  --load_in_4bit False \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --num_generations 4 \
  --gradient_accumulation_steps 1 \
  --max_completion_length 512
```

如果显存压力大：

```text
per_device_train_batch_size 4 -> 2
num_generations 4 -> 2
max_completion_length 512 -> 256
```

## Step 12: 评测

### 12.1 模型对比

至少评这些：

| 模型 | 路径 |
|---|---|
| Smoke Base | `./models/Qwen3-4B-Base` |
| Formal Base | `./models/Qwen3-8B-Base` |
| PT | `outputs/pt_med_qwen3_8b_merged` |
| PT + Random-SFT | `outputs/sft_random_qwen3_8b` |
| PT + Retrieval-SFT | `outputs/sft_retrieval_qwen3_8b_merged` |
| PT + Retrieval-SFT + GRPO | `outputs/grpo_med_qwen3_8b` |

### 12.2 指标

| 指标 | 用途 |
|---|---|
| PT eval loss / PPL | 证明 PT 适应医学文本分布 |
| CEval 医学科目 | 标准医学选择题能力 |
| 自建病例 JSON valid | 结构化输出能力 |
| Diagnosis F1 | 诊断命中 |
| Evidence coverage | 证据引用 |
| Hallucination rate | 幻觉抑制 |
| Safety pass | 医疗安全 |

CEval 任务：

```text
ceval-valid_basic_medicine
ceval-valid_clinical_medicine
ceval-valid_physician
ceval-valid_veterinary_medicine
```

## 13. 最终结果表

| Model | PT PPL ↓ | CEval-basic | CEval-clinical | CEval-physician | JSON valid | Diagnosis F1 | Evidence coverage | Hallucination ↓ | Safety pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | x | x | x | x | x | x | x | x | x |
| PT | x | x | x | x | x | x | x | x | x |
| PT + Random-SFT | x | x | x | x | x | x | x | x | x |
| PT + Retrieval-SFT | x | x | x | x | x | x | x | x | x |
| PT + Retrieval-SFT + GRPO | x | x | x | x | x | x | x | x | x |

## 14. 最终交付物

```text
README.md
data/README.md
scripts/run_pt_medgrpo_h20.sh
scripts/run_sft_smoke_medgrpo_h20.sh
scripts/run_sft_random_medgrpo_h20.sh
scripts/run_sft_retrieval_medgrpo_h20.sh
scripts/run_grpo_medgrpo_h20.sh
tools/medgrpo_select_jsonl.py
tools/medgrpo_extract_feature.py
tools/medgrpo_retrieve_sft.py
tools/medgrpo_build_grpo.py
results/result_table.md
results/case_studies/
```

## 15. 当前执行顺序

不要一上来就做 Retrieval 和 GRPO。先按这个顺序：

1. 下载 `shibing624/medical` 和 `huatuo_medical_qa_sharegpt`。
2. 用 `tools/convert_dataset.py` 转 PT JSONL。
3. 用 H20 跑 4B PT smoke，再跑 8B PT 10 万样本正式版。
4. 合并 PT LoRA。
5. 用 `tools/convert_dataset.py` 转 SFT ShareGPT。
6. 用 `tools/validate_jsonl.py` 校验。
7. 用 H20 跑 4B SFT smoke。
8. 用 8B PT merged model 跑 Random-SFT。
9. 再做 HealthAI 风格结构化和 Retrieval-SFT。
10. 最后做 GRPO。

这样你的学习路径也最顺：

```text
先学 MedicalGPT 原生项目怎么跑，
再学数据格式怎么转，
再学领域 PT 和 SFT 的区别，
再学 HealthAI-2025 的数据治理为什么有用，
最后学 GRPO reward 怎么设计。
```
