# 阿里云 LLaMA-Factory 微调说明

本说明对应当前仅保留两类可用数据的启动版 SFT：

- `guide_generation`
- `travel_qa`

融合策略沿用项目原始阶段比例 `30:25` 的归一化结果。由于当前两份数据实际条数正好是 `1500:1250`，因此直接全量合并，无需额外过采样或下采样。

## 1. 本地产物

- 融合数据：`data/final/stage1_bootstrap_guide_qa_sft.json`
- 融合报告：`data/final/stage1_bootstrap_guide_qa_sft.report.json`
- 训练配置：`configs/llamafactory_bootstrap_guide_qa_sft.yaml`
- 数据注册：`configs/llamafactory_dataset_info_bootstrap_guide_qa.json`

## 2. 上传到阿里云

假设：

- 远端 LLaMA-Factory 目录：`/root/llama-factory`
- SSH 别名或主机：`<your_ssh_host>`

将融合数据上传到远端 `data/` 目录：

```bash
scp data/final/stage1_bootstrap_guide_qa_sft.json <your_ssh_host>:/root/llama-factory/data/soulv_bootstrap_guide_qa_sft.json
```

将训练配置上传到远端：

```bash
scp configs/llamafactory_bootstrap_guide_qa_sft.yaml <your_ssh_host>:/root/llama-factory/examples/train_lora/qwen3_bootstrap_guide_qa_sft.yaml
```

## 3. 注册数据集

登录远端后，编辑 `/root/llama-factory/data/dataset_info.json`，加入下面这段：

```json
{
  "soulv_bootstrap_guide_qa_sft": {
    "file_name": "soulv_bootstrap_guide_qa_sft.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

## 4. 启动训练

```bash
cd /root/llama-factory
llamafactory-cli train examples/train_lora/qwen3_bootstrap_guide_qa_sft.yaml
```

## 5. 训练前建议确认

- `model_name_or_path` 是否与远端基座模型路径一致
- 机器是否支持 `bf16`
- 显存是否足够跑 `Qwen3-8B-Instruct + 4bit QLoRA`
- `output_dir` 所在磁盘是否有足够空间

如果远端不支持 `bf16`，可将配置中的 `bf16: true` 改成 `fp16: true`。