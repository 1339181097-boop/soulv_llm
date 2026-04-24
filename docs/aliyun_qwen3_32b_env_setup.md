# 阿里云 Qwen3-32B 双环境安装指南

这份文档对应当前仓库准备中的 32B 轨道。

状态说明：

- 现在的目标是先把 `lf` 和 `qwen` 两个环境装稳
- 当前还没有开始正式训练 `Qwen3-32B`
- 32B 基座权重、训练数据、merge 产物、远端 dataset 注册都不默认存在
- 文中出现的 32B 路径，默认都按“未来约定路径”理解，不代表云端现在已经有内容

这份文档当前聚焦：

- 训练环境：`lf`
- 推理环境：`qwen`
- 训练框架：`LLaMA-Factory`
- 推理框架：`vLLM`
- 目标模型：`Qwen/Qwen3-32B`
- 目标机器：`2 x NVIDIA L20 48G`

适用默认路径：

- 项目仓库：`/root/soulv_llm`
- LLaMA-Factory：`/root/llama-factory`
- 模型与训练产物：`/root/soulv_assets`
- Conda：`/root/miniconda3/bin/conda`

## 1. 版本口径

本仓库固定成“两套环境分离”的方式：

- `lf` 只负责训练、merge、导出
- `qwen` 只负责 vLLM 推理、网关、评测

这样做是为了避免 `torch`、`nccl`、`vllm`、`deepspeed` 在同一个环境里互相污染。

本次默认版本：

- `lf`
  - `Python 3.11`
  - `LLaMA-Factory v0.9.4`
  - `torch 2.6.0 + cu124`
  - `transformers 4.51.3`
  - `datasets 3.2.0`
  - `accelerate 1.2.1`
  - `peft 0.15.1`
  - `trl 0.9.6`
  - `deepspeed 0.16.4`
  - `bitsandbytes 0.48.2`
- `qwen`
  - `Python 3.11`
  - `vllm 0.10.2`
  - `transformers 4.56.2`
  - `tokenizers >=0.22,<0.24`
  - `numpy 2.2.6`
  - `modelscope 1.34.x`

说明：

- `flash-attn` 被拆成可选安装，默认不强装。原因不是它没用，而是它在云端经常因为 CUDA toolchain 编译问题拖慢首轮落地。
- 如果后续你确认机器驱动、CUDA、编译链都稳定，再执行 `INSTALL_FLASH_ATTN=1 bash scripts/00_setup_lf_env.sh` 即可。
- 当前仓库里 `lf` 环境的 `bitsandbytes` 已按阿里云双 L20 实测结果固定为 `0.48.2`。这是为了解决旧版 `0.43.1` 在新 Triton 环境下可能触发的 `No module named triton.ops` 报错。

## 2. 为什么这样拆

这套拆法直接对应官方要求：

- LLaMA-Factory 最新发布版已经把 Python 基线抬到 `3.11+`
- vLLM 官方安装文档明确建议使用“全新隔离环境”
- vLLM 官方还特别提醒：如果用 conda，也不要把 conda 装的 `pytorch` 和 pip 装的 `vllm` 混在同一个运行时里，否则可能因为 NCCL 差异导致问题
- Qwen 官方文档说明：
  - `Qwen3` 在 `vLLM >= 0.8.5` 下支持 Hermes function calling
  - `Qwen3` 的 reasoning parser 需要 `vLLM >= 0.9.0`

所以这里不把 `vllm` 装进 `lf`，而是单独放在 `qwen` 环境里。

## 3. 为什么之前写 `cd /root/soulv_llm`

之前那样写，只是为了让你直接用相对路径执行：

```bash
bash scripts/00_setup_lf_env.sh
```

不是因为创建 conda 环境本身必须先进入仓库目录。

现在这几个脚本已经改过了，会自动根据脚本自身位置推断 `PROJECT_ROOT`，所以你可以在任意目录执行：

```bash
bash /root/soulv_llm/scripts/00_setup_lf_env.sh
bash /root/soulv_llm/scripts/00_setup_qwen_env.sh
bash /root/soulv_llm/scripts/00_prepare_aliyun_layout.sh
bash /root/soulv_llm/scripts/00_doctor_envs.sh
```

如果你人在仓库目录里，继续用相对路径也没问题。

## 4. 一次性执行顺序

按顺序执行：

```bash
bash /root/soulv_llm/scripts/00_setup_lf_env.sh
bash /root/soulv_llm/scripts/00_setup_qwen_env.sh
bash /root/soulv_llm/scripts/00_prepare_aliyun_layout.sh
bash /root/soulv_llm/scripts/00_doctor_envs.sh
```

这四步现在分别负责：

1. 创建并安装 `lf`
2. 创建并安装 `qwen`
3. 只准备目录结构和可选软链接
4. 打印 GPU、Torch、LLaMA-Factory、vLLM 的自检信息

## 5. 目录准备现在会做什么

`bash scripts/00_prepare_aliyun_layout.sh` 会自动处理这些事：

- 创建目录：
  - `/root/soulv_assets/models`
  - `/root/soulv_assets/runs/checkpoints`
  - `/root/soulv_assets/runs/merged`
- `/root/llama-factory`

它不再做这些事：

- 不自动注册 dataset
- 不自动复制训练数据
- 不假设 `Qwen3-32B` 权重已经下载
- 不假设 stage1 / stage2 的 32B 产物已经存在

如果你将来想把“真实模型目录”挂到仓库约定路径，可以这样做：

```bash
MODEL_SOURCE_PATH=/root/model_store/Qwen3-32B \
bash /root/soulv_llm/scripts/00_prepare_aliyun_layout.sh
```

如果你将来想把训练数据目录挂到 LLaMA-Factory 默认数据目录，也可以这样做：

```bash
DATA_SOURCE_PATH=/root/data/llamafactory \
bash /root/soulv_llm/scripts/00_prepare_aliyun_layout.sh
```

数据注册这一步这次已经从脚本里去掉了，后面你有新数据时自己注册即可。

## 6. 模型路径

当前仓库里的 32B 相关配置，约定基座路径是：

```text
/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B
```

这只是“约定路径”，不是当前服务器已存在的事实。

开始训练或推理前，你只需要保证二选一：

1. 真实权重就放在这个路径
2. 你把这个路径做成软链接，指向真实权重目录

如果你的实际模型目录不同，有两种做法：

1. 保持仓库配置不变，用软链接对齐到这个路径
2. 直接改仓库现有配置里的 `model_name_or_path` 和 `TOKENIZER_PATH`

当前会受影响的地方主要有：

- `configs/llamafactory_stage1_32b_smoke_sft.yaml`
- `configs/llamafactory_stage1_32b_formal_sft.yaml`
- `configs/llamafactory_stage1_32b_merge_for_stage2.yaml`
- `scripts/03_run_vllm_api.sh`

## 7. 当前阶段建议只做环境验证

你现在还没有下载 32B 权重，也没有准备训练数据，所以当前更合理的目标不是直接开训，而是只确认：

- `lf` 环境能创建成功
- `qwen` 环境能创建成功
- `nvidia-smi`、`torch.cuda`、`vllm` 基础自检正常
- 路径约定和软链接方案已经定下来

先进入训练环境：

```bash
source /root/miniconda3/bin/activate lf
```

先只做基础检查，不直接跑 32B 训练：

```bash
bash /root/soulv_llm/scripts/00_doctor_envs.sh
```

等将来 32B 权重和训练数据都到位后，再进入正式训练。

## 8. 推理怎么跑

先进入推理环境：

```bash
source /root/miniconda3/bin/activate qwen
```

`scripts/03_run_vllm_api.sh` 现在也改了：

- 不再默认假设 32B 模型已经存在
- 启服务时如果 `MODEL_PATH` 不存在，会直接报错退出
- 如果你没传 `TOKENIZER_PATH`，默认回退到 `MODEL_PATH`

所以真正启动 vLLM 时，需要显式给模型路径，或者先把软链接建好。

比如未来基座下载完后可以这样起：

```bash
MODEL_VARIANT=32b \
MODEL_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B \
TOKENIZER_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B \
HOST=127.0.0.1 \
PORT=8000 \
bash scripts/03_run_vllm_api.sh
```

如果以后是拿 merge 后模型起服务，就把 `MODEL_PATH` 换成 merge 输出目录。

## 9. 前端网关

```bash
HOST=0.0.0.0 \
PORT=7860 \
UPSTREAM_VLLM_BASE_URL=http://127.0.0.1:8000 \
DEFAULT_MODEL_NAME=qwen3_32b_official \
bash scripts/07_run_frontend_gateway.sh
```

## 10. 首轮建议

第一次建议按下面顺序：

1. `bash scripts/00_doctor_envs.sh`
2. 确认两张 `L20 48G` 都正常
3. 装稳 `lf`
4. 装稳 `qwen`
5. 再决定权重下载路径和未来软链接路径
6. 等数据和模型到位后，再开始 32B smoke train / vLLM serve

这样能最快确定问题到底出在环境层，而不是训练配置层：

- GPU / driver
- torch / deepspeed
- LLaMA-Factory
- vLLM
- 路径约定

## 11. 常见调整

如果你要改版本，优先改这里：

- `requirements-lf.txt`
- `requirements-lf-flashattn.txt`
- `requirements-qwen.txt`

如果你要改机器路径，优先改这里：

- `scripts/00_prepare_aliyun_layout.sh`
- `scripts/03_run_vllm_api.sh`
- `configs/llamafactory_stage1_32b_*.yaml`
- `configs/llamafactory_stage2_32b_*.yaml`

## 12. 官方依据

截至 `2026-04-20`，这套方案主要对齐以下官方文档：

- LLaMA-Factory 安装与发布说明：
  - https://github.com/hiyouga/LLaMA-Factory
  - https://github.com/hiyouga/LLaMA-Factory/releases
  - https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html
- vLLM 安装与服务：
  - https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html
  - https://docs.vllm.ai/en/stable/cli/serve/
- Qwen 官方关于 vLLM / function calling：
  - https://qwen.readthedocs.io/en/stable/deployment/vllm.html
  - https://qwen.readthedocs.io/en/stable/framework/function_call.html
