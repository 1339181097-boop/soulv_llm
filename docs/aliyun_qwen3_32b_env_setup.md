# 阿里云 Qwen3-32B 双环境安装指南

这份文档对应当前仓库的正式主线：

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
  - `bitsandbytes 0.43.1`
- `qwen`
  - `Python 3.11`
  - `vllm>=0.9.0,<0.10.0`

说明：

- `flash-attn` 被拆成可选安装，默认不强装。原因不是它没用，而是它在云端经常因为 CUDA toolchain 编译问题拖慢首轮落地。
- 如果后续你确认机器驱动、CUDA、编译链都稳定，再执行 `INSTALL_FLASH_ATTN=1 bash scripts/00_setup_lf_env.sh` 即可。

## 2. 为什么这样拆

这套拆法直接对应官方要求：

- LLaMA-Factory 最新发布版已经把 Python 基线抬到 `3.11+`
- vLLM 官方安装文档明确建议使用“全新隔离环境”
- vLLM 官方还特别提醒：如果用 conda，也不要把 conda 装的 `pytorch` 和 pip 装的 `vllm` 混在同一个运行时里，否则可能因为 NCCL 差异导致问题
- Qwen 官方文档说明：
  - `Qwen3` 在 `vLLM >= 0.8.5` 下支持 Hermes function calling
  - `Qwen3` 的 reasoning parser 需要 `vLLM >= 0.9.0`

所以这里不把 `vllm` 装进 `lf`，而是单独放在 `qwen` 环境里。

## 3. 一次性执行顺序

先在云端进入项目目录：

```bash
cd /root/soulv_llm
```

然后按顺序执行：

```bash
bash scripts/00_setup_lf_env.sh
bash scripts/00_setup_qwen_env.sh
bash scripts/00_prepare_aliyun_layout.sh
bash scripts/00_doctor_envs.sh
```

这四步分别负责：

1. 创建并安装 `lf`
2. 创建并安装 `qwen`
3. 准备 `/root/llama-factory/data` 和 `/root/soulv_assets/...`
4. 打印 GPU、Torch、LLaMA-Factory、vLLM 的自检信息

## 4. 目录准备会做什么

`bash scripts/00_prepare_aliyun_layout.sh` 会自动处理这些事：

- 创建目录：
  - `/root/llama-factory/data`
  - `/root/soulv_assets/models`
  - `/root/soulv_assets/runs/checkpoints`
  - `/root/soulv_assets/runs/merged`
- 合并数据注册：
  - `configs/llamafactory_dataset_info_stage1_general_sft.json`
  - `configs/llamafactory_dataset_info_stage2_amap_tool.json`
- 复制训练数据到远端 LLaMA-Factory 数据目录：
  - `data/final/stage1_general_sft.json`
  - `data/final/stage2_amap_tool_use_sft.json`

所以这一步做完后，当前仓库里 32B 训练配置引用的 `dataset_dir: /root/llama-factory/data` 就能直接对上。

## 5. 模型路径

当前仓库默认约定的基座路径是：

```text
/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B
```

在开始训练或推理前，需要确认这个目录已经准备好。

如果你的实际模型目录不同，有两种做法：

1. 把模型整理到上面这个固定路径
2. 直接改仓库现有配置里的 `model_name_or_path` 和 `TOKENIZER_PATH`

当前会受影响的地方主要有：

- `configs/llamafactory_stage1_32b_smoke_sft.yaml`
- `configs/llamafactory_stage1_32b_formal_sft.yaml`
- `configs/llamafactory_stage1_32b_merge_for_stage2.yaml`
- `scripts/03_run_vllm_api.sh`

## 6. 训练怎么跑

先进入训练环境：

```bash
source /root/miniconda3/bin/activate lf
```

然后按仓库既有主线执行：

```bash
bash scripts/02_run_sft.sh stage1_32b smoke
bash scripts/02_run_sft.sh stage1_32b formal
bash scripts/04_merge_stage1_for_stage2.sh stage1_32b
bash scripts/02_run_sft.sh stage2_amap_32b smoke
bash scripts/02_run_sft.sh stage2_amap_32b formal
bash scripts/06_merge_stage2_for_deploy.sh stage2_amap_32b
```

默认双卡参数已经写在仓库里，不需要额外改：

- `FORCE_TORCHRUN=1`
- `NPROC_PER_NODE=2`

## 7. 推理怎么跑

先进入推理环境：

```bash
source /root/miniconda3/bin/activate qwen
```

再启动 vLLM：

```bash
HOST=127.0.0.1 \
PORT=8000 \
MODEL_VARIANT=32b \
bash scripts/03_run_vllm_api.sh
```

如果要开前端网关：

```bash
HOST=0.0.0.0 \
PORT=7860 \
UPSTREAM_VLLM_BASE_URL=http://127.0.0.1:8000 \
DEFAULT_MODEL_NAME=qwen3_32b_stage2_amap_tool_use \
bash scripts/07_run_frontend_gateway.sh
```

## 8. 首轮建议

第一次不要直接冲 full train，建议按下面顺序：

1. `bash scripts/00_doctor_envs.sh`
2. `conda activate lf`
3. `bash scripts/02_run_sft.sh stage1_32b smoke`
4. `conda activate qwen`
5. 用 `bash scripts/03_run_vllm_api.sh` 拉起 32B 服务
6. 跑通 `src/eval/` 和 `src/tool_eval/` 的最小验证

这样能最快确定问题到底出在：

- GPU / driver
- torch / deepspeed
- LLaMA-Factory
- vLLM
- 模型路径
- 数据注册

## 9. 常见调整

如果你要改版本，优先改这里：

- `requirements-lf.txt`
- `requirements-lf-flashattn.txt`
- `requirements-qwen.txt`

如果你要改机器路径，优先改这里：

- `scripts/00_prepare_aliyun_layout.sh`
- `scripts/03_run_vllm_api.sh`
- `configs/llamafactory_stage1_32b_*.yaml`
- `configs/llamafactory_stage2_32b_*.yaml`

## 10. 官方依据

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
