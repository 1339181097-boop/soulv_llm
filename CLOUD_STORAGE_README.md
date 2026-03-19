# 云端存储说明

## 一、目的

这份文档用于说明 `soulv_llm` 在云端服务器上的目录组织方式，方便团队成员快速理解：

- 代码放哪里
- 模型放哪里
- 最终数据放哪里
- 训练产物放哪里

核心原则只有一句话：

- `/root/soulv_llm` 只放代码和轻量配置
- 所有大体积运行资产统一放到 `/root/soulv_assets`

这样做可以保证仓库干净、Git 轻量、训练和部署更容易维护。

补充说明：

- 数据清洗和处理过程主要在本地完成
- 云端常驻保留“最终数据”
- 云端默认长期保留代码、模型缓存、最终数据和训练产物

## 二、推荐目录结构

```text
/root/
├── soulv_llm/
└── soulv_assets/
    ├── models/
    │   ├── modelscope/
    │   └── huggingface/
    ├── dataset/
    └── runs/
        ├── checkpoints/
        ├── logs/
        └── exports/
```

## 三、各目录分别存什么

### 1. 代码仓库

路径：

```text
/root/soulv_llm
```

这里存放：

- 项目源码
- 脚本
- 配置文件
- 文档
- 轻量级示例文件

这里不存放：

- 模型权重
- 大数据集
- 训练输出产物

### 2. 模型缓存

路径：

```text
/root/soulv_assets/models/modelscope
/root/soulv_assets/models/huggingface
```

这里存放：

- 从 ModelScope 下载的模型缓存
- 从 Hugging Face 下载的模型缓存
- tokenizer 文件
- 模型配置文件
- 权重分片文件

说明：

- `modelscope/` 用来放 ModelScope 缓存
- `huggingface/` 用来放 Hugging Face 缓存
- 这些目录属于运行资产，不属于 Git 仓库内容

### 3. 最终数据集

路径：

```text
/root/soulv_assets/dataset
```

这里存放：

- 最终训练数据
- 最终评测数据
- 已经整理好的可直接上云使用的数据文件

当前约定是：

- 原始数据处理主要在本地完成
- 云端不保留 `raw/` 和 `processed/` 中间过程目录
- 云端长期保留最终可训练数据

### 4. 训练产物

路径：

```text
/root/soulv_assets/runs/checkpoints
/root/soulv_assets/runs/logs
/root/soulv_assets/runs/exports
```

这里存放：

- LoRA checkpoint
- 训练日志
- 合并后的模型
- 部署前导出的模型文件

## 四、为什么这样组织

这样拆分有几个明显好处：

- 代码仓库保持干净，不会被大文件拖慢
- 不容易误把模型、数据、checkpoint 提交进 Git
- 模型、最终数据、训练产物更容易单独迁移和清理
- VS Code 里仍然只需要打开 `/root/soulv_llm`
- 代码和运行资产分离后，团队协作会更清晰
- 本地负责数据生产，云端负责训练和部署，职责更明确

## 五、推荐环境变量

建议在云端服务器的 `~/.bashrc` 中加入：

```bash
export MODELSCOPE_CACHE=/root/soulv_assets/models/modelscope
export HF_HOME=/root/soulv_assets/models/huggingface
export SOULV_DATASET_DIR=/root/soulv_assets/dataset
export SOULV_RUNS_DIR=/root/soulv_assets/runs
```

加入后执行：

```bash
source ~/.bashrc
```

这样以后：

- 模型缓存会优先走新目录
- 训练脚本可以统一读取数据目录
- 训练输出可以统一写入 runs 目录

## 六、典型使用方式

### 1. 推理服务

例如启动 `vLLM` 时：

```bash
conda activate qwen
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90
```

此时模型会从你设置好的缓存目录中读取，而不是继续使用默认的 `/root/.cache/...`。

### 2. 数据文件

例如最终训练数据可以放在：

```text
/root/soulv_assets/dataset/soulv_mixed_sft.json
```

### 3. 训练输出

例如某次训练可以输出到：

```text
/root/soulv_assets/runs/checkpoints/qwen3_8b_sft_001
```

对应日志可以放在：

```text
/root/soulv_assets/runs/logs/qwen3_8b_sft_001
```

## 七、团队协作约定

团队成员在云端使用本项目时，统一遵守以下规则：

- 不要把模型权重放进 `/root/soulv_llm`
- 不要把数据集和 checkpoint 提交进 Git
- 不要把缓存目录当成源码目录使用
- 代码放仓库，运行资产放仓库外
- 本地负责数据清洗，云端保留最终数据用于训练和部署

## 八、新机器初始化清单

当新开一台云端机器时，建议按下面顺序初始化：

1. 把代码仓库 clone 到 `/root/soulv_llm`
2. 创建 `/root/soulv_assets`
3. 配置 `MODELSCOPE_CACHE`、`HF_HOME`、`SOULV_DATASET_DIR`、`SOULV_RUNS_DIR`
4. 把最终数据同步到 `/root/soulv_assets/dataset`
5. 下载或迁移模型到 `/root/soulv_assets/models`
6. 后续训练产物统一输出到 `/root/soulv_assets/runs`

## 九、一句话总结

云端机器上，`/root/soulv_llm` 只负责放代码；模型、最终数据、训练产物全部放到 `/root/soulv_assets`。
