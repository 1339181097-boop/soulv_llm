#!/usr/bin/env bash
set -euo pipefail

CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
LF_ENV_NAME="${LF_ENV_NAME:-lf}"
QWEN_ENV_NAME="${QWEN_ENV_NAME:-qwen}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU]"
  nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
  echo
fi

echo "[lf]"
"$CONDA_EXE" run -n "$LF_ENV_NAME" python - <<'PY'
import importlib.metadata as md
import torch

packages = [
    "llamafactory",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "deepspeed",
    "bitsandbytes",
]

print("python ok")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
for name in packages:
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY
echo

echo "[qwen]"
"$CONDA_EXE" run -n "$QWEN_ENV_NAME" python - <<'PY'
import importlib.metadata as md
import torch

print("python ok")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
for name in ("vllm",):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY
