#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/soulv_llm}"
LLAMAFACTORY_HOME="${LLAMAFACTORY_HOME:-/root/llama-factory}"
ASSETS_ROOT="${ASSETS_ROOT:-/root/soulv_assets}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python is required to merge dataset_info.json."
    exit 1
  fi
fi

mkdir -p \
  "$LLAMAFACTORY_HOME/data" \
  "$ASSETS_ROOT/models" \
  "$ASSETS_ROOT/runs/checkpoints" \
  "$ASSETS_ROOT/runs/merged"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/register_llamafactory_datasets.py" \
  --target "$LLAMAFACTORY_HOME/data/dataset_info.json" \
  --source "$PROJECT_ROOT/configs/llamafactory_dataset_info_stage1_general_sft.json" \
  --source "$PROJECT_ROOT/configs/llamafactory_dataset_info_stage2_amap_tool.json" \
  --copy-dataset "$PROJECT_ROOT/data/final/stage1_general_sft.json" \
  --copy-dataset "$PROJECT_ROOT/data/final/stage2_amap_tool_use_sft.json"

echo "Prepared remote layout:"
echo "  project_root=$PROJECT_ROOT"
echo "  llamafactory_home=$LLAMAFACTORY_HOME"
echo "  assets_root=$ASSETS_ROOT"
