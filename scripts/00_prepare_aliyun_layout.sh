#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
LLAMAFACTORY_HOME="${LLAMAFACTORY_HOME:-/root/llama-factory}"
ASSETS_ROOT="${ASSETS_ROOT:-/root/soulv_assets}"
MODEL_LINK_PATH="${MODEL_LINK_PATH:-$ASSETS_ROOT/models/modelscope/models/Qwen/Qwen3-32B}"
MODEL_SOURCE_PATH="${MODEL_SOURCE_PATH:-}"
DATA_LINK_PATH="${DATA_LINK_PATH:-$LLAMAFACTORY_HOME/data}"
DATA_SOURCE_PATH="${DATA_SOURCE_PATH:-}"

ensure_symlink() {
  local target="$1"
  local link_path="$2"

  mkdir -p "$(dirname "$link_path")"

  if [[ -L "$link_path" ]]; then
    local current_target
    current_target="$(readlink "$link_path")"
    if [[ "$current_target" == "$target" ]]; then
      echo "[OK] symlink already exists: $link_path -> $target"
      return 0
    fi
    rm -f "$link_path"
  elif [[ -e "$link_path" ]]; then
    echo "[ERROR] path exists and is not a symlink: $link_path"
    echo "Remove it manually or choose another *_LINK_PATH."
    exit 1
  fi

  ln -s "$target" "$link_path"
  echo "[OK] created symlink: $link_path -> $target"
}

mkdir -p \
  "$ASSETS_ROOT/models" \
  "$ASSETS_ROOT/runs/checkpoints" \
  "$ASSETS_ROOT/runs/merged" \
  "$LLAMAFACTORY_HOME"

if [[ -n "$MODEL_SOURCE_PATH" ]]; then
  ensure_symlink "$MODEL_SOURCE_PATH" "$MODEL_LINK_PATH"
else
  mkdir -p "$(dirname "$MODEL_LINK_PATH")"
  echo "[INFO] MODEL_SOURCE_PATH is empty, skipped model symlink."
  echo "       Example:"
  echo "       MODEL_SOURCE_PATH=/root/model_store/Qwen3-32B bash $SCRIPT_DIR/00_prepare_aliyun_layout.sh"
fi

if [[ -n "$DATA_SOURCE_PATH" ]]; then
  ensure_symlink "$DATA_SOURCE_PATH" "$DATA_LINK_PATH"
else
  mkdir -p "$DATA_LINK_PATH"
  echo "[INFO] DATA_SOURCE_PATH is empty, kept plain directory: $DATA_LINK_PATH"
  echo "       Future training data can be linked later without touching dataset registration."
fi

echo "Prepared remote layout:"
echo "  project_root=$PROJECT_ROOT"
echo "  llamafactory_home=$LLAMAFACTORY_HOME"
echo "  assets_root=$ASSETS_ROOT"
echo "  model_link_path=$MODEL_LINK_PATH"
echo "  data_link_path=$DATA_LINK_PATH"
