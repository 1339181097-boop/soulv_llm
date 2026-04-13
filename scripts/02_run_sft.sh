#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-stage1}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

case "$STAGE" in
  stage1)
    CONFIG="configs/llamafactory_stage1_sft.yaml"
    ;;
  stage2_amap)
    CONFIG="configs/llamafactory_stage2_amap_tool_use_sft.yaml"
    ;;
  *)
    echo "Usage: bash scripts/02_run_sft.sh [stage1|stage2_amap]"
    exit 1
    ;;
esac

echo "Using config: $CONFIG"
llamafactory-cli train "$CONFIG"
