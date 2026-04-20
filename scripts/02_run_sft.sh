#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-stage1}"
PROFILE="${2:-formal}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

USE_DISTRIBUTED=0

case "$STAGE" in
  stage1)
    CONFIG="configs/llamafactory_stage1_sft.yaml"
    ;;
  stage2_amap)
    CONFIG="configs/llamafactory_stage2_amap_tool_use_sft.yaml"
    ;;
  stage1_32b)
    USE_DISTRIBUTED=1
    case "$PROFILE" in
      smoke)
        CONFIG="configs/llamafactory_stage1_32b_smoke_sft.yaml"
        ;;
      formal)
        CONFIG="configs/llamafactory_stage1_32b_formal_sft.yaml"
        ;;
      *)
        echo "Usage: bash scripts/02_run_sft.sh stage1_32b [smoke|formal]"
        exit 1
        ;;
    esac
    ;;
  stage2_amap_32b)
    USE_DISTRIBUTED=1
    case "$PROFILE" in
      smoke)
        CONFIG="configs/llamafactory_stage2_32b_amap_tool_use_smoke_sft.yaml"
        ;;
      formal)
        CONFIG="configs/llamafactory_stage2_32b_amap_tool_use_formal_sft.yaml"
        ;;
      *)
        echo "Usage: bash scripts/02_run_sft.sh stage2_amap_32b [smoke|formal]"
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Usage: bash scripts/02_run_sft.sh [stage1|stage2_amap|stage1_32b|stage2_amap_32b] [formal|smoke]"
    exit 1
    ;;
esac

if [[ "$USE_DISTRIBUTED" == "1" ]]; then
  export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
  export NNODES="${NNODES:-1}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
fi

echo "Using config: $CONFIG"
echo "Profile: $PROFILE"
echo "Distributed: $USE_DISTRIBUTED"

if [[ "$USE_DISTRIBUTED" == "1" ]]; then
  echo "Torchrun env: FORCE_TORCHRUN=$FORCE_TORCHRUN NPROC_PER_NODE=$NPROC_PER_NODE NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
fi

llamafactory-cli train "$CONFIG"
