#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-stage2_amap}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

case "$TARGET" in
  stage2_amap)
    CONFIG="configs/llamafactory_stage2_merge_for_deploy.yaml"
    ;;
  stage2_amap_32b)
    CONFIG="configs/llamafactory_stage2_32b_merge_for_deploy.yaml"
    ;;
  *)
    if [[ -f "$TARGET" ]]; then
      CONFIG="$TARGET"
    else
      echo "Usage: bash scripts/06_merge_stage2_for_deploy.sh [stage2_amap|stage2_amap_32b|config_path]"
      exit 1
    fi
    ;;
esac

echo "Using merge config: $CONFIG"
llamafactory-cli export "$CONFIG"
