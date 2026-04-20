#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-stage1}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

case "$TARGET" in
  stage1)
    CONFIG="configs/llamafactory_stage1_merge_for_stage2.yaml"
    ;;
  stage1_32b)
    CONFIG="configs/llamafactory_stage1_32b_merge_for_stage2.yaml"
    ;;
  *)
    if [[ -f "$TARGET" ]]; then
      CONFIG="$TARGET"
    else
      echo "Usage: bash scripts/04_merge_stage1_for_stage2.sh [stage1|stage1_32b|config_path]"
      exit 1
    fi
    ;;
esac

echo "Using merge config: $CONFIG"
llamafactory-cli export "$CONFIG"
