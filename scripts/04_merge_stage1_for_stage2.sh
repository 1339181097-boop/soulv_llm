#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/llamafactory_stage1_merge_for_stage2.yaml}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

echo "Using merge config: $CONFIG"
llamafactory-cli export "$CONFIG"
