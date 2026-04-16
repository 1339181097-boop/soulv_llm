#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/llamafactory_stage2_merge_for_deploy.yaml}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

echo "Using merge config: $CONFIG"
llamafactory-cli export "$CONFIG"
