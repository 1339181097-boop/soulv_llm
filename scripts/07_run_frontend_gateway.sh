#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
UPSTREAM_VLLM_BASE_URL="${UPSTREAM_VLLM_BASE_URL:-http://127.0.0.1:8000}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-qwen3_8b_stage2_amap_tool_use}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-300}"
UPSTREAM_VLLM_API_KEY="${UPSTREAM_VLLM_API_KEY:-}"

cmd=(
  python -m src.deploy.frontend_server
  --host "$HOST"
  --port "$PORT"
  --upstream-base-url "$UPSTREAM_VLLM_BASE_URL"
  --default-model "$DEFAULT_MODEL_NAME"
  --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS"
)

if [[ -n "$UPSTREAM_VLLM_API_KEY" ]]; then
  cmd+=(--upstream-api-key "$UPSTREAM_VLLM_API_KEY")
fi

echo "Launching frontend gateway with:"
echo "  host=$HOST port=$PORT"
echo "  upstream_vllm_base_url=$UPSTREAM_VLLM_BASE_URL"
echo "  default_model_name=$DEFAULT_MODEL_NAME"
printf '  command='
printf '%q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
