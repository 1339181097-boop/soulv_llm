#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
UPSTREAM_VLLM_BASE_URL="${UPSTREAM_VLLM_BASE_URL:-http://127.0.0.1:8000}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-qwen3_32b_official}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-300}"
UPSTREAM_VLLM_API_KEY="${UPSTREAM_VLLM_API_KEY:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cmd=(
  "$PYTHON_BIN" -m src.deploy.frontend_server
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
echo "  project_root=$PROJECT_ROOT"
printf '  command='
printf '%q ' "${cmd[@]}"
printf '\n'

cd "$PROJECT_ROOT"
exec "${cmd[@]}"
