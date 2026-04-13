#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/path/to/merged-or-base-plus-lora-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Example stage2 vLLM command:"
echo "vllm serve $MODEL_PATH --host $HOST --port $PORT --max-model-len 8192"
