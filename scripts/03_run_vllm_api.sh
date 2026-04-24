#!/usr/bin/env bash
set -euo pipefail

MODEL_VARIANT="${MODEL_VARIANT:-custom}"

case "${MODEL_VARIANT,,}" in
  8b)
    DEFAULT_MODEL_PATH=""
    DEFAULT_TOKENIZER_PATH=""
    DEFAULT_SERVED_MODEL_NAME="qwen3_8b_stage2_amap_tool_use"
    DEFAULT_TENSOR_PARALLEL_SIZE="1"
    ;;
  32b)
    DEFAULT_MODEL_PATH=""
    DEFAULT_TOKENIZER_PATH=""
    DEFAULT_SERVED_MODEL_NAME="qwen3_32b_stage2_amap_tool_use"
    DEFAULT_TENSOR_PARALLEL_SIZE="2"
    ;;
  custom)
    DEFAULT_MODEL_PATH=""
    DEFAULT_TOKENIZER_PATH=""
    DEFAULT_SERVED_MODEL_NAME="qwen_custom_model"
    DEFAULT_TENSOR_PARALLEL_SIZE="1"
    ;;
  *)
    echo "Unsupported MODEL_VARIANT: $MODEL_VARIANT"
    echo "Expected one of: custom, 8b, 32b"
    exit 1
    ;;
esac

MODEL_PATH="${1:-${MODEL_PATH:-$DEFAULT_MODEL_PATH}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$DEFAULT_SERVED_MODEL_NAME}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TENSOR_PARALLEL_SIZE}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
GENERATION_CONFIG_MODE="${GENERATION_CONFIG_MODE:-vllm}"
ENABLE_REASONING_PARSER="${ENABLE_REASONING_PARSER:-1}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
ENABLE_CORS="${ENABLE_CORS:-0}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required."
  echo "Example:"
  echo "  MODEL_VARIANT=32b MODEL_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B \\"
  echo "  TOKENIZER_PATH=/root/soulv_assets/models/modelscope/models/Qwen/Qwen3-32B \\"
  echo "  SERVED_MODEL_NAME=qwen3_32b_base PORT=8000 bash scripts/03_run_vllm_api.sh"
  exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH"
  echo "Download the model first or create the expected symlink before serving."
  exit 1
fi

if [[ -n "$TOKENIZER_PATH" && ! -e "$TOKENIZER_PATH" ]]; then
  echo "TOKENIZER_PATH does not exist: $TOKENIZER_PATH"
  exit 1
fi

if [[ -z "$TOKENIZER_PATH" ]]; then
  TOKENIZER_PATH="$MODEL_PATH"
fi

cmd=(
  vllm serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tokenizer "$TOKENIZER_PATH"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --generation-config "$GENERATION_CONFIG_MODE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ "$ENABLE_REASONING_PARSER" == "1" ]]; then
  cmd+=(--reasoning-parser "$REASONING_PARSER")
fi

# vLLM auto tool choice needs both the switch and a parser. We default to the
# Hermes parser because the model family is Qwen-based and this repo already
# captured valid tool_calls on earlier stage2 baseline runs.
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
  cmd+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ -n "$CHAT_TEMPLATE" ]]; then
  cmd+=(--chat-template "$CHAT_TEMPLATE")
fi

if [[ "$ENABLE_CORS" == "1" ]]; then
  cmd+=(--allowed-origins "*" --allowed-methods "*" --allowed-headers "*")
fi

if [[ -n "$VLLM_API_KEY" ]]; then
  cmd+=(--api-key "$VLLM_API_KEY")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $EXTRA_ARGS )
  cmd+=("${extra_args[@]}")
fi

echo "Launching vLLM with:"
echo "  model_variant=$MODEL_VARIANT"
echo "  model_path=$MODEL_PATH"
echo "  tokenizer_path=$TOKENIZER_PATH"
echo "  host=$HOST port=$PORT"
echo "  served_model_name=$SERVED_MODEL_NAME"
echo "  tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
echo "  enable_auto_tool_choice=$ENABLE_AUTO_TOOL_CHOICE parser=${TOOL_CALL_PARSER:-<none>}"
echo "  generation_config=$GENERATION_CONFIG_MODE"
printf '  command='
printf '%q ' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
