#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-qwen}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

if [[ -z "${VLLM_PACKAGE_SPEC:-}" ]]; then
  if [[ ! -f "$PROJECT_ROOT/requirements-qwen.txt" ]]; then
    echo "Missing requirements file: $PROJECT_ROOT/requirements-qwen.txt"
    exit 1
  fi
  VLLM_PACKAGE_SPEC="$(grep -Ev '^[[:space:]]*($|#)' "$PROJECT_ROOT/requirements-qwen.txt" | paste -sd ' ' -)"
fi

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "conda executable not found: $CONDA_EXE"
  echo "Set CONDA_EXE explicitly, for example: CONDA_EXE=/root/miniconda3/bin/conda"
  exit 1
fi

if ! "$CONDA_EXE" run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  "$CONDA_EXE" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

"$CONDA_EXE" run -n "$ENV_NAME" python -m pip install \
  --upgrade \
  -i "$PIP_INDEX_URL" \
  --trusted-host "$PIP_TRUSTED_HOST" \
  pip setuptools wheel uv

ENV_PYTHON="$("$CONDA_EXE" run -n "$ENV_NAME" python -c "import sys; print(sys.executable)")"

if ! UV_DEFAULT_INDEX="$UV_DEFAULT_INDEX" UV_TORCH_BACKEND="$UV_TORCH_BACKEND" \
  "$CONDA_EXE" run -n "$ENV_NAME" uv pip install --python "$ENV_PYTHON" $VLLM_PACKAGE_SPEC; then
  echo "uv installation failed, falling back to pip install $VLLM_PACKAGE_SPEC"
  "$CONDA_EXE" run -n "$ENV_NAME" python -m pip install \
    -i "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    $VLLM_PACKAGE_SPEC
fi

echo "qwen environment is ready."
echo "  conda env: $ENV_NAME"
echo "  python: $PYTHON_VERSION"
echo "  vllm spec: $VLLM_PACKAGE_SPEC"
echo "  uv index: $UV_DEFAULT_INDEX"
echo "  uv torch backend: $UV_TORCH_BACKEND"
